"""Server-only source acquisition and statistical contract audit; no model calls."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import urllib.request

import numpy as np
from scipy.special import betainc
from scipy.stats import binom, beta


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(url, target, maximum=30_000_000):
    request = urllib.request.Request(url, headers={'User-Agent': 'watermark2-research-audit/1.0'})
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError('Source exceeds the bounded metadata/data budget')
    target.write_bytes(data)
    return dict(url=url, file=target.name, bytes=len(data), sha256=sha(target))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    assert str(args.root.resolve()) == '/data0/liyanlei/watermark_research'
    out = args.root / 'runs/phase1_contract_20261113'
    out.mkdir(exist_ok=False)
    sources, errors = {}, {}
    urls = {
        'sembind_icml2026.pdf': 'https://raw.githubusercontent.com/mlresearch/v306/main/assets/zhang26hj/zhang26hj.pdf',
        'gs_watermark.py': 'https://raw.githubusercontent.com/bsmhmmlf/Gaussian-Shading/09c678fadc7545acf7be12647ddf2a5e66f6a9dc/watermark.py',
        'gs_run.sh': 'https://raw.githubusercontent.com/bsmhmmlf/Gaussian-Shading/09c678fadc7545acf7be12647ddf2a5e66f6a9dc/scripts/run.sh',
        'sembind_tree.json': 'https://api.github.com/repos/XinZhang1999/SemBind/git/trees/main?recursive=1',
        'sdp_metadata.json': 'https://huggingface.co/api/datasets/Gustavosta/Stable-Diffusion-Prompts',
    }
    for name, url in urls.items():
        try:
            sources[name] = fetch(url, out/name)
            print('SOURCE', name, sources[name]['bytes'], flush=True)
        except Exception as exc:
            errors[name] = repr(exc)
            print('SOURCE_UNAVAILABLE', name, type(exc).__name__, flush=True)
    if 'sdp_metadata.json' in sources:
        metadata = json.loads((out/'sdp_metadata.json').read_text())
        revision = metadata['sha']
        paths = [x['rfilename'] for x in metadata['siblings']
                 if x['rfilename'].startswith('data/test-') and x['rfilename'].endswith('.parquet')]
        for index, path in enumerate(paths):
            name = f'sdp_test_{index}.parquet'
            url = f'https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts/resolve/{revision}/{path}'
            try:
                sources[name] = fetch(url, out/name)
                print('SOURCE', name, sources[name]['bytes'], flush=True)
            except Exception as exc:
                errors[name] = repr(exc)
    n = 256
    nominal = 1e-6
    minimum = next(k for k in range(n+1) if binom.sf(k-1, n, .5) <= nominal)
    official_i = next(i for i in range(n) if betainc(i+1, n-i, .5) <= nominal)
    tie_probability = math.comb(64, 32) / 2**64
    decoded_one = (1-tie_probability)/2
    fixed_messages = []
    for ones in [0, 112, 120, 128, 136, 144, 256]:
        # Given an ideal independent decryption stream and a fixed target message.
        pmf = np.convolve(binom.pmf(np.arange(ones+1), ones, decoded_one),
                          binom.pmf(np.arange(n-ones+1), n-ones, 1-decoded_one))
        fixed_messages.append(dict(message_ones=ones,
            expected_matching_bits=ones*decoded_one+(n-ones)*(1-decoded_one),
            fpr_at_ideal_minimum=float(pmf[minimum:].sum()),
            fpr_at_official_comparator=float(pmf[official_i:].sum())))
    result = dict(status='complete', time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        host='249', gpu_used=False, model_experiments=0, sources=sources, source_errors=errors,
        available_modules={x: importlib.util.find_spec(x) is not None
                           for x in ['pypdf', 'fitz', 'Crypto', 'pyarrow', 'requests']},
        theoretical_working_point=dict(assumption='256 independent fair agreement bits, averaged over fresh uniform messages',
            target_fpr=nominal, minimum_matching_bits=minimum, strict_ba_threshold=(minimum-1)/n,
            inclusive_ba_threshold=minimum/n, achieved_ideal_fpr=float(binom.sf(minimum-1,n,.5)),
            previous_minimum_matching_bits=182, previous_ideal_fpr=float(binom.sf(181,n,.5))),
        original_gs_comparator=dict(source='gs_watermark.py', computed_index=official_i,
            threshold=official_i/n, comparison='>=', actual_minimum_matching_bits=official_i,
            ideal_fpr=float(binom.sf(official_i-1,n,.5)),
            note='The source selects i using P(K>i), then accepts K>=i; report both without changing upstream.'),
        majority_tie=dict(repetitions=64, tie_probability=tie_probability, decoded_one_probability=decoded_one,
            fixed_message_examples=fixed_messages,
            scope='Ideal null calculation, not empirical natural-image FPR; fresh-message averaging hides per-message bias.'),
        statistical_resolution=dict(zero_of_50_two_sided_cp_upper=float(beta.ppf(.975,1,50)),
            zero_of_100_two_sided_cp_upper=float(beta.ppf(.975,1,100)),
            iid_zero_event_trials_needed_for_one_sided_95_upper_1e6=math.ceil(math.log(.05)/math.log1p(-nominal)),
            warning='Repeated images/keys need clustering; pooled correlated pairs do not supply this number of independent trials.'),
        script_sha256=sha(__file__))
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['theoretical_working_point','original_gs_comparator','statistical_resolution']}), flush=True)


if __name__ == '__main__':
    main()
