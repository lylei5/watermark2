"""Fresh semantic-component measurements with unchanged published distortions."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def stats(values):
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                min=min(values), max=max(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    plan = json.loads((run / 'protocol.json').read_text())
    root = Path(plan['root'])
    assert not (run / 'masker').exists(), 'Fresh output required'
    for name, digest in plan['source_sha256'].items():
        assert sha(run / name) == digest, name
    for name, digest in plan['model_sha256'].items():
        assert sha(root / name) == digest, name
    assert sha(__file__) == plan['source_sha256']['revalidate_masker.py']
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    output = run / 'masker'
    output.mkdir()
    (output / 'transform_work').mkdir()
    os.chdir(output / 'transform_work')
    upstream = module('model', run / 'sembind_official_model.py')
    distortions = module('official_gs_distortions', run / 'gs_image_utils.py')
    adapter = module('sembind_adapter', run / 'sembind_adapter.py')
    dino_path = root / 'models/sembind/dinov2-giant'
    checkpoint = root / 'models/sembind/sembind_masker.pth'
    processor = AutoImageProcessor.from_pretrained(str(dino_path), local_files_only=True)
    dino = AutoModel.from_pretrained(str(dino_path), local_files_only=True).to('cuda:0').eval()
    official = upstream.AttnSupConHashingModel(input_dim=1536, feat_dim=8192, hash_dim=1024)
    official.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    official.to('cuda:0').eval()
    ours = adapter.SemanticMasker(str(dino_path), str(checkpoint), device='cuda:0')

    @torch.no_grad()
    def official_code(image):
        inputs = processor(images=image, return_tensors='pt').to('cuda:0')
        cls = F.normalize(dino(**inputs).last_hidden_state[:, 0, :], dim=1)
        return official(cls, stage='binarize', s=12.0).view(-1).to(torch.int64).cpu()

    rows, clean_codes = [], []
    with (output / 'rows.jsonl').open('w') as log:
        for idx, item in enumerate(plan['masker_inputs']):
            path = root / item['path']
            assert sha(path) == item['sha256']
            image = Image.open(path).convert('RGB')
            clean = official_code(image)
            clean_codes.append(clean)
            for condition in [{'name': 'clean', 'value': None}] + plan['distortions']:
                name, value = condition['name'], condition['value']
                seed = plan['distortion_seed_base'] + idx
                if name == 'clean':
                    query, ob = image, clean
                else:
                    options = SimpleNamespace(**{c['name']: None for c in plan['distortions']})
                    setattr(options, name, value)
                    distortions.set_random_seed(seed)
                    query = distortions.image_distortion(image.copy(), seed, options).convert('RGB')
                    ob = official_code(query)
                mb = ours.code(query).to(torch.int64).cpu()
                assert ob.numel() == mb.numel() == 1024
                row = dict(image_index=idx, input_sha256=item['sha256'], condition=name, value=value,
                           seed=seed, pixels_sha256=hashlib.sha256(query.tobytes()).hexdigest(),
                           official_vs_adapter_hamming=int((ob != mb).sum()),
                           clean_vs_condition_hamming=int((clean != ob).sum()))
                rows.append(row)
                log.write(json.dumps(row) + '\n')
                log.flush()
            print('SOURCE_COMPLETE', idx + 1, len(plan['masker_inputs']), flush=True)
    pairs = [dict(i=i, j=j, hamming=int((clean_codes[i] != clean_codes[j]).sum()))
             for i in range(len(clean_codes)) for j in range(i + 1, len(clean_codes))]
    (output / 'pairs.json').write_text(json.dumps(pairs) + '\n')
    conditions = sorted({r['condition'] for r in rows})
    summary = dict(sources=len(clean_codes), observations=len(rows), pairs=len(pairs),
                   official_vs_adapter_max=max(r['official_vs_adapter_hamming'] for r in rows),
                   conditions={c: stats([r['clean_vs_condition_hamming'] for r in rows if r['condition'] == c]) for c in conditions},
                   cross_image=stats([r['hamming'] for r in pairs]),
                   protocol_sha256=sha(run / 'protocol.json'),
                   output_sha256={n: sha(output / n) for n in ['rows.jsonl', 'pairs.json']},
                   independent_confirmation=False, full_sembind_reproduction=False,
                   semantic_collision_established=False)
    assert len(rows) == 1000 and len(pairs) == 4950
    (output / 'complete.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
