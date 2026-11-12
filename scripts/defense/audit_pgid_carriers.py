"""Exercise all four released PGID carriers on249 CPU, without model inference."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback


def main():
    root = Path('/data0/liyanlei/watermark_research')
    assert root.is_dir() and os.name == 'posix'
    os.environ.update(CUDA_VISIBLE_DEVICES='', NUMBA_NUM_THREADS='4', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    source = root / 'third_party/frozen/PGID_dc4ad96'
    sys.path.insert(0, str(source))
    import torch
    torch.set_num_threads(4)
    from watermarks.gs_watermark import parser as gs_parser
    from watermarks.tr_watermark import parser as tr_parser
    from watermarks.prc_watermark import parser as prc_parser
    from watermarks.t2smark_watermark import parser as t2s_parser
    from watermarks.watermark_utils import SemanticWatermarks, validate, get_detection_threshold
    from utils import get_watermark_args, add_detection_result, set_random_seed, configure_PGID
    args = argparse.ArgumentParser(parents=[gs_parser, tr_parser, prc_parser, t2s_parser],
                                   conflict_handler='resolve').parse_args([])
    private = root / 'private_eval/pgid_carrier_checks_20261113'
    private.mkdir(mode=0o700, exist_ok=False)
    args.prc_key_path = str(private / 'prc_keys.pkl')
    rows = []
    parameters = []
    for name in ['GS', 'TR', 'PRC', 'T2SMark']:
        args.wm_type = name
        for attack_type in ['forgery', 'removal']:
            args.attack_type = attack_type
            configure_PGID(args)
            parameters.append(dict(carrier=name, variant=attack_type, steps=args.num_defense_steps,
                                   k=args.stopping_timestep, skip=args.num_skip_inversion, gamma=args.guidance_strength))
        try:
            set_random_seed(1)
            kwargs = get_watermark_args(args, 'cpu')
            for wm, idx in SemanticWatermarks[name].value.generate_watermark_signals(
                    total_num_signals=2, latent_shape=(1, 4, 64, 64), w_seed=1, **kwargs):
                initial = wm.get_wm_latents()
                z = initial['zT_torch']
                assert list(z.shape) == [1, 4, 64, 64] and z.isfinite().all()
                message = initial.get('message_bits_str_list', [None])[0]
                result = validate(None, wm, None, 50, message_bits_str_initial=message, provided_zT=z)
                result = add_detection_result(result, name, get_detection_threshold(name, 'Manojb/stable-diffusion-2-1-base'))
                numeric = {k: float(v) if v is not None else None for k, v in result.items()
                           if k in ['bit_accuracy', 'p_value', 'l1_norm_w']}
                rows.append(dict(carrier=name, task=idx, status='complete',
                                 accepted=bool(result['detection_successful']), metrics=numeric))
                print(json.dumps(rows[-1]), flush=True)
        except Exception as exc:
            rows.append(dict(carrier=name, status='failed', error=repr(exc), traceback=traceback.format_exc()))
            print(json.dumps(dict(carrier=name, status='failed', error=repr(exc))), flush=True)
    record = dict(rows=rows, native_parameters=parameters,
                  dependencies={n: importlib.metadata.version(n) for n in ['torch', 'numpy', 'galois', 'ldpc', 'diffusers', 'transformers']},
                  source_commit='dc4ad963ffefcb4cf84340084db617472983d3e0',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='CPU latent encode/validate and official parameter dispatch only; no image pipeline or attacks')
    (root / 'runs/three_baselines_20261113/pgid_carrier_checks.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(completed=len([x for x in rows if x['status']=='complete']), failed=len([x for x in rows if x['status']=='failed']))))


if __name__ == '__main__':
    main()
