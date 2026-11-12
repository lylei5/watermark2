"""Paired published distortions, full released DPM inversion, and score decomposition."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--shard', type=int, choices=[0, 1, 2], required=True)
    args = p.parse_args()
    run = args.run.resolve()
    plan = json.loads((run / 'protocol.json').read_text())
    parent = Path(plan['parent_run'])
    paths = {k: Path(v) for k, v in json.loads((parent / 'paths.json').read_text()).items()}
    out = run / f'shard_{args.shard}'
    out.mkdir(exist_ok=False)
    state = dict(status='preflight', pid=os.getpid(), protocol_sha256=sha(run/'protocol.json'))
    try:
        assert sha(__file__) == plan['script_sha256']
        for path, digest in plan['input_sha256'].items():
            assert sha(path) == digest, path
        for path, digest in plan['source_sha256'].items():
            assert sha(path) == digest, path
        assert plan['inversion_steps'] == 50 and plan['decoder_optimization_steps'] == 100
        gpu = plan['gpu_indices'][args.shard]
        assert gpu in [7, 5, 4]
        info = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
            '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
        assert int(info[0]) == gpu and int(info[2]) < 50 and int(info[3]) == 0, info
        os.environ.update(CUDA_VISIBLE_DEVICES=info[1], CUDA_DEVICE_ORDER='PCI_BUS_ID',
            HF_HOME=plan['hf_home'], HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
            PIP_INDEX_URL='https://pypi.tuna.tsinghua.edu.cn/simple/')
        sys.path.insert(0, str(parent/'scripts'))
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import numpy as np
        import torch
        from PIL import Image
        from sembind_official_audit import load_official, carrier
        from codeword_binding import sembind_mask_bits, bit_accuracy
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        spec = importlib.util.spec_from_file_location('official_gs_distortions', plan['distortion_source'])
        distort = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(distort)
        spec = importlib.util.spec_from_file_location('model', paths['official']/'model.py')
        official_model = importlib.util.module_from_spec(spec)
        sys.modules['model'] = official_model
        spec.loader.exec_module(official_model)
        from semantic_code import SemanticCodeExtractor
        model = SemanticCodeExtractor(paths['semantic']/'dinov2-giant', paths['semantic']/'sembind_masker.pth')
        jobs = [j for n, j in enumerate(plan['jobs']) if n % 3 == args.shard]
        for folder in ['images', 'codes', 'inversions', 'work']:
            (out/folder).mkdir()
        os.chdir(out/'work')
        state.update(status='running', stage='query_codes', gpu=gpu, gpu_uuid=info[1], jobs=len(jobs))
        write(out/'state.json', state)
        for job in jobs:
            source = paths['output']/'images'/f"{job['task']:03}_{job['method']}.png"
            options = SimpleNamespace(**{c['name']: None for c in plan['conditions']})
            setattr(options, job['condition'], job['value'])
            distort.set_random_seed(job['seed'])
            query = distort.image_distortion(Image.open(source).convert('RGB'), job['seed'], options).convert('RGB')
            query.save(out/'images'/(job['id']+'.png'))
            code = model.code(query).cpu().numpy().astype(np.uint8)
            assert code.shape == (1024,)
            np.savez_compressed(out/'codes'/(job['id']+'.npz'), sembind_code=code)
        del model, official_model
        gc.collect()
        torch.cuda.empty_cache()
        inversion, pipe = load_official(SimpleNamespace(**paths))
        state.update(stage='full_inversion')
        write(out/'state.json', state)
        rows = []
        for job in jobs:
            task, method, jid = job['task'], job['method'], job['id']
            imagepath = out/'images'/(jid+'.png')
            query = Image.open(imagepath).convert('RGB')
            _, cipher, message, perm = carrier(paths['root'], task)
            stream = cipher ^ np.tile(message, 64)
            codes = {'auxiliary': np.load(paths['output']/'codes'/f'{task:03}_aux.npz')['sembind_code'],
                'clean': np.load(paths['output']/'codes'/f'{task:03}_{method}.npz')['sembind_code'],
                'query': np.load(out/'codes'/(jid+'.npz'))['sembind_code']}
            source_mask = sembind_mask_bits(codes['auxiliary'][None], perm)[0] if method == 'sembind' else np.zeros(16384, dtype=np.uint8)
            for setting, decoder in [('blind_encoder', False), ('blind_decoder', True)]:
                torch.manual_seed(job['seed'])
                z = inversion.exact_inversion(query, prompt='', guidance_scale=3., num_inference_steps=50,
                    test_num_inference_steps=50, inv_order=0, decoder_inv=decoder, pipe=pipe)
                array = z.detach().cpu().numpy()
                assert array.shape == (1, 4, 64, 64) and np.isfinite(array).all()
                latent = out/'inversions'/(jid+'_'+setting+'.npy')
                np.save(latent, array)
                signs = (array.reshape(-1)>0).astype(np.uint8)
                row = dict(**job, setting=setting, image_sha256=sha(imagepath), code_sha256=sha(out/'codes'/(jid+'.npz')),
                    latent=str(latent.relative_to(run)), latent_sha256=sha(latent),
                    auxiliary_query_agreement=float((codes['auxiliary']==codes['query']).mean()),
                    clean_query_agreement=float((codes['clean']==codes['query']).mean()),
                    channel_sign_error=float((signs != (cipher ^ source_mask)).mean()))
                for label, code in codes.items():
                    mask = sembind_mask_bits(code[None], perm)[0] if method == 'sembind' else np.zeros(16384, dtype=np.uint8)
                    ba = float(bit_accuracy(signs ^ mask ^ stream, message)[0])
                    row[label+'_ba'] = ba
                    row[label+'_accepted'] = ba > 181/256
                rows.append(row)
                write(out/'progress.json', dict(rows=rows))
                print('INVERT_COMPLETE', jid, setting, row['query_ba'], flush=True)
        assert len(rows) == len(jobs)*2 == 48
        state.update(status='complete', rows=rows)
        write(out/'complete.json', state)
    except Exception as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        write(out/'state.json', state)


if __name__ == '__main__':
    main()
