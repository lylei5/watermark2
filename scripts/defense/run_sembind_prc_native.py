"""Paper-spec SemBind PRC-S using released masker, PRC and exact DPM components."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--shard', type=int, choices=range(3), required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    root = Path('/data0/liyanlei/watermark_research')
    assert run.parent == root / 'runs'
    plan = json.loads((run / 'protocol.json').read_text())
    for name, digest in plan['local_sha256'].items():
        assert sha(run / name) == digest, name
    out = run / f'native_{args.shard}'
    out.mkdir(exist_ok=False)
    state = dict(status='preflight', pid=os.getpid(), protocol_sha256=sha(run / 'protocol.json'))
    rows = []
    started = time.monotonic()
    try:
        gpu = plan['gpu_indices'][args.shard]
        assert gpu in [7, 5, 4]
        fields = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                  '--query-gpu=index,uuid,memory.used,utilization.gpu',
                  '--format=csv,noheader,nounits'], text=True).strip().split(', ')
        assert int(fields[0]) == gpu and int(fields[2]) < 100 and int(fields[3]) == 0, fields
        os.environ.update(CUDA_VISIBLE_DEVICES=fields[1], HF_HOME=str(root / 'cache/huggingface'),
                          HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4',
                          NUMBA_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
        sys.path.insert(0, str(root / 'envs/phase1_prc_extra'))
        import numpy as np
        import torch
        from PIL import Image
        from sembind_binding import SemBindMask
        from semantic_code import SemanticCodeExtractor
        from sembind_official_audit import load_official
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        paths = {k: Path(v) for k, v in plan['paths'].items()}
        for name, digest in plan['official_sha256'].items():
            assert sha(paths['official'] / name) == digest, name
        assert sha(paths['scheduler']) == plan['scheduler_sha256']
        load('model', paths['official'] / 'model.py')
        prc = load('native_prc', paths['official'] / 'src/prc.py')
        gaussians = load('native_pseudogaussians', paths['official'] / 'src/pseudogaussians.py')
        private = Path(plan['private'])
        assert sha(private / 'keys.pkl') == plan['private_key_sha256']
        assert sha(private / 'binding.json') == plan['binding_key_sha256']
        with (private / 'keys.pkl').open('rb') as stream:
            encoding_key, decoding_key = pickle.load(stream)
        mask = SemBindMask(bytes.fromhex(json.loads((private / 'binding.json').read_text())['key']), .5)
        extractor = SemanticCodeExtractor(paths['semantic'] / 'dinov2-giant', paths['semantic'] / 'sembind_masker.pth')
        inversion, pipe = load_official(SimpleNamespace(**paths))
        pipe.set_progress_bar_config(disable=True)
        state.update(status='running', gpu=gpu, gpu_uuid=fields[1])
        save(out / 'state.json', state)
        (out / 'images').mkdir()
        (out / 'latents').mkdir()

        def code(image):
            return extractor.code(image).detach().cpu().numpy().astype(np.uint8)

        def inverse(image):
            return inversion.exact_inversion(image, prompt='', guidance_scale=3., num_inference_steps=50,
                                             test_num_inference_steps=50, inv_order=0, decoder_inv=True, pipe=pipe).detach()

        for item in (x for x in plan['sources'] if x['task'] % 3 == args.shard):
            task = item['task']
            tick = time.monotonic()
            aux_path = Path(item['auxiliary'])
            assert sha(aux_path) == item['auxiliary_sha256']
            aux = Image.open(aux_path).convert('RGB')
            embedding_code = code(aux)
            secret_path = private / f'{task:03}.npz'
            assert sha(secret_path) == item['private_carrier_sha256']
            with np.load(secret_path) as data:
                initial = torch.tensor(data['initial'], device='cuda', dtype=torch.float32).reshape(1, 4, 64, 64)
                message = data['message'].copy()
                codeword = data['codeword'].copy()
            images = {'null': aux}
            image_paths = {'null': aux_path}
            for arm in ['prc', 'prc_s']:
                latent = initial if arm == 'prc' else initial * torch.tensor(mask.signs(embedding_code), device='cuda')
                pipe.to('cuda')
                generated, _ = pipe(item['prompt'], latents=latent, guidance_scale=7.5, num_inference_steps=50,
                                    height=512, width=512)
                assert not any(generated.nsfw_content_detected or [])
                image = generated.images[0]
                path = out / 'images' / f'{task:03}_{arm}.png'
                image.save(path)
                images[arm], image_paths[arm] = image, path
            for kind, image in images.items():
                query_code = code(image)
                torch.manual_seed(202611130000 + task)
                recovered = inverse(image)
                path = out / 'latents' / f'{task:03}_{kind}.npz'
                np.savez_compressed(path, recovered=recovered.cpu().numpy(), query_code=query_code)
                arms = ['prc', 'prc_s'] if kind == 'null' else [kind]
                for arm in arms:
                    unbound = recovered if arm == 'prc' else recovered * torch.tensor(mask.signs(query_code), device='cuda')
                    flattened = unbound.flatten().cpu().double()
                    posterior = gaussians.recover_posteriors(flattened, variances=1.5)
                    detected = bool(prc.Detect(decoding_key, posterior))
                    decoded = prc.Decode(decoding_key, posterior)
                    decodable = decoded is not None
                    rows.append(dict(task=task, source_task=item['source_task'], dataset=item['dataset'],
                                     image_kind=kind, detector=arm, detected=detected,
                                     decodable=decodable, accepted=detected or decodable,
                                     message_ba=float(np.mean(decoded == message)) if decodable else None,
                                     codeword_ba=float(np.mean((flattened.numpy() > 0) == (codeword > 0))),
                                     semantic_agreement=float(np.mean(embedding_code == query_code)),
                                     image=str(image_paths[kind]), image_sha256=sha(image_paths[kind]),
                                     latent=str(path.relative_to(run)), latent_sha256=sha(path)))
                save(out / 'progress.json', dict(rows=rows))
            print('SOURCE_COMPLETE', task, round(time.monotonic() - tick, 2), flush=True)
        state.update(status='complete', rows=rows, seconds=time.monotonic()-started,
                     completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save(out / 'complete.json', state)
    except Exception as exc:
        state.update(status='failed', error=repr(exc), completed_decisions=len(rows))
        raise
    finally:
        save(out / 'state.json', {k: v for k, v in state.items() if k != 'rows'})


if __name__ == '__main__':
    main()
