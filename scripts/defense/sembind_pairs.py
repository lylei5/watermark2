"""Baseline Imprint attack against SemBind-GS on the frozen 30 pairs (sharded).

For each pair: generate an auxiliary clean image from the pair prompt, build the
semantic sign mask, generate the SemBind-bound GS reference, run the pinned
Imprint optimisation toward that reference, then verify with SemBind unbinding.
"""
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path('/data0/liyanlei/watermark_research')
sys.path.insert(0, str(ROOT / 'scripts/confirmation100'))
sys.path.insert(0, str(ROOT / 'scripts/defense'))
import common100 as m  # noqa: E402
from sembind import SemanticMasker  # noqa: E402
from sembind_smoke import imprint_attack  # noqa: E402


def main(out_dir, shard, nshard, n_tasks, steps, device='cuda:0'):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shard, nshard, n, steps = int(shard), int(nshard), int(n_tasks), int(steps)
    design = m.c.read(ROOT / 'protocols/confirmation100_design_v7.json')
    proxy = dict(design['proxy'])
    provider = m.r.official_provider(proxy)
    from utils.wm.gs_provider import GsProvider
    masker = SemanticMasker(ROOT / 'models/sembind/dinov2-giant',
                            ROOT / 'models/sembind/sembind_masker.pth', device=device)
    rows = []
    for task in design['tasks'][:n]:
        sid = task['id']
        if (sid - 1) % nshard != shard:
            continue
        start = time.time()
        m.r.deterministic(task['reference_seed'])
        wm = GsProvider(latent_shape=(1, 4, 64, 64), device=device, dtype=torch.float32,
                        message_width_in_bytes=32, num_replications=64, l=1, offset=0)
        zT = wm.get_wm_latents()['zT_torch']
        torch.manual_seed(task['reference_seed'])
        aux_z = torch.randn(1, 4, 64, 64, device=device)
        prompt = task['reference_prompt']
        aux = provider.generate(prompts=[prompt], num_inference_steps=50, guidance_scale=7.5,
                                latents=aux_z, return_latents=False)['images_PIL'][0]
        sign = masker.sign_mask(aux).to(device)
        ref = provider.generate(prompts=[prompt], num_inference_steps=50, guidance_scale=7.5,
                                latents=sign * zT, return_latents=False)['images_PIL'][0]
        (out / (str(sid) + '_reference.png')).parent.mkdir(exist_ok=True)
        ref.save(out / (str(sid) + '_reference.png'))
        with Image.open(ROOT / task['cover']) as im:
            cover = im.convert('RGB')
        cand = imprint_attack(provider, proxy['snapshot'], cover, ref, steps, device)
        cand.save(out / (str(sid) + '_forged.png'))
        s_hat = masker.sign_mask(cand).to(device)
        z_inv = provider.invert_images(cand, num_inference_steps=50)['zT_torch']
        ba_bound = float(wm.get_accuracies(z_inv)['bit_accuracies'][0])
        ba_unbound = float(wm.get_accuracies(s_hat * z_inv)['bit_accuracies'][0])
        row = dict(task_id=sid, ba_bound=ba_bound, ba_unbound=ba_unbound,
                   gs_sembind_pass=bool(ba_unbound > 0.70703125),
                   gs_vanilla_pass=bool(ba_bound > 0.70703125),
                   seconds=time.time() - start)
        rows.append(row)
        m.c.write(out / (str(sid) + '_result.json'), row)
        print('PAIR_DONE', sid, row['ba_bound'], row['ba_unbound'], flush=True)
    m.c.write(out / ('summary_shard_' + str(shard) + '.json'), dict(completed=True, shard=shard, rows=rows))
    print('SHARD_DONE', shard, len(rows))


if __name__ == '__main__':
    main(*sys.argv[1:])
