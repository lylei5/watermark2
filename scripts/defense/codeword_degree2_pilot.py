"""Exploratory degree-two sensitivity pilot with newly generated references.

Same-model public proxy reprompting uses only the released reference pixels.
This is a pipeline diagnostic, not a reproduction of SemBind's exact inversion.
The shared process is not a security isolation boundary; the reprompt function
accepts no watermark material and its candidate is saved before scoring.
"""
import argparse
import gc
import hashlib
import io
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from codeword_binding import CodewordBinding, keyed_order, sembind_mask_bits, bit_accuracy


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--semantic',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n',type=int,default=5)
    p.add_argument('--reuse',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    sys.path.insert(0,str(args.semantic))
    from model import AttnSupConHashingModel
    device='cuda:0'
    pipe=StableDiffusionPipeline.from_pretrained(args.model,torch_dtype=torch.float16,
                                                safety_checker=None,local_files_only=True)
    pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    proc=AutoImageProcessor.from_pretrained(args.semantic/'dinov2-giant',local_files_only=True)
    dino=AutoModel.from_pretrained(args.semantic/'dinov2-giant',local_files_only=True,
                                  torch_dtype=torch.float32).eval()
    masker=AttnSupConHashingModel(input_dim=1536,feat_dim=8192,hash_dim=1024)
    masker.load_state_dict(torch.load(args.semantic/'sembind_masker.pth',weights_only=True,map_location='cpu'))
    masker.eval()
    # Keep only one large model family on the 8GB GPU at any point.
    def mode(family):
        if family=='diffusion':
            dino.cpu(); masker.cpu(); torch.cuda.empty_cache(); pipe.to(device)
        else:
            pipe.unet.cpu(); pipe.vae.cpu(); pipe.text_encoder.cpu(); torch.cuda.empty_cache(); dino.to(device); masker.to(device)

    @torch.inference_mode()
    def code(im):
        x=proc(images=im,return_tensors='pt').to(device)
        cls=F.normalize(dino(**x).last_hidden_state[:,0],dim=1)
        return masker(cls,stage='binarize',s=12)[0].cpu().numpy().astype(np.uint8)

    @torch.inference_mode()
    def generate(prompt,z):
        pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
        return pipe(prompt,latents=z.to(device,dtype=torch.float16),num_inference_steps=50,
                    guidance_scale=7.5,height=512,width=512).images[0]

    @torch.inference_mode()
    def invert(im):
        inv=DDIMInverseScheduler.from_config(pipe.scheduler.config)
        inv.set_timesteps(50,device=device)
        x=pipe.image_processor.preprocess(im).to(device,dtype=torch.float16)
        z=pipe.vae.encode(x).latent_dist.mode()*pipe.vae.config.scaling_factor
        embeds=pipe.encode_prompt('',device,1,False)[0]
        for t in inv.timesteps:
            pred=pipe.unet(inv.scale_model_input(z,t),t,encoder_hidden_states=embeds).sample
            z=inv.step(pred,t,z).prev_sample
        return z.float().cpu()

    def reprompt(reference, target_prompt):
        return generate(target_prompt,invert(reference))

    prompts=['A cat sitting on a wooden bench in a garden.',
             'A red bus parked next to a brick building.',
             'A bowl of apples and oranges on a kitchen table.',
             'A white sailboat on a blue lake with mountains.',
             'A train passing through a rural landscape.'][:args.n]
    attack_prompts=json.loads((Path(__file__).resolve().parents[2]/'protocols/codeword_hard_reprompt_20261109.json').read_text())['targets'][:args.n]
    manifest=dict(purpose='exploratory degree-two pilot; selected after observing degree-four JPEG rejection; not confirmatory',n=len(prompts),resolution=512,steps=50,
                  scheduler='DDIM / DDIMInverseScheduler',generation_cfg=7.5,inversion_cfg=1,
                  inversion_prompt='',diffusion_dtype='float16',semantic_dtype='float32',
                  threshold=181/256,comparison='>',target_and_proxy='same SD2.1 snapshot',
                  permission_isolation_implemented=False,methods=['aligned_parity2'],
                  prompts=prompts,reprompt_prompts=attack_prompts,
                  key_note='deterministic development keys, not final secret-key evaluation')
    (args.output/'protocol.json').write_text(json.dumps(manifest,indent=2))
    rows=[]; start=time.monotonic()
    for i,(prompt,target) in enumerate(zip(prompts,attack_prompts)):
        key=hashlib.sha256(('pilot-watermark-'+str(i)).encode()).digest()
        nonce=hashlib.sha256(('pilot-nonce-'+str(i)).encode()).digest()[:16]
        msgbytes=hashlib.sha256(('pilot-message-'+str(i)).encode()).digest()
        encrypted=Cipher(algorithms.ChaCha20(key,nonce),mode=None).encryptor().update(msgbytes*64)
        cipherbits=np.unpackbits(np.frombuffer(encrypted,dtype=np.uint8))
        msg=np.unpackbits(np.frombuffer(msgbytes,dtype=np.uint8))
        stream=cipherbits ^ np.tile(msg,64)
        rng=np.random.default_rng(44000+i)
        z=(np.abs(rng.standard_normal(16384))*(2*cipherbits.astype(np.float32)-1)).astype(np.float32)
        z=torch.from_numpy(z).reshape(1,4,64,64)
        bkey=hashlib.sha256(('pilot-binding-'+str(i)).encode()).digest()
        perm=keyed_order(bkey,b'sembind/permutation/',16384)
        aligned=CodewordBinding(bkey,degree=2)
        def mask(c,method):
            bits=sembind_mask_bits(c[None],perm)[0] if method=='sembind' else aligned.mask_bits(c)
            return (1-2*bits.astype(np.float32)).reshape(1,4,64,64)
        mode('diffusion')
        aux=Image.open(args.reuse/f'{i}_aux.png').convert('RGB')
        aux.save(args.output/f'{i}_aux.png')
        mode('semantic'); ca=code(aux)
        for method in ['aligned_parity2']:
            mode('diffusion')
            ref=generate(prompt,z*torch.from_numpy(mask(ca,method)))
            ref.save(args.output/f'{i}_{method}_reference.png')
            forged=reprompt(ref,target)
            forged_path=args.output/f'{i}_{method}_reprompt.png'
            forged.save(forged_path)
            sealed=hashlib.sha256(forged_path.read_bytes()).hexdigest()
            buf=io.BytesIO(); ref.save(buf,format='JPEG',quality=50); buf.seek(0)
            jpeg=Image.open(buf).convert('RGB')
            images=[('clean',ref),('jpeg50',jpeg),('reprompt',forged)]
            inverses=[invert(im).numpy() for _,im in images]
            mode('semantic')
            for (label,im),inv in zip(images,inverses):
                c=code(im)
                unbound=(inv*mask(c,method)).reshape(-1)
                bits=(unbound>0).astype(np.uint8)^stream
                ba=float(bit_accuracy(bits,msg)[0])
                row=dict(task=i,method=method,label=label,ba=ba,accepted=bool(ba>181/256),
                         semantic_hamming=int((ca!=c).sum()),candidate_sha256=sealed if label=='reprompt' else None)
                rows.append(row)
                print(json.dumps(row),flush=True)
            (args.output/'results.json').write_text(json.dumps(dict(completed=False,rows=rows),indent=2))
    manifest.update(completed=True,seconds=time.monotonic()-start,rows=rows,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'results.json').write_text(json.dumps(manifest,indent=2))
    print('COMPLETE',manifest['seconds'],flush=True)


if __name__=='__main__':
    main()
