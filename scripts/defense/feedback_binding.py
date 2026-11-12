"""Full-capacity sign carrier for a semantic-feedback embedding experiment."""
from anchor_binding import embed as original_embed,decode as original_decode


def embed(aux_latent,code,root,task,arm):
    assert arm in ['full','feedback','consensus']
    return original_embed(aux_latent,code,root,task,'full')


def decode(latent,code,root,task,arm):
    assert arm in ['full','feedback','consensus']
    return original_decode(latent,code,root,task,'full')
