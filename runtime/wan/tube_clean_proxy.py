"""One T46 current-clean proxy gradient; no Transformer graph or terminal transport."""
import copy,math
from types import SimpleNamespace
import torch
from main.tube_state import objective_alignment as objective
from .tube_response_selection import prepare_direction,replay
from .tube_retention import native_step,continue_single
from .tube_multistep import fingerprint,measures

def load_transformer(config):
    from diffusers import WanTransformer3DModel
    from huggingface_hub import HfApi
    model=config['model'];revision=model.get('revision');record={'requested_revision':revision,'resolved_revision':None,'original_weight_identity':'UNVERIFIED_ORIGINAL_REVISION_NOT_RECORDED'}
    try:
        revision=HfApi().model_info(model['id'],revision=revision or 'main').sha
        record['resolved_revision']=revision
    except Exception as exc:record['revision_resolution_error']=repr(exc)
    transformer=WanTransformer3DModel.from_pretrained(model['id'],subfolder='transformer',revision=revision,torch_dtype=torch.bfloat16).eval()
    for p in transformer.parameters():p.requires_grad_(False)
    if hasattr(transformer,'disable_gradient_checkpointing'):transformer.disable_gradient_checkpointing()
    transformer.to('cuda');record['config_commit_hash']=getattr(transformer.config,'_commit_hash',None)
    dtype=getattr(getattr(transformer,'patch_embedding',None),'weight',next(transformer.parameters())).dtype
    return SimpleNamespace(transformer=transformer),dtype,record

def clean_direction(clean,book,message,name):
    if name not in ('hinge','tanh'):raise ValueError('fixed objectives only')
    directions=torch.from_numpy(book['directions']).double();codes=torch.from_numpy(book['codes']).double()
    with torch.enable_grad():
        leaf=clean.detach().cpu().double().clone().requires_grad_(True)
        value=objective.loss(leaf,directions,codes,message,name);gradient,=torch.autograd.grad(value,leaf)
    gradient=gradient.detach();norm=objective.rms(gradient)
    if not math.isfinite(norm) or norm<=0 or not torch.isfinite(gradient).all():raise ValueError('zero/nonfinite clean proxy gradient; no fallback')
    raw=-gradient.float()
    if not torch.isfinite(raw).all() or objective.rms(raw)==0:raise ValueError('FP32 gradient is zero/nonfinite; no fallback')
    return raw,dict(objective=name,loss=float(value.detach()),gradient_support_rms=norm,
        gradient_shares=objective.gradient_shares(clean.cpu(),gradient,directions,codes,message),
        clean_nominal=objective.metrics(clean.cpu(),directions,codes,message),
        meaning='CPU detached current clean46 leaf; no terminal gradient transport, model/VAE graph or receiver derivative')

@torch.no_grad()
def off_step(snapshot,z,v,count):
    before=fingerprint(vars(snapshot));s=copy.deepcopy(snapshot)
    after=native_step(s,z,v,46,count)
    if fingerprint(vars(snapshot))!=before:raise RuntimeError('OFF changed original history')
    return after,s,dict(input_fingerprint=fingerprint(z),history_fingerprint=before,formal_OFF='new zero-control tail from saved46, not reused historical media')
