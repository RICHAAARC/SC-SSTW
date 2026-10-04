"""Shared runtime compatibility; no model downloads or GPU calls."""
import hashlib
import inspect
from types import SimpleNamespace
import pytest
import torch
from runtime.wan import trajectory, generation
pytestmark=pytest.mark.unit

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64,torch.int64,torch.bool,torch.bfloat16])
@pytest.mark.parametrize('shape',['scalar','strided','empty'])
def test_tensor_fingerprint_raw_storage_and_scalar(dtype,shape):
    x=torch.tensor(1,dtype=dtype) if shape=='scalar' else (torch.empty(0,dtype=dtype) if shape=='empty' else torch.arange(12).reshape(3,4).to(dtype).t())
    out=trajectory._fingerprint_value(x)
    assert out['shape']==list(x.shape) and out['dtype']==str(dtype)
    raw=x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    assert out['sha256']==hashlib.sha256(raw).hexdigest()
    if dtype!=torch.bfloat16:
        assert out['sha256']==hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()

@pytest.mark.parametrize('device,expected',[(None,'cuda'),('cpu','cpu')])
def test_frozen_vae_device_default_and_override(monkeypatch,device,expected):
    import diffusers
    captured={}
    class VAE:
        def __init__(self):self.weight=torch.nn.Parameter(torch.ones(1))
        def eval(self):captured['eval']=True;return self
        def disable_tiling(self):captured['tiling_disabled']=True
        def parameters(self):return iter([self.weight])
        def to(self,target):captured['device']=str(target);return self
    model=VAE()
    def load(*args,**kwargs):captured['kwargs']=kwargs;return model
    monkeypatch.setattr(diffusers.AutoencoderKLWan,'from_pretrained',load)
    result=generation.load_frozen_vae({'model':{'id':'fixture/model','revision':'fixed'}},device=device)
    assert result is model and captured['device']==expected
    assert captured['kwargs']['torch_dtype']==torch.float32
    assert not model.weight.requires_grad and captured['eval']

def test_loader_signature_remains_optional():
    assert inspect.signature(generation.load_frozen_vae).parameters['device'].default is None
    for name in ('device','model_dtype'):
        assert inspect.signature(generation.prepare_generation).parameters[name].default is None
