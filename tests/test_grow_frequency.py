"""Small CPU/real-UniPC and fake-media checks, no pretrained models."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from main.tube_state import grow_frequency as method
from runtime.wan.grow_frequency import generate
from experiments.wan_state_clock import grow_frequency_run as runner
from test_velocity_direction import scheduler

pytestmark=pytest.mark.unit
torch.set_num_threads(1)


def test_orthonormal_dct_gradient_and_local_descent():
    book=method.codebook(b'fixed-test')
    z=torch.randn(method.SHAPE,dtype=torch.float64,generator=torch.Generator().manual_seed(5))*.2
    x=z[:,0:1]
    torch.testing.assert_close(method.idct2(method.dct2(x)),x,atol=1e-13,rtol=1e-12)
    assert float(method.dct2(x).square().sum())==pytest.approx(float(x.square().sum()),rel=1e-12)
    leaf=z.requires_grad_();loss=method.loss(leaf,book,0)
    gradient=torch.autograd.grad(loss,leaf)[0]
    expected=method.analytic_gradient(z.detach(),book,0)
    torch.testing.assert_close(gradient,expected,atol=1e-13,rtol=1e-12)
    assert torch.count_nonzero(gradient[:,1:])==0
    after=method.loss(z.detach()-.1*gradient,book,0)
    assert float(after)==pytest.approx(.81*float(loss),rel=1e-12)
    direction=expected/expected.norm();eps=1e-5
    fd=(method.loss(z.detach()+eps*direction,book,0)-method.loss(z.detach()-eps*direction,book,0))/(2*eps)
    assert float(fd)==pytest.approx(float((gradient*direction).sum()),rel=1e-8)


def test_primary_sign_votes_not_soft_score_and_erasures():
    book=method.codebook(b'fixed-test');uv=torch.tensor(book['coordinates'])
    spectrum=torch.zeros(method.SHAPE,dtype=torch.float64)
    payload=torch.tensor(book['payloads'][0],dtype=torch.float64)
    values=torch.cat([payload,payload,payload,-100*payload])
    spectrum[:,0:1,:,uv[:,0],uv[:,1]]=values
    readout=method.read(method.idct2(spectrum),book)
    assert readout['aggregate']['signs']==book['payloads'][0]
    soft=torch.tensor(readout['aggregate']['coefficient_mean_diagnostic'])
    assert torch.sign(soft).tolist()==book['payloads'][1]
    assert readout['aggregate']['votes_per_bit']==184 and len(readout['per_time'])==46
    comparison=method.compare_payloads(readout,book)
    assert comparison['aggregate'][0]['exact_payload_match'] and comparison['aggregate'][1]['bit_errors_including_erasures']==16
    values=torch.cat([payload,payload,-payload,-payload])
    spectrum[:,0:1,:,uv[:,0],uv[:,1]]=values
    tied=method.read(method.idct2(spectrum),book)
    assert tied['aggregate']['bit_erasures']==16
    zero=method.read(torch.zeros(method.SHAPE),book)
    assert zero['aggregate']['bit_erasures']==16 and zero['aggregate']['zero_coefficient_votes']==46*64


class Model(torch.nn.Module):
    def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.1));self.inputs=[]
    def forward(self,hidden_states,timestep,encoder_hidden_states,**kw):
        self.inputs.append(torch.is_grad_enabled())
        return (hidden_states*self.weight+encoder_hidden_states*.001,)


def test_native_history_once_per_step_no_transformer_backward():
    book=method.codebook(b'fixed-test');model=Model()
    pipe=SimpleNamespace(transformer=model,scheduler=scheduler())
    z=torch.randn(method.SHAPE,generator=torch.Generator().manual_seed(9))*.1
    calls=[];rows=[]
    result=generate(pipe,z,torch.tensor(1.),torch.tensor(-1.),torch.float32,5.,book,0,
        lambda k,d:calls.append((k,d)),rows.append)
    assert calls.count(('transformer',True))==100
    assert calls.count(('scheduler_step',True))==50
    assert calls.count(('local_gradient',True))==20
    assert [v['index'] for v in rows if v['controlled']]==list(range(10,30))
    assert pipe.scheduler.step_index==50 and pipe.scheduler.model_outputs[-1] is not None
    assert not any(model.inputs) and model.weight.grad is None and result.grad_fn is None
    assert all(v['loss_after_local']<v['loss_before'] for v in rows if v['controlled'])
    assert all(v['clean_update_error_rms']<1e-6 for v in rows if v['controlled'])
    pipe.scheduler=scheduler();pipe.scheduler.predict_x0=False
    with pytest.raises(ValueError,match='predict_x0'):generate(pipe,z,torch.tensor(1.),torch.tensor(-1.),torch.float32,5.,book,0,lambda *a:None,lambda *a:None)


def test_four_same_decode_layers_and_partial_media_kept(tmp_path,monkeypatch):
    pipe=SimpleNamespace(transformer=Model(),scheduler=scheduler())
    initial=torch.zeros(method.SHAPE)
    monkeypatch.setattr(runner,'prepare_generation',lambda *a,**kw:(pipe,initial,torch.tensor(1.),torch.tensor(-1.),torch.float32))
    def fake_generate(pipe,initial,prompt,negative,dtype,cfg,book,message,count,record):
        for i in range(50):record({'index':i,'controlled':message is not None and i in method.CONTROL_INDICES and not (message==0 and i==10)})
        return initial+.001*(0 if message is None else message+1)
    monkeypatch.setattr(runner,'generate',fake_generate)
    vae=Model();monkeypatch.setattr(runner,'load_frozen_vae',lambda *a:vae)
    monkeypatch.setattr(runner,'RGB_SHAPE',(5,4,4,3))
    decoded=[];encoded=[];saved={}
    def decode(vae,z):
        rgb=torch.full((5,4,4,3),.123+float(z.mean()));decoded.append(rgb);return rgb
    def encode(vae,rgb):encoded.append(rgb.clone());return initial+float(rgb.mean())
    def save(rgb,path,fps,crf):
        saved[path]=torch.round(rgb*255)/255+.001
        if path.stem=='B':raise RuntimeError('intentional save failure with residue')
    monkeypatch.setattr(runner,'decode_normalized_latent',decode)
    monkeypatch.setattr(runner,'reencode_rgb24_readback',encode)
    monkeypatch.setattr(runner,'encode_rgb',save)
    monkeypatch.setattr(runner,'read_mp4',lambda p:saved[p])
    result=runner.run_case(runner.CASES[0],tmp_path/'result')
    assert result['video_denominator']==3 and len(result['videos'])==3
    assert result['actual_calls']['vae_decode_completed']==3 and result['actual_calls']['vae_encode_completed']==8
    assert result['actual_calls']['mp4_save_attempted']==3 and result['actual_calls']['mp4_save_completed']==2
    torch.testing.assert_close(encoded[0],decoded[0])
    torch.testing.assert_close(encoded[1],torch.from_numpy(np.rint(decoded[0].numpy()*255).astype(np.uint8)).float()/255)
    torch.testing.assert_close(encoded[2],saved[next(p for p in saved if p.stem=='OFF')])
    for arm,item in result['videos'].items():
        assert item['actual_controlled_steps']=={'OFF':0,'A':19,'B':20}[arm]
        assert all(len(v['per_time'])==46 for v in item['layers'].values())
    assert result['videos']['A']['control_schedule_complete'] is False
    assert result['videos']['A']['status']=='WITH_RETAINED_FAILURES'
    assert result['videos']['B']['layers']['mp4']['status']=='FAILED_SAVE_OR_READBACK'
    assert result['videos']['B']['layers']['rgb8']['status']=='COMPLETE'
    assert result['status']=='WITH_RETAINED_FAILURES' and result['scientific_pass'] is None
    assert result['videos']['OFF']['layers']['terminal']['OFF_coincidental_exact_matches']==[]


def test_missing_case_result_keeps_four_case_roster(tmp_path,monkeypatch):
    launched=[]
    def fake(command,**kw):launched.append(command);return SimpleNamespace(returncode=1)
    monkeypatch.setattr(runner.subprocess,'run',fake)
    result=runner.run_all(tmp_path/'all')
    assert len(launched)==4 and len(result['cases'])==4
    assert result['video_denominator']==12 and result['layer_denominator']==48
    assert all(len(c['videos'])==3 for c in result['cases'].values())
    assert all(len(v['layers'])==4 for c in result['cases'].values() for v in c['videos'].values())
    assert result['status']=='WITH_RETAINED_FAILURES'


@pytest.mark.parametrize('field,value',[('prediction_type','epsilon'),('predict_x0',False),('thresholding',True),('solver_p',object())])
def test_scheduler_contract_rejects_before_any_forward(field,value):
    s=scheduler()
    if field in ('prediction_type','thresholding'):s.register_to_config(**{field:value})
    else:setattr(s,field,value)
    model=Model();pipe=SimpleNamespace(transformer=model,scheduler=s)
    with pytest.raises(ValueError,match='requires native UniPC'):
        generate(pipe,torch.empty(0),None,None,torch.float32,5.,None,None,lambda *a:None,lambda *a:None)
    assert model.inputs==[]
