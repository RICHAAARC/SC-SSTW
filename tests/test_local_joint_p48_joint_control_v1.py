"""Targeted CPU arithmetic/causal replay/recovery tests; no pretrained model calls."""
import copy
import json
from pathlib import Path
import types
import pytest
import torch
from main.tube_state import local_joint_p48_joint_control_v1 as m
from runtime.wan import local_joint_p48_joint_control_v1 as r
from runtime.wan import local_joint_p48_journal_v1 as journal
from runtime.wan.local_joint_p48_native_v1 import coefficients
from runtime.wan.local_joint_paired_transfer_native_v1 import snapshot,restore
from runtime.wan.local_joint_state_payload_provider_v1 import LatentSupport
from runtime.wan.local_joint_readout_checkpoint_v1 import checkpoint_call
from test_local_joint_readout_m0_v1 import TinyWan
from test_local_joint_p0_input_vjp_v1 import TinyTransformer

ROOT=Path(__file__).resolve().parents[1]
SHAPE=(1,3,3,4,4)
SUPPORT=LatentSupport((1,3),((0,1,0,1),(0,1,3,4),(3,4,0,1),(3,4,3,4)))
PROTOCOL=types.SimpleNamespace(video_shape=(9,4,4,3),rois=SUPPORT.rois)
pytestmark=pytest.mark.unit


@pytest.fixture(autouse=True)
def single_thread():
    old=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_objective_equal_ties_all32():
    c=torch.zeros(22,dtype=torch.double,requires_grad=True)
    margins=torch.full((4,8),-.01,dtype=torch.double,requires_grad=True)
    o=m.objectives(dict(correlations=c,margins=margins))
    gc=torch.autograd.grad(o["S"],c)[0]
    torch.testing.assert_close(gc,torch.tensor([-1.]+[1/21]*21,dtype=torch.double))
    gp=torch.autograd.grad(o["P"],margins)[0]
    gw=torch.autograd.grad(o["W"],margins)[0]
    torch.testing.assert_close(gp,torch.full_like(gp,-1/32)); torch.testing.assert_close(gw,gp)


@pytest.mark.parametrize("ties",[False,True])
def test_quality_cotangents_against_independent_autograd(ties):
    gen=torch.Generator().manual_seed(91)
    z=torch.randn(SHAPE,generator=gen,dtype=torch.double,requires_grad=True)
    rgb=torch.randn(PROTOCOL.video_shape,generator=gen,dtype=torch.double)
    if ties: rgb=torch.tensor([1.,-1.,1.,-1.,1.,-1.,1.,-1.,1.],dtype=torch.double)[:,None,None,None].expand_as(rgb).clone()
    rgb.requires_grad_(); zz=torch.zeros_like(z); rr=torch.zeros_like(rgb)
    lm=m.support_mask(SHAPE,SUPPORT); rm=torch.zeros((4,4),dtype=torch.bool)
    for y0,y1,x0,x1 in PROTOCOL.rois: rm[y0:y1,x0:x1]=True
    dt=rgb[1:]-rgb[:-1]
    energies=torch.stack((z[~lm].square().sum(),rgb[:,~rm,:].square().mean(),rgb.square().mean(),dt.square().mean(),
        rgb.square().mean((1,2,3)).amax(),dt.square().mean((1,2,3)).amax(),rgb.square().amax(),dt.square().amax()))
    den=torch.arange(1,9,dtype=torch.double)
    expected=torch.autograd.grad((energies/den).mean(),(z,rgb))
    gz,gr,receipt=m.quality_cotangents(z,rgb,zz,rr,den,SUPPORT,PROTOCOL)
    torch.testing.assert_close(gz,expected[0],rtol=1e-13,atol=1e-13)
    torch.testing.assert_close(gr,expected[1],rtol=1e-13,atol=1e-13)
    torch.testing.assert_close(m.quality_values(z,rgb,zz,rr,SUPPORT,PROTOCOL),energies.detach())
    assert gr[:,~rm,:].abs().sum()>0 and gz[~lm].abs().sum()>0
    if ties: assert len(receipt["ties"]["frames"])==9 and receipt["ties"]["pixels"]["count"]==rgb.numel()
    den[3]=0
    with pytest.raises(m.UndefinedConstruction): m.quality_cotangents(z,rgb,zz,rr,den,SUPPORT,PROTOCOL)


def native_source():
    from diffusers import UniPCMultistepScheduler
    scheduler=UniPCMultistepScheduler(prediction_type="flow_prediction",use_flow_sigmas=True,flow_shift=3,solver_order=2,solver_type="bh2",lower_order_final=True)
    scheduler.set_timesteps(50)
    z=torch.randn(SHAPE,generator=torch.Generator().manual_seed(19))*.03+.1
    for i in range(48): z=scheduler.step(.04*torch.sin(z),scheduler.timesteps[i],z,return_dict=False)[0]
    fork=dict(z=z,conditional=.04*torch.sin(z),unconditional=.03*torch.sin(z),scheduler=snapshot(scheduler))
    d=torch.zeros(SHAPE,dtype=torch.double); d[0,0,1,0,0]=1
    return dict(fork=fork,direction=d,prompt=torch.full((1,2,2),.1,dtype=torch.bfloat16),negative=torch.full((1,2,2),-.1,dtype=torch.bfloat16))


def test_native_full_history_corrector_and_final_pre_corrector_formula():
    source=native_source(); fork=source["fork"]; s=restore(fork["scheduler"],"cpu"); co=coefficients(s,SHAPE)
    theta=torch.zeros(SHAPE,requires_grad=True); z=fork["z"]
    v=fork["unconditional"]+5*(fork["conditional"]-theta/co["sigma48"]-fork["unconditional"])
    x49=s.step(v,s.timesteps[48],z,return_dict=False)[0]
    derivative=torch.autograd.grad(x49.sum(),theta)[0]
    torch.testing.assert_close(derivative,torch.full_like(derivative,co["k48"]),atol=2e-6,rtol=0)
    v49=.3*torch.sin(x49.detach()); actual=s.step(v49,s.timesteps[49],x49.detach(),return_dict=False)[0]
    torch.testing.assert_close(actual,x49.detach()-co["sigma49"]*v49,atol=0,rtol=0)
    wrong=restore(fork["scheduler"],"cpu"); wrong.register_to_config(use_flow_sigmas=False)
    with pytest.raises(ValueError,match="flow-sigma"): coefficients(wrong,SHAPE)
    wrong=restore(fork["scheduler"],"cpu"); wrong.disable_corrector=[47]
    with pytest.raises(ValueError,match="corrector48"): coefficients(wrong,SHAPE)


def test_socp_actual_fp32_bounds_and_fixed_degeneracy():
    a=torch.zeros(SHAPE); a[0,0,1,0,0]=.019
    candidate,report=m.solve_correction(a,[-a]*4,SUPPORT)
    assert report["status"]=="CANDIDATE_DEFINED"
    assert candidate.double().norm()<=.02 and (candidate.double()-a.double()).norm()<=.02
    assert report["primary_certificate"]["realized_t"]>0 and report["secondary_target"]<report["primary_certificate"]["realized_t"]
    assert all(v<0 for v in report["normalized_slopes"])
    json.dumps(report,allow_nan=False)
    candidate,report=m.solve_correction(a,[a,-a,a,-a],SUPPORT)
    assert candidate is None and report["status"] in ("NO_RESOLVED_COMMON_DESCENT","NUMERICALLY_UNRESOLVED")
    candidate,report=m.solve_correction(a,[a,a,a,torch.zeros_like(a)],SUPPORT)
    assert candidate is None and report["status"]=="UNDEFINED"


class MixingTransformer(TinyTransformer):
    def forward(self,*a,**kw):
        y=super().forward(*a,**kw)[0].float()
        return ((y+4*y.mean()).bfloat16(),)


def fake_read(rgb,**_):
    v=rgb.double().mean()
    return dict(correlations=torch.cat((v.reshape(1),torch.zeros(21,dtype=v.dtype,device=v.device))),margins=(v-.8).expand(4,8))


def fake_observe(path,rgb,protocol):
    # Truth-free numeric fixture only; production uses the unchanged raw-first observer.
    raw=[{k:[dict(q=float(rgb.mean())) for _ in range(8)] for k in ("state_chips","payload_chips")} for _ in range(88)]
    r.write_json(path/"raw.json",raw)
    read=fake_read(rgb)
    values=dict(correlations=read["correlations"].tolist(),margins=read["margins"].flatten().tolist())
    r.write_json(path/"scores.json",dict(state=dict(correlations=[dict(value=v) for v in values["correlations"]]),payload=dict(metrics=[dict(signed_mean=v) for v in values["margins"]])))
    r.write_json(path/"values.json",values)
    return values


def config(tmp):
    c=json.loads((ROOT/"experiments/wan_state_clock/configs/local_joint_p48_joint_control_v1.json").read_text())
    c.update(device="cpu",spool_root=str(tmp),source_run=str(tmp/"source")); return c


def execute(tmp,**kw):
    return r.run(config(tmp),tmp/"out",source_loader=lambda _:native_source(),preparer=lambda *_:MixingTransformer(),
        vae_loader=lambda *_,**__:TinyWan().eval(),protocol=PROTOCOL,shape=SHAPE,support=SUPPORT,expected_blocks=2,
        observer=fake_observe,reader=fake_read,solver=lambda a,g,s:(a*.5,dict(status="FIXTURE",physical_candidates=1)),**kw)


def test_complete_fixed_budget_same_objects_and_report_only(tmp_path):
    result=execute(tmp_path)
    assert result["status"]=="COMPLETE",result["failures"]
    assert {k:v["attempted"] for k,v in result["counts"].items()}==journal.LIMITS
    assert result["block_counts"]["forward"]["attempted"]==result["block_counts"]["recompute"]["attempted"]==16
    assert result["chunk_counts"]==dict(forward=dict(attempted=21,completed=21),recompute=dict(attempted=12,completed=12))
    assert all(v["same_object"] for v in result["object_moves"])
    assert all(v["boundary_storage"]["closed"] for v in result["dit_replay"].values())
    assert all(v["boundary_storage"]["closed"] for v in result["vae_replay"].values())
    assert all(v["rgb_comparison"]["exact_equal"] for v in result["objectives"].values())
    calls=copy.deepcopy(result["calls"])
    resumed=execute(tmp_path)
    assert resumed["calls"]==calls and resumed["status"]=="COMPLETE"
    (tmp_path/"out/objectives/Q/terminal_cotangent.pt").unlink()
    resumed=execute(tmp_path)
    assert resumed["status"]=="ENGINEERING_FAILURE" and resumed["calls"]==calls
    assert resumed["objectives"]["Q"]["status"]=="MISSING" and resumed["comparison"]["status"]=="INCOMPLETE_EVIDENCE"


def test_failure_does_not_retry_and_retains_three_four_rows(tmp_path,monkeypatch):
    original=torch.autograd.grad
    def fail(*a,**kw): raise RuntimeError("fixture VJP failure")
    monkeypatch.setattr(torch.autograd,"grad",fail)
    result=execute(tmp_path)
    assert result["status"]=="ENGINEERING_FAILURE"
    assert result["counts"]["decoder_vjp"]["attempted"]==1
    assert result["counts"]["dit_input_vjp"]["attempted"]==0
    assert set(result["endpoints"])==set(m.VIEWS) and set(result["objectives"])==set(m.OBJECTIVES)
    assert result["endpoints"]["CANDIDATE"]["status"]=="MISSING"
    monkeypatch.setattr(torch.autograd,"grad",original)
    again=execute(tmp_path)
    assert again["counts"]==result["counts"]


def test_precheck_failure_can_enter_first_attempt_only(tmp_path):
    def fail(_): raise OSError("missing source")
    result=r.run(config(tmp_path),tmp_path/"out",source_loader=fail)
    assert result["status"]=="ENGINEERING_FAILURE" and journal.needs_model_work(tmp_path/"out")
    completed=execute(tmp_path)
    assert completed["status"]=="COMPLETE" and completed["preflight_history"]
    assert len(completed["failures"])==1 and completed["current_attempt"]["failures"]==[]
    assert not journal.needs_model_work(tmp_path/"out")


def test_chunk_checkpoint_rejects_second_replay_before_entry(tmp_path):
    ledger=r.VaeLedger(None,str(tmp_path),1); ledger.new_decode()
    x=torch.tensor([.4],requires_grad=True)
    y=checkpoint_call(ledger,"vae_chunk",torch.sin,x)
    torch.autograd.grad(y,x,retain_graph=True)
    with pytest.raises(RuntimeError,match="replay budget"): torch.autograd.grad(y,x)
    assert ledger.counts["vae_chunk"]["recompute"]["attempted"]==1
    ledger.release_boundary_storage()


@pytest.mark.parametrize("gradient",[False,True])
def test_actual_non_bf16_forward_saved_then_rejected(tmp_path,monkeypatch,gradient):
    class WrongOutput(MixingTransformer):
        def forward(self,*a,**kw):
            y=super().forward(*a,**kw)[0]
            return (y.float() if (self.gradient_checkpointing if gradient else True) else y,)
    monkeypatch.setattr(__import__(__name__),"MixingTransformer",WrongOutput)
    result=execute(tmp_path)
    kind="dit_gradient_forward" if gradient else "dit_ordinary_forward"
    assert result["status"]=="ENGINEERING_FAILURE" and result["counts"][kind]["FAILED"]==1
    assert result["counts"]["dit_input_vjp"]["attempted"]==0
    directory=tmp_path/("out/objectives/S/conditional" if gradient else "out/ZERO")
    basename="output" if gradient else "conditional_output"
    assert (directory/(basename+".pt")).exists()
    assert json.loads((directory/(basename+"_arithmetic.json")).read_text())["actual_output_dtype"]=="torch.float32"
    calls=copy.deepcopy(result["calls"])
    assert execute(tmp_path)["calls"]==calls


@pytest.mark.parametrize("file",["ZERO/raw.json","ZERO/scores.json","ZERO/values.json","ANCHOR/quality.json",
    "objectives/Q/objective.json","objectives/S/conditional/comparison.json","CANDIDATE/control.json","solution.json"])
def test_method_json_missing_keeps_calls_and_invalidates_completion(tmp_path,file):
    result=execute(tmp_path); assert result["status"]=="COMPLETE"
    (tmp_path/"out"/file).unlink()
    resumed=execute(tmp_path)
    assert resumed["calls"]==result["calls"] and resumed["status"]=="ENGINEERING_FAILURE"
    assert resumed["comparison"]["status"]=="INCOMPLETE_EVIDENCE"


def test_json_cardinality_and_nonfinite_are_rejected(tmp_path):
    p=tmp_path/"values.json"
    r.write_json(p,dict(correlations=[0.]*22,margins=[0.]*31))
    with pytest.raises(ValueError,match="cardinality"): journal.check_json(p,"values")
    p.write_text('{"correlations": [NaN], "margins": []}')
    with pytest.raises(ValueError,match="nonfinite"): journal.check_json(p,"values")
    r.write_json(p,dict(state=dict(correlations=[{}]*22),payload=dict(metrics=[{}]*32)))
    with pytest.raises(KeyError): journal.check_json(p,"scores")


def test_gap_resource_peak_saved_before_cuda_reset_and_at_exit(tmp_path,monkeypatch):
    counter={"peak":10,"reset":False}
    def snapshot(_):
        return dict(time_unix=0,host_rss_bytes=counter["peak"],host_process_max_rss_bytes=counter["peak"],
            cuda_allocated_bytes=3,cuda_reserved_bytes=4,cuda_peak_allocated_bytes=counter["peak"],cuda_peak_reserved_bytes=counter["peak"])
    monkeypatch.setattr(journal,"resource_snapshot",snapshot)
    monkeypatch.setattr(torch.cuda,"is_initialized",lambda:True)
    def reset(_):
        saved=json.loads((tmp_path/"out/result.json").read_text())
        assert saved["resource_lifetime"]["cuda_peak_allocated_bytes_max"]>=counter["peak"]
        counter["reset"]=True; counter["peak"]=1
    monkeypatch.setattr(torch.cuda,"reset_peak_memory_stats",reset)
    state=journal.initial(config(tmp_path)); log=journal.Journal(tmp_path/"out",state,"cuda")
    with log:
        log.sample(); counter["peak"]=90
        log.start_stage("fake physical stage"); assert counter["reset"]
        log.end_stage(); counter["peak"]=120
    assert state["resource_lifetime"]["cuda_peak_allocated_bytes_max"]==120
    assert state["resource_boundaries"][0]["snapshot"]["cuda_peak_allocated_bytes"]==90
    assert json.loads((tmp_path/"out/report.json").read_text())["resource_lifetime"]["host_rss_bytes_max"]==120
