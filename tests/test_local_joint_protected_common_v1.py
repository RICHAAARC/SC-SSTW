from __future__ import annotations

import copy
import json

import pytest
import torch

from main.tube_state import local_joint_protected_common_v1 as method
from runtime.wan import local_joint_protected_common_v1 as runtime
from runtime.wan import local_joint_all_bit_resume_v1 as resume
from tests.test_local_joint_readout_m0_v1 import TinyWan, PROTOCOL, SHAPE, SUPPORT

pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def reference():
    margins = [.04+.003*i for i in range(32)]
    for i in method.BITS: margins[i] = -.1
    margins[29] = .0209909
    return dict(margins=margins,correlations=[.1]+[.02]*21,state_gap=.08)


def test_four_scalars_direct_derivatives_and_barrier_domain():
    parameters = method.barrier_parameters(reference())
    margins = torch.tensor(reference()["margins"],dtype=torch.double,requires_grad=True)
    correlations = torch.tensor([.1]+[.02]*21,dtype=torch.double,requires_grad=True)
    correlations.data[5] = .03
    read = dict(margins=margins.reshape(4,8),objectives=torch.cat(((correlations[0]-correlations[1:].max()).reshape(1),torch.zeros(4))),
                active=dict(other_offsets=[5]))
    g = torch.autograd.grad(method.protection_scalar("bit29",read,None,None,parameters),margins)[0]
    assert g[29] == 1 and g.count_nonzero() == 1
    barrier = method.protection_scalar("barrier25",read,None,None,parameters)
    grad = torch.autograd.grad(barrier,margins)[0]
    exact = -torch.log((margins[parameters["indices"]]-.02)/torch.tensor(parameters["start_gaps"],dtype=torch.double)).mean()
    torch.testing.assert_close(grad,torch.autograd.grad(exact,margins)[0])
    assert grad[29] == 0 and grad.count_nonzero() == 25
    state = torch.autograd.grad(method.protection_scalar("state_G",read,None,None,parameters),correlations)[0]
    assert state[0] == 1 and state[5] == -1 and state.count_nonzero() == 2
    rgb = torch.linspace(.1,.9,24,dtype=torch.double).reshape(2,2,2,3).requires_grad_()
    base = torch.full_like(rgb,.3)
    q = method.protection_scalar("quality_Q",read,rgb,base,parameters)
    torch.testing.assert_close(q,((rgb-base)**2).mean())
    torch.testing.assert_close(torch.autograd.grad(q,rgb)[0],2*(rgb-base)/rgb.numel())
    broken = reference()["margins"]; broken[0] = .02
    assert method.barrier_value(broken,parameters)["status"] == "OUTSIDE_LOG_DOMAIN"
    read["active"]["other_offsets"] = [5,6]
    with pytest.raises(method.UndefinedProtection,match="tied"):
        method.protection_scalar("state_G",read,None,None,parameters)


def problem(targets,protections):
    g = [torch.tensor(row,dtype=torch.float64) for row in targets+protections]
    return method.solve_direction(g,torch.zeros(3),torch.tensor([.9,0,0]))


def test_socp_raw_maxmin_tangent_and_tiny_quality_scale():
    # Raw objective optimum balances d1=2*d2, unlike separately normalized rows.
    targets = [[0,1,0],[0,0,2]]+[[0,3,3]]*4
    protections = [[0,1,0],[0,-1,-1],[0,0,1],[0,-1e-20,-1e-20]]
    direction,report = problem(targets,protections)
    assert report["status"] == "POSITIVE_COMMON_DIRECTION",report
    assert abs(float(direction[0])) < 1e-12
    assert direction.norm() == pytest.approx(1)
    assert float(direction[1]/direction[2]) == pytest.approx(2,abs=1e-5)
    assert report["min_target_slope"] == pytest.approx(2/(5**.5),abs=1e-7)
    assert report["residual"] <= method.RESIDUAL_TOL
    scaled = [protections[0],protections[1],protections[2],[0,-100,-100]]
    other,_ = problem(targets,scaled)
    torch.testing.assert_close(other,direction,rtol=0,atol=1e-7)


def test_socp_zero_and_failed_solver_preserve_unresolved(monkeypatch):
    direction,report = problem([[0,1,0]]*6,[[0,-1,0],[0,0,0],[0,0,0],[0,0,0]])
    assert direction is None and report["status"] in ("NO_CLEAR_POSITIVE_DIRECTION","NUMERICALLY_UNRESOLVED")
    assert report["zero_direction_feasible"]
    import cvxpy
    def failed(*a,**k): raise RuntimeError("solver fault")
    monkeypatch.setattr(cvxpy.Problem,"solve",failed)
    direction,report = problem([[0,1,0]]*6,[[0,0,0]]*4)
    assert direction is None and report["status"] == "NUMERICALLY_UNRESOLVED"
    assert "solver fault" in report["reason"]


def setup(tmp_path,monkeypatch,*,no_direction=False):
    source = tmp_path/"source"; source.mkdir()
    six = source/"six"; six.mkdir(); (six/"units").mkdir()
    tangent = source/"tangent"; (tangent/"inputs").mkdir(parents=True)
    base = torch.randn(SHAPE,generator=torch.Generator().manual_seed(16))*.2
    delta = runtime.masked(torch.randn(SHAPE,generator=torch.Generator().manual_seed(3)),SUPPORT).double()
    delta *= .9/delta.norm(); start = (base.double()+delta).float()
    actual = start.double()-base.double(); u = actual/actual.norm()
    direction = runtime.masked(torch.randn(SHAPE,generator=torch.Generator().manual_seed(2)),SUPPORT).double()
    direction -= (direction*u).sum()*u; direction /= direction.norm()
    candidate = (start.double()+.1*direction).float()
    torch.save(dict(base=base,final=start),six/"inputs.pt")
    torch.save(dict(base=base,start=start,candidate=candidate),tangent/"inputs/fixed_inputs.pt")
    for bit in method.BITS:
        torch.save(dict(bit_index=bit,input_terminal=start,gradient=direction.float(),
            readout=dict(derivative_sign="positive_margin")),six/"units"/f"bit_{bit:02d}.pt")
    for name,terminal in dict(BASE=base,START_FINAL=start,TANGENT=candidate).items():
        directory=tangent/name; directory.mkdir()
        rgb=runtime.adapter.decode_normalized_latent(TinyWan(),terminal)
        rows=runtime.observer.read_rgb(rgb,key=runtime.KEY,rho=.5,protocol=PROTOCOL)
        evaluation=runtime.observer.evaluate_saved(rows,key=runtime.KEY,message=bytes.fromhex("8001a55a"),rho=.5,protocol=PROTOCOL)
        # Deliberately synthetic saved scores give this execution fixture the adopted
        # START active set; numerical scalar correctness is independently tested above.
        if name == "START_FINAL":
            for row,value in zip(evaluation["payload"]["metrics"],reference()["margins"]): row["signed_mean"]=value
        torch.save(terminal,directory/"terminal_latent.pt"); torch.save(rgb,directory/"float_rgb.pt")
        torch.save(torch.ones(22,4,16),directory/"q.pt")
        (directory/"raw.json").write_text(json.dumps(rows)); (directory/"metrics.json").write_text(json.dumps(evaluation))
    config=dict(device="cpu",model={},carrier=dict(key=runtime.KEY,message_hex="8001a55a",rho=.5,cap=1.),
        step_size=.1,inputs=dict(six_bit_inputs=str(six/"inputs.pt"),gradient_units=str(six/"units"),tangent_run=str(tangent)))
    def solve(*a):
        return (None if no_direction else direction),dict(status="NO_CLEAR_POSITIVE_DIRECTION" if no_direction else "POSITIVE_COMMON_DIRECTION",fixture=True)
    monkeypatch.setattr(method,"solve_direction",solve)
    spool=tmp_path/"spool"; spool.mkdir(); monkeypatch.setenv("M0_BOUNDARY_SPOOL_ROOT",str(spool))
    return config,base,start,direction


def invoke(config,output,model):
    return runtime.run(config,output,loader=lambda *_a,**_k:model,protocol=PROTOCOL,latent_shape=SHAPE,support=SUPPORT)


def test_planned_pause_fresh_instance_resume_and_complete_cpu_fastpath(tmp_path,monkeypatch):
    config,base,start,direction=setup(tmp_path,monkeypatch); output=tmp_path/"out"
    model=TinyWan(); first=invoke(config,output,model)
    assert first["status"] == "PAUSED_FOR_SESSION_RESTART",first["failures"]
    assert first["counts"]["decode_completed"] == first["counts"]["decoder_vjp_completed"] == 1
    assert first["counts"]["vae_load_completed"] == 1 and model.calls == 1
    assert first["totals"]["observed_metrics"] == 165
    assert (output/"controlled_pause.json").is_file() and (output/"units/bit29.pt").is_file()
    assert first["replay"]["bit29"]["boundary_storage"]["closed"]
    # New model instance stands for a second process. This is not real Colab evidence.
    second_model=TinyWan(); result=invoke(config,output,second_model)
    assert result["status"] == "COMPLETE",result["failures"]
    assert second_model.calls == 4
    assert result["counts"]["vae_load_completed"] == 2
    assert result["counts"]["decode_completed"] == 5 and result["counts"]["decoder_vjp_completed"] == 4
    assert result["current_attempt_counts"]["decode_completed"] == 4
    assert result["ordinary_chunks"] == dict(attempted=46,completed=46)
    assert result["physical_replay_chunks"] == dict(forward=184,recompute=184)
    assert result["totals"]["expected_metrics"] == result["totals"]["observed_metrics"] == 220
    assert result["gradient_probe_totals"]["observed_metrics"] == 220
    assert all(x["boundary_storage"]["closed"] for x in result["replay"].values())
    assert not list((tmp_path/"spool").iterdir())
    terminal=torch.load(output/"candidate_unit.pt",weights_only=True)["terminal"]
    assert torch.equal(terminal,(start.double()+.1*direction).float())
    assert result["steps"]["step"]["actual_radius"] <= 1
    comparison=resume.read_json(output/"comparison.json")
    assert len(comparison["bits_32"]) == 32 and comparison["protections"]["PROTECTED"]["status"] == "SCORED"
    assert comparison["protections"]["PROTECTED"]["barrier"]["status"] == "OUTSIDE_LOG_DOMAIN"
    assert len(resume.read_json(output/"PROTECTED/quality_vs_BASE.json")["framewise"]) == 181
    for key in config["inputs"]: config["inputs"][key]=str(tmp_path/"missing"/key)
    def forbidden(*a,**k): raise AssertionError("completed task called model")
    monkeypatch.setattr(runtime.generation,"load_frozen_vae",forbidden)
    assert not runtime.model_work_remaining(output)
    again=runtime.run(config,output,protocol=PROTOCOL,latent_shape=SHAPE,support=SUPPORT)
    assert again["status"] == "COMPLETE",again["failures"]
    assert not any(again["current_attempt_counts"].values())
    assert again["counts"] == result["counts"]
    assert again["attempt_history"][0]["status"] == "PAUSED_FOR_SESSION_RESTART"


def test_no_clear_direction_preserves_fixed_slots_without_candidate_decode(tmp_path,monkeypatch):
    config,*_=setup(tmp_path,monkeypatch,no_direction=True); output=tmp_path/"out"
    assert invoke(config,output,TinyWan())["status"] == "PAUSED_FOR_SESSION_RESTART"
    result=invoke(config,output,TinyWan())
    assert result["status"] == "COMPLETE_NO_CANDIDATE",result["failures"]
    assert result["counts"]["decode_completed"] == result["counts"]["decoder_vjp_completed"] == 4
    assert result["views"]["PROTECTED"]["status"] == "NOT_RUN"
    assert result["totals"]["expected_metrics"] == 220 and result["totals"]["observed_metrics"] == 165
    assert not (output/"candidate_unit.pt").exists() and not runtime.model_work_remaining(output)
    assert not any(invoke(config,output,TinyWan())["current_attempt_counts"].values())


@pytest.mark.parametrize("boundary",["gradient","gradient_saved","rgb_uncommitted","rgb_saved"])
def test_interrupted_unit_resume_and_report_recovery(tmp_path,monkeypatch,boundary):
    config,*_=setup(tmp_path,monkeypatch); output=tmp_path/"out"
    invoke(config,output,TinyWan())
    atomic=resume.atomic_tensor
    def interruption(path,value):
        if boundary == "gradient" and str(path).endswith("units/barrier25.pt"):
            raise KeyboardInterrupt("expiry before gradient commit")
        if boundary == "rgb_uncommitted" and str(path).endswith("PROTECTED/float_rgb.pt"):
            raise KeyboardInterrupt("decode returned but RGB not committed")
        atomic(path,value)
        if boundary == "gradient_saved" and str(path).endswith("units/barrier25.pt"):
            raise KeyboardInterrupt("gradient saved before completion marker")
        if boundary == "rgb_saved" and str(path).endswith("PROTECTED/float_rgb.pt"):
            raise KeyboardInterrupt("expiry after RGB commit")
    monkeypatch.setattr(resume,"atomic_tensor",interruption)
    with pytest.raises(KeyboardInterrupt): invoke(config,output,TinyWan())
    monkeypatch.setattr(resume,"atomic_tensor",atomic)
    if boundary == "rgb_uncommitted":
        interrupted=runtime.read_state(output)
        assert interrupted["counts"]["decode_completed"] == 5
        assert not (output/"rgb_commits/PROTECTED.json").exists()
        assert runtime.model_work_remaining(output)
    model=TinyWan(); result=invoke(config,output,model)
    assert result["status"] == "COMPLETE",result["failures"]
    assert result["failure_history"]
    if boundary == "gradient":
        assert model.calls == 4 and result["counts"]["decoder_vjp_completed"] == 5
        assert result["counts"]["decode_completed"] == 6
    elif boundary == "gradient_saved":
        assert model.calls == 3 and result["counts"]["decoder_vjp_completed"] == 4
        assert result["counts"]["decode_completed"] == 5
        assert (output/"gradient_commits/barrier25.json").is_file()
    elif boundary == "rgb_uncommitted":
        assert model.calls == 1 and result["current_attempt_counts"]["decode_completed"] == 1
        assert result["counts"]["decode_completed"] == 6 and result["counts"]["decoder_vjp_completed"] == 4
    else:
        assert model.calls == 0 and not any(result["current_attempt_counts"].values())
        assert result["counts"]["decode_completed"] == 5


def test_wrong_old_gradient_point_fails_before_any_model_load(tmp_path,monkeypatch):
    config,*_=setup(tmp_path,monkeypatch); path=__import__('pathlib').Path(config['inputs']['gradient_units'])/'bit_04.pt'
    unit=torch.load(path,weights_only=True); unit['input_terminal']=torch.zeros_like(unit['input_terminal']); torch.save(unit,path)
    result=invoke(config,tmp_path/'out',TinyWan())
    assert result['status'] == 'ENGINEERING_FAILURE' and not result['counts']['vae_load_attempted']
    assert 'different bit or numerical START' in result['failures'][0]['reason']


def test_committed_rgb_missing_or_unreadable_is_never_redecoded(tmp_path,monkeypatch):
    import shutil
    config,*_=setup(tmp_path,monkeypatch); template=tmp_path/"template"
    invoke(config,template,TinyWan()); completed=invoke(config,template,TinyWan())
    assert completed["status"] == "COMPLETE"
    for case in ("missing","unreadable","legacy_scored"):
        output=tmp_path/case; shutil.copytree(template,output)
        (output/"PROTECTED/raw.json").unlink()
        if case == "unreadable": (output/"PROTECTED/float_rgb.pt").write_bytes(b"unreadable RGB")
        else: (output/"PROTECTED/float_rgb.pt").unlink()
        if case == "legacy_scored":
            (output/"rgb_commits/PROTECTED.json").unlink()
            state=resume.read_json(output/"result.json")
            state.pop("completed_rgb_views",None)
            resume.atomic_bytes(output/"result.json",resume.json_bytes(state))
        for _ in range(2):
            assert not runtime.model_work_remaining(output)
            model=TinyWan(); result=invoke(config,output,model)
            assert result["status"] == "ENGINEERING_FAILURE"
            assert model.calls == 0 and not any(result["current_attempt_counts"].values())
            assert result["counts"] == completed["counts"]
            assert result["failure_history"]
            assert "PROTECTED" in result["completed_rgb_views"]


def test_view_q_and_png_cpu_repair_from_saved_rgb_and_raw(tmp_path,monkeypatch):
    config,*_=setup(tmp_path,monkeypatch); output=tmp_path/"out"
    invoke(config,output,TinyWan()); completed=invoke(config,output,TinyWan())
    assert completed["status"] == "COMPLETE"
    (output/"BASE/q.pt").write_bytes(b"unreadable q")
    torch.save(torch.ones(4,16),output/"START_FINAL/q.pt")
    torch.save(torch.full((22,4,16),float("nan")),output/"TANGENT/q.pt")
    torch.save(torch.zeros(22,4,16),output/"PROTECTED/q.pt")
    index=resume.read_json(output/"PROTECTED/frames_vs_BASE.json")
    (output/"PROTECTED"/index["frames"][0]["path"]).write_bytes(b"bad PNG")
    assert not runtime.reports_complete(output,list(method.VIEWS),181)
    model=TinyWan(); result=invoke(config,output,model)
    assert result["status"] == "COMPLETE",result["failures"]
    assert model.calls == 0 and not any(result["current_attempt_counts"].values())
    assert result["counts"] == completed["counts"]
    for name in method.VIEWS:
        expected=runtime.raw_q(resume.read_json(output/name/"raw.json"))
        assert runtime.q_matches_raw(output/name/"q.pt",expected)
    assert runtime.reports_complete(output,list(method.VIEWS),181)
    # Missing derived q, including a missing source q, also recovers from raw.
    (output/"BASE/q.pt").unlink()
    __import__('pathlib').Path(config["inputs"]["tangent_run"],"BASE/q.pt").unlink()
    again=invoke(config,output,TinyWan())
    assert again["status"] == "COMPLETE" and not any(again["current_attempt_counts"].values())
    assert runtime.q_matches_raw(output/"BASE/q.pt",runtime.raw_q(resume.read_json(output/"BASE/raw.json")))


def test_paused_committed_gradient_loss_is_retained_without_model_work(tmp_path,monkeypatch):
    import shutil
    config,*_=setup(tmp_path,monkeypatch); template=tmp_path/"paused"
    paused=invoke(config,template,TinyWan())
    assert paused["status"] == "PAUSED_FOR_SESSION_RESTART"
    assert (template/"gradient_commits/bit29.json").exists()
    for kind in ("missing","corrupt","legacy_pause"):
        output=tmp_path/kind; shutil.copytree(template,output)
        if kind == "corrupt": (output/"units/bit29.pt").write_bytes(b"unreadable gradient")
        else: (output/"units/bit29.pt").unlink()
        if kind == "legacy_pause":
            (output/"gradient_commits/bit29.json").unlink()
            state=resume.read_json(output/"result.json")
            state.pop("completed_gradient_units",None)
            state["gradient_units"]["bit29"]["status"]="RUNNING"
            state.pop("controlled_pause",None)
            resume.atomic_bytes(output/"result.json",resume.json_bytes(state))
            # The durable pause sidecar still proves bit29 was committed/read back.
        for _ in range(2):
            assert not runtime.model_work_remaining(output)
            model=TinyWan(); result=invoke(config,output,model)
            assert result["status"] == "ENGINEERING_FAILURE"
            assert model.calls == 0 and not any(result["current_attempt_counts"].values())
            assert result["counts"] == paused["counts"]
            assert "bit29" in result["completed_gradient_units"]
            assert result["failures"][-1]["stage"] == "bit29"


def test_later_committed_gradient_loss_is_found_before_any_model_load(tmp_path,monkeypatch):
    config,*_=setup(tmp_path,monkeypatch); output=tmp_path/"out"
    invoke(config,output,TinyWan())
    atomic=resume.atomic_tensor
    def interrupt_before_state_unit(path,value):
        if str(path).endswith("units/state_G.pt"):
            raise KeyboardInterrupt("state unit is unfinished")
        atomic(path,value)
    monkeypatch.setattr(resume,"atomic_tensor",interrupt_before_state_unit)
    with pytest.raises(KeyboardInterrupt): invoke(config,output,TinyWan())
    monkeypatch.setattr(resume,"atomic_tensor",atomic)
    saved=runtime.read_state(output)
    assert saved["gradient_units"]["barrier25"]["status"] == "COMPLETE"
    assert (output/"gradient_commits/barrier25.json").is_file()
    (output/"units/barrier25.pt").unlink()
    for _ in range(2):
        assert not runtime.model_work_remaining(output)
        model=TinyWan(); result=invoke(config,output,model)
        assert result["status"] == "ENGINEERING_FAILURE"
        assert model.calls == 0 and not any(result["current_attempt_counts"].values())
        assert result["counts"] == saved["counts"]
        assert "barrier25" in result["completed_gradient_units"]
        assert result["failures"][-1]["stage"] == "barrier25"
