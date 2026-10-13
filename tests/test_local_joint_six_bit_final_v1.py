from __future__ import annotations

import copy
import json

import pytest
import torch

from main.tube_state import local_joint_six_bit_final_v1 as method
from main.tube_state import local_joint_all_bit_v1 as loss_method
from runtime.wan import local_joint_six_bit_final_v1 as runtime
from runtime.wan import local_joint_all_bit_resume_v1 as resume
from runtime.wan import vae as adapter
from tests.test_local_joint_readout_m0_v1 import TinyWan, PROTOCOL, SHAPE, SUPPORT

pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def values():
    margins = [.1]*32
    for i in method.BITS: margins[i] = -.1
    return dict(margins=margins,correlations=[.03]+[.01]*21,state_gap=.02,
                active=dict(fragment_bits=[[0,1]]*4,other_offsets=list(range(1,22))))


def test_exact_positive_flat_index_and_conditional_loss_sum():
    x = torch.linspace(-.5,.5,32,dtype=torch.double,requires_grad=True)
    read = dict(margins=x.reshape(4,8))
    for bit in method.BITS:
        gradient = torch.autograd.grad(method.selected_margin(read,bit),x,retain_graph=True)[0]
        assert gradient[bit] == 1 and gradient.sum() == 1
    v = values()
    assert method.loss_condition(v)["valid"]  # Fragment minima / wrong-offset ties do not gate.
    margins = torch.tensor(v["margins"],dtype=torch.double,requires_grad=True)
    state = torch.tensor(v["state_gap"],dtype=torch.double,requires_grad=True)
    read = dict(margins=margins.reshape(4,8),objectives=state.reshape(1),active=v["active"])
    g,state_g = torch.autograd.grad(loss_method.scalar_loss(read),(margins,state))
    expected = torch.zeros(32,dtype=torch.double); expected[list(method.BITS)] = -.025
    torch.testing.assert_close(g,expected,rtol=0,atol=0); assert state_g == 0
    for mutation in (lambda a:a["margins"].__setitem__(0,.01),
                     lambda a:a["margins"].__setitem__(4,.001),
                     lambda a:a.__setitem__("state_gap",.005)):
        other = copy.deepcopy(v); mutation(other)
        assert not method.loss_condition(other)["valid"]


def diagnose(vectors, final=None):
    g = [torch.tensor(row,dtype=torch.double) for row in vectors]
    base = torch.zeros_like(g[0]); final = final if final is not None else base
    return method.analyze(g,base,final,[dict(values=values()) for _ in method.BITS])


def test_unnormalized_common_gram_cosine_and_residuals():
    vectors = [[1,0],[2,0],[1,1],[1,-1],[3,2],[4,-3]]
    report,directions = diagnose(vectors,torch.tensor([.9,0],dtype=torch.double))
    expected = torch.tensor(vectors,dtype=torch.double)
    assert report["raw"]["gram"] == (expected@expected.T).tolist()
    assert report["raw"]["norms"][1] == 2
    assert report["common"]["status"] == "NUMERICALLY_SUPPORTED_STRICT_COMMON_DIRECTION"
    assert min(report["common"]["direction"]["slope_lower_bounds"]) > 0
    assert report["common"]["primal_residual"] < 1e-12
    assert report["common"]["dual_residual"] < 1e-12
    assert report["loss_gradient_formula_valid_on_all_six_reads"]
    assert report["tangent"]["status"] == "NUMERICALLY_UNRESOLVED"
    assert report["pure_tangent_distance"] == pytest.approx((1-.9**2)**.5)
    assert directions["common"][0] > 0
    # A negative pairwise cosine still permits a strict common ascent direction.
    other,_ = diagnose([[1,2],[1,-2],[2,0],[3,0],[4,1],[4,-1]])
    assert other["raw"]["cosine"][0][1] < 0
    assert other["common"]["status"] == "NUMERICALLY_SUPPORTED_STRICT_COMMON_DIRECTION"


@pytest.mark.parametrize("vectors",[
    [[0,0]]*6,
    [[1,0],[-1,0],[0,1],[0,1],[0,1],[0,1]],
    [[1,1e-9],[-1,1e-9],[0,1],[0,1],[0,1],[0,1]],
])
def test_zero_opposite_near_zero_never_claim_infeasibility(vectors):
    report,_ = diagnose(vectors)
    assert report["common"]["status"] == "NUMERICALLY_UNRESOLVED"
    assert report["tangent"]["status"] == "UNDEFINED_RADIAL_AXIS_AT_BASE"
    assert "infeasibility certificate" in report["common"]["ceiling"]
    if vectors == [[0,0]]*6:
        assert report["aggregate"]["status"] == "UNDEFINED_ZERO_VECTOR"
        assert report["raw"]["cosine"][0][0] is None


def test_pure_tangent_and_geometry_do_not_generate_terminal():
    report,directions = diagnose([[1,1],[2,1],[-1,2],[-2,2],[0,3],[4,4]],
                                 torch.tensor([.9,0],dtype=torch.double))
    assert report["tangent"]["status"] == "NUMERICALLY_SUPPORTED_STRICT_COMMON_DIRECTION"
    assert abs(float(directions["tangent_common"][0])) < 1e-15
    assert report["tangent"]["direction"]["ball_distance"] == pytest.approx((1-.9**2)**.5)


def setup(tmp_path):
    source = tmp_path/"source"; source.mkdir()
    base = torch.randn(SHAPE,generator=torch.Generator().manual_seed(4))*.3
    delta = runtime.masked(torch.randn(SHAPE,generator=torch.Generator().manual_seed(6)),SUPPORT)
    delta = delta/float(delta.double().norm())*.8
    final = base+delta
    model = TinyWan()
    with torch.no_grad():
        mean,std = adapter._scale_tensors(model,final)
        rgb = (model.decode(final*std+mean)[0][0].permute(1,2,3,0)/2+.5).clamp(0,1)
        reference = method.values55(method.tensor_readout(rgb,key=runtime.KEY,message=bytes.fromhex("8001a55a"),protocol=PROTOCOL))
    model.calls = 0; model.clear_cache()
    torch.save(base,source/"base.pt")
    torch.save(dict(terminal=final,input_terminal=torch.zeros_like(final)),source/"step_2_unit.pt")
    (source/"result.json").write_text(json.dumps(dict(views=dict(CONT_FINAL=reference))))
    config = dict(inputs=dict(terminal_latent=str(source/"base.pt"),final_step_unit=str(source/"step_2_unit.pt"),
                             saved_final_result=str(source/"result.json")),device="cpu",model={},
                  carrier=dict(key=runtime.KEY,message_hex="8001a55a",rho=.5,cap=1.),
                  bits=list(method.BITS),loss=method.LOSS_PARAMETERS)
    return config,base,final,model


def invoke(config,output,model):
    return runtime.run(config,output,loader=lambda *_a,**_k:model,protocol=PROTOCOL,latent_shape=SHAPE,support=SUPPORT)


def test_serial_saved_pair_resume_and_all_complete_zero_model_calls(tmp_path,monkeypatch):
    config,base,final,model = setup(tmp_path)
    spool = tmp_path/"spool"; spool.mkdir()
    monkeypatch.setenv("M0_BOUNDARY_SPOOL_ROOT",str(spool))
    ledgers = []
    original_ledger = runtime.ReplayLedger
    def ledger(*a,**kw):
        assert all(x.summary()["boundary_storage"]["closed"] for x in ledgers)
        current = original_ledger(*a,**kw); ledgers.append(current); return current
    monkeypatch.setattr(runtime,"ReplayLedger",ledger)
    atomic = resume.atomic_tensor
    def fail_second_unit(path,value):
        if str(path).endswith("bit_12.pt"):
            raise OSError("injected unit commit failure")
        atomic(path,value)
    monkeypatch.setattr(resume,"atomic_tensor",fail_second_unit)
    output = tmp_path/"out"
    failed = invoke(config,output,model)
    assert failed["status"] == "ENGINEERING_FAILURE" and model.calls == 2
    assert failed["views"]["bit_04"]["status"] == "COMPLETE"
    assert failed["views"]["bit_12"]["status"] == "FAILED"
    assert len(failed["views"]) == 6 and failed["totals"]["expected_metrics"] == 330
    assert all(x.summary()["boundary_storage"]["closed"] for x in ledgers)
    monkeypatch.setattr(resume,"atomic_tensor",atomic)
    # Restore from previous readable JSON; durable bit04 must still win over stale state.
    (output/"result.json").write_text("torn")
    complete = invoke(config,output,model)
    assert complete["status"] == "COMPLETE",complete["failures"]
    assert model.calls == 7 and complete["counts"]["decoder_vjp_completed"] == 7
    assert complete["current_attempt_counts"]["decode_completed"] == 5
    assert complete["counts"]["vae_load_completed"] == 2
    assert complete["totals"]["observed_metrics"] == 330
    assert complete["physical_replay_chunks"] == dict(forward=7*46,recompute=7*46)
    assert complete["counts"]["ordinary_decode"] == 0
    assert complete["attempt_history"][0]["failures"]
    assert not list(spool.iterdir())
    for name,bit in zip(method.VIEWS,method.BITS):
        unit = torch.load(output/"units"/(name+".pt"),weights_only=True)
        assert torch.equal(unit["input_terminal"],final) and not torch.equal(final,torch.zeros_like(final))
        assert unit["bit_index"] == bit and unit["readout"]["max_abs_difference_55"] == 0
        assert unit["readout"]["payload_denominators"] == [24,24,20,20]
    # Source files are unnecessary after durable backup; no CUDA/loader on report-only rerun.
    for key in config["inputs"]: config["inputs"][key] = str(tmp_path/"missing"/key)
    def forbidden(*a,**k): raise AssertionError("complete task loaded model")
    monkeypatch.setattr(runtime.generation,"load_frozen_vae",forbidden)
    assert not runtime.model_work_remaining(output)
    again = runtime.run(config,output,protocol=PROTOCOL,latent_shape=SHAPE,support=SUPPORT)
    assert again["status"] == "COMPLETE" and not any(again["current_attempt_counts"].values())
    assert (output/"checkpoints/latest.json").exists() and (output/"checkpoints/previous.json").exists()
    assert (output/"progress.json").exists()
    assert complete["counts"] == again["counts"]
    # A committed wrong pair is a data fault, with zero replacement model calls.
    path = output/"units/bit_04.pt"
    invalid = torch.load(path,weights_only=True)
    invalid["input_terminal"] = base
    torch.save(invalid,path)
    bad = runtime.run(config,output,protocol=PROTOCOL,latent_shape=SHAPE,support=SUPPORT)
    assert bad["status"] == "ENGINEERING_FAILURE"
    assert not any(bad["current_attempt_counts"].values())
    assert "different numerical input" in bad["failures"][0]["reason"]


def test_wrong_directory_partial_state_and_invalid_pairs_never_silent_restart(tmp_path):
    config,_,_,model = setup(tmp_path)
    output = tmp_path/"out"; output.mkdir()
    (output/"some_previous_task.pt").write_text("existing")
    with pytest.raises(OSError,match="silent restart"): invoke(config,output,model)
    assert model.calls == 0
    other = tmp_path/"old"; other.mkdir()
    (other/"result.json").write_text(json.dumps(dict(views={},steps={},counts={},model_calls={})))
    with pytest.raises(OSError,match="No readable"): invoke(config,other,model)
    assert model.calls == 0


def test_interrupt_inside_third_vjp_only_runs_remaining_four(tmp_path,monkeypatch):
    config,base,final,model = setup(tmp_path)
    output = tmp_path/"out"
    original = runtime.ReplayLedger
    ledgers = []
    def ledger(callback):
        number = len(ledgers)
        def progress(summary):
            callback(summary)
            if number == 2 and summary["counts"]["vae_chunk"]["recompute"]["attempted"]:
                raise KeyboardInterrupt("injected expiry inside third VJP")
        current = original(callback=progress); ledgers.append(current); return current
    monkeypatch.setattr(runtime,"ReplayLedger",ledger)
    with pytest.raises(KeyboardInterrupt,match="expiry"):
        invoke(config,output,model)
    assert model.calls == 3
    state = runtime.read_state(output)
    assert state["status"] == "INTERRUPTED" and state["failure_history"]
    assert all(x.summary()["boundary_storage"]["closed"] for x in ledgers)
    monkeypatch.setattr(runtime,"ReplayLedger",original)
    completed = invoke(config,output,model)
    assert completed["status"] == "COMPLETE"
    assert completed["current_attempt_counts"]["decode_completed"] == 4
    assert model.calls == 7 and completed["counts"]["decoder_vjp_completed"] == 6
    assert completed["counts"]["decoder_vjp_attempted"] == 7
    # Autograd may start other independent recompute branches while propagating
    # the first Python exception. Retain the observed attempts, not a guessed one.
    interrupted_attempts = ledgers[2].counts["vae_chunk"]["recompute"]["attempted"]
    assert 1 <= interrupted_attempts <= 46
    assert completed["physical_replay_chunks"] == dict(forward=322,recompute=276+interrupted_attempts)
    old_decode = [r for r in completed["call_history"] if r["unit"] == "bit_18_gradient_decode"][0]
    assert old_decode["replay"]["counts"]["vae_chunk"]["recompute"]["attempted"] == interrupted_attempts


def test_actual_margin_forward55_matches_independent_reader_and_sign(tmp_path):
    config,base,final,model = setup(tmp_path)
    z = final.clone().requires_grad_(True)
    mean,std = adapter._scale_tensors(model,z)
    rgb = (model.decode(z*std+mean)[0][0].permute(1,2,3,0)/2+.5).clamp(0,1)
    read = method.tensor_readout(rgb,key=runtime.KEY,message=bytes.fromhex("8001a55a"),protocol=PROTOCOL)
    from main.tube_state import local_joint_terminal_bridge_v1 as observer
    from runtime.wan.local_joint_readout_m0_v1 import metric_values
    raw = observer.read_rgb(rgb.detach(),key=runtime.KEY,rho=.5,protocol=PROTOCOL)
    evaluation = observer.evaluate_saved(raw,key=runtime.KEY,message=bytes.fromhex("8001a55a"),rho=.5,protocol=PROTOCOL)
    independent = metric_values(evaluation)
    report = method.readout_report(read,independent,4)
    assert report["max_abs_difference_55"] < 1e-15
    g = torch.autograd.grad(method.selected_margin(read,4),z)[0]
    direction = runtime.masked(g,SUPPORT); direction /= direction.double().norm()
    predicted = float((g.double()*direction.double()).sum())
    def margin(at):
        model.clear_cache()
        decoded = model.decode(at*std+mean)[0][0].permute(1,2,3,0)
        return float(method.tensor_readout((decoded/2+.5).clamp(0,1),key=runtime.KEY,
                     message=bytes.fromhex("8001a55a"),protocol=PROTOCOL)["margins"].reshape(-1)[4])
    finite_difference = (margin(final+1e-3*direction)-margin(final-1e-3*direction))/2e-3
    assert predicted > 0 and finite_difference == pytest.approx(predicted,rel=.003)
