from __future__ import annotations

import json

import pytest
import torch

from main.tube_state import local_joint_tangent_common_v1 as method
from runtime.wan import local_joint_tangent_common_v1 as runtime
from runtime.wan import local_joint_all_bit_resume_v1 as resume
from tests.test_local_joint_readout_m0_v1 import TinyWan, PROTOCOL, SHAPE, SUPPORT

pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def setup(tmp_path):
    source = tmp_path/"source"; source.mkdir()
    base = torch.randn(SHAPE,generator=torch.Generator().manual_seed(16))*.2
    d = runtime.masked(torch.randn(SHAPE,generator=torch.Generator().manual_seed(3)),SUPPORT).double()
    d *= .9/d.norm(); start = (base.double()+d).float()
    actual = start.double()-base.double(); u = actual/actual.norm()
    direction = runtime.masked(torch.randn(SHAPE,generator=torch.Generator().manual_seed(2)),SUPPORT).double()
    direction -= (direction*u).sum()*u; direction /= direction.norm()
    torch.save(dict(base=base,final=start),source/"fixed_inputs.pt")
    torch.save(dict(tangent_common=direction),source/"directions.pt")
    (source/"units").mkdir()
    for bit in method.BITS:
        torch.save(dict(bit_index=bit,input_terminal=start,gradient=direction.float()),source/"units"/f"bit_{bit:02d}.pt")
    config = dict(device="cpu",model={},carrier=dict(key=runtime.KEY,message_hex="8001a55a",rho=.5,cap=1.),
        step_size=.1,inputs=dict(six_bit_inputs=str(source/"fixed_inputs.pt"),directions=str(source/"directions.pt"),
                               gradient_units=str(source/"units")))
    return config,base,start,direction


def invoke(config,output,model):
    return runtime.run(config,output,loader=lambda *_a,**_k:model,protocol=PROTOCOL,latent_shape=SHAPE,support=SUPPORT)


def test_actual_terminal_straight_tangent_and_cap_without_projection(tmp_path):
    _,base,start,direction = setup(tmp_path)
    terminal,report = method.candidate(base,start,direction,.1)
    assert torch.equal(terminal,(start.double()+.1*direction).float())
    assert report["actual_radius"] > report["start_radius"] and report["actual_radius"] < 1
    assert abs(report["start_delta_dot_direction"]) < 1e-12
    assert report["addition_rounding_l2"] > 0
    with pytest.raises(ValueError,match="no automatic projection"): method.candidate(base,start,direction,.6)
    with pytest.raises(ValueError,match="tangent"): method.candidate(base,start,2*direction,.1)
    radial = (start.double()-base.double()); radial /= radial.norm()
    with pytest.raises(ValueError,match="tangent"): method.candidate(base,start,radial,.1)


def test_complete_and_cpu_report_repair_no_repeated_model_calls(tmp_path,monkeypatch):
    config,base,start,direction = setup(tmp_path); output = tmp_path/"out"; model = TinyWan()
    result = invoke(config,output,model)
    assert result["status"] == "COMPLETE", result["failures"]
    assert model.calls == 3 and result["counts"]["vae_load_completed"] == 1
    assert result["counts"]["decode_completed"] == 3 and result["counts"]["decoder_vjp_attempted"] == 0
    assert result["ordinary_chunks"] == dict(attempted=138,completed=138)
    assert result["totals"]["observed_metrics"] == 165 and result["totals"]["observed_chips"] == 4224
    bundle = torch.load(output/"inputs/fixed_inputs.pt",weights_only=True)
    assert torch.equal(bundle["base"],base) and torch.equal(bundle["start"],start)
    assert torch.equal(bundle["direction"],direction)
    assert set(bundle["step_report"]["predicted_margin_changes"]) == {str(i) for i in method.BITS}
    comparison = resume.read_json(output/"comparison.json")
    assert len(comparison["bits_32"]) == 32 and comparison["bits_32"][0]["predicted_change"] is None
    assert all(len(result["views"][name]["margins"]) == 32 for name in method.VIEWS)
    for name in method.VIEWS:
        assert torch.load(output/name/"q.pt",weights_only=True).shape == (22,4,16)
        for ref in method.VIEWS[:2]:
            q = resume.read_json(output/name/("quality_vs_"+ref+".json"))
            assert len(q["framewise"]) == 181 and len(q["temporal_framewise"]) == 180
    assert not runtime.model_work_remaining(output)
    # Inputs can disappear after durable backup. Repair reports/images on CPU.
    for key in config["inputs"]: config["inputs"][key] = str(tmp_path/"missing"/key)
    (output/"TANGENT/metrics.json").write_text("broken")
    (output/"TANGENT/raw.json").write_text("broken")
    (output/"TANGENT/quality_vs_START_FINAL.json").unlink()
    frames = resume.read_json(output/"TANGENT/frames_vs_BASE.json")
    (output/"TANGENT"/frames["frames"][0]["path"]).unlink()
    def forbidden(*a,**k): raise AssertionError("unexpected model")
    monkeypatch.setattr(runtime.generation,"load_frozen_vae",forbidden)
    again = runtime.run(config,output,protocol=PROTOCOL,latent_shape=SHAPE,support=SUPPORT)
    assert again["status"] == "COMPLETE",again["failures"]
    assert not any(again["current_attempt_counts"].values())
    assert again["counts"] == result["counts"]
    assert (output/"checkpoints/latest.json").exists() and (output/"checkpoints/previous.json").exists()
    third = runtime.run(config,output,protocol=PROTOCOL,latent_shape=SHAPE,support=SUPPORT)
    assert third["last_resume_action"].startswith("report_only")


def test_decode_interruption_replays_only_incomplete_unit_and_retains_calls(tmp_path,monkeypatch):
    config,_,_,_ = setup(tmp_path); output = tmp_path/"out"; model = TinyWan()
    original = runtime.adapter.decode_normalized_latent
    def interrupted(m,z):
        if model.calls == 1:
            raise KeyboardInterrupt("simulated runtime expiry")
        return original(m,z)
    monkeypatch.setattr(runtime.adapter,"decode_normalized_latent",interrupted)
    with pytest.raises(KeyboardInterrupt,match="expiry"): invoke(config,output,model)
    assert (output/"BASE/float_rgb.pt").is_file() and model.calls == 1
    assert runtime.read_state(output)["failure_history"]
    monkeypatch.setattr(runtime.adapter,"decode_normalized_latent",original)
    result = invoke(config,output,model)
    assert result["status"] == "COMPLETE",result["failures"]
    assert model.calls == 3 and result["counts"]["vae_load_completed"] == 2
    assert result["current_attempt_counts"]["decode_completed"] == 2
    assert result["attempt_history"][0]["status"] == "INTERRUPTED"
    assert result["failure_history"]


def test_failed_base_read_does_not_block_two_other_decodes(tmp_path,monkeypatch):
    config,_,_,_ = setup(tmp_path); output = tmp_path/"out"; model = TinyWan()
    original = runtime.observer.read_rgb; calls = []
    def reader(*a,**kw):
        calls.append(1)
        if len(calls) == 1: raise RuntimeError("BASE read failure")
        return original(*a,**kw)
    monkeypatch.setattr(runtime.observer,"read_rgb",reader)
    result = invoke(config,output,model)
    assert result["status"] == "ENGINEERING_FAILURE" and model.calls == 3
    assert result["views"]["START_FINAL"]["status"] == result["views"]["TANGENT"]["status"] == "SCORED"
    assert result["totals"]["expected_metrics"] == 165 and result["totals"]["observed_metrics"] == 110
    monkeypatch.setattr(runtime.observer,"read_rgb",original)
    recovered = invoke(config,output,model)
    assert recovered["status"] == "COMPLETE" and model.calls == 3
    assert recovered["failure_history"]


def test_missing_optional_gradients_does_not_block_candidate(tmp_path):
    config,_,_,_ = setup(tmp_path)
    config["inputs"]["gradient_units"] = str(tmp_path/"missing_units")
    result = invoke(config,tmp_path/"out",TinyWan())
    assert result["status"] == "COMPLETE", result["failures"]
    assert len(result["steps"]["step"]["prediction_notes"]) == 6
    assert not result["steps"]["step"]["predicted_margin_changes"]


def test_expiry_after_decode_before_rgb_commit_counts_physical_replay(tmp_path,monkeypatch):
    config,_,_,_ = setup(tmp_path); output = tmp_path/"out"; model = TinyWan()
    original = resume.atomic_tensor
    def expire(path,value):
        if str(path).endswith("START_FINAL/float_rgb.pt"): raise KeyboardInterrupt("expiry at commit")
        return original(path,value)
    monkeypatch.setattr(resume,"atomic_tensor",expire)
    with pytest.raises(KeyboardInterrupt): invoke(config,output,model)
    assert model.calls == 2
    monkeypatch.setattr(resume,"atomic_tensor",original)
    result = invoke(config,output,model)
    assert result["status"] == "COMPLETE", result["failures"]
    assert model.calls == 4 and result["counts"]["decode_completed"] == 4
    assert result["ordinary_chunks"] == dict(attempted=184,completed=184)
    assert result["current_attempt_counts"]["decode_completed"] == 2


def test_timer_inside_decode_and_torn_state_restore(tmp_path,monkeypatch):
    import time
    config,_,_,_ = setup(tmp_path); output = tmp_path/"out"; model = TinyWan()
    original_init = resume.Journal.__init__
    def short_timer(self,*a,**kw):
        original_init(self,*a,**kw); self.interval = .005
    monkeypatch.setattr(resume.Journal,"__init__",short_timer)
    original_forward = model.decoder.forward; chunk = 0; observed = []
    def expire(*a,**kw):
        nonlocal chunk
        if model.calls == 2:
            chunk += 1
            if chunk == 3:
                time.sleep(.04)
                progress = resume.read_json(output/"progress.json")
                observed.append(progress)
                raise KeyboardInterrupt("expiry inside decoder")
        return original_forward(*a,**kw)
    model.decoder.forward = expire
    with pytest.raises(KeyboardInterrupt): invoke(config,output,model)
    assert observed[0]["reason"] == "60-second last committed state"
    assert "START_FINAL" in observed[0]["active_calls"]
    model.decoder.forward = original_forward
    (output/"result.json").write_text("torn whole state")
    result = invoke(config,output,model)
    assert result["status"] == "COMPLETE",result["failures"]
    assert result["counts"]["decode_attempted"] == 4 and result["counts"]["decode_completed"] == 3
    assert result["ordinary_chunks"] == dict(attempted=141,completed=140)
    unknown = [r for r in result["call_history"] if r["status"] == "INTERRUPTED_COMPLETION_UNKNOWN"]
    assert len(unknown) == 1 and unknown[0]["unit"] == "START_FINAL"
    assert result["recovery_events"] and result["failure_history"]


def test_unreadable_png_enters_cpu_recovery_without_model_calls(tmp_path):
    from PIL import Image
    config,_,_,_ = setup(tmp_path); output = tmp_path/"out"; model = TinyWan()
    first = invoke(config,output,model)
    assert first["status"] == "COMPLETE" and model.calls == 3
    index = resume.read_json(output/"TANGENT/frames_vs_BASE.json")
    path = output/"TANGENT"/index["frames"][0]["path"]
    path.write_bytes(b"")
    assert not runtime.reports_complete(output)
    recovered = invoke(config,output,model)
    assert recovered["status"] == "COMPLETE" and model.calls == 3
    assert recovered["counts"] == first["counts"]
    assert not any(recovered["current_attempt_counts"].values())
    with Image.open(path) as image: image.verify()
    assert runtime.reports_complete(output)


def test_damaged_failure_sidecars_preserve_files_and_readable_history(tmp_path):
    config,_,_,_ = setup(tmp_path); output = tmp_path/"out"; model = TinyWan()
    first = invoke(config,output,model)
    assert first["status"] == "COMPLETE" and model.calls == 3
    directory = output/"failures"; directory.mkdir(exist_ok=True)
    readable = dict(id="earlier_failure",attempt_id="earlier",stage="decode",reason="retained error")
    (directory/"readable.json").write_text(json.dumps(readable))
    damaged = {"torn.json":'{"bad":',"wrong_shape.json":"[]","missing_id.json":'{"reason":"old"}'}
    for name,contents in damaged.items(): (directory/name).write_text(contents)
    state = runtime.read_state(output)
    assert readable in state["failure_history"]
    assert {str(directory/name) for name in damaged} <= {row["path"] for row in state["recovery_events"]}
    recovered = invoke(config,output,model)
    assert recovered["status"] == "COMPLETE" and model.calls == 3
    assert not any(recovered["current_attempt_counts"].values())
    assert readable in recovered["failure_history"]
    for name,contents in damaged.items(): assert (directory/name).read_text() == contents
    assert all(row["reason"] for row in recovered["recovery_events"])
