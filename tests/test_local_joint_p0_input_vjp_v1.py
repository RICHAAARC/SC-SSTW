"""Bounded CPU fixtures: real autograd/checkpoint, fake Transformer, no model downloads."""
import copy
import json
from pathlib import Path

import pytest
import torch

from main.tube_state import local_joint_p0_input_vjp_v1 as method
from runtime.wan import local_joint_p0_input_vjp_v1 as runtime
from runtime.wan.local_joint_p0_checkpoint_v1 import checkpointed_branch

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.unit
SHAPE = (1, 1, 2, 2, 2)


@pytest.fixture(autouse=True)
def tiny_evidence_contract(monkeypatch):
    # Only tests change the fixed geometry; the CLI has no shape/contract override.
    monkeypatch.setattr(method, "LATENT_SHAPE", SHAPE)
    monkeypatch.setattr(method, "ROIS", ((0, 2, 0, 2),))


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.7, dtype=torch.bfloat16))

    def forward(self, x, embedding, timestep, rotary):
        return torch.sin(x * self.weight + embedding.mean())


class NonfiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, gradient):
        return gradient * float("nan")


class TinyTransformer(torch.nn.Module):
    def __init__(self, mode="normal"):
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block(), Block()])
        self.gradient_checkpointing = False
        self.mode = mode
        self.eval().requires_grad_(False)

    def enable_gradient_checkpointing(self, gradient_checkpointing_func=None):
        self.gradient_checkpointing = True
        self._gradient_checkpointing_func = gradient_checkpointing_func

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def forward(self, hidden_states, timestep, encoder_hidden_states, attention_kwargs, return_dict):
        assert attention_kwargs is None and return_dict is False
        if self.mode == "forward_failure":
            raise RuntimeError("fixture forward failure")
        if self.mode == "conditional_failure" and float(encoder_hidden_states.mean()) > 0:
            raise RuntimeError("fixture conditional-only failure")
        if self.mode == "oom":
            raise torch.OutOfMemoryError("fixture OOM; no actual device allocation")
        value = hidden_states
        for block in self.blocks:
            args = (value, encoder_hidden_states, timestep, (None, None))
            value = self._gradient_checkpointing_func(block, *args) if self.gradient_checkpointing else block(*args)
        if self.mode == "zero":
            value = value * 0
        if self.mode == "nonfinite":
            value = NonfiniteGradient.apply(value)
        return (value,)


def inputs():
    x = torch.arange(8, dtype=torch.float32).reshape(SHAPE) / 11
    d = torch.zeros_like(x)
    d[:, :, 1] = .5
    prompt = torch.ones((1, 2, 2), dtype=torch.bfloat16) * .1
    negative = prompt * -1
    model = TinyTransformer()
    predictions = {b: model(x.bfloat16(), torch.tensor([1.]), e, None, False)[0].float()
                   for b, e in zip(method.BRANCHES, (prompt, negative))}
    return dict(input=x, direction=d, prompt=prompt, negative=negative,
                timestep=torch.tensor([1.]), predictions=predictions, context={"fixture": True})


def config(tmp_path):
    c = json.loads((ROOT / "experiments/wan_state_clock/configs/local_joint_p0_input_vjp_v1.json").read_text())
    c.update(device="cpu", spool_root=str(tmp_path), source_run=str(tmp_path / "source"), tail_run=str(tmp_path / "tail"))
    return c


def execute(tmp_path, mode="normal", **kwargs):
    return runtime.run(config(tmp_path), tmp_path / "output", expected_blocks=2,
                       preparer=lambda *_: TinyTransformer(mode), source_loader=lambda _: inputs(), **kwargs)


def forbidden(*_args, **_kwargs):
    raise AssertionError("resume attempted forbidden physical work")


def test_checkpoint_preserves_bf16_input_pullback_and_releases_spool(tmp_path):
    source = inputs()
    model = TinyTransformer()
    reference = source["input"].clone().requires_grad_(True)
    y0 = model(reference.bfloat16(), source["timestep"], source["prompt"], None, False)[0].float()
    g0 = torch.autograd.grad(y0, reference, grad_outputs=source["direction"])[0]
    x = source["input"].clone().requires_grad_(True)
    snapshots = []
    with checkpointed_branch(model, snapshots.append, root=tmp_path) as ledger:
        y = model(x.bfloat16(), source["timestep"], source["prompt"], None, False)[0].float()
        gradient = torch.autograd.grad(y, x, grad_outputs=source["direction"])[0]
        assert torch.equal(y, y0) and torch.equal(gradient, g0)
        assert all(r["forward"] == r["recompute"] == "RETURNED" for r in ledger.rows)
        assert ledger.spool.peak_disk_bytes > 0
        directory = ledger.spool.path
    assert snapshots[-1]["boundary_storage"]["closed"] and not directory.exists()
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())


def test_fixed_calls_completed_resume_and_missing_committed_evidence(tmp_path):
    result = execute(tmp_path)
    assert result["status"] == "COMPLETE"
    assert [r["kind"] for r in result["calls"]] == ["transformer_prepare", "dit_forward", "dit_input_vjp", "dit_forward", "dit_input_vjp"]
    assert result["block_counts"]["recompute"]["attempted"] == 4
    assert all(r["comparison"]["exact_equal"] for r in result["branches"].values())
    result2 = runtime.run(config(tmp_path), tmp_path / "output", preparer=forbidden, source_loader=forbidden)
    assert result2["status"] == "COMPLETE" and len(result2["calls"]) == 5
    (tmp_path / "output/conditional/input_cotangent.pt").unlink()
    result3 = runtime.run(config(tmp_path), tmp_path / "output", preparer=forbidden, source_loader=forbidden)
    assert result3["status"] == "ENGINEERING_FAILURE" and result3["branches"]["conditional"]["status"] == "MISSING"
    assert len(result3["calls"]) == 5


@pytest.mark.parametrize("mode", ["zero", "nonfinite", "forward_failure"])
def test_zero_and_failed_units_are_retained_without_retry(tmp_path, mode):
    result = execute(tmp_path, mode)
    row = result["branches"]["conditional"]
    if mode == "zero":
        assert row["status"] == "COMPLETE" and row["cotangent_statistics"]["zero"]
    elif mode == "nonfinite":
        assert row["status"] == "NONFINITE" and (tmp_path / "output/conditional/input_cotangent.pt").exists()
        assert result["branches"]["unconditional"]["status"] == "MISSING"
    else:
        assert row["status"] == "FAILED" and result["counts"]["dit_input_vjp"]["attempted"] == 0
        assert row["forward"] == "FAILED"
        assert result["counts"]["dit_forward"]["attempted"] == 2
    calls = copy.deepcopy(result["calls"])
    reread = runtime.run(config(tmp_path), tmp_path / "output", preparer=forbidden, source_loader=forbidden)
    assert reread["calls"] == calls and reread["branches"]["conditional"]["status"] == row["status"]
    assert all(r[p] != "RUNNING" for r in reread["branches"].values() for p in ("forward", "vjp"))


def test_interrupted_vjp_keeps_forward_and_consumes_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.autograd, "grad", lambda *_a, **_k: (_ for _ in ()).throw(KeyboardInterrupt("fixture interruption")))
    with pytest.raises(KeyboardInterrupt):
        execute(tmp_path)
    result = runtime.run(config(tmp_path), tmp_path / "output", preparer=forbidden, source_loader=forbidden)
    assert result["branches"]["conditional"]["forward"] == "SAVED"
    assert result["branches"]["conditional"]["vjp"] == "INTERRUPTED_COMPLETION_UNKNOWN"
    assert (tmp_path / "output/conditional/output.pt").exists()
    assert result["counts"]["dit_input_vjp"]["INTERRUPTED_COMPLETION_UNKNOWN"] == 1
    assert result["counts"]["transformer_prepare"]["attempted"] == 1
    assert result["branches"]["unconditional"]["status"] == "MISSING"


def test_saved_source_checks_numeric_point_not_labels(tmp_path):
    c, data = config(tmp_path), inputs()
    source, tail = Path(c["source_run"]), Path(c["tail_run"])
    runtime.save_tensor(tail / "native/CONTROL_TAIL_after48.pt", data["input"])
    prediction = dict(input=data["input"], **data["predictions"])
    runtime.save_tensor(tail / "native/CONTROL_TAIL_step49_predictions.pt", prediction)
    runtime.save_tensor(source / "native/context.pt", dict(dtype="bfloat16", prompt=data["prompt"], negative=data["negative"]))
    runtime.save_tensor(source / "native/fork48.pt", dict(scheduler={"state": {"timesteps": torch.arange(50)}}))
    runtime.save_tensor(source / "units/gradient.pt", dict(input_terminal=data["input"], direction=data["direction"], status="arbitrary label"))
    runtime.save_tensor(source / "ZERO/terminal_latent.pt", data["input"])
    c["schema"] = "optional changed label"
    runtime.validate_config(c)
    loaded = runtime.load_source(c, shape=SHAPE, rois=((0, 2, 0, 2),), time_range=(0, 2))
    assert torch.equal(loaded["direction"], data["direction"])
    for dtype_label in (None, "changed optional label"):
        context = dict(prompt=data["prompt"], negative=data["negative"])
        if dtype_label is not None: context["dtype"] = dtype_label
        runtime.save_tensor(source / "native/context.pt", context)
        assert runtime.load_source(c, shape=SHAPE, rois=((0, 2, 0, 2),), time_range=(0, 2))["context"]["saved_dtype_label"] == dtype_label
    runtime.save_tensor(source / "native/context.pt", dict(dtype="bfloat16", prompt=data["prompt"].float(), negative=data["negative"]))
    with pytest.raises(ValueError, match="arithmetic dtype"):
        runtime.load_source(c, shape=SHAPE, rois=((0, 2, 0, 2),), time_range=(0, 2))
    runtime.save_tensor(source / "native/context.pt", dict(prompt=data["prompt"], negative=data["negative"]))
    prediction["input"] = data["input"] + 1
    runtime.save_tensor(tail / "native/CONTROL_TAIL_step49_predictions.pt", prediction)
    with pytest.raises(ValueError, match="different numerical input"):
        runtime.load_source(c, shape=SHAPE, rois=((0, 2, 0, 2),), time_range=(0, 2))


def test_preparation_failure_stays_failed_with_both_fixed_rows(tmp_path):
    def failed(*_):
        raise RuntimeError("fixture load failure")
    result = runtime.run(config(tmp_path), tmp_path / "output", preparer=failed, source_loader=lambda _: inputs())
    assert result["status"] == "ENGINEERING_FAILURE" and set(result["branches"]) == set(method.BRANCHES)
    reread = runtime.run(config(tmp_path), tmp_path / "output", preparer=forbidden, source_loader=forbidden)
    assert reread["counts"]["transformer_prepare"]["attempted"] == 1
    assert reread["counts"]["dit_forward"]["attempted"] == 0


@pytest.mark.parametrize("artifact,damage", [
    ("conditional/output.pt", "nan"), ("conditional/output.pt", "wrong_dtype"),
    ("conditional/output.pt", "wrong_shape"), ("conditional/input_cotangent.pt", "wrong_dtype"),
    ("saved_input.pt", "missing"), ("saved_input.pt", "nan"),
    ("saved_cotangent.pt", "missing"), ("saved_cotangent.pt", "wrong_norm"),
    ("saved_cotangent.pt", "outside_support")])
def test_resume_invalid_actual_evidence_never_complete_or_reexecuted(tmp_path, artifact, damage):
    result = execute(tmp_path)
    calls = copy.deepcopy(result["calls"])
    path = tmp_path / "output" / artifact
    if damage == "missing":
        path.unlink()
    else:
        value = runtime.tensor(path)
        if damage == "nan": value.reshape(-1)[0] = float("nan")
        elif damage == "wrong_dtype": value = value.double()
        elif damage == "wrong_shape": value = value.flatten()
        elif damage == "wrong_norm": value = value * 2
        elif damage == "outside_support": value = torch.roll(value, 1, dims=2)
        runtime.save_tensor(path, value)
    reread = runtime.run(config(tmp_path), tmp_path / "output", preparer=forbidden, source_loader=forbidden)
    assert reread["status"] != "COMPLETE" and reread["calls"] == calls
    assert any(r["status"] == "MISSING" for r in reread["branches"].values())


def test_native_dtype_exceptions_allowed_but_main_path_change_rejected():
    model = TinyTransformer()
    runtime.validate_model_arithmetic(model)
    model.blocks[0].weight.data = model.blocks[0].weight.data.float()
    with pytest.raises(ValueError, match="parameter arithmetic"):
        runtime.validate_model_arithmetic(model)
    model._keep_in_fp32_modules = ["weight"]
    assert runtime.validate_model_arithmetic(model)["parameter_dtypes"]["torch.float32"] == 1


def test_vjp_exception_terminal_phase_matches_physical_call(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.autograd, "grad", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("fixture VJP failure")))
    result = execute(tmp_path)
    assert result["branches"]["conditional"]["forward"] == "SAVED"
    assert result["branches"]["conditional"]["vjp"] == "FAILED"
    reread = runtime.run(config(tmp_path), tmp_path / "output", preparer=forbidden, source_loader=forbidden)
    assert reread["branches"]["conditional"]["vjp"] == "FAILED"


def test_ordinary_branch_failure_continues_independent_unit_but_oom_stops(tmp_path):
    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()
    result = execute(ordinary, "conditional_failure")
    assert result["branches"]["conditional"]["status"] == "FAILED"
    assert result["branches"]["unconditional"]["status"] == "COMPLETE"
    assert result["counts"]["transformer_prepare"]["attempted"] == 1
    assert result["counts"]["dit_forward"]["attempted"] == 2
    assert result["counts"]["dit_input_vjp"]["attempted"] == 1
    oom = tmp_path / "oom"
    oom.mkdir()
    stopped = execute(oom, "oom")
    assert stopped["branches"]["unconditional"]["status"] == "MISSING"
    assert stopped["counts"]["dit_forward"]["attempted"] == 1
    assert stopped["counts"]["dit_input_vjp"]["attempted"] == 0


def test_native_wan_checkpoint_hook_with_tiny_random_cpu_weights(tmp_path):
    """Exercise the real installed forward/checkpoint interface, never pretrained weights."""
    diffusers = pytest.importorskip("diffusers")
    model = diffusers.WanTransformer3DModel(in_channels=2, out_channels=2, text_dim=8,
        freq_dim=8, ffn_dim=24, num_layers=2, num_attention_heads=2, attention_head_dim=8,
        rope_max_seq_len=16).eval().requires_grad_(False)
    # FP32 tiny fixture checks actual interface; the separate fixture above checks BF16 casts.
    x = torch.randn(1, 2, 2, 4, 4).requires_grad_(True)
    embedding = torch.randn(1, 3, 8)
    timestep = torch.tensor([1.])
    d = torch.randn_like(x)
    y0 = model(x, timestep, embedding, return_dict=False)[0]
    g0 = torch.autograd.grad(y0, x, grad_outputs=d)[0]
    with checkpointed_branch(model, lambda _: None, root=tmp_path) as ledger:
        x1 = x.detach().clone().requires_grad_(True)
        y1 = model(x1, timestep, embedding, return_dict=False)[0]
        g1 = torch.autograd.grad(y1, x1, grad_outputs=d)[0]
        torch.testing.assert_close(y1, y0, rtol=0, atol=0)
        torch.testing.assert_close(g1, g0, rtol=0, atol=0)
        assert all(r["forward"] == r["recompute"] == "RETURNED" for r in ledger.rows)
    assert all(p.grad is None for p in model.parameters())
