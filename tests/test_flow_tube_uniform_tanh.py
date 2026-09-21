"""CPU/fake tests for live-history two-step control, budgets and retained failures."""
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from main.tube_state import projection_margin, state_clock
from runtime.wan import tube_retention
from experiments.wan_state_clock import flow_tube_uniform_tanh_run as run
from scripts import build_flow_tube_uniform_tanh_notebook as notebook_builder
from test_flow_tube_state_guidance import Model, book, native

pytestmark = pytest.mark.unit
torch.set_num_threads(1)


def source_fixture(tmp_path, native, book):
    source = tmp_path / run.SOURCE_RUN
    manifest = run.load(run.MANIFEST)
    for fixed in manifest["cases"]:
        root = source / fixed["id"]
        root.mkdir(parents=True)
        pipe = SimpleNamespace(transformer=Model(), scheduler=native())
        prompt, negative = torch.tensor(1.0), torch.tensor(-1.0)
        nodes, snapshots, _ = tube_retention.reference(pipe, torch.zeros(projection_margin.SHAPE), prompt, negative, torch.float32, 5.0, lambda *args: None)
        config = dict(
            model=dict(id=manifest["model"]["id"], revision=None),
            key_utf8=manifest["key_utf8"],
            generation=dict(prompt=fixed["prompt"], negative_prompt=manifest["generation"]["negative_prompt"], seed=fixed["seed"],
                            height=320, width=512, frames=181, fps=8, steps=50, guidance_scale=5.0),
        )
        run.dump(root / "config.json", config)
        np.savez(root / "codebook.npz", **book)
        for filename, value in (("OFF_nodes.pt", nodes), ("OFF_snapshots.pt", snapshots), ("prompt.pt", prompt), ("negative.pt", negative)):
            torch.save(value, root / filename)
        generation = dict(
            source_commit=run.SOURCE_COMMIT,
            resolved_model_revision=None,
            file_sha256={path.name: run.sha(path) for path in root.iterdir()},
            source_sha256={str(module.__file__): run.sha(module.__file__) for module in (projection_margin, state_clock)},
            reference_fingerprints={"44": dict(input=run.runtime.fingerprint(nodes[44]["z"]), history=run.runtime.fingerprint(vars(snapshots[44])))},
            videos={"LOCAL_A": {"control": {"actual_D": {"support_rms": 999.0}}},
                    "LOCAL_B": {"control": {"actual_D": {"support_rms": 888.0}}}},
        )
        run.dump(root / "generation.json", generation)
    return source


def patch_model(monkeypatch):
    model = Model()
    record = dict(requested_revision=run.MODEL_REVISION, resolved_revision=run.MODEL_REVISION,
                  config_commit_hash=run.MODEL_REVISION, original_weight_identity="UNVERIFIED_HISTORICAL_REVISION_NULL")
    monkeypatch.setattr(run.runtime, "load_transformer", lambda config: (SimpleNamespace(transformer=model), torch.float32, record))
    return model


def patch_media(monkeypatch, fail_off=False):
    class VAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(_commit_hash=run.MODEL_REVISION)

    monkeypatch.setattr(run, "RGB_SHAPE", (181, 4, 4, 3))
    monkeypatch.setattr(run, "load_frozen_vae", lambda *args: VAE())
    monkeypatch.setattr(run, "_clear_cache", lambda *args: None)
    base = (torch.arange(181).float() / 181 + 0.0004).reshape(181, 1, 1, 1).expand(run.RGB_SHAPE).clone()
    monkeypatch.setattr(run, "decode_normalized_latent", lambda *args: base.clone())
    cache, inputs, blind, reads = {}, [], [], []

    def encode_video(rgb, path, *args):
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"fake_codec")
        cache[str(path)] = run.quantize_rgb8_no_codec(rgb).float() / 255 + 0.0001
        if fail_off and path.stem == "OFF":
            raise RuntimeError("partial OFF save")

    def read_video(path):
        reads.append(path.stem)
        return cache[str(path)]

    def encode_latent(vae, rgb):
        inputs.append(rgb.clone())
        return torch.zeros((1, 16, 1 + (len(rgb) - 1) // 4, 40, 64))

    def receiver(observations, codebook):
        blind.append(sorted(observations))
        by_message = {"0": {"message": 0, "score": 0.15}, "1": {"message": 1, "score": -0.05}}
        return {"rankings": {mode: dict(best=by_message["0"], best_by_message=by_message, message_unique=True) for mode in run.MODES}}

    monkeypatch.setattr(run, "encode_rgb", encode_video)
    monkeypatch.setattr(run, "read_mp4", read_video)
    monkeypatch.setattr(run, "reencode_rgb24_readback", encode_latent)
    monkeypatch.setattr(run.state_clock, "read", receiver)
    monkeypatch.setattr(run.state_clock, "report", lambda *args: {"reporting_only": True})
    return base, inputs, blind, reads


def test_uniform_budget_recomputes_second_live_history_and_fixed_counts(monkeypatch, tmp_path, native, book):
    source = source_fixture(tmp_path, native, book)
    model = patch_model(monkeypatch)
    output = tmp_path / "result"
    result = run.generate_case(run.CASES[0], source, output)
    assert result["status"] == "GENERATION_COMPLETE", result["failures"]
    assert result["budget"]["R_star"] == run.R_STAR
    assert result["future_budget_fields_consumed"] == []
    assert result["consumed_saved_nodes"] == ["44.z", "44.v"]
    for key in ("transformer", "scheduler_step", "unit_response_probe_step", "second_control_zero_shadow_step", "clean_leaf_backward"):
        assert result["actual_calls"][key + "_completed"] == run.PLAN[key]
        assert result["actual_calls"][key + "_attempted"] == run.PLAN[key]
    assert not any(model.flags)
    for suffix in ("A", "B"):
        single = result["videos"]["SINGLE46_" + suffix]
        multi = result["videos"]["MULTI44_46_" + suffix]
        assert len(single["control_steps"]) == 1 and len(multi["control_steps"]) == 2
        assert single["control_steps"][0]["actual_D"]["support_rms"] == pytest.approx(run.R_STAR, rel=2e-5)
        assert multi["cumulative_native_response"]["actual_D"]["support"]["sum_rms"] == pytest.approx(run.R_STAR, rel=2e-5)
        assert single["cumulative_native_response"]["actual_D"]["support"]["sum_rms_squared"] == pytest.approx(run.R_STAR ** 2, rel=4e-5)
        assert multi["cumulative_native_response"]["actual_D"]["support"]["sum_rms_squared"] == pytest.approx(run.R_STAR ** 2 / 2, rel=4e-5)
        first, second = multi["control_steps"]
        assert first["index"] == 44 and second["index"] == 46
        assert set(first["clean_before"]) >= {"tanh_correct_score", "tanh_wrong_score", "tanh_gap"}
        assert set(second["clean_after"]) >= {"hard_correct_score", "hard_wrong_score", "hard_gap"}
        assert second["history_fingerprint"] != single["control_steps"][0]["history_fingerprint"]
        assert second["input_fingerprint"] != single["control_steps"][0]["input_fingerprint"]
        assert multi["gradients"][1]["index"] == 46 and "clean46 leaf" in multi["gradients"][1]["meaning"]
        assert multi["unit_probes"][1]["history_fingerprint"] == second["history_fingerprint"]
        assert multi["terminal_vs_same_batch_OFF"]["support_rms"] >= 0
        assert set(multi["terminal_nominal_by_message"]) == {"0", "1"}


def test_three_real_layers_and_absolute_receiver_scores(monkeypatch, tmp_path, native, book):
    source = source_fixture(tmp_path, native, book)
    patch_model(monkeypatch)
    output = tmp_path / "result"
    generated = run.generate_case(run.CASES[0], source, output)
    assert generated["status"] == "GENERATION_COMPLETE"
    base, inputs, blind, reads = patch_media(monkeypatch)
    result = run.media_case(run.CASES[0], output)
    assert result["status"] == "EXECUTION_COMPLETE", result["failures"]
    assert len(inputs) == 60 and len(blind) == 15
    assert all(phases == [0, 1, 2, 3] for phases in blind)
    assert [len(value) for value in inputs] == [181, 177, 177, 177] * 15
    torch.testing.assert_close(inputs[0], base)
    torch.testing.assert_close(inputs[4], run.quantize_rgb8_no_codec(base).float() / 255)
    summary = run.summarize({run.CASES[0]: result})
    record = summary["rows"][0]["media"]["MP4"]["local_state"]["SINGLE46_A"]
    assert record == dict(correct_score=0.15, wrong_score=-0.05, gap=0.2, unique_correct=True)
    assert summary["rows"][0]["terminal"]["OFF"] is not None


def test_second_control_failure_retains_first_control_diagnostics(monkeypatch, tmp_path, native, book):
    source = source_fixture(tmp_path, native, book)
    patch_model(monkeypatch)
    original = run.runtime.prepare_direction
    index46_calls = 0

    def fail_first_multi46(*args, **kwargs):
        nonlocal index46_calls
        index = args[6]
        if index == 46:
            index46_calls += 1
            if index46_calls == 3:
                raise RuntimeError("injected second-control probe failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(run.runtime, "prepare_direction", fail_first_multi46)
    result = run.generate_case(run.CASES[0], source, tmp_path / "retained")
    item = result["videos"]["MULTI44_46_A"]
    assert result["status"] == "WITH_RETAINED_FAILURES"
    assert item["status"] == "FAILED"
    assert len(item["control_steps"]) == 1 and item["control_steps"][0]["index"] == 44
    assert len(item["gradients"]) == 2 and len(item["unit_probes"]) == 1
    assert item["cumulative_native_response"]["controlled_steps"] == 1


def test_missing_or_tampered_source_retains_full_denominator(monkeypatch, tmp_path, native, book):
    source = source_fixture(tmp_path, native, book)
    (source / run.CASES[0] / "prompt.pt").write_bytes(b"tampered")
    monkeypatch.setattr(run.runtime, "load_transformer", lambda *args: pytest.fail("model must not load after source hash failure"))
    result = run.generate_case(run.CASES[0], source, tmp_path / "failed")
    assert result["status"] == "WITH_RETAINED_FAILURES"
    assert all(item["status"] == "FAILED" for item in result["videos"].values())
    assert result["actual_calls"]["transformer_attempted"] == 0
    monkeypatch.setattr(run.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    root = run.run_all(source, tmp_path / "all")
    assert root["video_denominator"] == 10 and root["receiver_encode_denominator"] == 120
    assert sum(len(layer["observations"]) for case in root["cases"].values() for video in case["videos"].values() for layer in video["layers"].values()) == 120
    assert len(root["paired_summary"]["rows"]) == 4


def test_real_cli_module_entry_retains_missing_source_failure(tmp_path):
    output = tmp_path / "cli"
    completed = subprocess.run(
        [sys.executable, "-m", run.MODULE, "--source", str(tmp_path / "missing"), "--output", str(output),
         "--case-id", run.CASES[0], "--stage", "generate"],
        cwd=run.MANIFEST.parents[3], capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 1
    result = run.load(output / "generation.json")
    assert result["status"] == "WITH_RETAINED_FAILURES"
    assert all(item["status"] == "FAILED" for item in result["videos"].values())
    assert result["actual_calls"]["transformer_attempted"] == 0


def test_notebook_locator_runs_outside_repo_and_preserves_missing_assets(monkeypatch, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)
    preferred = tmp_path / "Video-WM" / "FlowTubeResponseSelection" / run.SOURCE_RUN
    preferred.mkdir(parents=True)
    def forbid_scan(*args, **kwargs):
        raise AssertionError('Fixed input must not enumerate Drive directories')
    for method in ('rglob', 'glob', 'iterdir'):
        monkeypatch.setattr(type(tmp_path), method, forbid_scan)
    namespace = {"DRIVE_ROOT": tmp_path}
    exec(notebook_builder.LOCATOR_SOURCE, namespace)
    assert namespace["INPUT"] == preferred.resolve()
    preferred.rmdir()
    namespace = {"DRIVE_ROOT": tmp_path}
    exec(notebook_builder.LOCATOR_SOURCE, namespace)
    assert namespace["INPUT"] == preferred.resolve()
    first = tmp_path / "archive-a" / run.SOURCE_RUN
    second = tmp_path / "archive-b" / run.SOURCE_RUN
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    namespace = {"DRIVE_ROOT": tmp_path}
    exec(notebook_builder.LOCATOR_SOURCE, namespace)
    assert namespace["INPUT"] == preferred
    assert len(namespace["missing_inputs"]) == 14
    for case in run.CASES:
        (preferred / case).mkdir(parents=True)
        for name in namespace["required_names"]:
            (preferred / case / name).touch()
    exec(notebook_builder.LOCATOR_SOURCE, namespace)
    assert namespace["missing_inputs"] == []
