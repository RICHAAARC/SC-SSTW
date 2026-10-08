from __future__ import annotations

import ast
import copy
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from main.tube_state import video_trajectory_attribution_v1 as method
from experiments.wan_state_clock import video_trajectory_attribution_uncertainty_v1_protocol as protocol
from experiments.wan_state_clock import video_trajectory_attribution_uncertainty_v1_run as runner

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]


def rules():
    return {
        "status": "FROZEN",
        "formula_version": method.RULE_FORMULA_VERSION,
        "thresholds": {
            "181": {"tau_M": 0.014778458822231031, "tau_m": None, "tau_I": 0.0},
            "177": {"tau_M": 0.02283275070993452, "tau_m": 0.003530171359717223, "tau_I": 0.0},
            "89": {"tau_M": 0.04896837552751242, "tau_m": 0.0039622077929535385, "tau_I": 0.0},
        },
        "rule_sha256": protocol.FROZEN_RULE_SHA256,
        "claim_message": "OKOK",
        "claim_bits": list(method.CLAIM_BITS),
    }


def complete_sync(row, *, M=0.5, m=0.5, actions=1, paths=1):
    chosen = (
        {"received_index_map": protocol.true_action(row["view"], row["parameters"])}
        if actions == 1
        else None
    )
    return {
        "status": "COMPLETE",
        "frames": row["frames"],
        "M": M,
        "m": None if row["frames"] == 181 else m,
        "top_paths": [{"fixture": index} for index in range(paths)],
        "top_action_ids": [f"a{index}" for index in range(actions)],
        "top_action_count": actions,
        "unique_action": actions == 1,
        "chosen_action": chosen,
        "truth_inputs": False,
    }


def complete_identity(I=0.5, exact=True, differences=None):
    if differences is None:
        differences = [1] * 32
    return {
        "status": "COMPLETE",
        "attempted": True,
        "I": I,
        "exact32": exact,
        "signed_vote_differences": differences,
        "zero_signed_vote_bits": sum(value == 0 for value in differences),
        "truth_inputs": False,
    }


def test_frozen_roster_budget_maps_and_shared_denominators():
    rows = protocol.query_roster()
    assert len(rows) == 22
    assert len({row["observation_id"] for row in rows}) == 20
    assert sum(row["role"] == "PRIMARY_PROBE" for row in rows) == 10
    assert sum(row["role"] == "ALIAS_CONTROL" for row in rows) == 4
    assert sum(row["role"] == "REGRESSION_CONTROL" for row in rows) == 8
    assert sum(protocol.role(row)["expected_accept"] for row in rows) == 16
    assert sum(not protocol.role(row)["expected_accept"] for row in rows) == 6
    assert protocol.FIXED_DENOMINATOR == {
        "sources": 2,
        "queries": 22,
        "observations": 20,
        "primary_queries": 10,
        "alias_controls": 4,
        "regression_controls": 8,
        "positive_queries": 16,
        "negative_queries": 6,
        "sync_candidates": 11902,
        "framewise_frames": 3012,
        "framewise_batch8": 394,
        "max_payload_reads": 22,
        "max_votes": 802560,
        "max_time_bit_rows": 26752,
        "max_bit_rows": 704,
        "sync_coverage_denominator": 10,
        "weak_identity_coverage_denominator": 10,
    }
    assert protocol.received_index_map("SHORT89", {"start": 92}) == tuple(range(92, 181))
    assert len(protocol.received_index_map("DELETE177", {"b": 2, "k": 1})) == 177
    assert len(protocol.received_index_map("DELETE177", {"b": 2, "k": 176})) == 177
    with pytest.raises(ValueError):
        protocol.received_index_map("DELETE177", {"b": 2, "k": 0})


def test_config_freezes_inputs_rules_slots_and_receiver_only_roles():
    cfg = runner.load_config()
    assert cfg["fixed_denominator"] == protocol.FIXED_DENOMINATOR
    assert cfg["frozen_rules"] == {
        "path": "/content/drive/MyDrive/Video-WM/Trajectory-Attribution-V1/20261008T085244020205Z/fixed_reference/seals/frozen_rules.json",
        "file_sha256": protocol.FROZEN_RULE_FILE_SHA256,
        "rule_sha256": protocol.FROZEN_RULE_SHA256,
    }
    assert cfg["root_protocol_freeze_sha256"] == protocol.ROOT_PROTOCOL_FREEZE_SHA256
    assert cfg["model"]["role"] == "Original Wan VAE receiver encoding only"
    assert "receiver encoding" in cfg["framewise_model"]["role"]
    assert cfg["coverage"]["same_ten_primary_queries"] is True
    assert {row["role"] for row in cfg["slots"]} == {
        "PRIMARY_PROBE",
        "ALIAS_CONTROL",
        "REGRESSION_CONTROL",
    }
    assert len({spec["sha256"] for source in cfg["inputs"].values() for spec in source.values()}) == 6


def test_frozen_rule_loader_checks_file_bytes_and_internal_rule(tmp_path):
    assert "rules_override" not in inspect.signature(runner.load_frozen_rules).parameters
    assert "rules_override" not in inspect.signature(runner.run).parameters
    value = rules()
    raw = (json.dumps(value, indent=2) + "\n").encode()
    path = tmp_path / "rules.json"
    path.write_bytes(raw)
    cfg = copy.deepcopy(runner.load_config())
    cfg["frozen_rules"]["path"] = str(path)
    cfg["frozen_rules"]["file_sha256"] = hashlib.sha256(raw).hexdigest()
    loaded, observed = runner.load_frozen_rules(cfg)
    assert loaded == value and observed == raw
    path.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="byte identity"):
        runner.load_frozen_rules(cfg)


def _patch_frozen_rules(monkeypatch, frozen=None):
    value = rules() if frozen is None else frozen
    monkeypatch.setattr(
        runner,
        "load_frozen_rules",
        lambda cfg: (copy.deepcopy(value), None),
    )


def _fake_sync_collection(store, frozen, *args):
    for observation_id, rows in runner._rows_by_observation().items():
        store.observation_records[observation_id] = {
            "observation_id": observation_id,
            "frames": rows[0]["frames"],
            "rgb_sha256": "f" * 64,
            "rgb_shape": [rows[0]["frames"], 320, 512, 3],
            "rgb_bytes": rows[0]["frames"] * 320 * 512 * 3,
            "rgb_dtype": "uint8",
        }
    for truth in protocol.query_roster():
        query = store.data["queries"][truth["query_id"]]
        sync = complete_sync(truth)
        if truth["query_id"] == "UQ000":
            sync = complete_sync(truth, actions=2, paths=2)
        elif truth["query_id"] == "UQ004":
            sync = complete_sync(
                truth,
                m=frozen["thresholds"]["177"]["tau_m"],
                actions=1,
                paths=1,
            )
        elif truth["role"] == "ALIAS_CONTROL":
            sync = complete_sync(truth, actions=1, paths=2)
        elif truth["role"] == "REGRESSION_CONTROL" and (
            (truth["condition"], truth["key"]) in (("A_M05", "K1"), ("OFF", "K0"))
        ):
            sync = complete_sync(truth, M=0.0)
        query.update(status="SYNC_READ", sync=sync)
        store.sync_records[truth["query_id"]] = {
            "query_id": truth["query_id"],
            "key_id": "fixture",
            "summary": sync,
        }
    runner.seal_sync(store)


def _fake_payload_collection(store, frozen, *args):
    for truth in protocol.query_roster():
        query = store.data["queries"][truth["query_id"]]
        if not runner._sync_eligible(query["sync"], truth["frames"], frozen):
            identity = runner._not_eligible_identity()
        elif truth["query_id"] == "UQ001":
            identity = complete_identity(I=0.0, exact=False, differences=[0] + [1] * 31)
        elif truth["condition"] == "B_M05":
            identity = complete_identity(I=-1.0, exact=False, differences=[-1] * 32)
        else:
            identity = complete_identity()
        query.update(
            status="PAYLOAD_READ" if identity["status"] == "COMPLETE" else "PAYLOAD_NOT_ELIGIBLE",
            identity=identity,
        )
        store.payload_records[truth["query_id"]] = {
            "query_id": truth["query_id"],
            "key_id": "fixture",
            "identity": identity,
        }
    runner.seal_payload(store)


def test_fake_complete_chain_covers_scientific_branches_and_seal_order(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "collect_sync", _fake_sync_collection)
    monkeypatch.setattr(runner, "collect_payload", _fake_payload_collection)
    _patch_frozen_rules(monkeypatch)
    result = runner.run(tmp_path / "run", runner.load_config())
    assert result["status"] == "COMPLETE"
    assert result["coverage"]["primary_denominator"] == 10
    assert result["coverage"]["sync_unreliable_action"] == {
        "denominator": 10,
        "actual_trigger": 2,
        "valid_coverage": 2,
        "technical": 0,
        "status": "OBSERVED",
    }
    assert result["coverage"]["weak_identity"] == {
        "denominator": 10,
        "actual_trigger": 1,
        "valid_coverage": 1,
        "technical": 0,
        "tau_I": 0.0,
        "status": "OBSERVED",
    }
    assert result["controls"]["positive"] == {"denominator": 2, "matched": 2}
    assert result["controls"]["negative"] == {"denominator": 6, "matched": 6}
    assert result["false_attribution"]["false_accept_queries"] == 0
    assert result["false_attribution"]["any_false_claim"] is False
    assert result["queries"]["UQ001"]["identity"]["exact32"] is False
    assert result["queries"]["UQ001"]["posthoc"]["valid_weak_identity_coverage"] is True
    assert all(
        not row["posthoc"]["valid_sync_uncertainty_coverage"]
        and not row["posthoc"]["valid_weak_identity_coverage"]
        for row in result["queries"].values()
        if row["posthoc"]["role"] == "ALIAS_CONTROL"
    )
    sync_seal = json.loads((tmp_path / "run/seals/blind_sync.json").read_text())
    decision_seal = json.loads((tmp_path / "run/seals/blind_decisions.json").read_text())
    assert "C1" not in json.dumps(sync_seal) and "A_M05" not in json.dumps(sync_seal)
    assert "C1" not in json.dumps(decision_seal) and "PRIMARY_PROBE" not in json.dumps(decision_seal)
    assert all(row["truth_inputs"] is False for row in [sync_seal, decision_seal])


def test_same_action_alias_and_technical_uncertain_never_count_as_coverage():
    frozen = rules()
    truth = protocol.query_roster()[0]
    same_action = {
        "sync": complete_sync(truth, actions=1, paths=2),
        "identity": complete_identity(),
        "decision": {"decision": "UNCERTAIN", "reason": "UNCERTAIN_SYNC"},
    }
    assert runner._sync_trigger(same_action, truth["frames"], frozen) is False
    technical = copy.deepcopy(same_action)
    technical["sync"]["status"] = "TECHNICAL_INCOMPLETE"
    technical["decision"]["reason"] = "UNCERTAIN_TECHNICAL_SYNC"
    assert runner._sync_trigger(technical, truth["frames"], frozen) is False
    weak = {
        "sync": complete_sync(truth),
        "identity": complete_identity(I=0.0, exact=True, differences=[0] + [1] * 31),
        "decision": {"decision": "UNCERTAIN", "reason": "UNCERTAIN_IDENTITY_WEAK"},
    }
    assert runner._weak_identity_trigger(weak, True, truth["frames"], frozen) is True
    assert runner._weak_identity_trigger(weak, False, truth["frames"], frozen) is False
    weak["identity"]["signed_vote_differences"][1] = -1
    assert runner._weak_identity_trigger(weak, True, truth["frames"], frozen) is False


def test_partial_observation_failure_settles_shared_slots_and_continues(tmp_path, monkeypatch):
    cfg = runner.load_config()
    failing_sha = cfg["inputs"]["C1"]["A_M05"]["sha256"]

    class Inputs:
        @staticmethod
        def read(spec):
            if spec["sha256"] == failing_sha:
                raise OSError("fixture input unavailable")
            return {"frames": 181}

        @staticmethod
        def construct(full, indices):
            return {"frames": len(indices)}

        @staticmethod
        def receipt(observed):
            return {
                "sha256": "d" * 64,
                "shape": [observed["frames"], 320, 512, 3],
                "bytes": observed["frames"] * 320 * 512 * 3,
                "dtype": "uint8",
            }

    class Framewise:
        def __init__(self, cfg):
            pass

        def encode(self, observed):
            return observed

        def score(self, latent, key, kind):
            return {"fixture": True}

        def close(self):
            pass

    store = runner.Store(tmp_path / "partial", cfg)
    monkeypatch.setattr(method, "summarize_sync", lambda raw, frames, key: {
        "status": "COMPLETE",
        "frames": frames,
        "M": 0.5,
        "m": 0.5,
        "top_paths": [{"fixture": True}],
        "top_action_ids": ["a"],
        "top_action_count": 1,
        "unique_action": True,
        "chosen_action": {"received_index_map": list(range(frames))},
        "truth_inputs": False,
    })
    runner.collect_sync(store, rules(), Inputs, Framewise)
    failed = [
        row for row in store.data["queries"].values()
        if row["sync"]["status"] == "TECHNICAL_INCOMPLETE"
    ]
    assert len(failed) == 9
    shared = [
        row for row in protocol.query_roster()
        if row["source"] == "C1" and row["condition"] == "A_M05"
        and row["view"] == "CROP177"
    ]
    assert len(shared) == 2
    assert all(store.data["queries"][row["query_id"]]["sync"]["status"] == "TECHNICAL_INCOMPLETE" for row in shared)
    assert any(row["sync"]["status"] == "COMPLETE" for row in store.data["queries"].values())
    assert len(store.data["queries"]) == 22


def test_new_sources_parse_and_do_not_modify_core_runtime():
    for path in (
        ROOT / "experiments/wan_state_clock/video_trajectory_attribution_uncertainty_v1_protocol.py",
        ROOT / "experiments/wan_state_clock/video_trajectory_attribution_uncertainty_v1_run.py",
    ):
        ast.parse(path.read_text())



def test_rule_read_failure_still_settles_all_22_and_reports_technical(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("fixture frozen rule missing")

    monkeypatch.setattr(runner, "load_frozen_rules", unavailable)
    with pytest.raises(FileNotFoundError):
        runner.run(tmp_path / "rule-missing", runner.load_config())
    result = json.loads((tmp_path / "rule-missing/result.json").read_text())
    assert len(result["queries"]) == 22
    assert all(row["decision"] is not None for row in result["queries"].values())
    assert result["coverage"]["sync_unreliable_action"]["status"] == "UNRESOLVED_TECHNICAL"
    assert result["coverage"]["weak_identity"]["status"] == "UNRESOLVED_TECHNICAL"
    assert result["controls"]["status"] == "UNRESOLVED_TECHNICAL"
    assert result["false_attribution"]["any_false_claim"] == "UNRESOLVED"
    for name in ("blind_sync.json", "blind_payload.json", "blind_decisions.json"):
        value = json.loads((tmp_path / "rule-missing/seals" / name).read_text())
        assert len(value["queries"] if "queries" in value else value["decisions"]) == 22
        if name == "blind_sync.json":
            assert len(value["observations"]) == 20


def test_backend_init_failure_seals_full_observation_sync_then_payload(tmp_path):
    cfg = runner.load_config()
    store = runner.Store(tmp_path / "backend-init", cfg)

    class BrokenFramewise:
        def __init__(self, cfg):
            raise RuntimeError("fixture backend init failure")

    class PayloadMustNotInitialize:
        def __init__(self, cfg):
            raise AssertionError("payload backend must not initialize without eligible sync")

    runner.collect_sync(store, rules(), framewise_type=BrokenFramewise)
    assert list(store.data["seals"]) == ["sync"]
    sync_seal = json.loads((tmp_path / "backend-init/seals/blind_sync.json").read_text())
    assert len(sync_seal["observations"]) == 20
    assert len(sync_seal["queries"]) == 22
    assert all(
        row["sync"]["status"] == "TECHNICAL_INCOMPLETE"
        for row in store.data["queries"].values()
    )

    runner.collect_payload(store, rules(), wan_type=PayloadMustNotInitialize)
    assert list(store.data["seals"]) == ["sync", "payload"]
    payload_seal = json.loads((tmp_path / "backend-init/seals/blind_payload.json").read_text())
    assert len(payload_seal["queries"]) == 22


def test_interruption_reconciles_both_half_commit_orders(tmp_path, monkeypatch):
    def interrupted(store, frozen, *args):
        first, second = protocol.query_roster()[:2]
        first_sync = complete_sync(first)
        second_sync = complete_sync(second)
        store.sync_records[first["query_id"]] = {
            "query_id": first["query_id"],
            "key_id": "record-first",
            "summary": first_sync,
        }
        store.data["queries"][second["query_id"]].update(
            status="SYNC_READ",
            sync=second_sync,
        )
        raise KeyboardInterrupt("fixture between record and query commits")

    monkeypatch.setattr(runner, "collect_sync", interrupted)
    _patch_frozen_rules(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        runner.run(tmp_path / "half-commit", runner.load_config())

    result = json.loads((tmp_path / "half-commit/result.json").read_text())
    sync_seal = json.loads((tmp_path / "half-commit/seals/blind_sync.json").read_text())
    payload_seal = json.loads((tmp_path / "half-commit/seals/blind_payload.json").read_text())
    assert len(sync_seal["observations"]) == 20
    assert len(sync_seal["queries"]) == 22
    assert len(payload_seal["queries"]) == 22
    assert result["queries"]["UQ000"]["sync"]["status"] == "COMPLETE"
    assert result["queries"]["UQ001"]["sync"]["status"] == "COMPLETE"
    assert sync_seal["queries"]["UQ000"]["summary"] == result["queries"]["UQ000"]["sync"]
    assert sync_seal["queries"]["UQ001"]["summary"] == result["queries"]["UQ001"]["sync"]


def test_interruption_seal_is_completed_after_partial_records(tmp_path, monkeypatch):
    def interrupted(store, frozen, *args):
        truth = protocol.query_roster()[0]
        sync = complete_sync(truth)
        store.data["queries"][truth["query_id"]].update(status="SYNC_READ", sync=sync)
        store.sync_records[truth["query_id"]] = {
            "query_id": truth["query_id"],
            "key_id": "fixture",
            "summary": sync,
        }
        raise KeyboardInterrupt("fixture interruption")

    monkeypatch.setattr(runner, "collect_sync", interrupted)
    _patch_frozen_rules(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        runner.run(tmp_path / "interrupted", runner.load_config())
    result = json.loads((tmp_path / "interrupted/result.json").read_text())
    sealed = json.loads((tmp_path / "interrupted/seals/blind_sync.json").read_text())
    assert len(sealed["queries"]) == 22
    assert len(result["queries"]) == 22
    assert all(row["decision"] is not None for row in result["queries"].values())
    assert result["coverage"]["sync_unreliable_action"]["valid_coverage"] == 0
    assert result["false_attribution"]["any_false_claim"] == "UNRESOLVED"


def test_interruption_preserves_scientific_low_sync_and_sync_uncertain(tmp_path):
    store = runner.Store(tmp_path / "decisions", runner.load_config())
    frozen = rules()
    truth_rows = protocol.query_roster()
    for index, truth in enumerate(truth_rows):
        sync = complete_sync(truth, M=0.0)
        if index == 1:
            sync = complete_sync(truth, actions=2, paths=2)
        store.data["queries"][truth["query_id"]].update(
            sync=sync,
            identity=runner._technical_identity("fixture", attempted=False),
        )
    runner.decide_all(store, frozen, interruption=True)
    assert store.data["queries"]["UQ000"]["decision"]["reason"] == "REJECT_LOW_SYNC"
    assert store.data["queries"]["UQ001"]["decision"]["reason"] == "UNCERTAIN_SYNC"
    assert store.data["queries"]["UQ002"]["decision"]["reason"] == "REJECT_LOW_SYNC"


def test_positive_regression_accept_wrong_action_is_not_capability_match(tmp_path, monkeypatch):
    def sync_with_wrong_control(store, frozen, *args):
        _fake_sync_collection(store, frozen, *args)
        for truth in protocol.query_roster():
            if (
                truth["role"] == "REGRESSION_CONTROL"
                and protocol.role(truth)["expected_accept"]
            ):
                chosen = store.data["queries"][truth["query_id"]]["sync"]["chosen_action"]
                chosen["received_index_map"] = list(reversed(chosen["received_index_map"]))

    monkeypatch.setattr(runner, "collect_sync", sync_with_wrong_control)
    monkeypatch.setattr(runner, "collect_payload", _fake_payload_collection)
    _patch_frozen_rules(monkeypatch)
    result = runner.run(tmp_path / "wrong-action", runner.load_config())
    assert result["controls"]["positive"]["matched"] == 0
    assert result["false_attribution"]["false_accept_queries"] == 2
    assert result["false_attribution"]["any_false_claim"] is True


def test_real_payload_scheduler_uses_qualified_action_and_retains_one_read_failure(tmp_path, monkeypatch):
    cfg = runner.load_config()
    store = runner.Store(tmp_path / "payload", cfg)
    frozen = rules()
    for truth in protocol.query_roster():
        store.data["queries"][truth["query_id"]]["sync"] = complete_sync(truth)

    class Inputs:
        @staticmethod
        def read(spec):
            return {"full": spec["sha256"]}

        @staticmethod
        def construct(full, indices):
            return {"frames": len(indices), "full": full["full"]}

        @staticmethod
        def receipt(observed):
            return {
                "sha256": "e" * 64,
                "shape": [observed["frames"], 320, 512, 3],
                "bytes": observed["frames"] * 320 * 512 * 3,
                "dtype": "uint8",
            }

    class Wan:
        calls = 0
        closed = False

        def __init__(self, cfg):
            pass

        def encode(self, corrected):
            return corrected

        def read(self, latent, key, frames):
            type(self).calls += 1
            if type(self).calls == 2:
                raise RuntimeError("fixture payload read failure")
            rows = []
            for index, bit in enumerate(method.CLAIM_BITS):
                ones, zeros = ((6, 4) if bit else (4, 6))
                rows.append(
                    {
                        "bit_index": index,
                        "ones": ones,
                        "zeros": zeros,
                        "count": 10,
                        "decoded": bit,
                    }
                )
            return {"status": "READ", "truth_inputs": False, "bit_rows": rows}

        def close(self):
            type(self).closed = True

    applied = []

    def apply_action(observed, action):
        applied.append((observed["frames"], len(action["received_index_map"])))
        return observed

    monkeypatch.setattr(runner.runtime, "apply_action", apply_action)
    runner.collect_payload(store, frozen, Inputs, Wan)
    assert len(applied) == 22
    assert Wan.calls == 22 and Wan.closed is True
    assert sum(
        row["identity"]["status"] == "COMPLETE"
        for row in store.data["queries"].values()
    ) == 21
    assert sum(
        row["identity"]["status"] == "TECHNICAL_INCOMPLETE"
        for row in store.data["queries"].values()
    ) == 1
    complete = next(
        row["identity"]
        for row in store.data["queries"].values()
        if row["identity"]["status"] == "COMPLETE"
    )
    assert len(complete["signed_vote_differences"]) == 32
    assert all(type(value) is int and value == 2 for value in complete["signed_vote_differences"])
    sealed = json.loads((tmp_path / "payload/seals/blind_payload.json").read_text())
    assert len(sealed["queries"]) == 22



def test_notebook_schema_ast_mount_and_byte_rebuild(tmp_path):
    import nbformat
    from scripts import build_video_trajectory_attribution_uncertainty_notebook as builder

    path = ROOT / "notebooks/video_trajectory_attribution_uncertainty_v1_colab.ipynb"
    raw = path.read_bytes()
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    binding = dict(notebook.metadata["candidate_binding"])
    assert binding["candidate"] == "trajectory-attribution-uncertainty-v1"
    assert binding["source_sha"] is None or (
        isinstance(binding["source_sha"], str)
        and len(binding["source_sha"]) == 40
        and all(ch in "0123456789abcdef" for ch in binding["source_sha"])
    )
    assert binding["status"] == (
        "UNPUBLISHED_DRAFT" if binding["source_sha"] is None else "PUBLISHED_SHA_BOUND"
    )
    rebuilt = builder.build(binding["source_sha"], tmp_path / path.name)
    assert rebuilt.read_bytes() == raw
    draft_path = builder.build(None, tmp_path / "draft.ipynb")
    draft = nbformat.read(draft_path, as_version=4)
    assert draft.metadata["candidate_binding"] == {
        "candidate": "trajectory-attribution-uncertainty-v1",
        "source_sha": None,
        "status": "UNPUBLISHED_DRAFT",
    }
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code[0].source == "from google.colab import drive\ndrive.mount('/content/drive')\n"
    joined = "\n".join(cell.source for cell in code)
    assert "video_trajectory_attribution_uncertainty_v1_run" in joined
    assert "CONFIG_PATH=REPO/" in joined
    for forbidden in ("ffmpeg", "ffprobe", "WanPipeline"):
        assert forbidden not in joined
    for cell in code:
        ast.parse(cell.source)
        assert cell.outputs == [] and cell.execution_count is None


def test_no_git_release_copy_cli_and_missing_inputs_settle(tmp_path):
    from runtime.wan import provenance

    release = tmp_path / "release"
    for name in (*provenance.SOURCE_FILES, "release_manifest.json"):
        target = release / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    module = "experiments.wan_state_clock.video_trajectory_attribution_uncertainty_v1_run"
    help_code = (
        "import runpy,sys;sys.path.insert(0,"
        + repr(str(release))
        + ");sys.argv=['entry','--help'];runpy.run_module("
        + repr(module)
        + ",run_name='__main__')"
    )
    help_result = subprocess.run(
        [sys.executable, "-I", "-c", help_code],
        cwd=release,
        text=True,
        capture_output=True,
    )
    assert help_result.returncode == 0 and "--config" in help_result.stdout
    output = tmp_path / "missing-run"
    run_code = (
        "import runpy,sys;sys.path.insert(0,"
        + repr(str(release))
        + ");sys.argv=['entry','--config',"
        + repr(str(release / "experiments/wan_state_clock/configs/video_trajectory_attribution_uncertainty_v1.json"))
        + ",'--output',"
        + repr(str(output))
        + "];runpy.run_module("
        + repr(module)
        + ",run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", run_code],
        cwd=release,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    saved = json.loads((output / "result.json").read_text())
    assert saved["source_sha"] is None
    assert saved["source_provenance"]["manifest_status"] == "MATCH"
    assert len(saved["queries"]) == 22
    assert saved["coverage"]["sync_unreliable_action"]["status"] == "UNRESOLVED_TECHNICAL"
    assert saved["coverage"]["weak_identity"]["status"] == "UNRESOLVED_TECHNICAL"
