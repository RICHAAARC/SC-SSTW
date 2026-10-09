"""Staged real-evaluation runner using repository Wan/runtime interfaces.

The CLI in :mod:`real_cli` invokes one phase per process so the Wan
transformer, Wan VAE, framewise VAE, and native baselines never need to reside
together.  Importing this module performs no heavyweight imports or model I/O.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import time
from collections import Counter
from pathlib import Path

from experiments.paper_results_v1.real_backends import (
    close_adapter,
    file_sha256,
    load_local_framewise_backend,
    load_rivagan_adapter,
    load_videoseal_adapter,
)
from experiments.paper_results_v1.report import read_json


SCHEMA_VERSION = "paper-real-eval-v1"
MAIN_ARMS = ("OFF_NATIVE", "PAYLOAD_NATIVE", "PAYLOAD_FRAMEWISE_RECON", "PAYLOAD_FRAMEWISE_M05")
BASELINES = ("videoseal", "rivagan")
RULE_STATUS_ACTIVE = ("ADOPTED_FOR_EXECUTION", "ADOPTED_METHOD_DEFINITION")
VIDEOSEAL_STRICT_RULE = "CHANNEL_J_MOD_32_REPEAT_MEAN_STRICT_GT_ZERO_ZERO_TIE_UNEVALUABLE"
VIDEOSEAL_NATIVE_TIE_RULE = "CHANNEL_J_MOD_32_REPEAT_MEAN_STRICT_GT_ZERO_NATIVE_TIE_RETAINED"
RIVAGAN_STRICT_RULE = "ALL_DECLARED_FRAMES_EQUAL_LOGIT_MEAN_GE_ZERO_ZERO_TIE_UNEVALUABLE"
RIVAGAN_NATIVE_TIE_RULE = "ALL_DECLARED_FRAMES_EQUAL_LOGIT_MEAN_GE_ZERO_NATIVE_TIE_RETAINED"
MAIN_VOTE_TIE_POLICY = "COUNTER_VOTE_COUNTS_ONES_EQUALS_ZEROS_DECODE_UNCHANGED"
BASELINE_TIE_SEMANTICS = "REDUCED_EFFECTIVE_SOFT_EXACT_ZERO"
PHASES = (
    "generate", "decode", "framewise", "baseline-embed-videoseal",
    "baseline-embed-rivagan", "codec", "quality", "baseline-extract-videoseal",
    "baseline-extract-rivagan", "receiver-sync", "receiver-read", "evaluate",
)

QUALITY_PAIRS = (
    ("PAYLOAD_NATIVE", "OFF_NATIVE"),
    ("PAYLOAD_FRAMEWISE_RECON", "PAYLOAD_NATIVE"),
    ("PAYLOAD_FRAMEWISE_RECON", "OFF_NATIVE"),
    ("PAYLOAD_FRAMEWISE_M05", "PAYLOAD_FRAMEWISE_RECON"),
    ("PAYLOAD_FRAMEWISE_M05", "OFF_NATIVE"),
    ("videoseal", "OFF_NATIVE"),
    ("rivagan", "OFF_NATIVE"),
)


class RealEvalConfigError(ValueError):
    """The explicit execution manifest is incomplete or ambiguous."""


def _require(row, fields, where):
    if not isinstance(row, dict):
        raise RealEvalConfigError(f"{where} must be an object")
    missing = [field for field in fields if field not in row]
    if missing:
        raise RealEvalConfigError(f"{where} missing fields: {', '.join(missing)}")


def _identifier(value, where):
    if not isinstance(value, str) or not value.strip():
        raise RealEvalConfigError(f"{where} must be a nonempty string")


def _bits(value, length, where):
    if not isinstance(value, list) or len(value) != length:
        raise RealEvalConfigError(f"{where} must contain exactly {length} bits")
    if any(type(bit) is not int or bit not in (0, 1) for bit in value):
        raise RealEvalConfigError(f"{where} must contain integer 0/1 values")


def _strict_path(value, where):
    _identifier(value, where)
    if value.startswith(("http://", "https://")):
        raise RealEvalConfigError(f"{where} must be a local path, not a URL")


def _observe_file_identity(receipt, *, path_field="path", sha_field="sha256"):
    """Record a file digest comparison without making provenance a data gate."""

    if not isinstance(receipt, dict) or not isinstance(receipt.get(path_field), str):
        raise ValueError(f"saved record receipt must contain string {path_field}")
    hash_error = None
    try:
        actual = file_sha256(receipt[path_field])
    except OSError as exc:
        actual = None
        hash_error = f"{type(exc).__name__}: {exc}"
    expected = receipt.get(sha_field)
    receipt[f"{sha_field}_observation"] = {
        "expected": expected,
        "actual": actual,
        "status": (
            "OBSERVATION_UNAVAILABLE" if hash_error
            else "MATCH" if actual and expected == actual
            else "UNDECLARED" if not isinstance(expected, str) or not expected
            else "RECORDED_DIFFERENCE"
        ),
        "observation_error": hash_error,
        "blocking": False,
    }
    return actual


def _content_identity_token(receipt, actual_sha256):
    if actual_sha256:
        return actual_sha256
    return "UNHASHED:" + hashlib.sha256(json.dumps(
        [
            receipt.get("artifact_id"), receipt.get("path"),
            receipt.get("bytes"), receipt.get("shape"),
        ],
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _expand_edit_map(spec, where):
    if isinstance(spec, list):
        mapping = list(spec)
    elif isinstance(spec, dict) and spec.get("kind") == "RANGE":
        if any(type(spec.get(field)) is not int for field in ("start", "stop")):
            raise RealEvalConfigError(f"{where} RANGE requires integer start/stop")
        mapping = list(range(spec["start"], spec["stop"]))
    elif isinstance(spec, dict) and spec.get("kind") == "RANGE_DROP_ONE":
        if any(type(spec.get(field)) is not int for field in ("start", "stop", "drop")):
            raise RealEvalConfigError(f"{where} RANGE_DROP_ONE requires integer start/stop/drop")
        mapping = [index for index in range(spec["start"], spec["stop"]) if index != spec["drop"]]
    else:
        raise RealEvalConfigError(f"{where} must be an explicit list, RANGE, or RANGE_DROP_ONE")
    if len(mapping) not in (181, 177, 89):
        raise RealEvalConfigError(f"{where} must expand to 181, 177, or 89 indices")
    if any(type(item) is not int or item not in range(181) for item in mapping):
        raise RealEvalConfigError(f"{where} contains an invalid source index")
    return mapping


def validate_real_config(config):
    _require(
        config,
        (
            "schema_version", "study_id", "adoption", "method", "payload_bits",
            "keys", "models", "codec", "edit_maps", "cases", "evaluation_rules",
        ),
        "real evaluation config",
    )
    if config["schema_version"] != SCHEMA_VERSION:
        raise RealEvalConfigError(f"schema_version must be {SCHEMA_VERSION}")
    _identifier(config["study_id"], "study_id")
    _require(config["adoption"], ("status",), "adoption")
    if config["adoption"]["status"] not in (
        "PENDING_USER_ADOPTION", "ADOPTED_FOR_EXECUTION", "ADOPTED_METHOD_DEFINITION_LOCAL_ONLY",
    ):
        raise RealEvalConfigError("adoption.status is invalid")
    _require(config["method"], ("name", "version", "claim_ceiling"), "method")
    if config["method"]["name"] != "trajectory-payload-framewise-sync-v1":
        raise RealEvalConfigError("method.name must identify the frozen current main method")
    _bits(config["payload_bits"], 32, "payload_bits")
    _require(config["keys"], ("K0", "K1"), "keys")
    for label in ("K0", "K1"):
        _identifier(config["keys"][label], f"keys.{label}")
    if config["keys"]["K0"] == config["keys"]["K1"]:
        raise RealEvalConfigError("K0 and K1 must differ")

    _require(config["models"], ("wan", "framewise", "videoseal", "rivagan"), "models")
    wan = config["models"]["wan"]
    _require(wan, ("upstream_id", "local_snapshot_path"), "models.wan")
    _strict_path(wan["local_snapshot_path"], "models.wan.local_snapshot_path")
    framewise = config["models"]["framewise"]
    _require(framewise, ("upstream_id", "local_snapshot_path", "batch_frames", "device"), "models.framewise")
    _strict_path(framewise["local_snapshot_path"], "models.framewise.local_snapshot_path")
    if (
        framewise["upstream_id"] != "stabilityai/sd-vae-ft-mse"
        or framewise["batch_frames"] != 8
    ):
        raise RealEvalConfigError("models.framewise must match the frozen runtime adapter")
    vs = config["models"]["videoseal"]
    _require(
        vs,
        (
            "source_root", "card_path", "checkpoint_path", "model_card_name", "device",
            "native_message_length", "native_message_bits", "lowres_attenuation",
            "detect_output_layout",
        ),
        "models.videoseal",
    )
    for field in ("source_root", "card_path", "checkpoint_path"):
        _strict_path(vs[field], f"models.videoseal.{field}")
    if type(vs["native_message_length"]) is not int or vs["native_message_length"] <= 0:
        raise RealEvalConfigError("models.videoseal.native_message_length must be positive")
    _bits(vs["native_message_bits"], vs["native_message_length"], "models.videoseal.native_message_bits")
    riva = config["models"]["rivagan"]
    _require(
        riva,
        (
            "source_root", "checkpoint_path",
            "checkpoint_provenance", "model_name", "device", "input_color_layout",
            "native_message_bits",
        ),
        "models.rivagan",
    )
    for field in ("source_root", "checkpoint_path"):
        _strict_path(riva[field], f"models.rivagan.{field}")
    _bits(riva["native_message_bits"], 32, "models.rivagan.native_message_bits")
    if riva["input_color_layout"] not in ("BGR_UINT8", "RGB_UINT8_TO_BGR"):
        raise RealEvalConfigError("models.rivagan.input_color_layout is invalid")

    codec = config["codec"]
    if codec != {"codec": "libx264", "fps": 8, "crf": 18, "pix_fmt": "yuv420p"}:
        raise RealEvalConfigError("codec must explicitly match the shared fixed runtime transport")
    if not isinstance(config["edit_maps"], dict):
        raise RealEvalConfigError("edit_maps must be an object")
    for map_id, spec in config["edit_maps"].items():
        _identifier(map_id, "edit_maps key")
        _expand_edit_map(spec, f"edit_maps.{map_id}")
    if not isinstance(config["cases"], list):
        raise RealEvalConfigError("cases must be a list")
    case_ids = set()
    observation_ids = set()
    for index, case in enumerate(config["cases"]):
        where = f"cases[{index}]"
        _require(case, ("case_id", "cohort", "source_status", "prompt", "negative_prompt", "seed", "observations"), where)
        _identifier(case["case_id"], f"{where}.case_id")
        if case["case_id"] in case_ids:
            raise RealEvalConfigError(f"duplicate case_id {case['case_id']}")
        case_ids.add(case["case_id"])
        if case["cohort"] not in ("PILOT_EXCLUDED_FROM_CONFIRMATION", "CONFIRMATION_CANDIDATE"):
            raise RealEvalConfigError(f"{where}.cohort is invalid")
        if case["source_status"] not in (
            "USER_ADOPTED",
            "PROPOSAL_UNSEEN_STATUS_UNVERIFIED",
            "USER_ADOPTED_LIMITED_LOCAL_NO_MATCH_UNPROVEN",
        ):
            raise RealEvalConfigError(f"{where}.source_status is invalid")
        if type(case["seed"]) is not int:
            raise RealEvalConfigError(f"{where}.seed must be integer")
        _identifier(case["prompt"], f"{where}.prompt")
        _identifier(case["negative_prompt"], f"{where}.negative_prompt")
        if not isinstance(case["observations"], list):
            raise RealEvalConfigError(f"{where}.observations must be a list")
        for obs_index, observation in enumerate(case["observations"]):
            obs_where = f"{where}.observations[{obs_index}]"
            _require(observation, ("observation_id", "arm", "protocol", "map_id", "modes"), obs_where)
            _identifier(observation["observation_id"], f"{obs_where}.observation_id")
            full_id = f"{case['case_id']}/{observation['observation_id']}"
            if full_id in observation_ids:
                raise RealEvalConfigError(f"duplicate observation {full_id}")
            observation_ids.add(full_id)
            if observation["arm"] not in MAIN_ARMS:
                raise RealEvalConfigError(f"{obs_where}.arm is invalid")
            if observation["protocol"] not in ("GLOBAL", "SINGLE_JUMP"):
                raise RealEvalConfigError(f"{obs_where}.protocol is invalid")
            if observation["map_id"] not in config["edit_maps"]:
                raise RealEvalConfigError(f"{obs_where}.map_id is not declared")
            allowed = ("RAW", "GLOBAL") if observation["protocol"] == "GLOBAL" else ("RAW", "GLOBAL", "PATH")
            if not isinstance(observation["modes"], list) or not observation["modes"]:
                raise RealEvalConfigError(f"{obs_where}.modes must be nonempty")
            if len(set(observation["modes"])) != len(observation["modes"]) or any(mode not in allowed for mode in observation["modes"]):
                raise RealEvalConfigError(f"{obs_where}.modes do not match protocol")
        views = Counter(item["observation_id"].split("/", 1)[1] for item in case["observations"])
        if len(case["observations"]) != 36 or len(views) != 9 or set(views.values()) != {4}:
            raise RealEvalConfigError(f"{where}.observations must declare nine logical views for each of four main arms")

    rules = config["evaluation_rules"]
    _require(rules, ("videoseal_32", "rivagan_sequence"), "evaluation_rules")
    for name in ("videoseal_32", "rivagan_sequence"):
        _require(rules[name], ("status", "rule"), f"evaluation_rules.{name}")
        if rules[name]["status"] not in ("PENDING_USER_ADOPTION", *RULE_STATUS_ACTIVE):
            raise RealEvalConfigError(f"evaluation_rules.{name}.status is invalid")
    if rules["videoseal_32"]["rule"] not in (VIDEOSEAL_STRICT_RULE, VIDEOSEAL_NATIVE_TIE_RULE):
        raise RealEvalConfigError("unrecognized VideoSeal 32-task rule")
    if rules["rivagan_sequence"]["rule"] not in (RIVAGAN_STRICT_RULE, RIVAGAN_NATIVE_TIE_RULE):
        raise RealEvalConfigError("unrecognized RivaGAN sequence rule")
    if rules["videoseal_32"]["status"] in RULE_STATUS_ACTIVE:
        length = vs["native_message_length"]
        if length % 32 or vs["native_message_bits"] != [config["payload_bits"][index % 32] for index in range(length)]:
            raise RealEvalConfigError("adopted VideoSeal 32-task rule requires exact j mod 32 repeated message and K divisible by 32")
    if (
        rules["rivagan_sequence"]["status"] in RULE_STATUS_ACTIVE
        and riva["native_message_bits"] != config["payload_bits"]
    ):
        raise RealEvalConfigError("adopted RivaGAN sequence rule requires the same explicit 32-bit task")
    comparison = config.get("paper_comparison")
    if comparison is not None:
        _require(
            comparison,
            (
                "status", "main_arm", "main_key_label", "global_mode",
                "single_jump_mode", "full_mode", "full_separate_control",
            ),
            "paper_comparison",
        )
        if comparison["status"] not in ("PENDING_USER_ADOPTION", *RULE_STATUS_ACTIVE):
            raise RealEvalConfigError("paper_comparison.status is invalid")
        expected = {
            "main_arm": "PAYLOAD_FRAMEWISE_M05",
            "main_key_label": "K0",
            "global_mode": "GLOBAL",
            "single_jump_mode": "PATH",
            "full_mode": "RAW",
            "full_separate_control": True,
        }
        if any(comparison.get(key) != value for key, value in expected.items()):
            raise RealEvalConfigError("paper_comparison must match the adopted fixed M05/K0 readout")
    return config


def _json_dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return {"path": str(path), "sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded)}


def _artifact_id(case_id, stage, phase):
    return f"{case_id}/{stage}/{phase}"


def _baseline_views(case):
    """Return the nine declared logical views once, independent of main arm."""

    by_view = {}
    for observation in case["observations"]:
        view_id = observation["observation_id"].split("/", 1)[1]
        signature = (
            observation["protocol"], observation["map_id"],
            observation.get("analysis_role", "UNSPECIFIED"),
        )
        if view_id in by_view and by_view[view_id][0] != signature:
            raise RealEvalConfigError(f"baseline view {view_id} differs across main arms")
        by_view.setdefault(view_id, (signature, observation))
    return [by_view[key][1] for key in sorted(by_view)]


def _baseline_soft_artifact_id(case_id, method, view_id):
    return f"{case_id}/{method}/NATIVE_SOFT/{view_id}"


def _main_comparison_mode(observation):
    if observation["map_id"] == "full181":
        return "RAW"
    if observation["protocol"] == "GLOBAL":
        return "GLOBAL"
    if observation["protocol"] == "SINGLE_JUMP":
        return "PATH"
    raise RealEvalConfigError("unsupported comparison protocol")


def _build_comparison_slots(config):
    rows = []
    for case in config["cases"]:
        case_id = case["case_id"]
        for baseline in BASELINES:
            for observation in _baseline_views(case):
                view_id = observation["observation_id"].split("/", 1)[1]
                main_observation_id = f"payload_framewise_m05/{view_id}"
                mode = _main_comparison_mode(observation)
                rows.append({
                    "comparison_id": f"{case_id}/{baseline}/{view_id}",
                    "case_id": case_id,
                    "cohort": case["cohort"],
                    "source_status": case["source_status"],
                    "baseline_method": baseline,
                    "view_id": view_id,
                    "map_id": observation["map_id"],
                    "protocol": observation["protocol"],
                    "analysis_role": observation.get("analysis_role", "UNSPECIFIED"),
                    "main_arm": "PAYLOAD_FRAMEWISE_M05",
                    "main_key_label": "K0",
                    "main_mode": mode,
                    "main_slot_id": f"{case_id}/{main_observation_id}/K0/{mode}",
                    "baseline_slot_id": f"{case_id}/{baseline}/{view_id}",
                    "status": "PLANNED",
                })
    return rows


def _comparison_enabled(config):
    comparison = config.get("paper_comparison")
    return isinstance(comparison, dict) and comparison.get("status") in RULE_STATUS_ACTIVE


def _receiver_resource_plan(config, case_ids=None):
    selected = [
        case for case in config["cases"]
        if case_ids is None or case["case_id"] in set(case_ids)
    ]
    sync_encodes = sum(len(case["observations"]) for case in selected)
    reads = 0
    raw_identities = set()
    nonraw_upper = 0
    for case in selected:
        for observation in case["observations"]:
            key_count = len(config["keys"])
            reads += key_count * len(observation["modes"])
            if "RAW" in observation["modes"]:
                mapping = tuple(_expand_edit_map(
                    config["edit_maps"][observation["map_id"]],
                    f"edit_maps.{observation['map_id']}",
                ))
                raw_identities.add((case["case_id"], observation["arm"], mapping))
            nonraw_upper += key_count * sum(mode != "RAW" for mode in observation["modes"])
    return {
        "sync_framewise_encodes": sync_encodes,
        "raw_unique_map_arm_encodes": len(raw_identities),
        "aligned_or_path_encode_upper_bound": nonraw_upper,
        "wan_physical_encode_upper_bound": len(raw_identities) + nonraw_upper,
        "wan_reads": reads,
    }


def build_plan(config):
    """Expand the complete fixed denominator without importing model code."""

    validate_real_config(config)
    artifacts = []
    costs = []
    receiver_slots = []
    baseline_slots = []
    comparison_slots = []
    quality_rows = []
    for case in config["cases"]:
        case_id = case["case_id"]
        first_artifact = len(artifacts)
        for arm in MAIN_ARMS:
            if arm in ("OFF_NATIVE", "PAYLOAD_NATIVE"):
                artifacts.append({"artifact_id": _artifact_id(case_id, arm, "TERMINAL"), "status": "PLANNED"})
            artifacts.extend([
                {"artifact_id": _artifact_id(case_id, arm, "PRE"), "status": "PLANNED"},
                {"artifact_id": _artifact_id(case_id, arm, "POST"), "status": "PLANNED"},
            ])
        artifacts.append({"artifact_id": _artifact_id(case_id, "SHARED_FRAMEWISE", "LATENT"), "status": "PLANNED"})
        for baseline in BASELINES:
            artifacts.extend([
                {"artifact_id": _artifact_id(case_id, baseline, "NATIVE_PRE"), "status": "PLANNED"},
                {"artifact_id": _artifact_id(case_id, baseline, "NATIVE_POST"), "status": "PLANNED"},
            ])
            for observation in _baseline_views(case):
                view_id = observation["observation_id"].split("/", 1)[1]
                frames = len(_expand_edit_map(
                    config["edit_maps"][observation["map_id"]],
                    f"edit_maps.{observation['map_id']}",
                ))
                artifact_id = _baseline_soft_artifact_id(case_id, baseline, view_id)
                artifacts.append({"artifact_id": artifact_id, "status": "PLANNED"})
                baseline_slots.append({
                    "slot_id": f"{case_id}/{baseline}/{view_id}",
                    "case_id": case_id,
                    "cohort": case["cohort"],
                    "method": baseline,
                    "view_id": view_id,
                    "protocol": observation["protocol"],
                    "map_id": observation["map_id"],
                    "analysis_role": observation.get("analysis_role", "UNSPECIFIED"),
                    "frames": frames,
                    "artifact_id": artifact_id,
                    "status": "PLANNED",
                })
        for left, right in QUALITY_PAIRS:
            quality_rows.append({
                "quality_id": f"{case_id}/{left}_vs_{right}",
                "case_id": case_id,
                "cohort": case["cohort"],
                "left": left,
                "right": right,
                "phase": "POST",
                "status": "PLANNED",
            })
        for phase in PHASES[:-1]:
            costs.append({"cost_id": f"{case_id}/{phase}", "status": "PLANNED", "seconds": None})
        for observation in case["observations"]:
            artifacts.append({
                "artifact_id": f"{case_id}/observation/{observation['observation_id']}",
                "status": "PLANNED",
            })
            for key_label in ("K0", "K1"):
                for mode in observation["modes"]:
                    receiver_slots.append({
                        "slot_id": f"{case_id}/{observation['observation_id']}/{key_label}/{mode}",
                        "case_id": case_id,
                        "cohort": case["cohort"],
                        "observation_id": observation["observation_id"],
                        "arm": observation["arm"],
                        "protocol": observation["protocol"],
                        "analysis_role": observation.get("analysis_role", "UNSPECIFIED"),
                        "frames": len(_expand_edit_map(config["edit_maps"][observation["map_id"]], f"edit_maps.{observation['map_id']}")),
                        "key_label": key_label,
                        "mode": mode,
                        "planned_bits": 32,
                        "status": "PLANNED",
                    })
        for artifact in artifacts[first_artifact:]:
            artifact.update(
                case_id=case_id,
                source_id=case_id,
                source_status=case["source_status"],
                noise_seed=case["seed"],
                codec=config["codec"],
                cohort=case["cohort"],
            )
    if _comparison_enabled(config):
        comparison_slots.extend(_build_comparison_slots(config))
    return {
        "artifacts": artifacts,
        "costs": costs,
        "receiver_slots": receiver_slots,
        "baseline_slots": baseline_slots,
        "comparison_slots": comparison_slots,
        "quality_rows": quality_rows,
        "receiver_resource_plan": _receiver_resource_plan(config),
    }


class RunStore:
    def __init__(self, output, config, *, create=False):
        self.output = Path(output).resolve()
        self.path = self.output / "run_state.json"
        self.config = config
        config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if create:
            self.output.mkdir(parents=True, exist_ok=False)
            planned = build_plan(config)
            case_states = {case["case_id"]: {"status": "PLANNED", "failures": []} for case in config["cases"]}
            self.data = {
                "schema_version": SCHEMA_VERSION,
                "study_id": config["study_id"],
                "status": "INITIALIZED",
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "config_identity_observation": {
                    "expected": hashlib.sha256(config_bytes).hexdigest(),
                    "actual": hashlib.sha256(config_bytes).hexdigest(),
                    "status": "MATCH",
                    "blocking": False,
                },
                "method": config["method"],
                "message_contract": {
                    "bits": config["payload_bits"],
                    "support": "NEW_OUTER_EXPLICIT_32BIT_INTERFACE_NOT_HISTORICAL_ARBITRARY_MESSAGE_EVIDENCE",
                },
                "phases": {
                    phase: (
                        {"status": "PLANNED", "failures": []}
                        if phase == "evaluate"
                        else {"status": "PLANNED", "cases": copy.deepcopy(case_states)}
                    )
                    for phase in PHASES
                },
                **planned,
                "records": {},
            }
            self.save()
        else:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            actual = hashlib.sha256(config_bytes).hexdigest()
            expected = self.data.get("config_sha256")
            self.data["config_identity_observation"] = {
                "expected": expected,
                "actual": actual,
                "status": "MATCH" if expected == actual else "RECORDED_DIFFERENCE",
                "blocking": False,
            }
            self.save()

    def save(self):
        _json_dump(self.path, self.data)

    def artifact(self, artifact_id, *, status, **fields):
        row = next(row for row in self.data["artifacts"] if row["artifact_id"] == artifact_id)
        row.update(status=status, **fields)
        self.save()
        return row

    @staticmethod
    def _refresh_case_phase_status(row):
        states = [item["status"] for item in row["cases"].values()]
        if "RUNNING" in states:
            row["status"] = "RUNNING"
        elif "FAILED" in states:
            row["status"] = "FAILED"
        elif states and all(state == "COMPLETE" for state in states):
            row["status"] = "COMPLETE"
        elif "COMPLETE" in states:
            row["status"] = "PARTIAL"
        else:
            row["status"] = "PLANNED"

    def phase_start(self, phase, case_id=None):
        row = self.data["phases"][phase]
        target = row if case_id is None else row["cases"][case_id]
        if phase != "evaluate" and target["status"] != "PLANNED":
            raise RealEvalConfigError(
                f"{phase}/{case_id} already attempted with status {target['status']}; "
                "start a new explicit run for another attempt"
            )
        target.update(status="RUNNING", started_at_unix=time.time())
        if case_id is None:
            row["status"] = "RUNNING"
        else:
            self._refresh_case_phase_status(row)
        self.save()

    def phase_finish(self, phase, case_id=None):
        row = self.data["phases"][phase]
        target = row if case_id is None else row["cases"][case_id]
        target.update(status="COMPLETE", finished_at_unix=time.time())
        if case_id is None:
            row["status"] = "COMPLETE"
        else:
            self._refresh_case_phase_status(row)
        self.save()

    def phase_failure(self, phase, exc, case_id=None):
        reason = f"{type(exc).__name__}: {exc}"
        row = self.data["phases"][phase]
        target = row if case_id is None else row["cases"][case_id]
        target.update(status="FAILED", finished_at_unix=time.time())
        target["failures"].append(reason)
        if case_id is None:
            row["status"] = "FAILED"
        else:
            self._refresh_case_phase_status(row)
        if case_id is not None:
            for artifact_id in self._phase_artifacts(phase, case_id):
                artifact = next(item for item in self.data["artifacts"] if item["artifact_id"] == artifact_id)
                if artifact["status"] == "PLANNED":
                    artifact.update(status="FAILED", reason=reason)
            if phase.startswith("baseline-extract-"):
                method = phase.removeprefix("baseline-extract-")
                for slot in self.data["baseline_slots"]:
                    if slot["case_id"] == case_id and slot["method"] == method and slot["status"] == "PLANNED":
                        slot.update(status="FAILED", reason=reason)
            if phase == "quality":
                for quality in self.data["quality_rows"]:
                    if quality["case_id"] == case_id and quality["status"] == "PLANNED":
                        quality.update(status="FAILED", reason=reason)
        self.save()
        return reason

    @staticmethod
    def _phase_artifacts(phase, case_id):
        if phase == "generate":
            return [_artifact_id(case_id, arm, "TERMINAL") for arm in ("OFF_NATIVE", "PAYLOAD_NATIVE")]
        if phase == "decode":
            return [_artifact_id(case_id, arm, "PRE") for arm in ("OFF_NATIVE", "PAYLOAD_NATIVE")]
        if phase == "framewise":
            return [
                _artifact_id(case_id, "SHARED_FRAMEWISE", "LATENT"),
                _artifact_id(case_id, "PAYLOAD_FRAMEWISE_RECON", "PRE"),
                _artifact_id(case_id, "PAYLOAD_FRAMEWISE_M05", "PRE"),
            ]
        if phase.startswith("baseline-embed-"):
            return [_artifact_id(case_id, phase.removeprefix("baseline-embed-"), "NATIVE_PRE")]
        if phase == "codec":
            return [
                *[_artifact_id(case_id, arm, "POST") for arm in MAIN_ARMS],
                *[_artifact_id(case_id, method, "NATIVE_POST") for method in BASELINES],
            ]
        if phase.startswith("baseline-extract-"):
            method = phase.removeprefix("baseline-extract-")
            return [
                row["artifact_id"] for row in self.data["baseline_slots"]
                if row["case_id"] == case_id and row["method"] == method
            ]
        if phase == "receiver-sync":
            return [
                row["artifact_id"] for row in self.data["artifacts"]
                if row["case_id"] == case_id and "/observation/" in row["artifact_id"]
            ]
        return []

    def cost(self, case_id, phase, *, status, seconds=None, reason=None):
        cost_id = f"{case_id}/{phase}"
        row = next(row for row in self.data["costs"] if row["cost_id"] == cost_id)
        row.update(status=status, seconds=seconds, reason=reason, budget_status="NO_BUDGET_ADOPTED")
        self.save()


class ArtifactFiles:
    def __init__(self, store, case_id):
        self.store = store
        self.case_id = case_id
        self.root = store.output / "artifacts" / case_id

    def _receipt(self, path, **extra):
        receipt = {"path": str(path), "bytes": path.stat().st_size, **extra}
        _observe_file_identity(receipt)
        receipt["sha256"] = receipt["sha256_observation"]["actual"]
        return receipt

    def save_rgb8(self, artifact_id, value, name):
        import numpy as np

        candidate = value
        for method in ("detach", "cpu"):
            function = getattr(candidate, method, None)
            if callable(function):
                candidate = function()
        numpy_method = getattr(candidate, "numpy", None)
        if callable(numpy_method):
            candidate = numpy_method()
        array = np.asarray(candidate)
        if array.dtype != np.uint8 or array.ndim != 4 or int(array.shape[-1]) != 3:
            raise ValueError("RGB8 artifact must be uint8 [T,H,W,3]")
        path = self.root / f"{name}.rgb8"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(array.tobytes(order="C"))
        os.replace(temporary, path)
        receipt = self._receipt(path, shape=[int(part) for part in array.shape], dtype="uint8")
        self.store.artifact(artifact_id, status="AVAILABLE", **receipt)
        return receipt

    @staticmethod
    def load_rgb8(receipt):
        import numpy as np
        import torch

        path = Path(receipt["path"])
        _observe_file_identity(receipt)
        shape = tuple(receipt["shape"])
        raw = path.read_bytes()
        if len(raw) != math.prod(shape):
            raise ValueError("RGB8 artifact byte count does not match shape")
        return torch.from_numpy(np.frombuffer(raw, np.uint8).reshape(shape).copy())

    def save_numpy(self, artifact_id, value, name):
        import numpy as np

        path = self.root / f"{name}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
        array = np.asarray(value)
        receipt = self._receipt(path, shape=[int(part) for part in array.shape], dtype=str(array.dtype))
        self.store.artifact(artifact_id, status="AVAILABLE", **receipt)
        return receipt

    def native_output_store(self, label, value, descriptor):
        import numpy as np

        arrays = {}

        def collect(prefix, item):
            if isinstance(item, dict):
                for key, child in item.items():
                    collect(f"{prefix}.{key}" if prefix else str(key), child)
                return
            candidate = item
            for method in ("detach", "cpu"):
                function = getattr(candidate, method, None)
                if callable(function):
                    candidate = function()
            numpy_method = getattr(candidate, "numpy", None)
            if callable(numpy_method):
                candidate = numpy_method()
            array = np.asarray(candidate)
            if array.dtype.kind not in "biuf":
                raise ValueError(f"native output {prefix} is not a numeric lossless array")
            arrays[prefix or "value"] = array

        collect("", value)
        path = self.root / "native" / f"{label}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        identity_receipt = {"path": str(path)}
        actual_sha256 = _observe_file_identity(identity_receipt)
        return {
            "lossless": True,
            "uri": str(path),
            "format": "npz",
            "sha256": actual_sha256,
            "sha256_observation": identity_receipt["sha256_observation"],
            "bytes": path.stat().st_size,
            "arrays": {
                key: {"shape": [int(part) for part in array.shape], "dtype": str(array.dtype)}
                for key, array in arrays.items()
            },
            "descriptor": descriptor,
        }


def static_preflight(config):
    """Check paths and import availability without importing model libraries."""

    validate_real_config(config)
    checks = []

    def check(name, ok, detail, **fields):
        checks.append({
            "name": name,
            "status": "READY" if ok else "BLOCKED",
            "detail": str(detail),
            **fields,
        })

    def observe_digest(path):
        try:
            return file_sha256(path), None
        except OSError as exc:
            return None, f"{type(exc).__name__}: {exc}"

    for executable in ("ffmpeg", "ffprobe"):
        found = shutil.which(executable)
        check(f"executable:{executable}", bool(found), found or "not found")
    for module in ("torch", "numpy", "diffusers", "transformers", "safetensors", "accelerate", "omegaconf"):
        spec = importlib.util.find_spec(module)
        check(f"python_module:{module}", spec is not None, spec.origin if spec else "not found")
    for name, required in (
        (
            "wan",
            (
                "model_index.json", "scheduler/scheduler_config.json", "tokenizer/tokenizer_config.json",
                "text_encoder/config.json", "transformer/config.json", "vae/config.json",
            ),
        ),
        ("framewise", ("config.json", "diffusion_pytorch_model.safetensors")),
    ):
        path = Path(config["models"][name]["local_snapshot_path"]).expanduser()
        missing = [item for item in required if not (path / item).exists()]
        check(f"local_snapshot:{name}", path.is_dir() and not missing, f"{path}; missing={missing}")
        weights = sorted(path.rglob("*.safetensors")) if path.is_dir() else []
        check(f"local_snapshot:{name}:weights", bool(weights), f"{len(weights)} local safetensors files")
    for method in BASELINES:
        row = config["models"][method]
        root = Path(row["source_root"]).expanduser()
        package = root / ("videoseal/__init__.py" if method == "videoseal" else "rivagan/rivagan.py")
        check(f"{method}:source", package.is_file(), package)
        for field in (("card_path", "card_sha256"),) if method == "videoseal" else ():
            path = Path(row[field[0]]).expanduser()
            actual, hash_error = observe_digest(path) if path.is_file() else (None, None)
            expected = row.get(field[1])
            check(
                f"{method}:{field[0]}", path.is_file(), path,
                expected_sha256=expected, actual_sha256=actual,
                sha256_status=(
                    "MATCH" if expected == actual and actual
                    else "UNDECLARED" if not expected
                    else "RECORDED_DIFFERENCE" if actual
                    else "UNAVAILABLE"
                ),
                sha256_observation_error=hash_error,
                sha256_blocking=False,
            )
        path = Path(row["checkpoint_path"]).expanduser()
        actual, hash_error = observe_digest(path) if path.is_file() else (None, None)
        expected = row.get("checkpoint_sha256")
        check(
            f"{method}:checkpoint", path.is_file(), path,
            expected_sha256=expected, actual_sha256=actual,
            sha256_status=(
                "MATCH" if expected == actual and actual
                else "UNDECLARED" if not expected
                else "RECORDED_DIFFERENCE" if actual
                else "UNAVAILABLE"
            ),
            sha256_observation_error=hash_error,
            sha256_blocking=False,
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "study_id": config["study_id"],
        "status": "READY" if all(row["status"] == "READY" for row in checks) else "BLOCKED",
        "loads_models": False,
        "executes_media": False,
        "checks": checks,
    }


def _runtime_generation_config(config, case):
    return {
        "model": {
            "id": str(Path(config["models"]["wan"]["local_snapshot_path"]).expanduser().resolve()),
            "revision": None,
            "upstream_id": config["models"]["wan"]["upstream_id"],
            "upstream_revision": config["models"]["wan"].get("revision"),
            "local_files_only_by_path": True,
        },
        "generation": {
            "height": 320,
            "width": 512,
            "frames": 181,
            "fps": config["codec"]["fps"],
            "steps": 50,
            "guidance_scale": 5.0,
            "max_sequence_length": 512,
            "prompt": case["prompt"],
            "negative_prompt": case["negative_prompt"],
            "seed": case["seed"],
        },
    }


def run_off_trajectory(pipe, initial, scheduler, prompt, negative, dtype, count, record):
    """Exact no-payload sibling of the current conditional-joint P0 loop."""

    import torch
    from runtime.wan import video_trajectory_conditional_joint_v1 as runtime

    runtime.trajectory.validate_scheduler(scheduler)
    if scheduler.step_index is not None:
        raise ValueError("fresh full native history required")
    z = initial.detach().float().clone()
    count = count or (lambda name, done: None)
    for index in range(50):
        conditional, unconditional = runtime.predict_branches(
            pipe, z, scheduler, prompt, negative, dtype, index, count,
        )
        sigma = float(scheduler.sigmas[index])
        velocity = unconditional.float() + 5.0 * (conditional.float() - unconditional.float())
        if not bool(torch.isfinite(velocity).all()):
            raise FloatingPointError("nonfinite OFF CFG")
        z = runtime.trajectory.native_step(scheduler, z, velocity.detach(), index, count, kind="native_step")
        record({
            "index": index,
            "sigma": sigma,
            "cursor_after": scheduler.step_index,
            "arm": "OFF_NATIVE",
            "payload_enabled": False,
            "cfg_dtype": "float32",
        })
    if scheduler.step_index != 50:
        raise RuntimeError("OFF native trajectory incomplete")
    return z.detach().cpu(), {
        "arm": "OFF_NATIVE",
        "payload_enabled": False,
        "steps": 50,
        "cfg": "unconditional + 5*(conditional-unconditional)",
        "scheduler_path": "runtime.wan.conditional_joint.trajectory.native_step",
    }


def _case(config, case_id):
    try:
        return next(case for case in config["cases"] if case["case_id"] == case_id)
    except StopIteration as exc:
        raise RealEvalConfigError(f"unknown case_id {case_id}") from exc


def _artifact(store, artifact_id):
    return next(row for row in store.data["artifacts"] if row["artifact_id"] == artifact_id)


def _run_timed(store, phase, case_id, function):
    store.phase_start(phase, case_id)
    started = time.perf_counter()
    try:
        value = function()
        elapsed = time.perf_counter() - started
        store.cost(case_id, phase, status="SUCCEEDED", seconds=elapsed)
        store.phase_finish(phase, case_id)
        return value
    except BaseException as exc:
        elapsed = time.perf_counter() - started
        reason = store.phase_failure(phase, exc, case_id)
        store.cost(case_id, phase, status="FAILED", seconds=elapsed, reason=reason)
        raise


def phase_generate(store, config, case_id):
    def execute():
        import torch
        from runtime.wan import video_trajectory_conditional_joint_v1 as runtime

        case = _case(config, case_id)
        cfg = _runtime_generation_config(config, case)
        device, dtype = runtime.execution_device_dtype()
        pipe, initial, prompt, negative, input_dtype = runtime.generation.prepare_generation(
            cfg, load_vae=False, device=device, model_dtype=dtype,
        )
        pristine = copy.deepcopy(pipe.scheduler)
        records = {}
        try:
            for arm in ("OFF_NATIVE", "PAYLOAD_NATIVE"):
                steps = []
                if arm == "OFF_NATIVE":
                    terminal, receipt = run_off_trajectory(
                        pipe, initial, copy.deepcopy(pristine), prompt, negative,
                        input_dtype, None, steps.append,
                    )
                else:
                    terminal, receipt = runtime.run_trajectory(
                        pipe, initial, copy.deepcopy(pristine), prompt, negative,
                        input_dtype, config["keys"]["K0"], list(config["payload_bits"]),
                        lambda name, done: None, steps.append,
                    )
                    receipt["message_interface"] = "EXPLICIT_32_BITS_OUTER_INTERFACE_NEW_NOT_HISTORICAL_CAPACITY_EVIDENCE"
                path = store.output / "artifacts" / case_id / arm / "terminal.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(terminal, path)
                artifact_id = _artifact_id(case_id, arm, "TERMINAL")
                file_receipt = {
                    "path": str(path), "bytes": path.stat().st_size,
                    "shape": list(terminal.shape), "dtype": str(terminal.dtype),
                }
                file_receipt["sha256"] = _observe_file_identity(file_receipt)
                store.artifact(
                    artifact_id, status="AVAILABLE", **file_receipt,
                )
                records[arm] = {"receipt": receipt, "steps": steps}
                del terminal
        finally:
            pipe = initial = prompt = negative = pristine = None
            runtime.release()
        store.data["records"].setdefault(case_id, {})["generation"] = records
        store.save()

    return _run_timed(store, "generate", case_id, execute)


def phase_decode(store, config, case_id):
    def execute():
        import torch
        from runtime.wan import video_trajectory_conditional_joint_v1 as runtime

        case = _case(config, case_id)
        cfg = _runtime_generation_config(config, case)
        files = ArtifactFiles(store, case_id)
        device, _dtype = runtime.execution_device_dtype()
        vae = runtime.generation.load_frozen_vae(cfg, device=device)
        try:
            for arm in ("OFF_NATIVE", "PAYLOAD_NATIVE"):
                terminal_row = _artifact(store, _artifact_id(case_id, arm, "TERMINAL"))
                if terminal_row["status"] != "AVAILABLE":
                    raise ValueError(f"{arm} terminal unavailable")
                _observe_file_identity(terminal_row)
                terminal = torch.load(terminal_row["path"], map_location="cpu", weights_only=True)
                rgb = runtime.vae.decode_normalized_latent(
                    vae, terminal.to(device=next(vae.parameters()).device, dtype=torch.float32),
                )
                q8 = runtime.vae.quantize_rgb8_no_codec(rgb)
                files.save_rgb8(_artifact_id(case_id, arm, "PRE"), q8, f"{arm}/pre")
                del terminal, rgb, q8
        finally:
            vae = None
            runtime.release()

    return _run_timed(store, "decode", case_id, execute)


def phase_framewise(store, config, case_id):
    def execute():
        from runtime.wan import video_trajectory_conditional_joint_v1 as runtime

        case = _case(config, case_id)
        cfg = _runtime_generation_config(config, case)
        files = ArtifactFiles(store, case_id)
        p0 = _artifact(store, _artifact_id(case_id, "PAYLOAD_NATIVE", "PRE"))
        source = files.load_rgb8(p0)
        model = load_local_framewise_backend(config["models"]["framewise"])
        try:
            latent = model.encode(source)
            before = hashlib.sha256(latent.tobytes()).hexdigest()
            files.save_numpy(_artifact_id(case_id, "SHARED_FRAMEWISE", "LATENT"), latent, "framewise/shared_latent")
            written, write_receipt = model.write(latent.copy(), config["keys"]["K0"])
            if hashlib.sha256(latent.tobytes()).hexdigest() != before:
                raise RuntimeError("M05 write changed shared framewise latent")
            outputs = {
                "PAYLOAD_FRAMEWISE_RECON": model.decode(latent.copy()),
                "PAYLOAD_FRAMEWISE_M05": model.decode(written.copy()),
            }
            for arm, rgb in outputs.items():
                files.save_rgb8(_artifact_id(case_id, arm, "PRE"), rgb, f"{arm}/pre")
            receipt = _json_dump(files.root / "framewise" / "m05_write_receipt.json", write_receipt)
            store.data["records"].setdefault(case_id, {})["framewise"] = {
                "single_encode": True,
                "independent_decode_inputs": True,
                "shared_latent_sha256": before,
                "write_receipt": receipt,
            }
            store.save()
        finally:
            model.close()

    return _run_timed(store, "framewise", case_id, execute)


def _baseline_adapter(config, method, files):
    if method == "videoseal":
        return load_videoseal_adapter(
            config["models"][method], native_output_store=files.native_output_store,
        )
    if method == "rivagan":
        return load_rivagan_adapter(config["models"][method])
    raise ValueError(method)


def phase_baseline_embed(store, config, case_id, method):
    phase = f"baseline-embed-{method}"

    def execute():
        files = ArtifactFiles(store, case_id)
        source_row = _artifact(store, _artifact_id(case_id, "OFF_NATIVE", "PRE"))
        source = files.load_rgb8(source_row)
        adapter = _baseline_adapter(config, method, files)
        try:
            media = source.numpy() if method == "rivagan" else source
            embedded, record = adapter.embed(
                media, list(config["models"][method]["native_message_bits"]),
            )
            files.save_rgb8(_artifact_id(case_id, method, "NATIVE_PRE"), embedded, f"{method}/native_pre")
            record_path = _json_dump(files.root / method / "embed_record.json", record)
            store.data["records"].setdefault(case_id, {}).setdefault("native", {}).setdefault(method, {})["embed"] = record_path
            store.save()
        finally:
            close_adapter(adapter)

    return _run_timed(store, phase, case_id, execute)


def phase_codec(store, config, case_id):
    def execute():
        from runtime.wan import fixed_rgb_media

        files = ArtifactFiles(store, case_id)
        targets = [(arm, "PRE", "POST") for arm in MAIN_ARMS]
        targets += [(method, "NATIVE_PRE", "NATIVE_POST") for method in BASELINES]
        records = {}
        for stage, pre_phase, post_phase in targets:
            artifact_id = _artifact_id(case_id, stage, pre_phase)
            row = _artifact(store, artifact_id)
            post_id = _artifact_id(case_id, stage, post_phase)
            if row["status"] != "AVAILABLE":
                store.artifact(post_id, status="BLOCKED_DEPENDENCY", reason=f"{artifact_id} is {row['status']}")
                records[stage] = {"status": "BLOCKED_DEPENDENCY"}
                continue
            events = {}
            calls = Counter()
            root = files.root / stage / "codec"
            started = time.perf_counter()
            try:
                actual_raster_sha = _observe_file_identity(row)
                fixed_rgb_media.mp4_roundtrip(
                    row["path"], actual_raster_sha, root / "source.mp4", root / "received.rgb8",
                    count=lambda name, done: calls.update([f"{name}:{'completed' if done else 'attempted'}"]),
                    event=lambda name, value: events.__setitem__(name, value),
                )
                receipt = events["rgb24"]
                store.artifact(
                    post_id, status="AVAILABLE", path=receipt["path"], sha256=receipt["sha256"],
                    bytes=receipt["bytes"], shape=receipt["shape"], dtype="uint8",
                    mp4=events["mp4"], codec=config["codec"],
                )
                records[stage] = {
                    "status": "SUCCEEDED", "events": events, "calls": dict(calls),
                    "seconds": time.perf_counter() - started,
                }
            except Exception as exc:
                store.artifact(post_id, status="FAILED", reason=f"{type(exc).__name__}: {exc}")
                records[stage] = {
                    "status": "FAILED", "reason": f"{type(exc).__name__}: {exc}",
                    "events": events, "calls": dict(calls), "seconds": time.perf_counter() - started,
                }
        store.data["records"].setdefault(case_id, {})["codec"] = records
        store.save()

    return _run_timed(store, "codec", case_id, execute)


def phase_baseline_extract(store, config, case_id, method):
    phase = f"baseline-extract-{method}"

    def execute():
        files = ArtifactFiles(store, case_id)
        source_row = _artifact(store, _artifact_id(case_id, method, "NATIVE_POST"))
        source = files.load_rgb8(source_row)
        adapter = _baseline_adapter(config, method, files)
        slots = [
            row for row in store.data["baseline_slots"]
            if row["case_id"] == case_id and row["method"] == method
        ]
        records = store.data["records"].setdefault(case_id, {}).setdefault("native", {}).setdefault(method, {}).setdefault("extracts", {})
        cache = {}
        try:
            for slot in slots:
                view_id = slot["view_id"]
                try:
                    mapping = _expand_edit_map(
                        config["edit_maps"][slot["map_id"]],
                        f"edit_maps.{slot['map_id']}",
                    )
                    map_token = hashlib.sha256(json.dumps(mapping, separators=(",", ":")).encode()).hexdigest()
                    reused = map_token in cache
                    if not reused:
                        received = source[mapping]
                        if method == "videoseal":
                            adapter.native_output_store = (
                                lambda label, value, descriptor, token=map_token:
                                files.native_output_store(
                                    f"{method}/{token}/{label}", value, descriptor,
                                )
                            )
                        media = received.numpy() if method == "rivagan" else received
                        cache[map_token] = adapter.extract(media)
                    record = cache[map_token]
                    record_path = _json_dump(
                        files.root / method / "extract" / f"{view_id}.json", record,
                    )
                    store.artifact(
                        slot["artifact_id"], status="AVAILABLE", **record_path,
                        map_id=slot["map_id"], map_sha256=map_token,
                        reused_native_call=reused,
                    )
                    slot.update(status="AVAILABLE", record=record_path)
                    records[view_id] = record_path
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    slot.update(status="FAILED", reason=reason)
                    store.artifact(slot["artifact_id"], status="FAILED", reason=reason)
                store.save()
        finally:
            close_adapter(adapter)

    return _run_timed(store, phase, case_id, execute)


def _quality_metrics(left, right, *, chunk_elements=4_000_000):
    """Compute RGB8 quality without materializing a full float video copy."""

    import numpy as np

    if left.get("status") != "AVAILABLE" or right.get("status") != "AVAILABLE":
        raise ValueError(f"quality dependency unavailable: {left.get('status')}/{right.get('status')}")
    if left.get("shape") != right.get("shape") or left.get("dtype") != right.get("dtype"):
        raise ValueError("quality inputs must have identical shape and dtype")
    if left.get("dtype") != "uint8":
        raise ValueError("quality inputs must be uint8")
    for row in (left, right):
        _observe_file_identity(row)
    count = math.prod(left["shape"])
    left_values = np.memmap(left["path"], dtype=np.uint8, mode="r", shape=(count,))
    right_values = np.memmap(right["path"], dtype=np.uint8, mode="r", shape=(count,))
    squared_sum = 0
    maximum = 0
    for start in range(0, count, chunk_elements):
        stop = min(start + chunk_elements, count)
        difference = left_values[start:stop].astype(np.int16) - right_values[start:stop].astype(np.int16)
        squared_sum += int(np.square(difference, dtype=np.int64).sum(dtype=np.int64))
        maximum = max(maximum, int(np.abs(difference).max()))
    mse = squared_sum / count
    rmse = math.sqrt(mse)
    result = {
        "element_count": count,
        "squared_error_sum": squared_sum,
        "mse": mse,
        "rmse": rmse,
        "max_absolute_error": maximum,
        "identical": squared_sum == 0,
        "psnr_db": None if squared_sum == 0 else 20.0 * math.log10(255.0 / rmse),
        "psnr_status": "IDENTICAL_ZERO_MSE" if squared_sum == 0 else "FINITE",
    }
    if not all(math.isfinite(value) for key, value in result.items() if key in ("mse", "rmse", "max_absolute_error")):
        raise ValueError("quality calculation produced non-finite output")
    return result


def phase_quality(store, config, case_id):
    def execute():
        rows = [row for row in store.data["quality_rows"] if row["case_id"] == case_id]
        for row in rows:
            try:
                def post(stage):
                    phase = "NATIVE_POST" if stage in BASELINES else "POST"
                    return _artifact(store, _artifact_id(case_id, stage, phase))

                row.update(status="EVALUATED", **_quality_metrics(post(row["left"]), post(row["right"])))
            except Exception as exc:
                row.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
            store.save()

    return _run_timed(store, "quality", case_id, execute)


def _mode_internal(protocol, mode):
    if protocol == "GLOBAL":
        return {"RAW": "BASELINE", "GLOBAL": "EST_ALIGN"}[mode]
    return {"RAW": "RAW", "GLOBAL": "GLOBAL_ALIGN", "PATH": "PATH_ALIGN"}[mode]


def phase_receiver_sync(store, config, case_id):
    def execute():
        from main.tube_state import video_trajectory_conditional_joint_v1 as method
        from runtime.wan import video_trajectory_conditional_joint_v1 as runtime

        case = _case(config, case_id)
        files = ArtifactFiles(store, case_id)
        model = load_local_framewise_backend(config["models"]["framewise"])
        plans = {"truth_inputs": False, "observations": {}, "slots": {}}

        def persist():
            receipt = _json_dump(files.root / "receiver" / "blind_plan.json", plans)
            store.data["records"].setdefault(case_id, {})["blind_plan"] = receipt
            store.save()

        persist()
        try:
            for observation in case["observations"]:
                observation_artifact = f"{case_id}/observation/{observation['observation_id']}"
                try:
                    post = _artifact(store, _artifact_id(case_id, observation["arm"], "POST"))
                    full = files.load_rgb8(post)
                    source_sha_observation = post.get("sha256_observation")
                    if not isinstance(source_sha_observation, dict):
                        source_sha_observation = {
                            "expected": post.get("sha256"), "actual": None,
                            "status": "OBSERVATION_UNAVAILABLE_NOT_RECORDED",
                            "blocking": False,
                        }
                    actual_source_sha = source_sha_observation.get("actual")
                    source_identity_token = _content_identity_token(post, actual_source_sha)
                    construction_map = _expand_edit_map(
                        config["edit_maps"][observation["map_id"]],
                        f"edit_maps.{observation['map_id']}",
                    )
                    received = runtime.Inputs.construct(full, construction_map)
                    map_sha = hashlib.sha256(json.dumps(construction_map, separators=(",", ":")).encode()).hexdigest()
                    virtual_identity = hashlib.sha256(
                        json.dumps([source_identity_token, map_sha], separators=(",", ":")).encode()
                    ).hexdigest()
                    receipt = {
                        "storage": "VIRTUAL_DETERMINISTIC_INDEX_VIEW",
                        "source_artifact_id": post["artifact_id"],
                        "source_sha256": actual_source_sha,
                        "source_identity_token": source_identity_token,
                        "source_declared_sha256": post.get("sha256"),
                        "source_sha256_observation": source_sha_observation,
                        "map_id": observation["map_id"],
                        "map_sha256": map_sha,
                        "virtual_identity_sha256": virtual_identity,
                        "shape": [len(construction_map), 320, 512, 3],
                        "dtype": "uint8",
                    }
                    store.artifact(observation_artifact, status="AVAILABLE", **receipt)
                    latent = model.encode(received)
                    observation_row = {
                        "status": "AVAILABLE",
                        "protocol": observation["protocol"],
                        "frames": len(received),
                        "artifact": receipt,
                        "source_arm": observation["arm"],
                        "construction_map_sha256": map_sha,
                        "construction_map_not_supplied_to_blind_estimator": True,
                        "key_estimates": {},
                    }
                    for key_label, key in config["keys"].items():
                        raw = model.score(latent, key, observation["protocol"])
                        estimates = method.estimates(raw, observation["protocol"], len(received), key)
                        observation_row["key_estimates"][key_label] = estimates
                        for mode in observation["modes"]:
                            slot_id = f"{case_id}/{observation['observation_id']}/{key_label}/{mode}"
                            logical = {
                                "frames": len(received),
                                "protocol": observation["protocol"],
                                "mode": _mode_internal(observation["protocol"], mode),
                            }
                            operation = method.blind_operation(logical, estimates)
                            plans["slots"][slot_id] = {
                                "status": "PLANNED" if operation is not None else "UNRESOLVED",
                                "operation": operation,
                                "truth_inputs": False,
                            }
                    plans["observations"][observation["observation_id"]] = observation_row
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    artifact = _artifact(store, observation_artifact)
                    if artifact["status"] == "PLANNED":
                        store.artifact(observation_artifact, status="FAILED", reason=reason)
                    plans["observations"][observation["observation_id"]] = {
                        "status": "FAILED", "reason": reason, "truth_inputs": False,
                    }
                    for key_label in config["keys"]:
                        for mode in observation["modes"]:
                            slot_id = f"{case_id}/{observation['observation_id']}/{key_label}/{mode}"
                            plans["slots"][slot_id] = {
                                "status": "FAILED", "reason": reason, "truth_inputs": False,
                            }
                persist()
        finally:
            model.close()
        persist()

    return _run_timed(store, "receiver-sync", case_id, execute)


def phase_receiver_read(store, config, case_id):
    def execute():
        from runtime.wan import video_trajectory_conditional_joint_v1 as runtime

        case = _case(config, case_id)
        files = ArtifactFiles(store, case_id)
        plan_receipt = store.data["records"][case_id]["blind_plan"]
        _observe_file_identity(plan_receipt)
        plan = json.loads(Path(plan_receipt["path"]).read_text(encoding="utf-8"))
        model = runtime.WanBackend(_runtime_generation_config(config, case))
        reads = {"truth_inputs": False, "slots": {}, "physical_encodes": {}}
        cache = {}

        def persist():
            receipt = _json_dump(files.root / "receiver" / "blind_reads.json", reads)
            store.data["records"].setdefault(case_id, {})["blind_reads"] = receipt
            store.save()

        persist()
        try:
            for slot in [row for row in store.data["receiver_slots"] if row["case_id"] == case_id]:
                slot_id = slot["slot_id"]
                try:
                    planned = plan["slots"].get(slot_id, {"status": "UNRESOLVED", "operation": None})
                    if planned["status"] != "PLANNED":
                        reads["slots"][slot_id] = {
                            "status": planned.get("status", "UNRESOLVED"),
                            "reason": planned.get("reason", "blind operation unavailable"),
                            "truth_inputs": False,
                        }
                        persist()
                        continue
                    observation = plan["observations"][slot["observation_id"]]
                    declaration = next(
                        item for item in case["observations"]
                        if item["observation_id"] == slot["observation_id"]
                    )
                    source = _artifact(store, _artifact_id(case_id, declaration["arm"], "POST"))
                    full = files.load_rgb8(source)
                    source_sha_observation = source.get("sha256_observation")
                    if not isinstance(source_sha_observation, dict):
                        source_sha_observation = {
                            "expected": source.get("sha256"), "actual": None,
                            "status": "OBSERVATION_UNAVAILABLE_NOT_RECORDED",
                            "blocking": False,
                        }
                    actual_source_sha = source_sha_observation.get("actual")
                    source_identity_token = _content_identity_token(source, actual_source_sha)
                    construction_map = _expand_edit_map(
                        config["edit_maps"][declaration["map_id"]],
                        f"edit_maps.{declaration['map_id']}",
                    )
                    received = runtime.Inputs.construct(full, construction_map)
                    map_sha = hashlib.sha256(
                        json.dumps(construction_map, separators=(",", ":")).encode()
                    ).hexdigest()
                    actual_virtual_identity = hashlib.sha256(
                        json.dumps(
                            [source_identity_token, map_sha],
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest()
                    declared_virtual_identity = observation["artifact"].get("virtual_identity_sha256")
                    observation["artifact"]["virtual_identity_observation"] = {
                        "expected": declared_virtual_identity,
                        "actual": actual_virtual_identity,
                        "status": (
                            "MATCH" if declared_virtual_identity == actual_virtual_identity
                            else "UNDECLARED" if not declared_virtual_identity
                            else "RECORDED_DIFFERENCE"
                        ),
                        "blocking": False,
                    }
                    mapping = planned["operation"]["received_index_map"]
                    token = hashlib.sha256(json.dumps(
                        [actual_virtual_identity, mapping], separators=(",", ":"),
                    ).encode()).hexdigest()
                    if token not in cache:
                        corrected = model.operate(received, mapping)
                        cache[token] = model.cache(model.encode(corrected))
                        reads["physical_encodes"][token] = {
                            "observation_id": slot["observation_id"],
                            "received_virtual_identity_sha256": actual_virtual_identity,
                            "declared_virtual_identity_sha256": declared_virtual_identity,
                            "map_sha256": hashlib.sha256(json.dumps(mapping, separators=(",", ":")).encode()).hexdigest(),
                        }
                    normalized = model.restore(cache[token])
                    detail = model.read(normalized, config["keys"][slot["key_label"]], slot["frames"])
                    reads["slots"][slot_id] = {
                        "status": "READ",
                        "decoded_bits": detail["original_readout"]["decoded_bits"],
                        "detail": detail,
                        "truth_inputs": False,
                        "physical_encode": token,
                    }
                except Exception as exc:
                    reads["slots"][slot_id] = {
                        "status": "FAILED", "reason": f"{type(exc).__name__}: {exc}",
                        "truth_inputs": False,
                    }
                persist()
        finally:
            cache.clear()
            model.close()
        persist()

    return _run_timed(store, "receiver-read", case_id, execute)


def _load_json_receipt(receipt):
    _observe_file_identity(receipt)
    value = read_json(receipt["path"])
    if not isinstance(value, dict):
        raise ValueError("saved record root must be an object")
    return value


def _effective32_decision(soft, expected, *, rule, zero_decodes_one, strict_zero_tie):
    values = [float(value) for value in soft]
    if len(values) != 32 or any(not math.isfinite(value) for value in values):
        return {"status": "FAILED", "reason": "effective soft output must be 32 finite values"}
    native_bits = [int(value >= 0.0) if zero_decodes_one else int(value > 0.0) for value in values]
    ties = sum(value == 0.0 for value in values)
    base = {
        "rule": rule,
        "native_decoded_bits": native_bits,
        "native_bit_errors": sum(a != b for a, b in zip(native_bits, expected)),
        "tie_count": ties,
        "tie_policy": "UNEVALUABLE" if strict_zero_tie else "NATIVE_BIT_RETAINED",
        "effective_soft": values,
    }
    if ties and strict_zero_tie:
        return {
            **base,
            "status": "UNEVALUABLE_ZERO_TIE",
            "decoded_bits": None,
            "bit_errors": None,
            "exact_recovery": False,
            "reason": "at least one reduced effective bit is exactly zero",
        }
    return {
        **base,
        "status": "EVALUATED",
        "decoded_bits": native_bits,
        "bit_errors": base["native_bit_errors"],
        "exact_recovery": native_bits == expected,
    }


def _videoseal_32_result(record, expected, native_length, expected_frames, rule):
    if native_length % 32:
        return {"status": "FAILED", "reason": "native K is not divisible by 32"}
    sidecar = record.get("lossless_native_output")
    if record.get("storage") != "LOSSLESS_SIDECAR" or not isinstance(sidecar, dict):
        return {"status": "FAILED", "reason": "full VideoSeal native output sidecar unavailable"}
    import numpy as np

    path = Path(sidecar["uri"])
    sidecar_receipt = {"path": str(path), "sha256": sidecar.get("sha256")}
    try:
        _observe_file_identity(sidecar_receipt)
    except Exception as exc:
        return {"status": "FAILED", "reason": f"VideoSeal sidecar unavailable: {type(exc).__name__}: {exc}"}
    sidecar["sha256_observation"] = sidecar_receipt["sha256_observation"]
    with np.load(path, allow_pickle=False) as values:
        candidates = [values[key] for key in values.files if key.endswith("preds")]
        if len(candidates) != 1:
            return {"status": "FAILED", "reason": "VideoSeal preds array is not uniquely identified"}
        preds = candidates[0]
    if (
        preds.ndim < 2
        or int(preds.shape[0]) != expected_frames
        or int(preds.shape[1]) != native_length + 1
        or not np.isfinite(preds).all()
    ):
        return {"status": "FAILED", "reason": "VideoSeal preds shape/nonfinite mismatch"}
    native_soft = preds[:, 1:]
    reduce_axes = tuple(index for index in range(native_soft.ndim) if index != 1)
    channel_soft = native_soft.mean(axis=reduce_axes, dtype=np.float64)
    task_soft = np.asarray([
        channel_soft[list(range(bit, native_length, 32))].mean(dtype=np.float64)
        for bit in range(32)
    ])
    return {
        **_effective32_decision(
            task_soft.tolist(), expected,
            rule=rule,
            zero_decodes_one=False,
            strict_zero_tie=rule == VIDEOSEAL_STRICT_RULE,
        ),
        "effective_bits": 32,
        "native_channels": native_length,
        "expected_frames": expected_frames,
        "rate": f"32/{native_length}",
        "sidecar_sha256_observation": sidecar["sha256_observation"],
    }


def _rivagan_sequence_result(record, expected, expected_frames, rule=RIVAGAN_STRICT_RULE):
    frames = record.get("frame_soft_outputs")
    if not isinstance(frames, list) or len(frames) != expected_frames:
        return {"status": "FAILED", "reason": "missing or extra RivaGAN decoded frame"}
    if any(not isinstance(frame, list) or len(frame) != 32 for frame in frames):
        return {"status": "FAILED", "reason": "RivaGAN frame shape is not [32]"}
    if any(not math.isfinite(float(value)) for frame in frames for value in frame):
        return {"status": "FAILED", "reason": "RivaGAN frame logits contain nonfinite values"}
    means = [sum(float(frame[bit]) for frame in frames) / len(frames) for bit in range(32)]
    return {
        **_effective32_decision(
            means, expected,
            rule=rule,
            zero_decodes_one=True,
            strict_zero_tie=rule == RIVAGAN_STRICT_RULE,
        ),
        "expected_frames": expected_frames,
    }


def _validated_vote_ties(rows, decoded_bits, *, require_bit_rows):
    if not isinstance(rows, list) or len(rows) != 32:
        raise ValueError("vote evidence must contain exactly 32 rows")
    counts = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"vote evidence row {index} must be an object")
        ones, zeros = row.get("ones"), row.get("zeros")
        if any(type(value) is not int or value < 0 for value in (ones, zeros)):
            raise ValueError(f"vote evidence row {index} has invalid ones/zeros")
        if ones + zeros <= 0:
            raise ValueError(f"vote evidence row {index} has no votes")
        if "count" in row and (type(row["count"]) is not int or row["count"] != ones + zeros):
            raise ValueError(f"vote evidence row {index} has inconsistent count")
        if require_bit_rows:
            if type(row.get("decoded")) is not int or row["decoded"] not in (0, 1):
                raise ValueError(f"bit_rows row {index} has invalid decoded bit")
            if row["decoded"] != decoded_bits[index]:
                raise ValueError(f"bit_rows row {index} does not match saved decoded_bits")
            if type(row.get("tie")) is not bool or row["tie"] != (ones == zeros):
                raise ValueError(f"bit_rows row {index} has inconsistent tie marker")
        counts.append((ones, zeros))
    return counts


def _main_vote_tie_evidence(observed):
    base = {
        "main_vote_tie_count": None,
        "tie_count": None,
        "tie_policy": MAIN_VOTE_TIE_POLICY,
    }
    if not isinstance(observed, dict):
        return {
            **base,
            "tie_evidence_status": "UNAVAILABLE_INVALID",
            "tie_evidence_reason": "blind read slot is not an object",
        }
    decoded_bits = observed.get("decoded_bits")
    if (
        not isinstance(decoded_bits, list)
        or len(decoded_bits) != 32
        or any(type(bit) is not int or bit not in (0, 1) for bit in decoded_bits)
    ):
        return {
            **base,
            "tie_evidence_status": "UNAVAILABLE_MISSING",
            "tie_evidence_reason": "valid saved decoded_bits are unavailable for vote evidence",
        }
    detail = observed.get("detail")
    if not isinstance(detail, dict):
        return {
            **base,
            "tie_evidence_status": "UNAVAILABLE_MISSING",
            "tie_evidence_reason": "saved receiver detail is missing",
        }
    original = detail.get("original_readout")
    original_rows = original.get("votes") if isinstance(original, dict) else None
    bit_rows = detail.get("bit_rows")
    if original_rows is None and bit_rows is None:
        return {
            **base,
            "tie_evidence_status": "UNAVAILABLE_MISSING",
            "tie_evidence_reason": "saved receiver detail has no vote rows",
        }
    try:
        original_counts = (
            _validated_vote_ties(original_rows, decoded_bits, require_bit_rows=False)
            if original_rows is not None else None
        )
        bit_counts = (
            _validated_vote_ties(bit_rows, decoded_bits, require_bit_rows=True)
            if bit_rows is not None else None
        )
        if original_counts is not None and bit_counts is not None and original_counts != bit_counts:
            raise ValueError("original_readout.votes and bit_rows disagree")
        selected = original_counts if original_counts is not None else bit_counts
    except (TypeError, ValueError) as exc:
        return {
            **base,
            "tie_evidence_status": "UNAVAILABLE_INVALID",
            "tie_evidence_reason": str(exc),
        }
    count = sum(ones == zeros for ones, zeros in selected)
    return {
        **base,
        "main_vote_tie_count": count,
        "tie_count": count,
        "tie_evidence_status": (
            "AVAILABLE_ORIGINAL_READOUT_VOTES"
            if original_counts is not None else "AVAILABLE_VERIFIED_BIT_ROWS"
        ),
        "tie_evidence_reason": None,
    }


def _exact_success_difference_bounds(main_exact, baseline_exact):
    main = int(main_exact) if type(main_exact) is bool else None
    baseline = int(baseline_exact) if type(baseline_exact) is bool else None
    if main is not None and baseline is not None:
        return main - baseline, main - baseline
    if main is not None:
        return main - 1, main
    if baseline is not None:
        return -baseline, 1 - baseline
    return -1, 1


def _evaluate_comparison_rows(comparison_slots, receiver_rows, baseline_rows):
    main_by_id = {row["slot_id"]: row for row in receiver_rows}
    baseline_by_id = {row["slot_id"]: row for row in baseline_rows}
    rows = []
    for slot in comparison_slots:
        main = main_by_id.get(slot["main_slot_id"])
        baseline = baseline_by_id.get(slot["baseline_slot_id"])
        main_status = main.get("status") if isinstance(main, dict) else "MISSING_ROW"
        baseline_status = baseline.get("status") if isinstance(baseline, dict) else "MISSING_ROW"
        main_exact = (
            main.get("exact_recovery")
            if isinstance(main, dict) and main_status == "EVALUATED_TRUTH" else None
        )
        baseline_exact = (
            baseline.get("exact_recovery")
            if isinstance(baseline, dict) and baseline_status == "EVALUATED" else None
        )
        lower, upper = _exact_success_difference_bounds(main_exact, baseline_exact)
        baseline_tie_count = baseline.get("tie_count") if isinstance(baseline, dict) else None
        baseline_tie_available = type(baseline_tie_count) is int and baseline_tie_count >= 0
        row = {
            **slot,
            "main_status": main_status,
            "main_reason": main.get("reason") if isinstance(main, dict) else "main fixed row missing",
            "baseline_status": baseline_status,
            "baseline_reason": baseline.get("reason") if isinstance(baseline, dict) else "baseline fixed row missing",
            "main_exact_recovery": main_exact,
            "baseline_exact_recovery": baseline_exact,
            "main_bit_errors": main.get("bit_errors") if isinstance(main, dict) else None,
            "baseline_bit_errors": baseline.get("bit_errors") if isinstance(baseline, dict) else None,
            "main_vote_tie_count": main.get("main_vote_tie_count") if isinstance(main, dict) else None,
            "main_tie_policy": main.get("tie_policy", MAIN_VOTE_TIE_POLICY) if isinstance(main, dict) else MAIN_VOTE_TIE_POLICY,
            "main_tie_evidence_status": (
                main.get("tie_evidence_status", "UNAVAILABLE_NOT_REPORTED")
                if isinstance(main, dict) else "UNAVAILABLE_MISSING_ROW"
            ),
            "main_tie_evidence_reason": (
                main.get("tie_evidence_reason", "main tie evidence not reported")
                if isinstance(main, dict) else "main fixed row missing"
            ),
            "main_tie_semantics": "FINAL_BIT_COUNTER_VOTE_EQUALITY",
            "baseline_tie_count": baseline_tie_count if baseline_tie_available else None,
            "baseline_tie_policy": baseline.get("tie_policy") if isinstance(baseline, dict) else None,
            "baseline_tie_evidence_status": (
                "AVAILABLE_REDUCED_SOFT" if baseline_tie_available else "UNAVAILABLE"
            ),
            "baseline_tie_evidence_reason": (
                None if baseline_tie_available
                else baseline.get("reason", "baseline tie evidence unavailable") if isinstance(baseline, dict)
                else "baseline fixed row missing"
            ),
            "baseline_tie_semantics": BASELINE_TIE_SEMANTICS,
            "exact_success_difference_lower_bound": lower,
            "exact_success_difference_upper_bound": upper,
        }
        if main_status == "EVALUATED_TRUTH" and baseline_status == "EVALUATED":
            main_exact = int(main["exact_recovery"] is True)
            baseline_exact = int(baseline["exact_recovery"] is True)
            row.update(
                status="EVALUATED_PAIR",
                exact_success_difference_main_minus_baseline=main_exact - baseline_exact,
                bit_error_difference_main_minus_baseline=main["bit_errors"] - baseline["bit_errors"],
            )
        else:
            row.update(
                status="UNEVALUABLE_PAIR",
                exact_success_difference_main_minus_baseline=None,
                bit_error_difference_main_minus_baseline=None,
                reason=f"main={main_status}; baseline={baseline_status}",
            )
        rows.append(row)
    return rows


def _comparison_source_summaries(comparison_rows):
    summaries = []
    keys = sorted({(row["cohort"], row["case_id"], row["baseline_method"]) for row in comparison_rows})
    for cohort, case_id, method in keys:
        rows = [
            row for row in comparison_rows
            if row["cohort"] == cohort and row["case_id"] == case_id and row["baseline_method"] == method
        ]
        nonfull = [row for row in rows if row["analysis_role"] != "FULL_GEOMETRY_CONTROL_EXCLUDED_FROM_SYNC_GAIN"]
        full = [row for row in rows if row["analysis_role"] == "FULL_GEOMETRY_CONTROL_EXCLUDED_FROM_SYNC_GAIN"]
        evaluable = [row for row in nonfull if row["status"] == "EVALUATED_PAIR"]
        unavailable = len(nonfull) - len(evaluable)
        observed_difference = sum(row["exact_success_difference_main_minus_baseline"] for row in evaluable)
        compatible_lower = sum(row["exact_success_difference_lower_bound"] for row in nonfull)
        compatible_upper = sum(row["exact_success_difference_upper_bound"] for row in nonfull)
        summaries.append({
            "source_summary_id": f"{case_id}/{method}",
            "cohort": cohort,
            "case_id": case_id,
            "baseline_method": method,
            "fixed_nonfull_view_denominator": len(nonfull),
            "evaluable_nonfull_pairs": len(evaluable),
            "unavailable_nonfull_pairs": unavailable,
            "main_failed_or_missing_nonfull": sum(row["main_status"] != "EVALUATED_TRUTH" for row in nonfull),
            "baseline_failed_or_missing_nonfull": sum(row["baseline_status"] != "EVALUATED" for row in nonfull),
            "main_unavailable_nonfull": sum(row["main_status"] != "EVALUATED_TRUTH" for row in nonfull),
            "baseline_unavailable_nonfull": sum(row["baseline_status"] != "EVALUATED" for row in nonfull),
            "main_exact_successes_fixed_nonfull": sum(row["main_exact_recovery"] is True for row in nonfull),
            "baseline_exact_successes_fixed_nonfull": sum(row["baseline_exact_recovery"] is True for row in nonfull),
            "main_observed_errors_nonfull": sum(
                row["main_status"] == "EVALUATED_TRUTH" and row["main_exact_recovery"] is False
                for row in nonfull
            ),
            "baseline_observed_errors_nonfull": sum(
                row["baseline_status"] == "EVALUATED" and row["baseline_exact_recovery"] is False
                for row in nonfull
            ),
            "observed_exact_success_difference_sum": observed_difference,
            "exact_success_difference_compatible_range": [compatible_lower, compatible_upper],
            "exact_success_difference_lower_bound": compatible_lower,
            "exact_success_difference_upper_bound": compatible_upper,
            "complete_fixed_nonfull_pairs": unavailable == 0,
            "full_control_planned": len(full),
            "full_control_evaluable": sum(row["status"] == "EVALUATED_PAIR" for row in full),
            "independence_unit": "SOURCE_WITH_EIGHT_CLUSTERED_NONFULL_VIEWS",
        })
    return summaries


def _comparison_cohort_summaries(source_summaries):
    output = {}
    keys = sorted({(row["cohort"], row["baseline_method"]) for row in source_summaries})
    for cohort, method in keys:
        rows = [row for row in source_summaries if row["cohort"] == cohort and row["baseline_method"] == method]
        observed = sum(row["observed_exact_success_difference_sum"] for row in rows)
        unavailable = sum(row["unavailable_nonfull_pairs"] for row in rows)
        compatible_lower = sum(row["exact_success_difference_lower_bound"] for row in rows)
        compatible_upper = sum(row["exact_success_difference_upper_bound"] for row in rows)
        output[f"{cohort}|{method}"] = {
            "cohort": cohort,
            "baseline_method": method,
            "fixed_source_denominator": len(rows),
            "complete_sources": sum(row["complete_fixed_nonfull_pairs"] for row in rows),
            "fixed_nonfull_row_denominator": sum(row["fixed_nonfull_view_denominator"] for row in rows),
            "evaluable_nonfull_pairs": sum(row["evaluable_nonfull_pairs"] for row in rows),
            "unavailable_nonfull_pairs": unavailable,
            "observed_exact_success_difference_sum": observed,
            "exact_success_difference_compatible_range": [compatible_lower, compatible_upper],
            "exact_success_difference_lower_bound": compatible_lower,
            "exact_success_difference_upper_bound": compatible_upper,
            "full_control_rows": sum(row["full_control_planned"] for row in rows),
            "evaluable_full_controls": sum(row["full_control_evaluable"] for row in rows),
            "independence_unit": "SOURCE; VIEW_ROWS_ARE_CLUSTERED",
        }
    return output


def phase_evaluate(store, config):
    store.phase_start("evaluate")
    started = time.perf_counter()
    try:
        expected = config["payload_bits"]
        receiver_rows = []
        baseline_rows = []
        receiver_evidence = {}
        for case in config["cases"]:
            case_id = case["case_id"]
            receipt = store.data["records"].get(case_id, {}).get("blind_reads")
            if not receipt:
                receiver_evidence[case_id] = {
                    "slots_error": "blind read receipt missing",
                    "physical_error": "blind read receipt missing",
                }
                continue
            try:
                saved = _load_json_receipt(receipt)
                evidence = {}
                if isinstance(saved.get("slots"), dict):
                    evidence["slots"] = saved["slots"]
                else:
                    evidence["slots_error"] = "blind read record slots must be an object"
                if isinstance(saved.get("physical_encodes"), dict):
                    evidence["physical_encodes"] = saved["physical_encodes"]
                else:
                    evidence["physical_error"] = "blind read record physical_encodes must be an object"
                receiver_evidence[case_id] = evidence
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                receiver_evidence[case_id] = {
                    "slots_error": reason, "physical_error": reason,
                }
        for slot in store.data["receiver_slots"]:
            row = {**slot}
            evidence = receiver_evidence[slot["case_id"]]
            if "slots_error" in evidence:
                row.update(
                    status="FAILED", reason=evidence["slots_error"],
                    **_main_vote_tie_evidence({}),
                )
                receiver_rows.append(row)
                continue
            observed = evidence["slots"].get(slot["slot_id"], {})
            if not isinstance(observed, dict):
                row.update(
                    status="FAILED", reason="blind read slot must be an object",
                    **_main_vote_tie_evidence(observed),
                )
                receiver_rows.append(row)
                continue
            row.update(_main_vote_tie_evidence(observed))
            if observed.get("status") == "READ":
                bits = observed.get("decoded_bits")
                if isinstance(bits, list) and len(bits) == 32 and all(type(bit) is int and bit in (0, 1) for bit in bits):
                    row.update(
                        status="EVALUATED_TRUTH", decoded_bits=bits,
                        bit_errors=sum(a != b for a, b in zip(bits, expected)),
                        exact_recovery=bits == expected,
                    )
                else:
                    row.update(status="FAILED", reason="blind read is not exactly 32 binary bits")
            else:
                row.update(status="FAILED", reason=observed.get("status", "blind read missing"))
            receiver_rows.append(row)
        for slot in store.data["baseline_slots"]:
            rule = config["evaluation_rules"][
                "videoseal_32" if slot["method"] == "videoseal" else "rivagan_sequence"
            ]
            base = {**slot, "rule_status": rule["status"]}
            if rule["status"] not in RULE_STATUS_ACTIVE:
                baseline_rows.append({**base, "status": "PENDING_RULE_NOT_ADOPTED"})
                continue
            receipt = slot.get("record")
            if not receipt:
                baseline_rows.append({**base, "status": "FAILED", "reason": slot.get("reason", "extract record missing")})
                continue
            try:
                record = _load_json_receipt(receipt)
                if slot["method"] == "videoseal":
                    result = _videoseal_32_result(
                        record, expected, config["models"]["videoseal"]["native_message_length"],
                        slot["frames"], rule["rule"],
                    )
                else:
                    result = _rivagan_sequence_result(record, expected, slot["frames"], rule["rule"])
            except Exception as exc:
                result = {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}
            baseline_rows.append({**base, **result})
        if "comparison_slots" in store.data:
            comparison_slots = store.data["comparison_slots"]
        elif _comparison_enabled(config):
            comparison_slots = _build_comparison_slots(config)
        else:
            comparison_slots = []
        comparison_rows = _evaluate_comparison_rows(comparison_slots, receiver_rows, baseline_rows)
        comparison_source_summaries = _comparison_source_summaries(comparison_rows)
        comparison_cohort_summaries = _comparison_cohort_summaries(comparison_source_summaries)
        counts = Counter(row["status"] for row in receiver_rows)
        artifact_counts = Counter(row["status"] for row in store.data["artifacts"])
        baseline_issues = any(row["status"] != "EVALUATED" for row in baseline_rows)
        receiver_issues = any(row["status"] != "EVALUATED_TRUTH" for row in receiver_rows)
        quality_issues = any(row["status"] != "EVALUATED" for row in store.data["quality_rows"])
        artifact_issues = any(row["status"] != "AVAILABLE" for row in store.data["artifacts"])
        cohort_summaries = {}
        for cohort in sorted({case["cohort"] for case in config["cases"]}):
            cohort_cases = [case["case_id"] for case in config["cases"] if case["cohort"] == cohort]
            cohort_receiver = [row for row in receiver_rows if row["cohort"] == cohort]
            cohort_baseline = [row for row in baseline_rows if row["cohort"] == cohort]
            cohort_quality = [row for row in store.data["quality_rows"] if row["cohort"] == cohort]
            receiver_groups = {}
            for row in cohort_receiver:
                key = f"{row['arm']}|{row['key_label']}|{row['mode']}|{row['analysis_role']}"
                group = receiver_groups.setdefault(key, {"planned": 0, "evaluable": 0, "exact_recoveries": 0})
                group["planned"] += 1
                if row["status"] == "EVALUATED_TRUTH":
                    group["evaluable"] += 1
                    group["exact_recoveries"] += int(row.get("exact_recovery") is True)
            resources = _receiver_resource_plan(config, cohort_cases)
            cohort_summaries[cohort] = {
                "cases": len(cohort_cases),
                "case_ids": cohort_cases,
                "receiver_slots": len(cohort_receiver),
                "receiver_state_counts": dict(sorted(Counter(row["status"] for row in cohort_receiver).items())),
                "receiver_groups_by_arm_key_mode_role": dict(sorted(receiver_groups.items())),
                "baseline_slots": len(cohort_baseline),
                "baseline_state_counts": dict(sorted(Counter(row["status"] for row in cohort_baseline).items())),
                "quality_rows": len(cohort_quality),
                "quality_state_counts": dict(sorted(Counter(row["status"] for row in cohort_quality).items())),
                "receiver_resource_plan": resources,
            }
        actual_physical_encodes = {}
        for case in config["cases"]:
            evidence = receiver_evidence[case["case_id"]]
            if "physical_error" in evidence:
                actual_physical_encodes[case["case_id"]] = {
                    "status": "FAILED", "reason": evidence["physical_error"], "count": None,
                }
            else:
                actual_physical_encodes[case["case_id"]] = {
                    "status": "COUNTED",
                    "count": len(evidence["physical_encodes"]),
                }
        report = {
            "schema_version": SCHEMA_VERSION,
            "study_id": config["study_id"],
            "status": "COMPLETE_WITH_RETAINED_ISSUES" if receiver_issues or baseline_issues or quality_issues or artifact_issues else "COMPLETE",
            "claim_ceiling": config["method"]["claim_ceiling"],
            "fixed_denominator": {
                "cases": len(config["cases"]),
                "planned_artifacts": len(store.data["artifacts"]),
                "planned_cost_rows": len(store.data["costs"]),
                "planned_receiver_slots": len(store.data["receiver_slots"]),
                "planned_receiver_bits": 32 * len(store.data["receiver_slots"]),
                "planned_baseline_rows": len(store.data["baseline_slots"]),
                "planned_comparison_rows": len(comparison_slots),
                "planned_quality_rows": len(store.data["quality_rows"]),
            },
            "artifact_state_counts": dict(sorted(artifact_counts.items())),
            "receiver_state_counts": dict(sorted(counts.items())),
            "receiver_rows": receiver_rows,
            "baseline_rows": baseline_rows,
            "comparison_rows": comparison_rows,
            "comparison_source_summaries": comparison_source_summaries,
            "comparison_cohort_summaries": comparison_cohort_summaries,
            "quality_rows": store.data["quality_rows"],
            "cohort_summaries": cohort_summaries,
            "receiver_resource_counts": {
                "planned_from_manifest": _receiver_resource_plan(config),
                "actual_unique_physical_encodes_by_case": actual_physical_encodes,
                "identity": "virtual source artifact plus blind received-index map; counts are neither source media nor independent samples",
            },
            "message_contract": store.data["message_contract"],
            "codec_roundtrip_count": sum(
                record.get("status") == "SUCCEEDED"
                for case_records in store.data["records"].values()
                for record in case_records.get("codec", {}).values()
            ),
            "cost_records": store.data["costs"],
            "claim_guard": "Saved execution and exact recovery are fixed-manifest evidence only; no threshold, FPR, independence, or population guarantee is inferred.",
        }
        with (store.output / "receiver_rows.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = (
                "slot_id", "case_id", "cohort", "observation_id", "arm", "protocol",
                "analysis_role", "frames", "key_label", "mode", "planned_bits", "status",
                "bit_errors", "exact_recovery", "main_vote_tie_count", "tie_count",
                "tie_policy", "tie_evidence_status", "tie_evidence_reason", "reason",
            )
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(receiver_rows)
        with (store.output / "baseline_rows.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = (
                "slot_id", "case_id", "cohort", "method", "view_id", "protocol", "map_id",
                "analysis_role", "frames", "rule_status", "rule", "status", "decoded_bits",
                "native_decoded_bits", "native_bit_errors", "bit_errors", "exact_recovery",
                "tie_count", "tie_policy", "reason",
            )
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(baseline_rows)
        with (store.output / "quality_rows.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ("quality_id", "case_id", "cohort", "left", "right", "phase", "status", "element_count", "mse", "rmse", "psnr_db", "psnr_status", "identical", "max_absolute_error", "reason")
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(store.data["quality_rows"])
        with (store.output / "comparison_rows.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = (
                "comparison_id", "case_id", "cohort", "baseline_method", "view_id", "map_id",
                "protocol", "analysis_role", "main_arm", "main_key_label", "main_mode",
                "main_status", "baseline_status", "status", "main_exact_recovery",
                "baseline_exact_recovery", "main_bit_errors", "baseline_bit_errors",
                "main_vote_tie_count", "main_tie_policy", "main_tie_evidence_status",
                "main_tie_evidence_reason", "main_tie_semantics", "baseline_tie_count",
                "baseline_tie_policy", "baseline_tie_evidence_status",
                "baseline_tie_evidence_reason", "baseline_tie_semantics",
                "exact_success_difference_main_minus_baseline",
                "bit_error_difference_main_minus_baseline", "exact_success_difference_lower_bound",
                "exact_success_difference_upper_bound", "reason", "main_reason", "baseline_reason",
            )
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(comparison_rows)
        with (store.output / "comparison_source_summaries.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = (
                "source_summary_id", "cohort", "case_id", "baseline_method",
                "fixed_nonfull_view_denominator", "evaluable_nonfull_pairs",
                "unavailable_nonfull_pairs", "main_failed_or_missing_nonfull",
                "baseline_failed_or_missing_nonfull", "main_unavailable_nonfull",
                "baseline_unavailable_nonfull", "main_exact_successes_fixed_nonfull",
                "baseline_exact_successes_fixed_nonfull", "main_observed_errors_nonfull",
                "baseline_observed_errors_nonfull", "observed_exact_success_difference_sum",
                "exact_success_difference_lower_bound", "exact_success_difference_upper_bound",
                "complete_fixed_nonfull_pairs",
                "full_control_planned", "full_control_evaluable", "independence_unit",
            )
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(comparison_source_summaries)
        elapsed = time.perf_counter() - started
        store.data["phases"]["evaluate"].update(status="COMPLETE", finished_at_unix=time.time(), seconds=elapsed)
        report["phase_records"] = copy.deepcopy(store.data["phases"])
        report_path = _json_dump(store.output / "evaluation_report.json", report)
        store.data["records"]["evaluation_report"] = report_path
        store.data["status"] = report["status"]
        store.save()
        return report
    except BaseException as exc:
        store.phase_failure("evaluate", exc)
        raise


def run_phase(store, config, phase, *, case_id=None):
    if phase == "generate":
        return phase_generate(store, config, case_id)
    if phase == "decode":
        return phase_decode(store, config, case_id)
    if phase == "framewise":
        return phase_framewise(store, config, case_id)
    if phase.startswith("baseline-embed-"):
        return phase_baseline_embed(store, config, case_id, phase.removeprefix("baseline-embed-"))
    if phase == "codec":
        return phase_codec(store, config, case_id)
    if phase == "quality":
        return phase_quality(store, config, case_id)
    if phase.startswith("baseline-extract-"):
        return phase_baseline_extract(store, config, case_id, phase.removeprefix("baseline-extract-"))
    if phase == "receiver-sync":
        return phase_receiver_sync(store, config, case_id)
    if phase == "receiver-read":
        return phase_receiver_read(store, config, case_id)
    if phase == "evaluate":
        return phase_evaluate(store, config)
    raise RealEvalConfigError(f"unknown phase {phase}")
