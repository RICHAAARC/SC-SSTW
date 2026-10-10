"""Independent explicit-config entry for the local joint real mechanism run.

The default execution path is real. ``--preflight-only`` validates config and
the fixed record denominator without loading any model.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from runtime.wan import local_joint_state_payload_experiment_v1 as runtime


ROOT = Path(__file__).resolve().parents[2]
RAW_KEY_LABELS = ("CORRECT", "WRONG")
SOURCE_CLOSURE = (
    "experiments/__init__.py",
    "experiments/wan_state_clock/__init__.py",
    "experiments/wan_state_clock/local_joint_state_payload_v1_run.py",
    "main/__init__.py",
    "main/tube_state/__init__.py",
    "main/tube_state/local_joint_state_payload_carrier_v1.py",
    "main/tube_state/local_joint_state_payload_v1.py",
    "main/tube_state/grow_video_reference.py",
    "runtime/__init__.py",
    "runtime/wan/__init__.py",
    "runtime/wan/generation.py",
    "runtime/wan/fixed_rgb_media.py",
    "runtime/wan/grow_video_reference.py",
    "runtime/wan/io.py",
    "runtime/wan/local_joint_state_payload_experiment_v1.py",
    "runtime/wan/local_joint_state_payload_provider_v1.py",
    "runtime/wan/local_joint_state_payload_v1.py",
    "runtime/wan/trajectory.py",
    "runtime/wan/vae.py",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not __import__("math").isfinite(value):
        return None
    return value


def source_identity(root: Path) -> dict[str, Any]:
    return dict(workspace=str(Path(root).resolve()))


class Store:
    def __init__(self, output: Path, config: dict[str, Any], config_path: Path, execution_kind: str):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        identity = source_identity(ROOT)
        self.data = dict(
            schema=runtime.SCHEMA,
            status="RUNNING",
            stage="INITIALIZED",
            source_identity=identity,
            config_path=str(config_path),
            config=config,
            calls={},
            lifecycle=[],
            failures=[],
            execution=dict(kind=execution_kind, attempted=False, completed=False,
                           scientific_interpretation=False, blind_recovery=False, fpr_evidence=False),
            arms={arm: dict(
                status="PENDING",
                steps=[dict(index=i, status="PENDING") for i in range(50)],
                layers={name: dict(status="PENDING") for name in runtime.LAYERS},
                observations={f"{layer}/{label}": dict(status="PENDING", layer=layer, key_label=label)
                              for layer in ("float_rgb", "rgb8", "mp4") for label in RAW_KEY_LABELS},
            ) for arm in config["arms"]},
            evidence_ceiling=("preflight is unexecuted; an ordinary explicit-config invocation records attempted execution, "
                              "not scientific PASS, blind recovery, FPR, or generalization"),
        )
        self.save()

    def save(self) -> None:
        path = self.output / "result.json"
        temp = path.with_suffix(".json.tmp")
        encoded = json.dumps(_jsonable(self.data), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
        with temp.open("wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)

    def event(self, name: str, row: dict[str, Any]) -> None:
        self.data["lifecycle"].append(dict(event=name, **row))
        self.save()

    def call(self, name: str, function: Any) -> Any:
        row = self.data["calls"].setdefault(name, dict(attempted=0, completed=0))
        row["attempted"] += 1
        self.save()
        try:
            value = function()
        except Exception as exc:
            self.best_effort_failure(name, exc)
            raise
        row["completed"] += 1
        self.save()
        return value

    def failure(self, stage: str, exc: BaseException, **extra: Any) -> None:
        self.data["failures"].append(dict(stage=stage, reason=f"{type(exc).__name__}: {exc}", retry=False, **extra))
        self.save()

    def best_effort_failure(self, stage: str, exc: BaseException, **extra: Any) -> None:
        """One finite persistence attempt; never masks an existing primary error."""
        self.data["failures"].append(dict(stage=stage, reason=f"{type(exc).__name__}: {exc}", retry=False, **extra))
        try:
            self.save()
        except Exception as record_exc:
            self.data.setdefault("unpersisted_record_failures", []).append(
                dict(stage=stage, reason=f"{type(record_exc).__name__}: {record_exc}"))

    def step(self, arm: str, row: dict[str, Any]) -> None:
        index = int(row["index"])
        self.data["arms"][arm]["steps"][index] = dict(status="COMPLETED", **row)
        self.save()

    def seal_incomplete(self) -> None:
        for arm in self.data["arms"].values():
            if arm["status"] == "PENDING":
                arm["status"] = "NOT_RUN"
            elif arm["status"] == "RUNNING":
                arm["status"] = "FAILED"
            for row in arm["steps"]:
                if row["status"] == "PENDING":
                    row["status"] = "NOT_COMPLETED"
            for rows in (arm["layers"], arm["observations"]):
                for row in rows.values():
                    if row["status"] == "PENDING":
                        row["status"] = "MISSING_DEPENDENCY"
        self.save()


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    runtime.validate_config(value)
    return value


def _save_json(path: Path, value: Any) -> dict[str, Any]:

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    with temp.open("wb") as stream:
        stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)
    return dict(status="SAVED", path=str(path), rows=len(value) if isinstance(value, list) else None)


def run(config: dict[str, Any], output: Path, *, preflight_only: bool = False,
        residency_type: Any = runtime.WanSerialResidency, codec_type: Any = runtime.ExplicitFFmpeg,
        public_protocol: Any = None, raster_saver: Any = None,
        execution_kind: str = "explicit_config_real_entry") -> Store:
    config = dict(config)
    config_path = Path(config.pop("_config_path")) if "_config_path" in config else Path("<in-memory>")
    runtime.validate_config(config)
    public_protocol = runtime.carrier.PUBLIC if public_protocol is None else public_protocol
    store = Store(output, config, config_path, execution_kind)
    if preflight_only:
        store.data.update(status="PREFLIGHT_COMPLETE", stage="PREFLIGHT", actual_model_calls=False)
        store.seal_incomplete()
        return store
    residency = None
    primary: BaseException | None = None
    try:
        store.data["stage"] = "RESIDENCY_CONSTRUCTION"; store.save()
        try:
            residency = residency_type(config, store.event)
        except Exception as exc:
            store.data.update(failed_stage="RESIDENCY_CONSTRUCTION", actual_model_calls=False)
            store.best_effort_failure("residency_construction", exc)
            raise
        store.data["execution"]["attempted"] = True
        store.data["stage"] = "MODEL_LOAD"; store.save()
        store.call("generation_load", residency.load_generation)
        initial_identity = residency._identity()["initial"]
        for arm_index, arm_name in enumerate(config["arms"]):
            arm = store.data["arms"][arm_name]
            arm["status"] = "RUNNING"; arm["initial_fingerprint"] = initial_identity; store.save()
            try:
                terminal, receipt, provider = store.call(
                    f"{arm_name}/trajectory",
                    lambda arm_name=arm_name: runtime.run_arm(
                        residency, config, arm_name,
                        count=lambda kind, completed, arm_name=arm_name: _count(store, f"{arm_name}/{kind}", completed),
                        record_step=lambda row, arm_name=arm_name: store.step(arm_name, row),
                    ),
                )
                arm["trajectory"] = receipt
                arm["terminal_fingerprint"] = receipt["terminal_sha256"]
                if provider is not None:
                    arm["provider"] = dict(calls=dict(provider.calls), failures=list(provider.failures))
                try:
                    rgb, rgb8, layers = store.call(
                        f"{arm_name}/terminal_media",
                        lambda arm_name=arm_name, terminal=terminal: runtime.save_terminal_and_rgb_layers(
                            residency, terminal, output / arm_name.lower(),
                            event=lambda name, row, arm_name=arm_name: _layer(store, arm_name, name, row),
                            restore_transformer=arm_index < len(config["arms"]) - 1,
                            public_protocol=public_protocol, raster_saver=raster_saver,
                        ),
                    )
                except Exception as exc:
                    _mark_first_pending_layer_failed(store, arm_name, exc)
                    raise
                arm["layers"].update(layers); store.save()
                for layer_name, observed in (("float_rgb", rgb), ("rgb8", rgb8)):
                    _observe_layer(store, arm_name, layer_name, observed, config, output, public_protocol)
                codec = codec_type(config["media"])
                received, media = store.call(
                    f"{arm_name}/mp4_roundtrip",
                    lambda arm_name=arm_name: codec.roundtrip(
                        Path(arm["layers"]["rgb8"]["path"]), arm["layers"]["rgb8"]["sha256"],
                        output / arm_name.lower() / "video.mp4",
                        event=lambda row, arm_name=arm_name: _media_event(store, arm_name, row),
                    ),
                )
                arm["layers"]["mp4"] = media; store.save()
                _observe_layer(store, arm_name, "mp4", received, config, output, public_protocol)
                arm["status"] = ("COMPLETE_WITH_OBSERVATION_FAILURE"
                                 if any(row["status"] == "FAILED" for row in arm["observations"].values())
                                 else "COMPLETE")
                store.save()
            except Exception as exc:
                arm["status"] = "FAILED"; store.best_effort_failure(f"{arm_name}/arm", exc, arm=arm_name); raise
        store.data["execution"]["completed"] = True
        final_status = ("COMPLETE_WITH_OBSERVATION_FAILURE"
                        if any(row["status"] == "COMPLETE_WITH_OBSERVATION_FAILURE"
                               for row in store.data["arms"].values()) else "COMPLETE")
        store.data.update(status=final_status, stage="COMPLETE",
                          actual_model_calls=(execution_kind == "explicit_config_real_entry"))
        store.save()
        return store
    except BaseException as exc:
        primary = exc
        store.data.update(status="FAILED", stage="FAILED")
        store.best_effort_failure("process", exc)
        try:
            store.seal_incomplete()
        except Exception as record_exc:
            store.best_effort_failure("seal_incomplete", record_exc, primary_reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if residency is not None:
            try:
                residency.release()
            except Exception as cleanup_exc:
                store.best_effort_failure("residency_release", cleanup_exc,
                                          primary_reason=None if primary is None else f"{type(primary).__name__}: {primary}")
                if primary is None:
                    store.data.update(status="FAILED", stage="RELEASE_FAILED")
                    store.data["execution"]["completed"] = False
                    try:
                        store.save()
                    except Exception:
                        pass
                    raise


def _count(store: Store, name: str, completed: bool) -> None:
    row = store.data["calls"].setdefault(name, dict(attempted=0, completed=0))
    row["completed" if completed else "attempted"] += 1
    store.save()


def _layer(store: Store, arm: str, name: str, row: dict[str, Any]) -> None:
    store.data["arms"][arm]["layers"][name] = dict(row)
    store.save()


def _media_event(store: Store, arm: str, row: dict[str, Any]) -> None:
    current = store.data["arms"][arm]["layers"]["mp4"]
    if "stages" not in current:
        current = {"status": "RUNNING", "stages": {}}
    stage = row["stage"]
    current["stages"][stage] = dict(row)
    if stage == "encode" and row["status"] == "COMPLETED":
        current["artifact"] = dict(row["artifact"])
    if row["status"] == "FAILED":
        current["status"] = "FAILED"
    elif stage == "read" and row["status"] == "COMPLETED":
        current["status"] = "SAVED"
    store.data["arms"][arm]["layers"]["mp4"] = current
    store.save()


def _mark_first_pending_layer_failed(store: Store, arm: str, exc: Exception) -> None:
    pending = [name for name in runtime.LAYERS if store.data["arms"][arm]["layers"][name]["status"] == "PENDING"]
    if pending:
        store.data["arms"][arm]["layers"][pending[0]] = dict(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
        for name in pending[1:]:
            store.data["arms"][arm]["layers"][name] = dict(status="MISSING_DEPENDENCY", dependency=pending[0])
        try:
            store.save()
        except Exception as record_exc:
            store.data.setdefault("unpersisted_record_failures", []).append(
                dict(stage=f"{arm}/layer_failure", reason=f"{type(record_exc).__name__}: {record_exc}"))


def _observe_layer(store: Store, arm: str, layer: str, rgb: Any, config: dict[str, Any], output: Path,
                   public_protocol: Any) -> None:
    target = store.data["arms"][arm]["observations"]
    keys = dict(CORRECT=config["carrier"]["key"], WRONG=config["carrier"]["wrong_key"])
    for label in RAW_KEY_LABELS:
        entry = f"{layer}/{label}"
        target[entry] = dict(status="ATTEMPTED", layer=layer, key_label=label, truth_used=False); store.save()
        try:
            rows = store.call(f"{arm}/{layer}/{label}_observe", lambda label=label: runtime.raw_observations(
                rgb, keys[label], public_protocol=public_protocol))
        except Exception as exc:
            # Store.call already recorded the extraction failure once.
            target[entry] = dict(status="FAILED", layer=layer, key_label=label, key=keys[label],
                                 truth_used=False, reason=f"{type(exc).__name__}: {exc}")
            try:
                store.save()
            except Exception as record_exc:
                store.data.setdefault("unpersisted_record_failures", []).append(
                    dict(stage=f"{arm}/{layer}/{label}_observe_receipt",
                         reason=f"{type(record_exc).__name__}: {record_exc}"))
            continue
        try:
            receipt = _save_json(
                output / arm.lower() / f"{layer}_{label.lower()}_raw_observations.json", rows)
            target[entry] = dict(**receipt, layer=layer, key_label=label, key=keys[label],
                                 truth_used=False, fixed_directory_rows=len(rows))
            store.save()
        except Exception as exc:
            target[entry] = dict(status="FAILED", layer=layer, key_label=label, key=keys[label],
                                 truth_used=False, reason=f"{type(exc).__name__}: {exc}")
            store.best_effort_failure(f"{arm}/{layer}/{label}_save", exc, arm=arm,
                                      layer=layer, key_label=label)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config["_config_path"] = str(args.config.resolve())
    try:
        run(config, args.output, preflight_only=args.preflight_only)
    except BaseException as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
