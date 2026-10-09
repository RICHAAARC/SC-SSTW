"""CPU/static checks for native adapters and explicit artifact workflow."""
from __future__ import annotations

import copy
import json
import math
import os
import shutil
import subprocess
import sys
import types
from itertools import product
from pathlib import Path

import pytest

from experiments.paper_results_v1 import report
from experiments.paper_results_v1.fixture_backends import (
    build_fixture_native_adapters,
    build_fixture_operations,
)
from experiments.paper_results_v1.native_adapters import (
    CallableTensorBackend,
    NativeAdapterError,
    RivaGANLoadedTensorBackend,
    RivaGANNativeAdapter,
    RivaGANPathBackend,
    VideoSealLoadedBackend,
    VideoSealNativeAdapter,
)
from experiments.paper_results_v1.workflow import (
    WorkflowManifestError,
    _quality_record,
    run_workflow,
    validate_workflow_manifest,
)


pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments/paper_results_v1"


def workflow_manifest():
    return report.read_json(PACKAGE / "workflow_fixture.manifest.json")


def test_explicit_workflow_runs_callbacks_and_retains_fixed_denominators():
    trace = {}
    value = run_workflow(
        workflow_manifest(),
        operations=build_fixture_operations(trace),
        native_adapters=build_fixture_native_adapters(),
        base_dir=PACKAGE,
        main_report_link={"status": "LINKED", "report_id": "unit"},
    )
    assert value["report_status"] == "COMPLETE_WITH_RETAINED_ISSUES"
    assert value["manifest_denominator"] == {
        "cases": 2,
        "source_artifacts": 2,
        "plan_steps": 18,
        "native_jobs": 4,
        "quality_pairs": 10,
        "cost_rows": 30,
        "source_state_counts": {"AVAILABLE": 1, "MISSING": 1},
        "plan_state_counts": {"BLOCKED_DEPENDENCY": 9, "SUCCEEDED": 9},
        "native_state_counts": {"BLOCKED_DEPENDENCY": 2, "SUCCEEDED": 2},
        "quality_state_counts": {"BLOCKED_DEPENDENCY": 5, "OBSERVED": 5},
    }
    assert len(value["artifacts"]) == 28
    assert {row["state"] for row in value["quality_records"]} == {"OBSERVED", "BLOCKED_DEPENDENCY"}
    observed = [row for row in value["quality_records"] if row["state"] == "OBSERVED"]
    assert len(observed) == 5
    assert all(row["metric_semantics"] == "PAIRWISE_ABSOLUTE_NOT_ADDITIVE_DECOMPOSITION" for row in observed)
    assert all(row["pairing_verification_scope"].endswith("NOT_PHYSICAL_MEDIA_VERIFICATION") for row in observed)
    assert trace["framewise_encode_calls"] == 1
    assert trace["framewise_decode_calls"] == 2
    assert trace["p1_input"] == trace["shared_latent_original"]
    assert trace["m05_input_before_write"] == trace["shared_latent_original"]
    assert trace["m05_private_written_latent"] != trace["shared_latent_original"]
    assert trace["shared_latent_reference"] == trace["shared_latent_original"]

    videoseal = next(row for row in value["native_records"] if row["job_id"] == "fixture_ok/videoseal")
    vs_extract = videoseal["native"]["extract"]
    assert vs_extract["truth_input_supplied"] is False
    assert vs_extract["output_shapes"]["preds"] == [2, 5, 1, 1]
    assert vs_extract["output_element_counts"]["preds"] == 10
    assert len(vs_extract["raw_native_output"]["preds"]) == 2
    assert videoseal["native"]["main_32bit_mapping"] == "PENDING_NOT_APPLIED"

    rivagan = next(row for row in value["native_records"] if row["job_id"] == "fixture_ok/rivagan")
    riva_extract = rivagan["native"]["extract"]
    assert riva_extract["truth_input_supplied"] is False
    assert riva_extract["frame_shapes"] == [[32], [32]]
    assert len(riva_extract["frame_soft_outputs"][0]) == 32
    assert len(riva_extract["frame_native_zero_threshold_bits"][1]) == 32
    assert riva_extract["soft_value_count"] == 64
    assert rivagan["native"]["sequence_32bit_recovery"] == "PENDING_REDUCER_NOT_APPLIED"
    assert "bit_accuracy" not in json.dumps(rivagan["native"])


def test_missing_callbacks_and_failed_callback_never_create_success_rows():
    manifest = workflow_manifest()
    no_backends = run_workflow(manifest, base_dir=PACKAGE)
    assert no_backends["manifest_denominator"]["plan_steps"] == 18
    assert no_backends["manifest_denominator"]["plan_state_counts"].get("SUCCEEDED", 0) == 0
    assert len(no_backends["artifacts"]) == 28
    assert all(row["status"] != "SUCCEEDED" for row in no_backends["native_records"])

    no_codec_operations = build_fixture_operations()
    del no_codec_operations["fixture_codec_roundtrip"]
    no_codec = run_workflow(
        manifest,
        operations=no_codec_operations,
        native_adapters=build_fixture_native_adapters(),
        base_dir=PACKAGE,
    )
    native_without_codec = next(
        row for row in no_codec["native_records"] if row["job_id"] == "fixture_ok/videoseal"
    )
    assert native_without_codec["status"] == "SKIPPED_BACKEND_UNAVAILABLE"
    assert native_without_codec["native"]["status"] == "PARTIAL_WITH_RETAINED_FAILURE"
    assert native_without_codec["native"]["embed"]["submitted_msgs"] == {
        "bits": [0, 1, 1, 0], "shape": [1, 4], "element_count": 4,
    }
    assert "imgs_w" not in json.dumps(native_without_codec["native"]["embed"])
    pre = next(row for row in no_codec["artifacts"] if row["artifact_id"] == "fixture_ok/videoseal/pre")
    post = next(row for row in no_codec["artifacts"] if row["artifact_id"] == "fixture_ok/videoseal/post")
    assert pre["status"] == "AVAILABLE" and post["status"] == "SKIPPED"

    operations = build_fixture_operations()

    def fail_payload(value, context):
        del value, context
        raise RuntimeError("fixture transform failure")

    operations["fixture_payload_native"] = fail_payload
    failed = run_workflow(
        manifest,
        operations=operations,
        native_adapters=build_fixture_native_adapters(),
        base_dir=PACKAGE,
    )
    p0 = next(row for row in failed["plan"] if row["step_id"] == "fixture_ok/PAYLOAD_NATIVE/TRANSFORM")
    p0_codec = next(row for row in failed["plan"] if row["step_id"] == "fixture_ok/PAYLOAD_NATIVE/CODEC")
    p1 = next(row for row in failed["plan"] if row["step_id"] == "fixture_ok/PAYLOAD_FRAMEWISE_RECON/TRANSFORM")
    assert p0["state"] == "FAILED"
    assert p0_codec["state"] == p1["state"] == "BLOCKED_DEPENDENCY"
    assert len(failed["plan"]) == 18 and len(failed["quality_records"]) == 10

    malformed_source = workflow_manifest()
    malformed_source["cases"][0]["source"]["input"]["value"] = [[0.0], [0.0, 1.0]]
    retained = run_workflow(malformed_source, base_dir=PACKAGE)
    source_row = next(row for row in retained["artifacts"] if row["artifact_id"] == "fixture_ok/source")
    assert source_row["status"] == "FAILED" and "ragged" in source_row["reason"]
    assert len(retained["plan"]) == 18


def test_manifest_dependency_and_native_message_contracts_are_explicit():
    manifest = workflow_manifest()
    manifest["cases"][0]["stages"]["PAYLOAD_FRAMEWISE_RECON"]["input_artifact_id"] = "wrong"
    with pytest.raises(WorkflowManifestError, match="input_artifact_id"):
        validate_workflow_manifest(manifest)

    manifest = workflow_manifest()
    manifest["cases"][0]["stages"]["PAYLOAD_FRAMEWISE_M05"]["input_artifact_id"] = (
        "fixture_ok/p1/pre"
    )
    with pytest.raises(WorkflowManifestError, match="input_artifact_id"):
        validate_workflow_manifest(manifest)

    manifest = workflow_manifest()
    manifest["native_jobs"][1]["native_message_bits"] = [0, 1]
    with pytest.raises(WorkflowManifestError, match="exactly 32"):
        validate_workflow_manifest(manifest)

    manifest = workflow_manifest()
    manifest["native_jobs"][0]["input_artifact_id"] = "fixture_ok/p0/pre"
    with pytest.raises(WorkflowManifestError, match="OFF_NATIVE PRE"):
        validate_workflow_manifest(manifest)

    template = report.read_json(PACKAGE / "workflow.template.json")
    validate_workflow_manifest(template)
    assert template["cases"] == [] and template["native_jobs"] == []
    assert run_workflow(template)["report_status"] == "PENDING_EMPTY_ROSTER"


def test_videoseal_loaded_backend_uses_native_calls_without_truth_or_capacity_mapping():
    calls = []

    class Model:
        def embed(self, media, *, msgs, is_video, lowres_attenuation):
            calls.append(("embed", copy.deepcopy(msgs), is_video, lowres_attenuation))
            return {"imgs_w": media, "msgs": msgs}

        def detect(self, media, *, is_video):
            calls.append(("detect", media, is_video))
            return {"preds": [[[[0.1]], [[-0.2]], [[0.3]]]]}

    adapter = VideoSealNativeAdapter(
        VideoSealLoadedBackend(Model(), lambda bits: [bits]),
        native_message_length=2,
        backend_metadata={
            "source_version": None, "model_version": "unit", "weight_identity": None,
            "detect_output_layout": "T,1+K,H,W",
        },
    )
    _media, record = adapter.run([[0.2]], [0, 1])
    assert calls[0] == ("embed", [[0, 1]], True, False)
    assert calls[1][0] == "detect" and len(calls[1]) == 3
    assert record["extract"]["raw_native_output"]["preds"] == [[[[0.1]], [[-0.2]], [[0.3]]]]
    assert record["extract"]["truth_input_supplied"] is False
    assert record["main_32bit_mapping"] == "PENDING_NOT_APPLIED"
    assert record["embed"]["submitted_msgs"] == {
        "bits": [0, 1], "shape": [1, 2], "element_count": 2,
    }
    assert record["embed"]["embedded_media"]["storage"] == "ARTIFACT_VALUE_NOT_INLINED"
    assert "imgs_w" not in json.dumps(record["embed"])
    with pytest.raises(NativeAdapterError, match="exactly 2"):
        adapter.run([[0.2]], [0])
    with pytest.raises(NativeAdapterError, match="backend_metadata missing"):
        VideoSealNativeAdapter(
            VideoSealLoadedBackend(Model(), lambda bits: [bits]),
            native_message_length=2,
            backend_metadata={"model_version": "unit"},
        )


@pytest.mark.parametrize(
    "builder",
    [
        lambda bits: [bits + bits],
        lambda bits: [bits[:-1]],
        lambda bits: [[1 - bit for bit in bits]],
    ],
)
def test_videoseal_message_builder_cannot_repeat_truncate_or_change_bits(builder):
    calls = []

    class Model:
        def embed(self, *args, **kwargs):
            calls.append((args, kwargs))
            return {"imgs_w": args[0]}

    adapter = VideoSealNativeAdapter(
        VideoSealLoadedBackend(Model(), builder),
        native_message_length=2,
        backend_metadata={
            "source_version": "unit", "model_version": "unit", "weight_identity": "unit",
            "detect_output_layout": "T,1+K,H,W",
        },
    )
    with pytest.raises(NativeAdapterError, match="preserve declared bits exactly"):
        adapter.embed([[0.2]], [0, 1])
    assert calls == []


def test_videoseal_large_detect_output_requires_explicit_lossless_sidecar():
    class Backend:
        def extract(self, media):
            del media
            return {"preds": [[[[0.1]], [[-0.1]], [[0.2]]]]}

    metadata = {
        "source_version": "unit", "model_version": "unit", "weight_identity": "unit",
        "detect_output_layout": "T,1+K,H,W",
    }
    without_store = VideoSealNativeAdapter(
        Backend(), native_message_length=2, backend_metadata=metadata, inline_element_limit=1,
    )
    with pytest.raises(NativeAdapterError, match="lossless native_output_store"):
        without_store.extract([[0.2]])

    received = []

    def store(label, value, descriptor):
        received.append((label, value, descriptor))
        return {"lossless": True, "uri": "artifact://detect-full.npz", "format": "npz"}

    with_store = VideoSealNativeAdapter(
        Backend(), native_message_length=2, backend_metadata=metadata,
        inline_element_limit=1, native_output_store=store,
    )
    record = with_store.extract([[0.2]])
    assert record["storage"] == "LOSSLESS_SIDECAR"
    assert record["total_element_count"] == 3
    assert record["lossless_native_output"]["uri"] == "artifact://detect-full.npz"
    assert received[0][0] == "videoseal_detect_output"


def test_rivagan_path_backend_discloses_hidden_native_codec():
    calls = []

    class Model:
        def encode(self, source, bits, target):
            calls.append(("encode", source, bits, target))

        def decode(self, source):
            calls.append(("decode", source))
            return [[-0.1, 0.1] * 16]

    backend = RivaGANPathBackend(Model())
    output = backend.embed("source.mp4", [0, 1] * 16, output_uri="native.mp4")
    decoded = list(backend.extract(output))
    assert calls[0][0] == "encode" and calls[1] == ("decode", "native.mp4")
    assert len(decoded[0]) == 32
    assert backend.transport_metadata["native_path_codec"] == "opencv_mp4v"
    assert backend.transport_metadata["native_path_fps"] == 20
    assert backend.transport_metadata["codec_comparability"] == "NATIVE_PATH_CODEC_NOT_SHARED_PLANNED_CODEC"


def test_rivagan_rejects_non_32_frame_logits_without_sequence_reducer():
    backend = CallableTensorBackend(
        lambda media, bits, output_uri=None: media,
        lambda media: [[0.0] * 31],
        transport_metadata={
            "transport_mode": "FIXTURE", "color_layout": "BGR",
            "decoder_normalization": "NONE", "codec_comparability": "NO_CODEC",
        },
    )
    adapter = RivaGANNativeAdapter(
        backend,
        backend_metadata={
            "source_version": None, "model_version": "unit", "weight_identity": None,
        },
    )
    with pytest.raises(NativeAdapterError, match=r"strict shape \[32\]"):
        adapter.extract([[0.0]])


def test_rivagan_loaded_tensor_backend_matches_pinned_frame_wiring():
    def shape_of(value):
        if not isinstance(value, list):
            return ()
        return (len(value), *shape_of(value[0])) if value else (0,)

    def flatten(value):
        if isinstance(value, list):
            return [item for child in value for item in flatten(child)]
        return [float(value)]

    def nested(flat, shape):
        if not shape:
            return flat[0]
        stride = math.prod(shape[1:])
        return [nested(flat[index * stride:(index + 1) * stride], shape[1:]) for index in range(shape[0])]

    class FakeTensor:
        def __init__(self, value, shape=None):
            if shape is None:
                self._shape = shape_of(value)
                self.flat = flatten(value)
            else:
                self._shape = tuple(shape)
                self.flat = [float(item) for item in value]

        @property
        def shape(self):
            return self._shape

        def _map(self, function):
            return FakeTensor([function(item) for item in self.flat], self.shape)

        def __truediv__(self, other):
            return self._map(lambda item: item / other)

        def __sub__(self, other):
            return self._map(lambda item: item - other)

        def __add__(self, other):
            return self._map(lambda item: item + other)

        def __mul__(self, other):
            return self._map(lambda item: item * other)

        def permute(self, *axes):
            old_shape = self.shape
            new_shape = tuple(old_shape[axis] for axis in axes)
            old_strides = [math.prod(old_shape[index + 1:]) for index in range(len(old_shape))]
            values = []
            for new_index in product(*(range(size) for size in new_shape)):
                old_index = [0] * len(old_shape)
                for new_axis, old_axis in enumerate(axes):
                    old_index[old_axis] = new_index[new_axis]
                flat_index = sum(index * stride for index, stride in zip(old_index, old_strides))
                values.append(self.flat[flat_index])
            return FakeTensor(values, new_shape)

        def unsqueeze(self, axis):
            shape = list(self.shape)
            shape.insert(axis, 1)
            return FakeTensor(self.flat, shape)

        def clamp(self, low, high):
            return self._map(lambda item: min(high, max(low, item)))

        def __getitem__(self, key):
            if isinstance(key, tuple) and len(key) == 5:
                assert key == (0, slice(None), 0, slice(None), slice(None))
                assert self.shape[0] == self.shape[2] == 1
                value = nested(self.flat, self.shape)[0]
                return FakeTensor([channel[0] for channel in value])
            if not isinstance(key, tuple):
                key = (key,)
            value = nested(self.flat, self.shape)
            for selector in key:
                if isinstance(selector, int):
                    value = value[selector]
                elif selector == slice(None):
                    value = list(value)
                else:
                    raise AssertionError(f"unsupported fixture selector {selector!r}")
            return FakeTensor(value)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return FakeArray(nested(self.flat, self.shape), dtype="float32")

    class FakeArray:
        def __init__(self, value, *, dtype="uint8"):
            self.value = value
            self.shape = shape_of(value)
            self.dtype = dtype

        def __iter__(self):
            return iter(self.value)

        def tolist(self):
            return copy.deepcopy(self.value)

        def astype(self, dtype):
            def convert(value):
                if isinstance(value, list):
                    return [convert(item) for item in value]
                return int(value) if dtype == "uint8" else float(value)
            return FakeArray(convert(self.value), dtype=dtype)

    class FakeArrayModule:
        @staticmethod
        def stack(values, axis):
            assert axis == 0
            return FakeArray([value.value for value in values], dtype=values[0].dtype)

    class FakeTorch:
        float32 = "float32"

        @staticmethod
        def as_tensor(value, *, dtype, device):
            assert dtype == "float32" and device == "cpu-fixture"
            return FakeTensor(value)

    class Model:
        def __init__(self):
            self.encoder_calls = []
            self.decoder_calls = []

        def encoder(self, frame, data):
            self.encoder_calls.append((frame.shape, data.shape, data.numpy().tolist()))
            return frame

        def decoder(self, frame):
            self.decoder_calls.append(frame.shape)
            return FakeTensor([[0.0] * 32])

    model = Model()
    backend = RivaGANLoadedTensorBackend(
        model, tensor_module=FakeTorch, array_module=FakeArrayModule, device="cpu-fixture",
        input_color_layout="BGR_UINT8",
    )
    media = FakeArray([
            [[[0, 127, 255], [255, 127, 0]]],
            [[[10, 20, 30], [40, 50, 60]]],
        ])
    message = [0, 1] * 16
    embedded = backend.embed(media, message)
    assert embedded.shape == media.shape and embedded.dtype == "uint8"
    assert len(model.encoder_calls) == 2
    assert all(call[:2] == ((1, 3, 1, 1, 2), (1, 32)) for call in model.encoder_calls)
    assert model.encoder_calls[0][2] == [[float(bit) for bit in message]]
    decoded = list(backend.extract(embedded))
    assert [value.shape for value in decoded] == [(32,), (32,)]
    assert model.decoder_calls == [(1, 3, 1, 1, 2), (1, 3, 1, 1, 2)]


def test_quality_identical_psnr_is_null_and_declared_identity_mismatch_conflicts():
    metadata = {
        "case_id": "c", "source_id": "s", "source_content_id": "content", "noise_id": "n",
        "noise_seed": 1, "codec_id": "codec", "codec_parameters": {"x": 1},
    }
    artifacts = {
        "a": {"artifact_id": "a", "status": "AVAILABLE", "_value": [0.0, 0.0], **metadata},
        "b": {"artifact_id": "b", "status": "AVAILABLE", "_value": [0.0, 0.0], **metadata},
    }
    pair = {
        "quality_id": "q", "case_id": "c", "comparison": "IDENTICAL_FIXTURE",
        "reference_artifact_id": "a", "candidate_artifact_id": "b", "data_range": 1.0,
    }
    row = _quality_record(pair, artifacts)
    assert row["state"] == "OBSERVED" and row["mse"] == 0.0
    assert row["psnr_db"] is None and row["psnr_status"] == "IDENTICAL_INFINITE"
    artifacts["b"]["codec_id"] = "other"
    row = _quality_record(pair, artifacts)
    assert row["state"] == "CONFLICT" and "codec_id" in row["reason"]


def test_quality_accepts_array_and_tensor_like_values_and_rejects_nonfinite():
    class FakeDType:
        kind = "f"

    class FakeArray:
        dtype = FakeDType()

        def __init__(self, values, shape=(2, 2)):
            self.values = [float(value) for value in values]
            self.shape = shape
            self.size = len(self.values)

        def astype(self, dtype, copy=False):
            del dtype, copy
            return self

        def __sub__(self, other):
            return FakeArray([left - right for left, right in zip(self.values, other.values)], self.shape)

    class FiniteReceipt:
        def __init__(self, values):
            self.values = values

        def all(self):
            return all(math.isfinite(value) for value in self.values)

    fake_numpy = types.SimpleNamespace(
        float64="float64",
        asarray=lambda value: value,
        isfinite=lambda value: FiniteReceipt(value.values),
        square=lambda value: FakeArray([item * item for item in value.values], value.shape),
        mean=lambda value, dtype: sum(value.values) / len(value.values),
    )
    previous_numpy = sys.modules.get("numpy")
    sys.modules["numpy"] = fake_numpy
    metadata = {
        "case_id": "c", "source_id": "s", "source_content_id": "content", "noise_id": "n",
        "noise_seed": 1, "codec_id": "codec", "codec_parameters": {"x": 1},
    }

    class TensorLike:
        def __init__(self, value):
            self.value = value

        @property
        def shape(self):
            return self.value.shape

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    try:
        artifacts = {
            "a": {"artifact_id": "a", "status": "AVAILABLE", "_value": FakeArray([0, 0, 0, 0]), **metadata},
            "b": {"artifact_id": "b", "status": "AVAILABLE", "_value": TensorLike(FakeArray([1, 1, 1, 1])), **metadata},
        }
        pair = {
            "quality_id": "q", "case_id": "c", "comparison": "ARRAY_TENSOR",
            "reference_artifact_id": "a", "candidate_artifact_id": "b", "data_range": 1.0,
        }
        row = _quality_record(pair, artifacts)
        assert row["state"] == "OBSERVED" and row["mse"] == 1.0 and row["shape"] == [2, 2]
        artifacts["b"]["_value"] = FakeArray([float("nan"), 0, 0, 0])
        row = _quality_record(pair, artifacts)
        assert row["state"] == "FAILED" and "non-finite" in row["reason"]
    finally:
        if previous_numpy is None:
            del sys.modules["numpy"]
        else:
            sys.modules["numpy"] = previous_numpy


def test_native_adapter_method_mismatch_retains_rows_without_calling_adapter():
    class WrongAdapter:
        method = "rivagan"
        backend_metadata = {}

        def embed(self, *args, **kwargs):
            raise AssertionError("mismatched adapter must not be called")

    adapters = build_fixture_native_adapters()
    adapters["videoseal"] = WrongAdapter()
    value = run_workflow(
        workflow_manifest(), operations=build_fixture_operations(),
        native_adapters=adapters, base_dir=PACKAGE,
    )
    job = next(row for row in value["native_records"] if row["job_id"] == "fixture_ok/videoseal")
    assert job["status"] == "CONFLICT" and "adapter.method" in job["reason"]
    artifacts = {row["artifact_id"]: row for row in value["artifacts"]}
    assert artifacts["fixture_ok/videoseal/pre"]["status"] == "CONFLICT"
    assert artifacts["fixture_ok/videoseal/post"]["status"] == "CONFLICT"
    costs = [row for row in value["cost_records"] if row["cost_id"].startswith("native/fixture_ok/videoseal/")]
    assert len(costs) == 3 and all(row["state"] == "CONFLICT" for row in costs)


def test_workflow_cli_runs_from_no_git_source_copy(tmp_path):
    source = tmp_path / "source-copy"
    (source / "experiments").mkdir(parents=True)
    shutil.copy(ROOT / "experiments/__init__.py", source / "experiments/__init__.py")
    shutil.copytree(PACKAGE, source / "experiments/paper_results_v1")
    output = tmp_path / "workflow-output"
    interpreter = [sys.executable]
    if not os.access(sys.executable, os.X_OK):
        interpreter = ["/lib64/ld-linux-x86-64.so.2", sys.executable]
    completed = subprocess.run(
        interpreter + [
            "-m", "experiments.paper_results_v1.workflow_cli",
            "--manifest", "experiments/paper_results_v1/workflow_fixture.manifest.json",
            "--backend-factory", "experiments.paper_results_v1.fixture_backends:build_fixture_backend_bundle",
            "--output-dir", str(output),
        ],
        cwd=source,
        text=True,
        capture_output=True,
        check=True,
    )
    assert not (source / ".git").exists()
    summary = json.loads(completed.stdout)
    assert summary["denominator"]["plan_steps"] == 18
    assert summary["main_report_status"] == "LINKED"
    assert {path.name for path in output.iterdir()} == {
        "workflow_report.json", "workflow_report.md", "plan.csv", "artifacts.csv",
        "quality.csv", "cost.csv", "native_records.json", "main_report",
    }
    value = json.loads((output / "workflow_report.json").read_text(encoding="utf-8"))
    assert value["manifest_denominator"]["native_state_counts"] == {
        "BLOCKED_DEPENDENCY": 2, "SUCCEEDED": 2,
    }
