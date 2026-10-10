"""Persistence at model/reader boundaries, with no real model execution."""
import json
import types

import pytest

from runtime.wan import local_joint_terminal_bridge_v1 as runtime

pytestmark = pytest.mark.quick


def test_before_adapter_failure_returned_model_call_is_already_saved(tmp_path):
    result = runtime.initial_result({})
    def save():
        runtime.write_json(tmp_path/"result.json", result)
    tracked = runtime.TrackedVAE(types.SimpleNamespace(encode=lambda _: "returned-value"), result, save,
                                 execution_kind="INJECTED_TEST_DOUBLE")
    tracked.current = "encode_base"
    assert tracked.encode(None) == "returned-value"
    on_disk = json.loads((tmp_path/"result.json").read_text())
    assert on_disk["counts"]["encode_completed"] == 1
    assert on_disk["model_calls"]["encode_base"]["status"] == "RETURNED"
    with pytest.raises(RuntimeError, match="repeated"):
        tracked.encode(None)
    assert result["counts"]["encode_attempted"] == 1


def test_primary_model_error_survives_secondary_persistence_error():
    result = runtime.initial_result({})
    primary = RuntimeError("primary model failure")
    def encode(_):
        raise primary
    calls = []
    def save():
        calls.append(None)
        if len(calls) == 2:
            raise OSError("secondary disk failure")
    tracked = runtime.TrackedVAE(types.SimpleNamespace(encode=encode), result, save,
                                 execution_kind="INJECTED_TEST_DOUBLE")
    tracked.current = "encode_base"
    with pytest.raises(RuntimeError) as caught:
        tracked.encode(None)
    assert caught.value is primary
    assert result["counts"]["encode_attempted"] == 1
    assert result["counts"]["encode_completed"] == 0
    assert "secondary disk failure" in str(getattr(primary, "__notes__", []))
