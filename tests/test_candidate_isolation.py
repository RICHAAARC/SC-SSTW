"""The formal candidate must not resolve historical modules from a parent tree."""
from pathlib import Path
import subprocess
import sys

import numpy
import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = Path(numpy.__file__).resolve().parent.parent


def invoke(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, '-I', '-c', code], text=True, capture_output=True, check=False)


def test_formal_runner_imports_in_an_isolated_interpreter():
    result = invoke(f"import sys; sys.path.insert(0, {str(DEPENDENCIES)!r}); sys.path.insert(0, {str(ROOT)!r}); import experiments.wan_state_clock.run")
    assert result.returncode == 0, result.stderr


def test_missing_adapter_does_not_resolve_from_another_tree():
    result = invoke(f"import sys, importlib; sys.path.insert(0, {str(DEPENDENCIES)!r}); sys.path.insert(0, {str(ROOT)!r}); import experiments.wan_state_clock.run; importlib.import_module('runtime.missing_adapter')")
    assert result.returncode != 0
    assert 'ModuleNotFoundError' in result.stderr
