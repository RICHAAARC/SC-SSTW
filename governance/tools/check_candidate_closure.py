"""Verify the formal candidate imports only from its own isolated tree."""
from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FORBIDDEN_PATHS = ('runtime/c2a', 'runtime/c2t1', 'runtime/tstwv2', 'main/sc_sstw', 'experiments/feasibility')


def isolated(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, '-I', '-c', code], text=True, capture_output=True, check=False)


def main() -> None:
    missing = [name for name in FORBIDDEN_PATHS if (ROOT / name).exists()]
    if missing:
        raise SystemExit(f'forbidden candidate paths: {missing}')
    root = repr(str(ROOT))
    dependencies = repr(str(Path(numpy.__file__).resolve().parent.parent))
    prefix = f"import sys; sys.path.insert(0, {dependencies}); sys.path.insert(0, {root}); "
    positive = isolated(prefix + "import experiments.wan_state_clock.run as runner; import main.tube_state.state_clock as state; import runtime.wan.io as io; assert all(str(path).startswith(" + root + ") for path in (runner.__file__, state.__file__, io.__file__)); assert 'torch' not in sys.modules")
    if positive.returncode:
        raise SystemExit(f'isolated formal imports failed: {positive.stderr}')
    core_tests = isolated(prefix + "import os, pytest; os.chdir(" + root + "); raise SystemExit(pytest.main(['-q', 'tests/test_wan_projection.py', 'tests/test_state_clock.py']))")
    if core_tests.returncode:
        raise SystemExit(f'isolated core tests failed: {core_tests.stdout}{core_tests.stderr}')
    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary)
        target = staging / 'experiments/wan_state_clock/run.py'
        target.parent.mkdir(parents=True)
        target.write_text((ROOT / 'experiments/wan_state_clock/run.py').read_text(encoding='utf-8') + '\nimport runtime.missing_adapter\n', encoding='utf-8')
        (staging / 'experiments/__init__.py').write_text('', encoding='utf-8')
        (staging / 'experiments/wan_state_clock/__init__.py').write_text('', encoding='utf-8')
        bad_prefix = f"import sys; sys.path.insert(0, {dependencies}); sys.path.insert(0, {root}); sys.path.insert(0, {str(staging)!r}); "
        missing_adapter = isolated(bad_prefix + "import experiments.wan_state_clock.run")
        if missing_adapter.returncode == 0 or 'ModuleNotFoundError' not in missing_adapter.stderr:
            raise SystemExit('bad formal runner unexpectedly resolved a missing adapter')
    from governance.harness.checks import dependencies
    import json
    policy = json.loads((ROOT / 'governance/policies/validation.json').read_text(encoding='utf-8'))
    errors = dependencies(ROOT, policy)
    if errors:
        raise SystemExit('\n'.join(errors))
    print('candidate closure PASS: isolated imports, 10 core tests, and bad-runner negative control')


if __name__ == '__main__':
    main()
