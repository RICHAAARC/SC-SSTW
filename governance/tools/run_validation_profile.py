"""Explicit lightweight build profiles; never called by the model runtime."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from governance.harness import checks


def run_profile(root: Path, name: str) -> dict:
    policy = json.loads((root / 'governance/policies/validation.json').read_text(encoding='utf-8'))
    profiles = {
        'method': (True, ('dependencies',)),
        'notebook': (False, ('notebook_binding',)),
        'governance': (False, ('dependencies',)),
        'release': (True, ('dependencies', 'notebook_binding', 'release')),
    }
    project_tests, selected = profiles[name]
    errors = []
    if project_tests:
        result = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=root, check=False)
        if result.returncode:
            errors.append(f'project tests exited {result.returncode}')
    if name in ('governance', 'release'):
        result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'governance/tests', '-q'], cwd=root, check=False)
        if result.returncode:
            errors.append(f'harness tests exited {result.returncode}')
    if name == 'release':
        result = subprocess.run([sys.executable, 'governance/tools/check_candidate_closure.py'], cwd=root, check=False)
        if result.returncode:
            errors.append(f'candidate closure exited {result.returncode}')
    for check in selected:
        errors.extend(getattr(checks, check)(root, policy))
    return {'profile': name, 'status': 'FAIL' if errors else 'PASS', 'errors': errors,
            'evidence': 'engineering checks only; no model execution or scientific decision'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', choices=('method', 'notebook', 'governance', 'release'))
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    report = run_profile(args.root.resolve(), args.profile)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(report['status'] != 'PASS')
