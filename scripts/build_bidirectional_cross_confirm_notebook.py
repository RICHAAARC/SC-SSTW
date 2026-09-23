"""Build the immutable-source Bidirectional-Cross-Confirm-V1 user-run notebook."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import build_window_state_mse_paired_notebook as base

ROOT = Path(__file__).parents[1]
DEFAULT_OUTPUT = ROOT / "notebooks" / "bidirectional_cross_confirm_v1_colab.ipynb"


def build(source_sha: str, output: str | Path | None = None) -> Path:
    output = DEFAULT_OUTPUT if output is None else Path(output)
    base.build(source_sha, output)
    notebook = json.loads(output.read_text(encoding="utf-8"))
    notebook["cells"][1]["source"] = (
        "# Bidirectional-Cross-Confirm-V1 fixed run\n\n"
        "Run all once. Four fresh OFF sources independently calibrate ORIGINAL, C1, C2, and the bidirectional candidate; four fresh evaluation sources each produce OFF, legacy SINGLE46, and legacy MULTI44_46. FULL, DELETE90, and SPEED5_4 are fixed: 16 physical source-arms, 48 saved views, and 192 phase encodes. The candidate averages the two held-out C2 directions and is invalid if either direction is invalid. The user performs the GPU run. Process completion is not a method PASS."
    ).splitlines(keepends=True)
    setup = """from pathlib import Path
import datetime, json, sys
SOURCE_SHA = 'SOURCE_TOKEN'
DRIVE_ROOT = Path('/content/drive/MyDrive/Video-WM/Bidirectional-Cross-Confirm-V1')
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUTPUT = DRIVE_ROOT / ('bidirectional_cross_confirm_v1_' + stamp)
OUTPUT.mkdir(exist_ok=False)
(OUTPUT / 'setup_receipt.json').write_text(json.dumps(dict(source_commit=SOURCE_SHA, python=sys.version, executable=sys.executable, status='SETUP_STARTED'), indent=2))
print('fixed-run output:', OUTPUT, flush=True)
""".replace("SOURCE_TOKEN", source_sha)
    notebook["cells"][2]["source"] = setup.splitlines(keepends=True)
    checkout = """import subprocess
REPO = Path('/content/SC-SSTW-Bidirectional-Cross-Confirm-' + stamp)
logged_run(['git', 'clone', '--filter=blob:none', 'https://github.com/RICHAAARC/SC-SSTW.git', str(REPO)])
logged_run(['git', '-C', str(REPO), 'checkout', '--detach', SOURCE_SHA])
actual = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
assert actual == SOURCE_SHA
print('source commit:', actual, flush=True)
"""
    notebook["cells"][4]["source"] = checkout.splitlines(keepends=True)
    run = """import json, os, subprocess, sys
CONFIG = REPO / 'experiments/wan_state_clock/configs/bidirectional_cross_confirm_v1.json'
env = os.environ.copy(); env['PYTHONUNBUFFERED'] = '1'
command = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.bidirectional_cross_confirm_run', '--config', str(CONFIG), '--output', str(OUTPUT)]
print('fixed experiment argv:', command, flush=True)
print('fixed experiment output:', OUTPUT, flush=True)
completed = subprocess.run(command, cwd=REPO, env=env, check=False)
print('fixed experiment returncode:', completed.returncode, flush=True)
(OUTPUT / 'execution_receipt.json').write_text(json.dumps(dict(command=command, returncode=completed.returncode, result_path=str(OUTPUT / 'result.json')), indent=2))
if not (OUTPUT / 'result.json').exists():
    raise FileNotFoundError('fixed runner produced no retained result.json')
"""
    notebook["cells"][6]["source"] = run.splitlines(keepends=True)
    results = """import json
result_path = OUTPUT / 'result.json'
try:
    result = json.loads(result_path.read_text())
except Exception as exc:
    print('result.json is missing or invalid:', repr(exc), 'path:', result_path, flush=True)
    raise
print('status:', result['status'])
print('fixed denominator:', result.get('fixed_denominator', 'NOT_FINALIZED'))
print('call accounting:', result.get('call_accounting', 'NOT_FINALIZED'))
print('calibrations:', result.get('calibrations', 'NOT_FINALIZED'))
cases = result.get('cases', {})
receiver_names = ('ORIGINAL', 'C1_MATCHED_CONFIRM', 'C2_STATE_CONFIRM', 'BIDIRECTIONAL_C2_MEAN')
view_statuses = {receiver:{} for receiver in receiver_names}; source_statuses = {receiver:{} for receiver in receiver_names}
for case_id, case in cases.items():
    print('case:', case_id, 'status:', case.get('status'), 'generate_exit:', case.get('generate_exit_code'), 'media_exit:', case.get('media_exit_code'))
    print(' case failures:', case.get('failures', []), 'parent failures:', case.get('parent_failures', []))
    for arm in case.get('videos', {}).values():
        for receiver, decision in arm.get('receiver_source_decisions', {}).items():
            status = decision.get('status', 'NOT_RUN'); bucket = source_statuses.setdefault(receiver, {}); bucket[status] = bucket.get(status, 0) + 1
        for view in arm.get('views', {}).values():
            for receiver, row in view.get('receivers', {}).items():
                status = row.get('decision', {}).get('status', 'NOT_RUN'); bucket = view_statuses.setdefault(receiver, {}); bucket[status] = bucket.get(status, 0) + 1
for receiver in receiver_names:
    print('receiver slots:', receiver, dict(view_slots=sum(view_statuses[receiver].values()), expected_view_slots=48, view_statuses=view_statuses[receiver], source_slots=sum(source_statuses[receiver].values()), expected_source_slots=16, source_statuses=source_statuses[receiver]))
evaluation = {receiver:{'OFF_views':{}, 'marked_views':{}, 'OFF_sources':{}, 'marked_sources':{}} for receiver in receiver_names}
for case in cases.values():
    if case.get('role') != 'evaluation': continue
    for arm_name, arm in case.get('videos', {}).items():
        kind = 'OFF' if arm_name == 'OFF' else 'marked'
        for receiver, decision in arm.get('receiver_source_decisions', {}).items():
            bucket = evaluation[receiver][kind + '_sources']; status = decision.get('status', 'NOT_RUN'); bucket[status] = bucket.get(status, 0) + 1
        for view in arm.get('views', {}).values():
            for receiver, row in view.get('receivers', {}).items():
                bucket = evaluation[receiver][kind + '_views']; status = row.get('decision', {}).get('status', 'NOT_RUN'); bucket[status] = bucket.get(status, 0) + 1
for receiver in receiver_names:
    print('evaluation-only:', receiver, evaluation[receiver], 'expected:', dict(OFF_views=12, marked_views=24, OFF_sources=4, marked_sources=8))
for row in result.get('comparison_records', []):
    print('comparison:', row)
print('top-level retained failures:', result.get('failures', []))
print('full result:', result_path)
"""
    notebook["cells"][7]["source"] = results.splitlines(keepends=True)
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = "bidirectional-cross-confirm-" + str(index)
    notebook["metadata"]["colab"]["name"] = "Bidirectional-Cross-Confirm-V1"
    notebook["metadata"]["source_commit"] = source_sha
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_sha")
    parser.add_argument("--output")
    arguments = parser.parse_args()
    print(build(arguments.source_sha, arguments.output))
