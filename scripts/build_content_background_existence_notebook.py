"""Build the immutable-source Content-Background-Existence-V1 notebook."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import build_window_state_mse_paired_notebook as base_builder

DEFAULT_OUTPUT = ROOT / "notebooks" / "content_background_existence_v1_colab.ipynb"


def _source(cell):
    return "".join(cell["source"])


def _set_source(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def build(source_sha: str, output: str | Path | None = None) -> Path:
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("source_sha must be one lowercase 40-hex Git commit")
    output = DEFAULT_OUTPUT if output is None else Path(output)
    base_builder.build(source_sha, output)
    notebook = json.loads(output.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    assert _source(cells[0]) == "from google.colab import drive\ndrive.mount('/content/drive')"

    _set_source(cells[1], """# Content-Background-Existence-V1 fixed new-source run

Run all once. Four OFF sources independently calibrate ORIGINAL, C1, C2, and the fixed wrong-key normalized candidate before four evaluation sources run OFF, SINGLE46, and MULTI44_46. Every physical arm saves FULL, DELETE90, and SPEED5_4: 8 fresh sources, 16 physical arms, 48 views, and 192 phase encodes. All failures remain in `result.json`; a nonzero runner return code does not suppress the retained result. GPU execution is performed by the user. Process completion is not a method or low-FPR PASS.""")

    setup = _source(cells[2])
    setup = setup.replace("Video-WM/Window-State-MSE-V1", "Video-WM/Content-Background-Existence-V1")
    setup = setup.replace("window_state_mse_v1_", "content_background_existence_v1_")
    _set_source(cells[2], setup)

    install = _source(cells[3])
    _set_source(cells[3], install)

    checkout = _source(cells[4]).replace(
        "SC-SSTW-Window-State-MSE-", "SC-SSTW-Content-Background-Existence-"
    )
    _set_source(cells[4], checkout)

    run = """import json, os, subprocess, sys
CONFIG = REPO / 'experiments/wan_state_clock/configs/content_background_existence_v1.json'
env = os.environ.copy(); env['PYTHONUNBUFFERED'] = '1'
command = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.content_background_existence_run', '--config', str(CONFIG), '--output', str(OUTPUT)]
print('fixed experiment argv:', command, flush=True)
print('fixed experiment output:', OUTPUT, flush=True)
completed = subprocess.run(command, cwd=REPO, env=env, check=False)
print('fixed experiment returncode:', completed.returncode, flush=True)
(OUTPUT / 'execution_receipt.json').write_text(json.dumps(dict(command=command, returncode=completed.returncode, result_path=str(OUTPUT / 'result.json')), indent=2))
if not (OUTPUT / 'result.json').exists():
    raise FileNotFoundError('runner produced no retained result.json')
"""
    _set_source(cells[6], run)

    results = """import json
result_path = OUTPUT / 'result.json'
try:
    result = json.loads(result_path.read_text())
except Exception as exc:
    print('result.json is missing or invalid:', repr(exc), 'path:', result_path, flush=True)
    raise
print('status:', result.get('status'))
print('fixed denominator:', result.get('fixed_denominator'))
print('candidate search accounting:', result.get('candidate_search_accounting'))
print('call accounting:', result.get('call_accounting', 'NOT_FINALIZED'))
print('calibrations:', result.get('calibrations', 'NOT_FINALIZED'))
print('evaluation summary:', result.get('summary', 'NOT_FINALIZED'))
for case_id, case in result.get('cases', {}).items():
    print('case:', case_id, 'role:', case.get('role'), 'status:', case.get('status'), 'generate_exit:', case.get('generate_exit_code'), 'media_exit:', case.get('media_exit_code'))
    print(' case failures:', case.get('failures', []), 'parent failures:', case.get('parent_failures', []))
    for arm, item in case.get('videos', {}).items():
        print(' ', arm, 'writer:', item.get('writer_objective_identity'), 'sources:', item.get('receiver_source_decisions'))
        for view, view_row in item.get('views', {}).items():
            print('   ', view, {name: row.get('decision') for name, row in view_row.get('receivers', {}).items()})
print('top-level retained failures:', result.get('failures', []))
print('full result:', result_path)
"""
    _set_source(cells[7], results)

    for index, cell in enumerate(cells):
        cell["id"] = "content-background-existence-" + str(index)
    notebook["metadata"]["colab"]["name"] = "Content-Background-Existence-V1"
    notebook["metadata"]["source_commit"] = source_sha
    notebook["metadata"]["notebook_binding_kind"] = "immutable_source_commit"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_sha")
    parser.add_argument("--output")
    arguments = parser.parse_args()
    print(build(arguments.source_sha, arguments.output))
