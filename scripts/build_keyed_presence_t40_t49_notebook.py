"""Build a user-run notebook only after its immutable source SHA is published."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
GOLDEN = ROOT / "notebooks" / "integrated_payload_v1_colab.ipynb"


def _code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(keepends=True)}


def _markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def build(source_sha: str, output: Path):
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("published immutable 40-hex source SHA required")
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    setup = """from pathlib import Path
import datetime, json, sys
SOURCE_SHA = 'SOURCE_TOKEN'
DRIVE_ROOT = Path('/content/drive/MyDrive/Video-WM/Keyed-Presence-T40-T49-V1')
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUTPUT = DRIVE_ROOT / ('keyed_presence_t40_t49_v1_' + stamp)
OUTPUT.mkdir(exist_ok=False)
(OUTPUT / 'setup_receipt.json').write_text(json.dumps(dict(source_commit=SOURCE_SHA, python=sys.version, executable=sys.executable, status='SETUP_STARTED'), indent=2))
print('same-run output:', OUTPUT, flush=True)
""".replace("SOURCE_TOKEN", source_sha)
    logging = """import subprocess, sys
SETUP_LOG = OUTPUT / 'setup.log'
def logged_run(command, check=True, cwd=None, env=None):
    with SETUP_LOG.open('a', encoding='utf-8') as log:
        line = 'COMMAND ' + repr(command) + '\\n'
        print(line, end='', flush=True); log.write(line); log.flush()
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in child.stdout:
            print(line, end='', flush=True); log.write(line); log.flush()
        returncode = child.wait()
        log.write('EXIT ' + str(returncode) + '\\n'); log.flush()
    if check and returncode:
        (OUTPUT / 'setup_failure.json').write_text(json.dumps(dict(command=command, returncode=returncode), indent=2))
        raise subprocess.CalledProcessError(returncode, command)
    return subprocess.CompletedProcess(command, returncode)
print('Python:', sys.version, 'Executable:', sys.executable, flush=True)
"""
    install = "".join(golden["cells"][2]["source"]).replace("subprocess.run(", "logged_run(")
    checkout = """import subprocess
REPO = Path('/content/SC-SSTW-Keyed-Presence-' + stamp)
logged_run(['git', 'clone', '--filter=blob:none', 'https://github.com/RICHAAARC/SC-SSTW.git', str(REPO)])
logged_run(['git', '-C', str(REPO), 'checkout', '--detach', SOURCE_SHA])
actual = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
assert actual == SOURCE_SHA
print('source commit:', actual, flush=True)
"""
    cuda = "".join(golden["cells"][4]["source"]).replace("subprocess.run(", "logged_run(")
    run = """import os
CONFIG = REPO / 'experiments/wan_state_clock/configs/keyed_presence_t40_t49_v1.json'
env = os.environ.copy(); env['PYTHONUNBUFFERED'] = '1'
command = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.keyed_presence_t40_t49_run', '--config', str(CONFIG), '--output', str(OUTPUT)]
print('fixed experiment output:', OUTPUT, flush=True)
logged_run(command, cwd=REPO, env=env)
"""
    results = """import json
result = json.loads((OUTPUT / 'result.json').read_text())
print('status:', result['status'])
print('fixed denominator:', result['fixed_denominator'])
print('C2 calibration:', result['calibration']['C2_STATE_CONFIRM']['status'], result['calibration']['C2_STATE_CONFIRM']['threshold'])
print('target H0/H1:', result['target_summary'])
for case_id, row in result['cases'].items():
    print(case_id, row['status'], {arm: item['receivers']['C2_STATE_CONFIRM'].get('decision') for arm, item in row['videos'].items()})
print('full result:', OUTPUT / 'result.json')
"""
    cells = [
        _code("from google.colab import drive\ndrive.mount('/content/drive')"),
        _markdown("# Keyed presence T40–T49 V1\n\nRun all once. Nine independent OFF sources calibrate FULL-only C2 before four new evaluation sources. Each evaluation source generates OFF and the four fixed tanh timing arms; the sole target is keyed H0/H1 presence for T40–T49. ORIGINAL/C1 and terminal-to-MP4 partitions are diagnostics. This notebook does not run automatically and does not establish video quality, identity, payload, low FPR, or extra multistep gain."),
        _code(setup), _code(logging + install), _code(checkout), _code(cuda), _code(run), _code(results),
    ]
    for i, cell in enumerate(cells): cell["id"] = f"keyed-presence-{i}"
    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "colab": {"name": "Keyed Presence T40-T49 V1", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}, "source_commit": source_sha,
            "notebook_binding_kind": "immutable_source_commit"}}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(build(args.source_sha, args.output))
