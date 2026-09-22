"""Build the immutable-source paired candidate notebook; the user runs it on Colab."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
DEFAULT_OUTPUT = ROOT / "notebooks" / "window_state_mse_v1_paired_colab.ipynb"


def build(source_sha: str, output: str | Path | None = None) -> Path:
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("source_sha must be one lowercase 40-hex Git commit")
    output = DEFAULT_OUTPUT if output is None else Path(output)
    golden = json.loads((ROOT / "notebooks" / "integrated_payload_v1_colab.ipynb").read_text())

    def code(text):
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}

    def markdown(text):
        return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

    setup = """from pathlib import Path
import datetime, json, sys
SOURCE_SHA = 'SOURCE_TOKEN'
DRIVE_ROOT = Path('/content/drive/MyDrive/Video-WM/Window-State-MSE-V1')
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUTPUT = DRIVE_ROOT / ('window_state_mse_v1_' + stamp)
OUTPUT.mkdir(exist_ok=False)
(OUTPUT / 'setup_receipt.json').write_text(json.dumps(dict(source_commit=SOURCE_SHA, python=sys.version, executable=sys.executable, status='SETUP_STARTED'), indent=2))
print('same-run output:', OUTPUT, flush=True)
""".replace("SOURCE_TOKEN", source_sha)
    logging = r"""import subprocess, sys, json
SETUP_LOG = OUTPUT / 'setup.log'
def logged_run(command, check=True):
    with SETUP_LOG.open('a', encoding='utf-8') as log:
        line = 'COMMAND ' + repr(command) + '\n'
        print(line, end='', flush=True); log.write(line); log.flush()
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in child.stdout:
            print(line, end='', flush=True); log.write(line); log.flush()
        returncode = child.wait()
        log.write('EXIT ' + str(returncode) + '\n'); log.flush()
    if check and returncode:
        (OUTPUT / 'setup_failure.json').write_text(json.dumps(dict(command=command, returncode=returncode), indent=2))
        raise subprocess.CalledProcessError(returncode, command)
    return subprocess.CompletedProcess(command, returncode)
print('Python:', sys.version, 'Executable:', sys.executable, flush=True)
"""
    install = "".join(golden["cells"][2]["source"]).replace("subprocess.run(", "logged_run(")
    receipt = """from pathlib import Path
info = dict(python=sys.version, executable=sys.executable, packages={})
for package in ('torch', 'torchvision', 'diffusers', 'transformers', 'accelerate', 'ftfy', 'sentencepiece', 'safetensors', 'huggingface_hub', 'numpy', 'Pillow'):
    try: info['packages'][package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError: info['packages'][package] = None
import json
Path(RECEIPT_PATH).write_text(json.dumps(info, indent=2))
"""
    install = install.replace(
        'logged_run([sys.executable, "-u", "-c", check_code], check=True)',
        "check_code = check_code.replace(\"assert str(torch.__version__)\", "
        + repr(receipt).replace("RECEIPT_PATH", "RECEIPT_PATH")
        + ".replace('RECEIPT_PATH', repr(str(OUTPUT / 'environment_setup.json'))) + \"assert str(torch.__version__)\")\n"
        + "logged_run([sys.executable, '-u', '-c', check_code], check=True)",
    )
    checkout = """import subprocess
REPO = Path('/content/SC-SSTW-Window-State-MSE-' + stamp)
logged_run(['git', 'clone', '--filter=blob:none', 'https://github.com/RICHAAARC/SC-SSTW.git', str(REPO)])
logged_run(['git', '-C', str(REPO), 'checkout', '--detach', SOURCE_SHA])
actual = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
assert actual == SOURCE_SHA
print('source commit:', actual, flush=True)
"""
    cuda = "".join(golden["cells"][4]["source"]).replace("subprocess.run(", "logged_run(")
    run = """import json, os, subprocess, sys
CONFIG = REPO / 'experiments/wan_state_clock/configs/window_state_mse_v1_paired.json'
env = os.environ.copy(); env['PYTHONUNBUFFERED'] = '1'
command = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.window_state_mse_paired_run', '--config', str(CONFIG), '--output', str(OUTPUT)]
print('paired experiment argv:', command, flush=True)
print('paired experiment output:', OUTPUT, flush=True)
completed = subprocess.run(command, cwd=REPO, env=env, check=False)
print('paired experiment returncode:', completed.returncode, flush=True)
(OUTPUT / 'execution_receipt.json').write_text(json.dumps(dict(command=command, returncode=completed.returncode, result_path=str(OUTPUT / 'result.json')), indent=2))
if not (OUTPUT / 'result.json').exists():
    raise FileNotFoundError('paired runner produced no retained result.json')
"""
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
for case_id, case in cases.items():
    print('case:', case_id, 'status:', case.get('status'), 'generate_exit:', case.get('generate_exit_code'), 'media_exit:', case.get('media_exit_code'))
    print(' case failures:', case.get('failures', []), 'parent failures:', case.get('parent_failures', []))
receiver_names = ('ORIGINAL', 'C1_MATCHED_CONFIRM', 'C2_STATE_CONFIRM')
view_statuses = {receiver:{} for receiver in receiver_names}; source_statuses = {receiver:{} for receiver in receiver_names}
view_slots = source_slots = 0
for case in cases.values():
    for arm in case.get('videos', {}).values():
        for receiver, decision in arm.get('receiver_source_decisions', {}).items():
            source_slots += 1; status = decision.get('status', 'NOT_RUN'); bucket = source_statuses.setdefault(receiver, {}); bucket[status] = bucket.get(status, 0) + 1
        for view in arm.get('views', {}).values():
            for receiver, row in view.get('receivers', {}).items():
                view_slots += 1; status = row.get('decision', {}).get('status', 'NOT_RUN'); bucket = view_statuses.setdefault(receiver, {}); bucket[status] = bucket.get(status, 0) + 1
for receiver in receiver_names:
    print('receiver slots:', receiver, dict(view_slots=sum(view_statuses[receiver].values()), expected_view_slots=36, view_statuses=view_statuses[receiver], source_slots=sum(source_statuses[receiver].values()), expected_source_slots=12, source_statuses=source_statuses[receiver]))
print('all receiver slots:', dict(view_slots=view_slots, expected_view_slots=108, source_slots=source_slots, expected_source_slots=36))
for row in result.get('paired_comparisons', []):
    print(row['case_id'], row['receiver'], row['schedule'])
    print(' source:', row['source_arm'])
    for view, pair in row['views'].items(): print(' ', view, pair)
print('top-level retained failures:', result.get('failures', []))
print('full result:', result_path)
"""
    cells = [
        code("from google.colab import drive\ndrive.mount('/content/drive')"),
        markdown(
            "# Window-State-MSE-V1 paired fixed run\n\n"
            "Run all once. Two OFF sources independently calibrate ORIGINAL, C1, and C2; two evaluation sources share one fresh noise/state-44 snapshot across OFF and four paired legacy/MSE forks. FULL, DELETE90, and SPEED5_4 are fixed: 12 physical source-arms, 36 saved views, and 144 phase encodes. The user performs the GPU run. Process completion is not a method PASS."
        ),
        code(setup), code(logging + install), code(checkout), code(cuda), code(run), code(results),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = "window-state-mse-" + str(index)
    notebook = {
        "cells": cells,
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "Window-State-MSE-V1 Paired", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "source_commit": source_sha,
            "notebook_binding_kind": "immutable_source_commit",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_sha")
    parser.add_argument("--output")
    arguments = parser.parse_args()
    print(build(arguments.source_sha, arguments.output))
