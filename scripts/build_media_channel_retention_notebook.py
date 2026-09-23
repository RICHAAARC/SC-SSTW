"""Build the immutable-source Media-Channel-Retention-V1 notebook."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
DEFAULT_OUTPUT = ROOT / "notebooks" / "media_channel_retention_v1_colab.ipynb"


def build(source_sha: str, output: str | Path | None = None) -> Path:
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("source_sha must be one lowercase 40-hex Git commit")
    output = DEFAULT_OUTPUT if output is None else Path(output)
    golden = json.loads((ROOT / "notebooks" / "integrated_payload_v1_colab.ipynb").read_text())

    def code(text):
        return {
            "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True),
        }

    def markdown(text):
        return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

    setup = """from pathlib import Path
import datetime, json, sys
SOURCE_SHA = 'SOURCE_TOKEN'
INPUT_ROOT = Path('/content/drive/MyDrive/Video-WM/Window-State-MSE-V1/window_state_mse_v1_20260922T174349464768Z')
OUTPUT_PARENT = Path('/content/drive/MyDrive/Video-WM/Media-Channel-Retention-V1')
OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUTPUT = OUTPUT_PARENT / ('media_channel_retention_v1_' + stamp)
OUTPUT.mkdir(exist_ok=False)
(OUTPUT / 'setup_receipt.json').write_text(json.dumps(dict(source_commit=SOURCE_SHA, input_root=str(INPUT_ROOT), python=sys.version, executable=sys.executable, status='SETUP_STARTED'), indent=2))
print('fixed input:', INPUT_ROOT, flush=True)
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
import importlib.metadata, json, sys
info = dict(python=sys.version, executable=sys.executable, packages={})
for package in ('torch', 'torchvision', 'diffusers', 'transformers', 'accelerate', 'ftfy', 'sentencepiece', 'safetensors', 'huggingface_hub', 'numpy', 'Pillow'):
    try: info['packages'][package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError: info['packages'][package] = None
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
REPO = Path('/content/SC-SSTW-Media-Channel-Retention-' + stamp)
logged_run(['git', 'clone', '--filter=blob:none', 'https://github.com/RICHAAARC/SC-SSTW.git', str(REPO)])
logged_run(['git', '-C', str(REPO), 'checkout', '--detach', SOURCE_SHA])
actual = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
assert actual == SOURCE_SHA
print('source commit:', actual, flush=True)
"""
    cuda = "".join(golden["cells"][4]["source"]).replace("subprocess.run(", "logged_run(")
    run = """import json, os, subprocess, sys
CONFIG = REPO / 'experiments/wan_state_clock/configs/media_channel_retention_v1.json'
env = os.environ.copy(); env['PYTHONUNBUFFERED'] = '1'
command = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.media_channel_retention_run', '--config', str(CONFIG), '--output', str(OUTPUT)]
print('diagnostic argv:', command, flush=True)
print('diagnostic output:', OUTPUT, flush=True)
completed = subprocess.run(command, cwd=REPO, env=env, check=False)
print('diagnostic returncode:', completed.returncode, flush=True)
(OUTPUT / 'execution_receipt.json').write_text(json.dumps(dict(command=command, returncode=completed.returncode, result_path=str(OUTPUT / 'result.json')), indent=2))
if not (OUTPUT / 'result.json').exists():
    raise FileNotFoundError('runner produced no retained result.json')
"""
    results = """import json
result_path = OUTPUT / 'result.json'
result = json.loads(result_path.read_text())
print('status:', result.get('status'))
print('fixed input audit:', result.get('input_root_audit'))
print('call accounting:', result.get('call_accounting'))
print('slot accounting:', result.get('slot_accounting'))
for case_id, case in result.get('cases', {}).items():
    print('case:', case_id, 'status:', case.get('status'), 'subprocess:', case.get('subprocess'))
    print(' case calls:', case.get('call_accounting'), 'failures:', case.get('failures'))
    for arm, row in case.get('trajectories', {}).items():
        print(' ', arm, row.get('status'), {name: layer.get('status') for name, layer in row.get('layers', {}).items()})
        if row.get('failures'): print('   failures:', row['failures'])
print('marked-minus-OFF slots:', len(result.get('marked_off_increments', [])))
print('adjacent increment-change slots:', len(result.get('adjacent_increment_changes', [])))
print('terminal-to-RGB8 combined slots:', len(result.get('terminal_to_rgb8_combined_increment_changes', [])))
def delta(row, field):
    return row.get('comparison', {}).get('scalars', {}).get(field, {}).get('delta')
increments = {
    (row['case_id'], row['marked_arm'], row['partition'], row['layer']): row
    for row in result.get('marked_off_increments', [])
}
changes = {
    (row['case_id'], row['marked_arm'], row['partition'], row['before_layer'], row['after_layer']): row
    for row in result.get('adjacent_increment_changes', [])
}
layers = ('TERMINAL', 'FLOAT_RGB_REENCODE', 'RGB8_NO_CODEC_REENCODE', 'EXISTING_MP4_G0')
transitions = tuple(zip(layers, layers[1:]))
for case_id in ('eval_p2_s2', 'eval_p3_s3'):
    for arm in ('LEGACY_SINGLE46', 'MSE_SINGLE46', 'LEGACY_MULTI44_46', 'MSE_MULTI44_46'):
        for partition in ('TOTAL', 'A', 'B'):
            print('increment table:', case_id, arm, partition)
            for layer in layers:
                row = increments.get((case_id, arm, partition, layer), {})
                print(' ', layer, 'matched marked-OFF=', delta(row, 'matched_score'), 'innovation marked-OFF=', delta(row, 'state_innovation_mean'), 'status=', row.get('comparison', {}).get('status', 'UNDEFINED'))
            for before, after in transitions:
                row = changes.get((case_id, arm, partition, before, after), {})
                print(' ', before + ' -> ' + after, 'matched increment change=', delta(row, 'matched_score'), 'innovation increment change=', delta(row, 'state_innovation_mean'), 'status=', row.get('comparison', {}).get('status', 'UNDEFINED'))
print('All 11-window q values and their deltas are retained in result.json fields cases.*.trajectories.*.layers.*.measurements.*.q_by_window, marked_off_increments, and adjacent_increment_changes.')
print('top-level retained failures:', result.get('failures', []))
print('full result:', result_path)
"""
    cells = [
        code("from google.colab import drive\ndrive.mount('/content/drive')"),
        markdown(
            "# Media-Channel-Retention-V1 fixed diagnostic\n\n"
            "Run all once. It reads exactly the retained Window-State-MSE-V1 evaluation artifacts, performs 20 new VAE encodes, and records all fixed success/failure slots. It does not generate, decode a latent video, create/decode/re-encode an MP4, add an attack, calibrate, tune, or run a deployment receiver. Reading MP4 bytes is limited to SHA-256 binding. Process completion is not a method PASS."
        ),
        code(setup), code(logging + install), code(checkout), code(cuda), code(run), code(results),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = "media-channel-retention-" + str(index)
    notebook = {
        "cells": cells,
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "Media-Channel-Retention-V1", "provenance": []},
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
