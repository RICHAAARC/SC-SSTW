"""Build the fixed, immutable-source CPU Colab notebook after source publication."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "notebooks/rgb_dct_off_diagnostic_v1_colab.ipynb"


def _code(source: str, index: int) -> dict:
    return dict(
        cell_type="code", execution_count=None, metadata={}, outputs=[],
        source=source.splitlines(keepends=True), id=f"rgb-dct-off-{index}",
    )


def build(source_sha: str, output: str | Path | None = None) -> Path:
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("source_sha must be one lowercase 40-hex Git commit")
    output = DEFAULT_OUTPUT if output is None else Path(output)
    setup = """from pathlib import Path
import datetime, json, sys
SOURCE_SHA = SOURCE_SHA_TOKEN
INPUT_ROOT = Path('/content/drive/MyDrive/Video-WM/Content-Background-Existence-V1/content_background_existence_v1_20260923T024543022721Z')
OUTPUT_PARENT = Path('/content/drive/MyDrive/Video-WM/RGB-DCT-Receiver-Diagnostic-V1')
OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUTPUT = OUTPUT_PARENT / stamp
OUTPUT.mkdir(exist_ok=False)
(OUTPUT / 'setup_receipt.json').write_text(json.dumps(dict(status='SETUP_STARTED', source_sha=SOURCE_SHA, input_root=str(INPUT_ROOT), output_dir=str(OUTPUT), python=sys.version, executable=sys.executable), indent=2) + '\\n', encoding='utf-8')
print('fixed input:', INPUT_ROOT, flush=True)
print('fresh output:', OUTPUT, flush=True)
""".replace("SOURCE_SHA_TOKEN", repr(source_sha))
    logging = """import subprocess, sys, json
SETUP_LOG = OUTPUT / 'setup.log'
def logged_run(command, *, cwd=None, check=True):
    with SETUP_LOG.open('a', encoding='utf-8') as log:
        line = 'COMMAND ' + repr(command) + '\\n'
        print(line, end='', flush=True); log.write(line); log.flush()
        child = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in child.stdout:
            print(line, end='', flush=True); log.write(line); log.flush()
        returncode = child.wait()
        log.write('EXIT ' + str(returncode) + '\\n'); log.flush()
    if check and returncode:
        (OUTPUT / 'setup_failure.json').write_text(json.dumps(dict(command=command, returncode=returncode), indent=2) + '\\n', encoding='utf-8')
        raise subprocess.CalledProcessError(returncode, command)
    return subprocess.CompletedProcess(command, returncode)
"""
    checkout = """REPO = Path('/content/SC-SSTW-RGB-DCT-' + stamp)
logged_run(['git', 'clone', '--filter=blob:none', 'https://github.com/RICHAAARC/SC-SSTW.git', str(REPO)])
logged_run(['git', '-C', str(REPO), 'fetch', 'origin', 'dev/rgb-dct-temporal-balanced-v1'])
logged_run(['git', '-C', str(REPO), 'checkout', '--detach', SOURCE_SHA])
actual = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
if actual != SOURCE_SHA:
    raise RuntimeError('immutable source SHA readback mismatch')
(OUTPUT / 'source_receipt.json').write_text(json.dumps(dict(expected_sha=SOURCE_SHA, actual_sha=actual, repo=str(REPO)), indent=2) + '\\n', encoding='utf-8')
print('source commit:', actual, flush=True)
"""
    environment = """import shutil, numpy, torch
cpu_probe = torch.tensor([1.0], device='cpu')
environment_receipt = dict(ffmpeg=shutil.which('ffmpeg'), ffprobe=shutil.which('ffprobe'), numpy=numpy.__version__, torch=torch.__version__, torch_cpu_available=(cpu_probe.item() == 1.0))
(OUTPUT / 'environment_receipt.json').write_text(json.dumps(environment_receipt, indent=2) + '\\n', encoding='utf-8')
if not environment_receipt['ffmpeg'] or not environment_receipt['ffprobe'] or not environment_receipt['torch_cpu_available']:
    raise RuntimeError('ffmpeg, ffprobe, and torch CPU are required for fixed RGB24 MP4 readback')
(OUTPUT / 'setup_receipt.json').write_text(json.dumps(dict(status='SETUP_COMPLETE', source_sha=SOURCE_SHA, input_root=str(INPUT_ROOT), output_dir=str(OUTPUT), python=sys.version, executable=sys.executable), indent=2) + '\\n', encoding='utf-8')
print('CPU readback environment:', environment_receipt, flush=True)
"""
    run = """command = [sys.executable, '-u', '-m', 'scripts.rgb_dct_off_diagnostic', '--run-root', str(INPUT_ROOT), '--output', str(OUTPUT)]
completed = logged_run(command, cwd=REPO, check=False)
RESULT_PATH = OUTPUT / 'result.json'
(OUTPUT / 'execution_receipt.json').write_text(json.dumps(dict(command=command, returncode=completed.returncode, result_path=str(RESULT_PATH)), indent=2) + '\\n', encoding='utf-8')
if not RESULT_PATH.exists():
    raise FileNotFoundError('runner produced no retained result.json')
"""
    results = """result = json.loads(RESULT_PATH.read_text(encoding='utf-8'))
if result['fixed_denominator'] != 8 or len(result['rows']) != 8:
    raise RuntimeError('fixed eight-row result missing')
print('scored:', result['scored_count'], 'invalid:', result['invalid_count'], 'pending:', result['pending_count'], flush=True)
print('full result:', RESULT_PATH, flush=True)
"""
    cells = [
        _code("from google.colab import drive\ndrive.mount('/content/drive')", 0),
        dict(cell_type="markdown", metadata={}, id="rgb-dct-off-purpose", source=[
            "# RGB-DCT receiver: fixed old-OFF development diagnostic\n",
            "\n",
            "Run all once. This only reads eight already seen historical OFF FULL.mp4 files, checks each byte SHA-256, and applies the new uncalibrated receiver to matching files. Historical calibration/evaluation labels are report labels, not new calibration or independent evaluation. Old marked videos are not H1 for this receiver. There is no threshold, PASS, FPR, or TPR conclusion. The user runs this CPU notebook; it does not install model packages or run a model.\n",
        ]),
        _code(setup, 1), _code(logging, 2), _code(checkout, 3),
        _code(environment, 4), _code(run, 5), _code(results, 6),
    ]
    notebook = dict(
        cells=cells, nbformat=4, nbformat_minor=5,
        metadata=dict(
            colab={"name": "RGB-DCT old-OFF development diagnostic", "provenance": []},
            kernelspec={"display_name": "Python 3", "language": "python", "name": "python3"},
            language_info={"name": "python"}, source_commit=source_sha,
            notebook_binding_kind="immutable_source_commit",
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_sha")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(build(args.source_sha, args.output))
