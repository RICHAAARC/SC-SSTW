"""Build one immutable-source Run-all Colab notebook after S2 is published."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "notebooks/rgb_dct_t49_vae_lift_v1_colab.ipynb"


def _code(source: str, index: int) -> dict:
    return dict(cell_type="code", execution_count=None, metadata={}, outputs=[],
                source=source.splitlines(keepends=True), id=f"rgb-dct-t49-{index}")


def build(source_sha: str, output: str | Path | None = None) -> Path:
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("source_sha must be one lowercase 40-hex Git commit")
    output = DEFAULT_OUTPUT if output is None else Path(output)
    golden = json.loads((ROOT / "notebooks/integrated_payload_v1_colab.ipynb").read_text())
    setup = """from pathlib import Path
import datetime, json, sys
SOURCE_SHA = SOURCE_SHA_TOKEN
OUTPUT_PARENT = Path('/content/drive/MyDrive/Video-WM/RGB-DCT-T49-VAE-Lift-V1')
OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUTPUT = OUTPUT_PARENT / stamp
OUTPUT.mkdir(exist_ok=False)
(OUTPUT / 'setup_receipt.json').write_text(json.dumps(dict(status='SETUP_STARTED', source_sha=SOURCE_SHA, output_dir=str(OUTPUT), python=sys.version, executable=sys.executable), indent=2) + '\\n', encoding='utf-8')
print('fresh output:', OUTPUT, flush=True)
""".replace("SOURCE_SHA_TOKEN", repr(source_sha))
    logging = """import subprocess, sys, json
SETUP_LOG = OUTPUT / 'setup.log'
def logged_run(command, *, cwd=None, env=None, check=True):
    with SETUP_LOG.open('a', encoding='utf-8') as log:
        line = 'COMMAND ' + repr(command) + '\\n'
        print(line, end='', flush=True); log.write(line); log.flush()
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in child.stdout:
            print(line, end='', flush=True); log.write(line); log.flush()
        returncode = child.wait()
        log.write('EXIT ' + str(returncode) + '\\n'); log.flush()
    if check and returncode:
        (OUTPUT / 'setup_failure.json').write_text(json.dumps(dict(command=command, returncode=returncode), indent=2) + '\\n', encoding='utf-8')
        raise subprocess.CalledProcessError(returncode, command)
    return subprocess.CompletedProcess(command, returncode)
"""
    install = "".join(golden["cells"][2]["source"]).replace("subprocess.run(", "logged_run(")
    checkout = """REPO = Path('/content/SC-SSTW-RGB-DCT-T49-' + stamp)
logged_run(['git', 'clone', '--filter=blob:none', 'https://github.com/RICHAAARC/SC-SSTW.git', str(REPO)])
logged_run(['git', '-C', str(REPO), 'fetch', 'origin', 'dev/rgb-dct-temporal-balanced-v1'])
logged_run(['git', '-C', str(REPO), 'checkout', '--detach', SOURCE_SHA])
actual = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
if actual != SOURCE_SHA:
    raise RuntimeError('immutable source SHA readback mismatch')
(OUTPUT / 'source_receipt.json').write_text(json.dumps(dict(expected_sha=SOURCE_SHA, actual_sha=actual, repo=str(REPO)), indent=2) + '\\n', encoding='utf-8')
print('source commit:', actual, flush=True)
"""
    environment = """import importlib.metadata, shutil, torch, numpy, diffusers
environment_receipt = dict(
    ffmpeg=shutil.which('ffmpeg'), ffprobe=shutil.which('ffprobe'),
    torch=torch.__version__, numpy=numpy.__version__, diffusers=diffusers.__version__,
    cuda_available=torch.cuda.is_available(),
    device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
)
(OUTPUT / 'environment_receipt.json').write_text(json.dumps(environment_receipt, indent=2) + '\\n', encoding='utf-8')
if not environment_receipt['ffmpeg'] or not environment_receipt['ffprobe'] or not environment_receipt['cuda_available']:
    raise RuntimeError('ffmpeg, ffprobe, and a CUDA runtime are required for this fixed Wan run')
(OUTPUT / 'setup_receipt.json').write_text(json.dumps(dict(status='SETUP_COMPLETE', source_sha=SOURCE_SHA, output_dir=str(OUTPUT), python=sys.version, executable=sys.executable), indent=2) + '\\n', encoding='utf-8')
print('fixed Wan environment:', environment_receipt, flush=True)
"""
    run = """import os
env = os.environ.copy()
env['PYTHONUNBUFFERED'] = '1'
command = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.rgb_dct_t49_vae_lift_run', '--output', str(OUTPUT)]
completed = logged_run(command, cwd=REPO, env=env, check=False)
RESULT_PATH = OUTPUT / 'result.json'
(OUTPUT / 'execution_receipt.json').write_text(json.dumps(dict(command=command, returncode=completed.returncode, result_path=str(RESULT_PATH)), indent=2) + '\\n', encoding='utf-8')
if not RESULT_PATH.exists():
    raise FileNotFoundError('runner produced no retained result.json')
"""
    results = """result = json.loads(RESULT_PATH.read_text(encoding='utf-8'))
if result['fixed_denominator'] != {'sources': 2, 'mp4_score_slots': 8, 'frames': 1448}:
    raise RuntimeError('fixed denominator mismatch')
print('status:', result['status'], flush=True)
print('attempted/scored/invalid/pending:', result['attempted_media_slots'], result['scored_media_slots'], result['invalid_media_slots'], result['pending_media_slots'], flush=True)
for case_id, case in result['cases'].items():
    for arm, row in case['slots'].items():
        print(case_id, arm, row['status'], row['score'], row['reason'], flush=True)
print('full result:', RESULT_PATH, flush=True)
"""
    cells = [
        _code("from google.colab import drive\ndrive.mount('/content/drive')", 0),
        dict(cell_type="markdown", metadata={}, id="rgb-dct-t49-purpose", source=[
            "# RGB-DCT fixed T49 VAE-lift development trial\n", "\n",
            "Run all once. Two fixed new prompt/seed pairs produce eight planned MP4 slots. The direct RGB +/- controls gate one positive VAE-lift T49 native arm per source. Continuous actual-MP4 scores are persisted before same-source comparisons. Zero lift/response are method negatives. This is development controllability only: no binary detection, low FPR, quality, payload, attribution, or multi-step gain claim. The user performs the GPU/Colab run.\n",
        ]),
        _code(setup, 1), _code(logging + install, 2), _code(checkout, 3),
        _code(environment, 4), _code(run, 5), _code(results, 6),
    ]
    notebook = dict(
        cells=cells, nbformat=4, nbformat_minor=5,
        metadata=dict(
            accelerator="GPU",
            colab={"name": "RGB-DCT fixed T49 VAE-lift development", "provenance": []},
            kernelspec={"display_name": "Python 3", "language": "python", "name": "python3"},
            language_info={"name": "python"},
            source_commit=source_sha,
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
