"""User-run four-case saved-prefix zero diagnostic; no model execution here."""
from pathlib import Path
import argparse
import json
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def build(source_commit=None):
    if source_commit is not None:
        if not re.fullmatch('[0-9a-f]{40}',source_commit):raise ValueError('full source SHA required')
        subprocess.run(['git','-C',str(ROOT),'cat-file','-e',source_commit+':experiments/wan_state_clock/zero_path_run.py'],check=True)
    cells=[]
    def cell(kind,name,text):
        row={'cell_type':kind,'id':name,'metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code':row.update(execution_count=None,outputs=[])
        cells.append(row)
    cell('code','drive-mount',"from google.colab import drive\ndrive.mount('/content/drive')\n")
    cell('markdown','scope',"""# Same-path no-grad zero from the original saved prefix

Fixed original run: `velocity_calibration_20260917T011448482341Z` under
`MyDrive/Video-WM/VelocityCalibration`. Its four original development cases each
receive exactly one shared A/B no-grad zero-coefficient tail44..49 from the saved
latent and complete UniPC state. Original data are read-only; new output is under
`MyDrive/Video-WM/VelocityZeroPath`. No prefix/gradient/strength scan/holdout/media.
48 Transformer calls,24 live scheduler steps,24 native shadow steps total;
0 backward,0 response probes,0 VAE,0 MP4. All four cases, failures and missing
original comparisons remain in the report. Source binding: """+str(source_commit)+""".

Run all uses the original torch2.11.0+cu128/diffusers0.40.0 software. Matching
torch is not reinstalled; mismatches are repaired before launching the children.
Fresh child processes verify the installed versions before launching the run.
No GPU model is required by name. The normal model/config/prompt encoding is
reused but its fresh initial noise is discarded; the prefix is never regenerated.

The snapshot's full scheduler history/order/cursor, tensor values and dtypes
are checked before continuation. Original prompt embeddings and resolved model
revision were not saved. Conditioning must be re-encoded, reported as
RESTORED_WITH_REENCODED_CONDITIONING, not a bitwise equivalence claim.
Z0-Zg differences cannot isolate grad/no-grad causality or establish a bug.

Report RMS of tensor differences Z0-ZgA/B and each saved ZrhoAB-Z0, original
loss/projection/competition statistics relative to Z0. Original absolute margins,
blind readouts and NO_SELECTION stay unchanged. No post-hoc PASS threshold.
There is no Z0 media zero baseline, so codec loss cannot be fully attributed.
This handoff has not executed models, GPU, Colab or Drive experiments.
""")
    cell('code','source-and-software',f"""from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata, json, os, signal, subprocess, sys
SOURCE_COMMIT = {source_commit!r}
if SOURCE_COMMIT is None:
    raise RuntimeError('Pending source publication: rebuild with the published full SHA')
SOURCE_URL = 'https://github.com/RICHAAARC/SC-SSTW.git'
RUN_ID = 'velocity_zero_path_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
SOURCE = Path('/content') / (RUN_ID + '_source')
subprocess.run(['git', 'init', str(SOURCE)], check=True)
subprocess.run(['git', '-C', str(SOURCE), 'remote', 'add', 'origin', SOURCE_URL], check=True)
subprocess.run(['git', '-C', str(SOURCE), 'fetch', '--depth', '1', 'origin', SOURCE_COMMIT], check=True)
subprocess.run(['git', '-C', str(SOURCE), 'checkout', '--detach', SOURCE_COMMIT], check=True)
if subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() != SOURCE_COMMIT:
    raise RuntimeError('Source checkout differs from pinned SHA')
def version(name):
    try: return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: return None
if version('torch') != '2.11.0+cu128':
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch==2.11.0', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cu128'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', 'diffusers==0.40.0', 'transformers', 'accelerate', 'ftfy', 'sentencepiece', 'safetensors', 'huggingface_hub', 'numpy', 'Pillow'], check=True)
subprocess.run([sys.executable, '-c', "import torch,diffusers; assert str(torch.__version__) == '2.11.0+cu128', torch.__version__; assert diffusers.__version__ == '0.40.0', diffusers.__version__"], check=True)
""")
    cell('code','four-original-cases',"""INPUT_RUN = Path('/content/drive/MyDrive/Video-WM/VelocityCalibration/velocity_calibration_20260917T011448482341Z')
OUTPUT = Path('/content/drive/MyDrive/Video-WM/VelocityZeroPath') / RUN_ID
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
ARCHIVE = OUTPUT.parent / (RUN_ID + '.source.zip')
subprocess.run(['git', '-C', str(SOURCE), 'archive', '--format=zip', '--output', str(ARCHIVE), SOURCE_COMMIT], check=True)
LOG = OUTPUT.parent / (RUN_ID + '.launcher.log')
command = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.zero_path_run', '--input-run', str(INPUT_RUN), '--output', str(OUTPUT)]
with LOG.open('w') as log:
    process = subprocess.Popen(command, cwd=SOURCE, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in process.stdout:
            print(line, end=''); log.write(line); log.flush()
        code = process.wait()
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM); process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait()
        except ProcessLookupError:
            pass
        raise
print('Output:', OUTPUT, 'Launcher log:', LOG, 'Source archive:', ARCHIVE)
if (OUTPUT / 'result.json').exists():
    result = json.loads((OUTPUT / 'result.json').read_text())
    print(json.dumps({'status': result['status'], 'fixed_total_calls': result['fixed_total_calls'], 'actual_calls': result.get('actual_calls'), 'cases': {k: v['status'] for k, v in result['cases'].items()}, 'original_selection': result['original_selection'].get('value', {}).get('status')}, indent=2))
if code:
    raise subprocess.CalledProcessError(code, command)
""")
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    target=ROOT/'notebooks/zero_path_colab.ipynb';target.write_text(json.dumps(notebook,indent=1)+'\n');return target


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-commit')
    print(build(parser.parse_args().source_commit))
