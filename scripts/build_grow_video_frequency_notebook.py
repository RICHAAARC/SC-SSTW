"""User-run fixed twelve-video GROW-style frequency candidate; no execution here."""
from pathlib import Path
import argparse
import json
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def build(source_commit=None):
    if source_commit is not None:
        if not re.fullmatch('[0-9a-f]{40}',source_commit):raise ValueError('full source SHA required')
        subprocess.run(['git','-C',str(ROOT),'cat-file','-e',source_commit+':experiments/wan_state_clock/grow_frequency_run.py'],check=True)
    cells=[]
    def cell(kind,name,text):
        row={'cell_type':kind,'id':name,'metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code':row.update(execution_count=None,outputs=[])
        cells.append(row)
    cell('code','drive-mount',"from google.colab import drive\ndrive.mount('/content/drive')\n")
    cell('markdown','scope',"""# GROW-style video frequency adaptation: fixed first validation

Run all generates the frozen 2 contents x 2 seeds x OFF/A/B =12 videos and
48 measurement layers. No mode selection, parameter scan, holdout or old-run
inputs are needed. All cases, arms, layers and46 latent slices retain failures.
New outputs, source archive and launcher/child logs persist under
`MyDrive/Video-WM/GROWVideoFrequency/grow_video_frequency_<UTC>`.

The fixed method uses channel0 whole-slice orthonormal spatial DCT-II,
frequencies2..9 on both axes,64 selected coefficients,16 payload bits repeated
four times per slice and across all46 slices. Primary readout is184 coefficient
sign votes/bit; zero/ties are erasures. Coefficient means are diagnostics only.
Target amplitude.5, eta.1, half-squared-error sum; local predicted-clean guidance
at explicit zero-based steps10..29 after CFG5. Native UniPC updates once per step,
with no Transformer backward. This is an explicit video adaptation, not official
GROW code or a resolution of the paper's ambiguous guidance-interval wording.

The same decoded output supplies floatRGB-to-VAE, RGB8-to-VAE and
CRF18 yuv420p MP4-to-VAE readouts alongside terminal latent readout. Execution
completeness, payload recovery and same-layer OFF quality are reported separately;
there is no invented quality threshold or scientific PASS. A failed case does not
remove other cases or shorten the fixed denominator; inspect result.json/logs.

Maximum calls:1200 Transformer forwards,600 native scheduler steps,160 local
DCT gradients,12 VAE decodes,12 MP4 saves and36 VAE encodes. GPU feasibility and
real recovery/quality remain to be measured by your run. The assistant has not
executed this notebook or pretrained models/GPU/Drive experiments.

The previously used torch2.11.0+cu128/diffusers0.40.0 environment is restored if
needed and checked in a fresh subprocess. Matching torch is not reinstalled.
Model loading uses the existing Wan1.3B loader; no GPU-model-name restriction.
Source SHA: """+str(source_commit)+""".
""")
    cell('code','source-and-software',f"""from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata, json, os, signal, subprocess, sys
SOURCE_COMMIT = {source_commit!r}
if SOURCE_COMMIT is None:
    raise RuntimeError('Pending source publication: rebuild with the published full SHA')
SOURCE_URL = 'https://github.com/RICHAAARC/SC-SSTW.git'
RUN_ID = 'grow_video_frequency_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
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
    cell('code','fixed-twelve-videos',"""OUTPUT = Path('/content/drive/MyDrive/Video-WM/GROWVideoFrequency') / RUN_ID
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
ARCHIVE = OUTPUT.parent / (RUN_ID + '.source.zip')
subprocess.run(['git', '-C', str(SOURCE), 'archive', '--format=zip', '--output', str(ARCHIVE), SOURCE_COMMIT], check=True)
LOG = OUTPUT.parent / (RUN_ID + '.launcher.log')
command = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.grow_frequency_run', '--output', str(OUTPUT)]
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
    print(json.dumps({'status': result['status'], 'video_denominator': result['video_denominator'], 'layer_denominator': result['layer_denominator'], 'fixed_calls': result['fixed_calls'], 'actual_calls_observed': result.get('actual_calls_observed'), 'call_count_case_coverage': result.get('call_count_case_coverage'), 'cases': {k: v['status'] for k, v in result['cases'].items()}}, indent=2))
if code:
    raise subprocess.CalledProcessError(code, command)
""")
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    target=ROOT/'notebooks/grow_video_frequency_colab.ipynb';target.write_text(json.dumps(notebook,indent=1)+'\n');return target


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-commit')
    print(build(parser.parse_args().source_commit))
