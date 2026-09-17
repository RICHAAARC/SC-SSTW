"""Build the fixed saved-prefix clipped-margin user-run notebook."""
from pathlib import Path
import argparse
import json
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def build(source_commit=None):
    if source_commit is not None:
        if not re.fullmatch('[0-9a-f]{40}',source_commit):raise ValueError('full source SHA required')
        subprocess.run(['git','-C',str(ROOT),'cat-file','-e',source_commit+':experiments/wan_state_clock/clipped_margin_run.py'],check=True)
    cells=[]
    def cell(kind,name,text):
        row={'cell_type':kind,'id':name,'metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code':row.update(execution_count=None,outputs=[])
        cells.append(row)
    cell('code','drive-mount',"from google.colab import drive\ndrive.mount('/content/drive')\n")
    cell('markdown','scope',"""# Saved-prefix clipped matched margin comparison

Run all executes **develop -> media -> report** on the original four development
cases and rho [.1,.3,1], using the saved latent and full UniPC history. The new
loss is negative nominal clipped matched margin; the original unclipped selector,
maximum two media candidates, budgets, and five-mode blind reader stay fixed.
No holdout, old hinge, prefix, or old media is regenerated.

Fixed read-only inputs under `MyDrive/Video-WM`:
- `VelocityCalibration/velocity_calibration_20260917T011448482341Z`
- `VelocityZeroPath/velocity_zero_path_20260917T092208999384Z`

New outputs, child logs, source archive and launcher log use
`MyDrive/Video-WM/VelocityClippedMargin`. All 24 new and 24 original terminal
rows remain in comparison.json. Missing/failed/old-not-run/new-not-selected media
remain explicit. Original NO_SELECTION is preserved. No candidate means no new
media calls; reporting still runs. A failed child is retained and later cases
continue; launcher errors are collected across stages, then shown after reporting.

The saved zero comparisons label historical controls MATCHED/NONMATCHING/UNVERIFIED.
Nonmatching/unverified selection is PROVISIONAL with raw_selector_result retained;
zero equality is not proof of original model or embedding identity. Original
conditioning is re-encoded. This objective aligns nominal matched statistics,
not observer penalties or the full blind search.

Maximum new development calls: 384 ordinary Transformer forwards,192 live steps,
192 shadow steps,8 backwards; checkpoint replay is counted separately. At most
16 new VAE decodes/MP4 saves and64 encodes. No GPU model name is required.
Original torch2.11.0+cu128/diffusers0.40.0 versions are checked in fresh child
processes; matching torch is not reinstalled. This published handoff has not been
executed by the assistant on models/GPU/Colab/Drive; CPU checks establish only
implementation behavior. Source SHA: """+str(source_commit)+""".
""")
    cell('code','source-and-software',f"""from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata, json, os, signal, subprocess, sys
SOURCE_COMMIT = {source_commit!r}
if SOURCE_COMMIT is None:
    raise RuntimeError('Pending source publication: rebuild with the published full SHA')
SOURCE_URL = 'https://github.com/RICHAAARC/SC-SSTW.git'
RUN_ID = 'velocity_clipped_margin_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
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
    cell('code','fixed-development-stages',"""INPUT_RUN = Path('/content/drive/MyDrive/Video-WM/VelocityCalibration/velocity_calibration_20260917T011448482341Z')
ZERO_RUN = Path('/content/drive/MyDrive/Video-WM/VelocityZeroPath/velocity_zero_path_20260917T092208999384Z')
OUTPUT = Path('/content/drive/MyDrive/Video-WM/VelocityClippedMargin') / RUN_ID
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
ARCHIVE = OUTPUT.parent / (RUN_ID + '.source.zip')
subprocess.run(['git', '-C', str(SOURCE), 'archive', '--format=zip', '--output', str(ARCHIVE), SOURCE_COMMIT], check=True)
LOG = OUTPUT.parent / (RUN_ID + '.launcher.log')
stage_codes = {}
with LOG.open('w') as log:
    for stage in ('develop', 'media', 'report'):
        command = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.clipped_margin_run', '--input-run', str(INPUT_RUN), '--zero-run', str(ZERO_RUN), '--output', str(OUTPUT), '--stage', stage]
        banner = 'Stage: ' + stage + '\\n'
        print(banner, end=''); log.write(banner); log.flush()
        process = subprocess.Popen(command, cwd=SOURCE, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end=''); log.write(line); log.flush()
            stage_codes[stage] = process.wait()
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM); process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            except ProcessLookupError:
                pass
            raise
        log.write('Exit code: ' + str(stage_codes[stage]) + '\\n'); log.flush()
print('Output:', OUTPUT, 'Launcher log:', LOG, 'Source archive:', ARCHIVE)
print('Stage exit codes:', stage_codes)
for name in ('terminal.json', 'media.json', 'terminal_selection.json', 'selection.json', 'comparison.json'):
    path = OUTPUT / name
    if path.exists():
        value = json.loads(path.read_text())
        summary = {k: value[k] for k in ('status', 'rho', 'selected_rhos', 'historical_control_comparability', 'paired_method_comparison_valid', 'flow_denominator_per_method') if k in value}
        if name in ('terminal.json', 'media.json'):
            summary['cases'] = {k: v['status'] for k, v in value['cases'].items()}
        if name == 'comparison.json':
            summary['original_selection'] = value['original_selection'].get('value', {}).get('status')
        print(name, json.dumps(summary, indent=2))
    else:
        print(name, 'NOT_RUN_OR_MISSING; inspect retained logs')
if any(stage_codes.values()):
    raise RuntimeError('One or more stages failed; all available fixed-denominator records and logs are retained: ' + str(stage_codes))
""")
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    target=ROOT/'notebooks/clipped_margin_colab.ipynb';target.write_text(json.dumps(notebook,indent=1)+'\n');return target


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-commit')
    print(build(parser.parse_args().source_commit))
