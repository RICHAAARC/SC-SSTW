"""Build local calibration handoff; bind an actually published source at release."""
from pathlib import Path
import argparse
import json
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def build(source_commit=None):
    if source_commit is not None:
        if not re.fullmatch('[0-9a-f]{40}',source_commit):raise ValueError('full source SHA required')
        for path in ('experiments/wan_state_clock/calibration_run.py','experiments/wan_state_clock/configs/velocity_calibration.json'):
            subprocess.run(['git','-C',str(ROOT),'cat-file','-e',source_commit+':'+path],check=True)
    cells=[]
    def cell(kind,name,text):
        value={'cell_type':kind,'id':name,'metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code':value.update(execution_count=None,outputs=[])
        cells.append(value)
    cell('code','drive-mount',"from google.colab import drive\ndrive.mount('/content/drive')\n")
    state='LOCAL PENDING PUBLICATION: source SHA is intentionally unset; this copy cannot launch the experiment.' if source_commit is None else 'Published source binding: '+source_commit
    cell('markdown','protocol',"""# Fixed development strength calibration

"""+state+"""

After source publication and SHA binding, Run all executes the frozen stages:
4 development cases (2 prompts × 2 seeds), rho [0.1, 0.3, 1.0]; at most 2 global
candidates enter saved MP4 readout; one development-selected rho is evaluated on
2 independent new prompt/seed cases. A missing/failed row stays in its denominator.
No eligible media candidate means NO_SELECTION and NOT_RUN_NO_SELECTION holdout.

Each case/message computes its zero-control gradient once and freezes q for all
listed strengths, epsilon=rho*0.999*the unchanged OFF-derived cap. Active steps
44/45/46 and full tail47..49, 5280 coefficients, codebook, loss, nested checkpoint,
BF16 CFG/FP32 controls and original five-mode blind reader are unchanged.
Terminal ranking uses worst margin, mean margin, then lower rho over all 8 rows;
negative margin is only a low-evidence candidate. Media selection requires all
fixed rows' five-mode unique correct readout and both saved RGB quality metrics
within1.5 times same-case/message terminal control; ties use worst then mean
quality ratio then lower rho. The tolerance is not a validated perception threshold.
Holdout never selects or tunes parameters; rank is not FPR.

Maximum ordinary calls: development736 Transformer + holdout272 =1008; 12
backwards, up to120 outer replays, block units separately counted. With2 media
candidates: at most38 VAE decodes,38 CRF18 yuv420p MP4 saves,152 four-origin VAE
encodes. Existing saved terminal latents feed media without regenerating tails.
Cases run in separate child processes; full runtime memory/timing is measured
only by the user run. This handoff has not executed pretrained models/GPU/Drive.
""")
    cell('code','source',f"""from pathlib import Path
from datetime import datetime, timezone
import json, os, signal, subprocess, sys
SOURCE_COMMIT = {source_commit!r}
if SOURCE_COMMIT is None:
    raise RuntimeError('Local unpublished calibration notebook: publish source and rebuild with its full SHA before Run all.')
SOURCE_URL = 'https://github.com/RICHAAARC/SC-SSTW.git'
RUN_ID = 'velocity_calibration_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
SOURCE = Path('/content') / (RUN_ID + '_source')
subprocess.run(['git', 'init', str(SOURCE)], check=True)
subprocess.run(['git', '-C', str(SOURCE), 'remote', 'add', 'origin', SOURCE_URL], check=True)
subprocess.run(['git', '-C', str(SOURCE), 'fetch', '--depth', '1', 'origin', SOURCE_COMMIT], check=True)
subprocess.run(['git', '-C', str(SOURCE), 'checkout', '--detach', SOURCE_COMMIT], check=True)
ACTUAL_SOURCE = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
if ACTUAL_SOURCE != SOURCE_COMMIT:
    raise RuntimeError('Source checkout differs from pinned SHA')
subprocess.run([sys.executable, '-m', 'pip', 'install', 'diffusers', 'transformers', 'accelerate', 'ftfy', 'sentencepiece', 'safetensors', 'huggingface_hub', 'numpy', 'Pillow'], check=True)
""")
    cell('code','fixed-stages',"""OUTPUT = Path('/content/drive/MyDrive/Video-WM/VelocityCalibration') / RUN_ID
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
ARCHIVE = OUTPUT.parent / (RUN_ID + '.source.zip')
subprocess.run(['git', '-C', str(SOURCE), 'archive', '--format=zip', '--output', str(ARCHIVE), SOURCE_COMMIT], check=True)
manifest = json.loads((SOURCE / 'experiments/wan_state_clock/configs/velocity_calibration.json').read_text())
manifest['base_config']['source_snapshot'] = SOURCE_URL + ' at ' + ACTUAL_SOURCE
manifest['base_config']['artifact_paths'] = {'source_archive': str(ARCHIVE), 'source_directory': str(SOURCE)}
MANIFEST = Path('/content') / (RUN_ID + '.manifest.json')
MANIFEST.write_text(json.dumps(manifest, indent=2))
command = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.calibration_run', '--manifest', str(MANIFEST), '--output', str(OUTPUT), '--stage', 'all']
LOG = OUTPUT.parent / (RUN_ID + '.launcher.log')
with LOG.open('w') as log:
    process = subprocess.Popen(command, cwd=SOURCE, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in process.stdout:
            print(line, end=''); log.write(line); log.flush()
        code = process.wait()
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait()
        except ProcessLookupError:
            pass
        raise
if code:
    raise subprocess.CalledProcessError(code, command)
for name in ('terminal_selection.json', 'selection.json', 'holdout.json'):
    path = OUTPUT / name
    if path.exists():
        value = json.loads(path.read_text())
        print(name, value.get('status'), 'rho:', value.get('rho'))
print('All fixed-roster records and child logs:', OUTPUT)
""")
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    target=ROOT/'notebooks/velocity_calibration_colab.ipynb'
    target.write_text(json.dumps(notebook,indent=1)+'\n')
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-commit')
    print(build(parser.parse_args().source_commit))
