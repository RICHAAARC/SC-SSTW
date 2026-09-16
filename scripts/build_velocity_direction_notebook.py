"""Self-contained user-run direction notebook. Does not run models."""
from pathlib import Path
import base64
import io
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def build():
    config=json.loads((ROOT/'experiments/wan_state_clock/configs/generate_replication.json').read_text())
    config['source_snapshot']='velocity-coefficient direction source bundled in this notebook; baseline 5736501151a0e454857e4e503e7008b6b6b27467'
    config['generation']['role']='shared_0_43_then_two_zero_and_four_signed_real_terminal_tails'
    config['output_drive_parent']='/content/drive/MyDrive/Video-WM/VelocityDirection'
    memory=io.BytesIO()
    with zipfile.ZipFile(memory,'w',zipfile.ZIP_DEFLATED) as archive:
        for folder in ('main','runtime','experiments/wan_state_clock'):
            for path in sorted((ROOT/folder).rglob('*.py')):
                info=zipfile.ZipInfo(str(path.relative_to(ROOT)))
                info.compress_type=zipfile.ZIP_DEFLATED
                archive.writestr(info,path.read_bytes())
        archive.writestr(zipfile.ZipInfo('experiments/wan_state_clock/configs/velocity_direction.json'),json.dumps(config,indent=2))
    payload=base64.b64encode(memory.getvalue()).decode()
    cells=[]
    def cell(kind,name,text):
        row=dict(cell_type=kind,id=name,metadata={},source=text.splitlines(keepends=True))
        if kind=='code':row.update(execution_count=None,outputs=[])
        cells.append(row)
    cell('code','drive-mount',"from google.colab import drive\ndrive.mount('/content/drive')\n")
    cell('markdown','scope', '''# Real-terminal velocity coefficient direction

Self-contained engineering handoff with its complete source. Open this notebook
in Colab and Run all. It does not fetch old published code as if it contained
the new differentiable implementation. No GPU/model execution has been performed
for this handoff. The original prompt, seed, key and 50-step schedule are fixed.

Run all creates one prefix (0–43), then ZERO_A, ZERO_B, PLUS_A, MINUS_A, PLUS_B,
MINUS_B, each with the complete real 44–49 Transformer/CFG/UniPC continuation.
The two zero paths each compute one coefficient-space terminal-loss gradient.
The unit negative-gradient direction uses a single symmetric amplitude derived
from same-run OFF terminal-write budget R and actual UniPC response coefficients.
The fixed probe fraction rho=0.1 (with 0.1% numerical reserve) is an engineering
convention, not a guarantee of locality. No search, clipping or automatic retries.
All six rows, including failures, remain in the results. BF16 AD and finite
differences are reported separately; a sign match is only local direction evidence.

Complete cost: 88 prefix + 72 tail Transformer forwards; two backwards; pure
Transformer checkpoint replays counted separately (up to 20 invocations, possibly
early-stopped); 80 live scheduler steps, 36 detached budget-shadow steps and six
scalar response-probe steps. No VAE loading, decode/encode, MP4, or VAE backward.
Actual memory peaks and elapsed time are measured by the user-run process.
''')
    cell('code','local-source',f'''from pathlib import Path
from datetime import datetime, timezone
import base64, io, json, os, signal, subprocess, sys, zipfile
RUN_ID = 'velocity_direction_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
SOURCE = Path('/content') / (RUN_ID + '_source')
SOURCE.mkdir(exist_ok=False)
PAYLOAD = {payload!r}
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as archive:
    archive.extractall(SOURCE)
subprocess.run([sys.executable, '-m', 'pip', 'install', 'diffusers', 'transformers', 'accelerate', 'ftfy', 'sentencepiece', 'safetensors', 'huggingface_hub', 'numpy', 'Pillow'], check=True)
print('Bundled source snapshot:', SOURCE)
''')
    cell('code','fixed-six-tails','''OUTPUT = Path('/content/drive/MyDrive/Video-WM/VelocityDirection') / RUN_ID
BUNDLED_CONFIG = SOURCE / 'experiments/wan_state_clock/configs/velocity_direction.json'
LOG = OUTPUT.parent / f'{RUN_ID}.launcher.log'
LOG.parent.mkdir(parents=True, exist_ok=True)
SOURCE_ARCHIVE = OUTPUT.parent / f'{RUN_ID}.source.zip'
SOURCE_ARCHIVE.write_bytes(base64.b64decode(PAYLOAD))
config = json.loads(BUNDLED_CONFIG.read_text())
config['artifact_paths'] = {'launcher_log': str(LOG), 'source_archive': str(SOURCE_ARCHIVE), 'source_directory': str(SOURCE)}
CONFIG = Path('/content') / f'{RUN_ID}.config.json'
CONFIG.write_text(json.dumps(config, indent=2) + '\\n')
cmd = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.velocity_direction_run', '--config', str(CONFIG), '--output', str(OUTPUT)]
with LOG.open('w') as log:
    process = subprocess.Popen(cmd, cwd=SOURCE, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in process.stdout:
            print(line, end='')
            log.write(line)
            log.flush()
        code = process.wait()
    except BaseException:
        try:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        except ProcessLookupError:
            pass
        raise
print('Exit:', code, 'Results:', OUTPUT, 'Log:', LOG)
if (OUTPUT / 'result.json').exists():
    result = json.loads((OUTPUT / 'result.json').read_text())
    print(json.dumps({k: result.get(k) for k in ('status', 'actual_calls', 'R', 'directions', 'direction_comparisons', 'zero_repeat_floor', 'over_budget_conditions', 'response_check_failed_conditions', 'final_resources', 'failures')}, indent=2))
if code:
    raise subprocess.CalledProcessError(code, cmd)
''')
    value=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},nbformat=4,nbformat_minor=5)
    (ROOT/'notebooks/velocity_direction_colab.ipynb').write_text(json.dumps(value,indent=1)+'\n')


if __name__=='__main__':build()
