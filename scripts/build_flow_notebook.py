"""Build the local, self-contained Run-all handoff; no publishing or execution."""
from pathlib import Path
import base64
import io
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build():
    config = json.loads((ROOT/'experiments/wan_state_clock/configs/generate_replication.json').read_text())
    config['source_snapshot'] = 'unpublished Flow-Tube-State local source bundled in this notebook; base 3f0a5fafa7c2aa56fbc69bed17649accf49ae152'
    config['generation']['role'] = 'shared_prefix_0_43_then_off_flow_a_flow_b'
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for folder in ('main', 'runtime', 'experiments/wan_state_clock'):
            for path in sorted((ROOT/folder).rglob('*.py')):
                archive.writestr(str(path.relative_to(ROOT)), path.read_bytes())
        archive.writestr('experiments/wan_state_clock/configs/flow_replication.json', json.dumps(config, indent=2))
    payload = base64.b64encode(buffer.getvalue()).decode()
    cells = []
    def code(name, source):
        cells.append(dict(cell_type='code', id=name, execution_count=None, metadata={}, outputs=[], source=source.splitlines(keepends=True)))
    def markdown(name, source):
        cells.append(dict(cell_type='markdown', id=name, metadata={}, source=source.splitlines(keepends=True)))
    code('drive-mount', "from google.colab import drive\ndrive.mount('/content/drive')\n")
    markdown('scope', '# Flow tube-state: fixed five conditions\n\nLocal static handoff, not an executed result. Run all runs OFF, TERMINAL_A/B, FLOW_A/B at the fixed prompt and seed. This notebook contains its complete unpublished implementation snapshot, rather than fetching an old main commit. GitHub publication and binding to a new immutable release SHA remain a separate authorized action.\n\n50 steps; guidance only at indices 44/45/46, then normal 47/48/49. No model or VAE backward. R is fixed by same-run terminal A/B before Flow. Fixed budget: 124 Transformer forwards, 62 real and 12 shadow scheduler steps, 5 VAE decodes, 5 normal MP4 saves, 20 receiver encodes. Failures stay in all five rows. First-round saved RGB and residual temporal MSE limits are 1.5 times the corresponding terminal reference; these are not perceptual thresholds.\n')
    code('source', f'''from pathlib import Path
from datetime import datetime, timezone
import base64, io, json, os, signal, subprocess, sys, zipfile
RUN_ID = 'flow_tube_state_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
SOURCE = Path('/content') / (RUN_ID + '_source')
SOURCE.mkdir(exist_ok=False)
PAYLOAD = {payload!r}
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as archive:
    archive.extractall(SOURCE)
subprocess.run([sys.executable, '-m', 'pip', 'install', 'diffusers', 'transformers', 'accelerate', 'ftfy', 'sentencepiece', 'safetensors', 'huggingface_hub', 'numpy', 'Pillow'], check=True)
subprocess.run(['ffmpeg', '-version'], check=True)
subprocess.run(['ffprobe', '-version'], check=True)
print('Local bundled source:', SOURCE)
''')
    code('fixed-run', '''OUTPUT = Path('/content/drive/MyDrive/Video-WM/FlowTubeState') / RUN_ID
CONFIG = SOURCE / 'experiments/wan_state_clock/configs/flow_replication.json'
LOG = OUTPUT.parent / f'{RUN_ID}.launcher.log'
LOG.parent.mkdir(parents=True, exist_ok=True)
# Retain exact source beside the results without a published-source claim.
SOURCE_ARCHIVE = OUTPUT.parent / f'{RUN_ID}.source.zip'
SOURCE_ARCHIVE.write_bytes(base64.b64decode(PAYLOAD))
cmd = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.flow_run', '--config', str(CONFIG), '--output', str(OUTPUT)]
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
    print(json.dumps({k: result.get(k) for k in ('status', 'actual_calls', 'first_round_criteria_met', 'saved_quality_comparisons', 'resources', 'failures')}, indent=2))
if code:
    raise subprocess.CalledProcessError(code, cmd)
''')
    notebook = dict(cells=cells, metadata={'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'}, 'language_info':{'name':'python'}}, nbformat=4, nbformat_minor=5)
    (ROOT/'notebooks/flow_tube_state_colab.ipynb').write_text(json.dumps(notebook, indent=1)+'\n')


if __name__ == '__main__':
    build()
