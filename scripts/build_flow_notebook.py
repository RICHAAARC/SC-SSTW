"""Build the Run-all notebook bound to the published immutable source commit."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = 'fca6f1f4a447da8f3b725425f0db960541cb6a74'
SOURCE_URL = 'https://github.com/RICHAAARC/SC-SSTW.git'


def build():
    cells = []
    def code(name, source):
        cells.append(dict(cell_type='code', id=name, execution_count=None, metadata={}, outputs=[], source=source.splitlines(keepends=True)))
    def markdown(name, source):
        cells.append(dict(cell_type='markdown', id=name, metadata={}, source=source.splitlines(keepends=True)))
    code('drive-mount', "from google.colab import drive\ndrive.mount('/content/drive')\n")
    markdown('scope', '# Flow tube-state: fixed five conditions\n\nPublished source handoff, not an executed result. Run all runs OFF, TERMINAL_A/B, FLOW_A/B at the fixed prompt and seed. This notebook fetches the complete immutable GitHub source commit shown below, independently of the current branch tip.\n\n50 steps; guidance only at indices 44/45/46, then normal 47/48/49. No model or VAE backward. R is fixed by same-run terminal A/B before Flow. Fixed budget: 124 Transformer forwards, 62 real and 12 shadow scheduler steps, 5 VAE decodes, 5 normal MP4 saves, 20 receiver encodes. Failures stay in all five rows. First-round saved RGB and residual temporal MSE limits are 1.5 times the corresponding terminal reference; these are not perceptual thresholds.\n')
    code('source', f'''from pathlib import Path
from datetime import datetime, timezone
import json, os, signal, subprocess, sys
SOURCE_URL = {SOURCE_URL!r}
SOURCE_COMMIT = {SOURCE_COMMIT!r}
RUN_ID = 'flow_tube_state_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
SOURCE = Path('/content') / (RUN_ID + '_source')
SOURCE.mkdir(exist_ok=False)
subprocess.run(['git', 'init', str(SOURCE)], check=True)
subprocess.run(['git', '-C', str(SOURCE), 'remote', 'add', 'origin', SOURCE_URL], check=True)
subprocess.run(['git', '-C', str(SOURCE), 'fetch', '--depth', '1', 'origin', SOURCE_COMMIT], check=True)
subprocess.run(['git', '-C', str(SOURCE), 'checkout', '--detach', SOURCE_COMMIT], check=True)
actual_commit = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
if actual_commit != SOURCE_COMMIT:
    raise RuntimeError('Fetched source does not match the fixed source commit')
subprocess.run([sys.executable, '-m', 'pip', 'install', 'diffusers', 'transformers', 'accelerate', 'ftfy', 'sentencepiece', 'safetensors', 'huggingface_hub', 'numpy', 'Pillow'], check=True)
subprocess.run(['ffmpeg', '-version'], check=True)
subprocess.run(['ffprobe', '-version'], check=True)
print('Published source:', actual_commit, SOURCE)
''')
    code('fixed-run', '''OUTPUT = Path('/content/drive/MyDrive/Video-WM/FlowTubeState') / RUN_ID
config = json.loads((SOURCE / 'experiments/wan_state_clock/configs/generate_replication.json').read_text())
config['source_snapshot'] = 'published GitHub source ' + SOURCE_COMMIT
config['generation']['role'] = 'shared_prefix_0_43_then_off_flow_a_flow_b'
CONFIG = Path('/content') / f'{RUN_ID}.config.json'
CONFIG.write_text(json.dumps(config, indent=2) + '\\n')
LOG = OUTPUT.parent / f'{RUN_ID}.launcher.log'
LOG.parent.mkdir(parents=True, exist_ok=True)
# Retain the exact published source beside the user-run results.
SOURCE_ARCHIVE = OUTPUT.parent / f'{RUN_ID}.source.zip'
subprocess.run(['git', '-C', str(SOURCE), 'archive', '--format=zip', '--output', str(SOURCE_ARCHIVE), SOURCE_COMMIT], check=True)
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
