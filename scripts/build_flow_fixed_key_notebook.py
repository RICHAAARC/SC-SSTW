"""Build the fixed development candidate notebook; GPU execution belongs to the user."""
import json
from pathlib import Path
SOURCE_SHA = '14596cf6301cd11242400ed617254ff3e537f65b'
ROOT = Path(__file__).parents[1]
OUTPUT = ROOT / 'notebooks/flow_fixed_key_v1_colab.ipynb'

def build():
    golden = json.loads((ROOT/'notebooks/integrated_payload_v1_colab.ipynb').read_text())
    def code(text): return dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=text.splitlines(keepends=True))
    def md(text): return dict(cell_type='markdown',metadata={},source=text.splitlines(keepends=True))
    setup = """from pathlib import Path
import datetime, json, sys
SOURCE_SHA = 'SOURCE_TOKEN'
DRIVE_ROOT = Path('/content/drive/MyDrive/Video-WM/Flow-Fixed-Key-V1')
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUTPUT = DRIVE_ROOT / ('flow_fixed_key_v1_' + stamp)
OUTPUT.mkdir(exist_ok=False)
(OUTPUT / 'setup_receipt.json').write_text(json.dumps(dict(source_commit=SOURCE_SHA, python=sys.version, executable=sys.executable, status='SETUP_STARTED'), indent=2))
print('same-run output:', OUTPUT, flush=True)
""".replace('SOURCE_TOKEN', SOURCE_SHA)
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
    install = ''.join(golden['cells'][2]['source']).replace('subprocess.run(', 'logged_run(')
    # Exact N5 dependency argv and fresh-process checks; add a structured receipt before assertions.
    receipt = """from pathlib import Path
info = dict(python=sys.version, executable=sys.executable, packages={})
for package in ('torch', 'torchvision', 'diffusers', 'transformers', 'accelerate', 'ftfy', 'sentencepiece', 'safetensors', 'huggingface_hub', 'numpy', 'Pillow'):
    try: info['packages'][package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError: info['packages'][package] = None
import json
Path(RECEIPT_PATH).write_text(json.dumps(info, indent=2))
"""
    install = install.replace('logged_run([sys.executable, "-u", "-c", check_code], check=True)', "check_code = check_code.replace(\"assert str(torch.__version__)\", " + repr(receipt) + ".replace('RECEIPT_PATH', repr(str(OUTPUT / 'environment_setup.json'))) + \"assert str(torch.__version__)\")\nlogged_run([sys.executable, '-u', '-c', check_code], check=True)")
    checkout = """import subprocess
REPO = Path('/content/SC-SSTW-Fixed-Key-' + stamp)
logged_run(['git', 'clone', '--filter=blob:none', 'https://github.com/RICHAAARC/SC-SSTW.git', str(REPO)])
logged_run(['git', '-C', str(REPO), 'checkout', '--detach', SOURCE_SHA])
actual = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
assert actual == SOURCE_SHA
print('source commit:', actual, flush=True)
"""
    cuda = ''.join(golden['cells'][4]['source']).replace('subprocess.run(', 'logged_run(')
    run = """import os, subprocess, sys
CONFIG = REPO / 'experiments/wan_state_clock/configs/flow_fixed_key_v1.json'
env = os.environ.copy(); env['PYTHONUNBUFFERED'] = '1'
command = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.flow_fixed_key_run', '--config', str(CONFIG), '--output', str(OUTPUT)]
print('fixed experiment output:', OUTPUT, flush=True)
subprocess.run(command, cwd=REPO, env=env, check=True)
"""
    results = """import json
result = json.loads((OUTPUT / 'result.json').read_text())
print('status:', result['status'])
print('fixed denominator:', result['fixed_denominator'])
print('calibration:', result['calibration'])
for case_id, case in result['cases'].items():
    print(case_id, case['status'])
    for arm, item in case['videos'].items():
        print(arm, 'source:', item.get('decision'))
        for view, row in item['views'].items():
            alignment = row.get('alignment_reporting_only', {})
            summary = {k: alignment.get(k) for k in ('reference_eligible_windows', 'exact_allocation_matches', 'best_missing_reference_count', 'nominal_scale_match')}
            print(' ', view, row.get('decision'), 'alignment:', summary)
print('full result:', OUTPUT / 'result.json')
"""
    cells=[code("from google.colab import drive\ndrive.mount('/content/drive')"),
           md('# Fixed-key Flow watermark V1\n\nRun all once. A single fixed 11-window key marker is generated afresh on four predeclared development prompt/seed cases. Two OFF cases calibrate the full search family; two cases evaluate OFF, SINGLE46 and MULTI44_46. FULL, DELETE90 and SPEED5_4 are all retained: 8 arms, 24 saved views, 96 encodes. This is a new, unrun existence candidate; historical attribution success does not validate it. GPU execution is performed by the user.'),
           code(setup),code(logging+install),code(checkout),code(cuda),code(run),code(results)]
    # Stable cell IDs satisfy nbformat 4.5 without executing the GPU experiment.
    for i, cell in enumerate(cells): cell['id']='fixed-key-'+str(i)
    nb=dict(cells=cells,nbformat=4,nbformat_minor=5,metadata=dict(accelerator='GPU',colab=dict(name='Fixed-key Flow V1',provenance=[]),kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python'),source_commit=SOURCE_SHA,notebook_binding_kind='immutable_source_commit'))
    OUTPUT.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
    return OUTPUT
if __name__ == '__main__': print(build())
