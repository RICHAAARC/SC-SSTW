"""Build the direct Run-all saved-terminal diagnostic, reusing B handoff helpers."""
from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

from scripts import build_local_joint_state_payload_v1_notebook as previous

ROOT = previous.ROOT
CONFIG_PATH = "experiments/wan_state_clock/configs/local_joint_terminal_bridge_v1.json"
NAME = "local_joint_state_payload_v1_terminal_bridge"
PORTABLE_FILES = previous.PORTABLE_FILES + (
    "main/tube_state/local_joint_terminal_bridge_v1.py",
    "runtime/wan/local_joint_terminal_bridge_v1.py",
    "experiments/wan_state_clock/local_joint_terminal_bridge_v1_run.py", CONFIG_PATH,
)


def portable_archive():
    stream = io.BytesIO()
    manifest = dict(schema="saved-joint-terminal-bridge-source-v1", git_commit=None, blocking=False,
                    files={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in PORTABLE_FILES})
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in (*PORTABLE_FILES, "portable_source_manifest.json"):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            data = (ROOT/name).read_bytes() if name in PORTABLE_FILES else json.dumps(manifest, sort_keys=True).encode()
            archive.writestr(info, data)
    return stream.getvalue(), manifest


def reused_setup_functions() -> str:
    # Reuse the already-run B process-group cleanup/logging verbatim through AST,
    # rather than maintaining a second subtly different subprocess implementation.
    tree = ast.parse("".join(previous.build_notebook()["cells"][2]["source"]))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ("write_json", "record_failure", "logged")]
    for node in functions:
        if node.name == "record_failure":
            node.name = "original_record_failure"
    return "\n\n".join(ast.unparse(n) for n in functions)


def reused_environment() -> str:
    source = "".join(previous.build_notebook()["cells"][3]["source"])
    tree = ast.parse(source)
    setup_try = next(n for n in tree.body if isinstance(n, ast.Try))
    # Only remove the codec installation branch; same dependency repair and
    # advisory version records remain. No pipeline import is needed for this run.
    setup_try.body = [n for n in setup_try.body if not (
        isinstance(n, ast.If) and "shutil.which" in ast.unparse(n.test))]
    source = ast.unparse(tree).replace("WanPipeline, AutoencoderKLWan", "AutoencoderKLWan")
    return source.replace("current 27D/50E/27-load memory and latency remain unverified",
                          "saved-terminal 2E/4D/one-VAE-load memory and latency remain unverified")


def build_notebook():
    package, manifest = portable_archive()
    config = json.loads((ROOT/CONFIG_PATH).read_text(encoding="utf-8"))
    c, m = previous._code, previous._markdown
    setup = f'''\
from pathlib import Path
import datetime, hashlib, json, os, signal, subprocess, sys, time, traceback
FIXED_CONFIG = {config!r}
EXPECTED_COST = {{"encode":2,"decode":4,"dit":0,"native":0,"codec":0,"backward":0,"vae_load":1}}
OUTPUT_PARENT = Path('/content/drive/MyDrive/Video-WM/Local-Joint-State-Payload-V1-Terminal-Bridge')
OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUTPUT = OUTPUT_PARENT / STAMP
OUTPUT.mkdir(exist_ok=False)
RUN_OUTPUT = OUTPUT / 'run'
WORKSPACE = Path('/content/Video-WM-Local-Joint-Terminal-Bridge-' + STAMP)
PYTHON = sys.executable
'''
    setup += reused_setup_functions() + '''

def record_failure(stage, exc):
    original_record_failure(stage, exc)
    fixed_slots.update(status='ENGINEERING_FAILURE', reason=repr(exc))
    write_json(OUTPUT / 'fixed_slots.json', fixed_slots)
    if (RUN_OUTPUT / 'result.json').exists():
        from runtime.wan.local_joint_terminal_bridge_v1 import finalize_interrupted
        finalize_interrupted(RUN_OUTPUT, 'notebook takeover after child cleanup: ' + repr(exc))

fixed_slots = dict(status='SETUP_STARTED', expected_cost=EXPECTED_COST,
    stages={name: dict(status='NOT_RUN', expected_windows_per_view=88, expected_chips_per_view=1408)
            for name in ('base','candidate','posterior_reconstruction','raw_reinjection','masked_reinjection','capped_reinjection')})
write_json(OUTPUT / 'fixed_slots.json', fixed_slots)
write_json(OUTPUT / 'setup_receipt.json', dict(config=FIXED_CONFIG, expected_cost=EXPECTED_COST,
    output=str(OUTPUT), python=sys.version, scientific_pass=False, automatic_retry=False))
print('Saved-terminal diagnostic output:', OUTPUT, flush=True)
'''
    encoded = base64.b64encode(package).decode()
    chunks = "\n".join(repr(encoded[i:i+100]) for i in range(0, len(encoded), 100))
    source = f'''\
import base64, io, zipfile
SOURCE_PACKAGE_B64 = (
{chunks}
)
try:
    WORKSPACE.mkdir(parents=True, exist_ok=False)
    package_bytes = base64.b64decode(SOURCE_PACKAGE_B64)
    with zipfile.ZipFile(io.BytesIO(package_bytes)) as archive:
        for member in archive.infolist():
            target = (WORKSPACE / member.filename).resolve()
            if not target.is_relative_to(WORKSPACE.resolve()):
                raise RuntimeError('unsafe embedded archive path')
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
    sys.path.insert(0, str(WORKSPACE))
    # Optional identity is descriptive. Missing provenance is not an admission gate.
    try:
        files = {{name: hashlib.sha256((WORKSPACE/name).read_bytes()).hexdigest() for name in {list(PORTABLE_FILES)!r}
                 if (WORKSPACE/name).is_file()}}
        write_json(OUTPUT / 'source_receipt.json', dict(files=files, git_commit=None, blocking=False,
            package_sha256=hashlib.sha256(package_bytes).hexdigest(), workspace=str(WORKSPACE)))
    except Exception as exc:
        print('Optional source receipt unavailable:', repr(exc), flush=True)
except BaseException as exc:
    record_failure('SOURCE_EXTRACTION', exc)
    raise
'''
    run = f'''\
try:
    from google.colab import userdata
    hf_token = userdata.get('HF_TOKEN')
    if hf_token: os.environ['HF_TOKEN'] = hf_token
except Exception as exc:
    print('Optional HF token unavailable:', type(exc).__name__, flush=True)

run_returncode = None
try:
    command = [PYTHON, '-u', '-m', 'experiments.wan_state_clock.local_joint_terminal_bridge_v1_run',
               '--config', str(WORKSPACE / {CONFIG_PATH!r}), '--output', str(RUN_OUTPUT)]
    run_returncode = logged(command, 'FIXED_TERMINAL_BRIDGE', cwd=WORKSPACE, check=False)
    from runtime.wan.local_joint_terminal_bridge_v1 import finalize_interrupted
    # The logger has reaped the child/process group before any takeover writes.
    finalize_interrupted(RUN_OUTPUT, 'child exited with return code ' + str(run_returncode))
    result_path = RUN_OUTPUT / 'result.json'
    result = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else None
    write_json(OUTPUT / 'execution_receipt.json', dict(returncode=run_returncode,
        status=None if result is None else result['status'],
        actual_counts=None if result is None else result['counts'], expected_cost=EXPECTED_COST,
        scientific_pass=False, automatic_retry=False))
    if run_returncode != 0 or result is None or result['status'] != 'COMPLETE':
        raise RuntimeError('fixed terminal bridge incomplete; retain entire output: ' + str(OUTPUT))
    fixed_slots.update(status='SUPERSEDED_BY_RUN_RESULT', result='run/result.json')
    write_json(OUTPUT / 'fixed_slots.json', fixed_slots)
except BaseException as exc:
    try: record_failure('FIXED_TERMINAL_BRIDGE', exc)
    except BaseException as record_error:
        if hasattr(exc, 'add_note'): exc.add_note('failure record error: ' + repr(record_error))
    raise
'''
    summary = '''\
result = json.loads((RUN_OUTPUT / 'result.json').read_text(encoding='utf-8'))
print('Engineering status:', result['status'])
print('Actual calls:', result['counts'])
for name, stage in result['stages'].items():
    row = stage['views']['postclip']
    means = row.get('payload_signed_means', [])
    print(name, stage['status'], 'state gap=', row.get('state_gap'),
          'positive payload=', sum(x is not None and x > 0 for x in means), '/32',
          'missing payload=', 32-sum(x is not None for x in means))
print(result['ceiling'])
print('Return the entire output directory, including failures:', OUTPUT)
'''
    return dict(nbformat=4, nbformat_minor=5, metadata=dict(
        kernelspec=dict(display_name="Python 3", language="python", name="python3"),
        language_info=dict(name="python", version="3"), colab=dict(name=NAME+"_colab.ipynb", provenance=[])),
        cells=[c("from google.colab import drive\ndrive.mount('/content/drive')"), m('''
        # B: fixed saved-JOINT terminal bridge (2E + 4D)

        Select a CUDA runtime and Run all. This reuses run `20261009T132316545817Z` JOINT
        `terminal_latent.pt` and `float_rgb.pt` from the fixed Drive path. It reads that run only.
        Missing/incompatible float input is an engineering failure; no fallback decode is added.
        One frozen FP32 Wan VAE is loaded; no transformer/pipeline, sampling, codec or backward runs.
        GPU names and reference dependency versions are informational, not admission gates.

        Fixed order: read base and original-carrier candidate; encode both; decode
        E(candidate), z_T+raw, z_T+mask(raw), z_T+cap(mask(raw)). Key/message/ROIs/rho=.5/cap=1
        and known-grid reader stay fixed. All executable layers run despite negative statistics.
        The cap and norms describe the terminal intervention, not signal retention.

        Each of six layers retains 88 windows × 16 chips (704 state + 704 payload),
        energies/support/q, 22 offset correlations, 32 fragment-bit means, paired changes and
        weakest bits. Candidate and four decoded layers retain pre/post-clamp observations
        from the same calls. Saved base has **no pre-clamp observation**. No OFF/wrong-key grid.

        Output is a fresh UTC directory under `MyDrive/Video-WM/Local-Joint-State-Payload-V1-Terminal-Bridge/`.
        Return the entire directory, even on failure: setup/fixed slots, source/environment receipts,
        execution log/failure, `run/result.json`, per-layer raw/metrics, tensors, paired comparisons.
        Counts are saved at model call entry/return; interruption preserves completed observations.
        A killed kernel cannot run final cleanup; the durable RUNNING record remains honest until inspected.

        This is one seen JOINT terminal point. It cannot locate original step25–49 conditional-clean,
        CFG/native or trajectory causality, or establish blind decoding/FPR/scientific PASS.
        CPU and notebook-stub checks do not establish real-model memory or latency.
        '''), c(setup), c(source), c(reused_environment()), c(run), c(summary)])


def main():
    notebook = build_notebook()
    (ROOT/"notebooks"/(NAME+"_colab.ipynb")).write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
    package, _ = portable_archive()
    (ROOT/"notebooks"/(NAME+"_portable_source.zip")).write_bytes(package)


if __name__ == "__main__":
    main()
