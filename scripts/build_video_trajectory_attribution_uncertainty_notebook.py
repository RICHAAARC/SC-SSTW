"""Build fixed saved-RGB attribution-uncertainty V1 notebook; no agent execution."""
from pathlib import Path
import argparse
import ast
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import build_video_trajectory_conditional_joint_notebook as base

CONFIG = ROOT / "experiments/wan_state_clock/configs/video_trajectory_attribution_uncertainty_v1.json"
OUTPUT = ROOT / "notebooks/video_trajectory_attribution_uncertainty_v1_colab.ipynb"

DISPLAY = """result=json.loads(RESULT_PATH.read_text())
if result['source_sha']!=SOURCE_SHA or result['config_sha256']!=CONFIG_SHA or result['fixed_denominator']!=FIXED:
    raise RuntimeError('result source/config/denominator identity mismatch')
print('Runtime:',result['status'],'Stage:',result['stage'])
print('Coverage denominator: the same 10 PRIMARY_PROBE queries are evaluated for both branches; they are not 20 independent samples.')
for branch,row in result['coverage'].items():
    if isinstance(row,dict) and 'denominator' in row:
        print('Coverage:',branch,'denominator=',row['denominator'],'actual_trigger=',row['actual_trigger'],'valid_coverage=',row['valid_coverage'],'technical=',row['technical'],'status=',row['status'])
print('Probe focus:',result['coverage']['probe_focus'],'Excluded alias/control:',result['coverage']['alias_controls_excluded'],result['coverage']['regression_controls_excluded'])
print('Regression controls:',result['controls'])
print('False attribution across all 22:',result['false_attribution'])
for query_id,row in sorted(result['queries'].items()):
    post=row['posthoc'];sync=row['sync'];identity=row['identity'];decision=row['decision']
    print(query_id,post['source'],post['condition'],post['key'],post['view'],post['parameters'],post['role'],post.get('focus'),
          decision['decision'],decision['reason'],'M=',sync.get('M'),'m=',sync.get('m'),'actions=',sync.get('top_action_count'),
          'I=',identity.get('I'),'exact32=',identity.get('exact32'),'action_correct=',post.get('action_map_correct'),
          'sync_coverage=',post['valid_sync_uncertainty_coverage'],'weak_coverage=',post['valid_weak_identity_coverage'],
          'false_claim=',post['false_claim'],'technical_complete=',post['technical_complete'])
print('Failures:',result['failures'])
print('Evidence ceiling:',result['evidence_ceiling'])
print('Result:',RESULT_PATH)
"""


def build(source_sha=None, output=OUTPUT):
    if source_sha is not None and not re.fullmatch("[0-9a-f]{40}", source_sha):
        raise ValueError("immutable source SHA")
    cfg = json.loads(CONFIG.read_text())
    pins = cfg["environment_pins"]
    setup = (
        base.SETUP_TEMPLATE.replace(
            "Trajectory-Payload-Framewise-Sync-M05",
            "Trajectory-Attribution-Uncertainty-V1",
        )
        .replace(
            "SC-SSTW-TRAJECTORY-PAYLOAD-FRAMEWISE-SYNC-M05-",
            "SC-SSTW-TRAJECTORY-ATTRIBUTION-UNCERTAINTY-V1-",
        )
    )
    probe = (
        "import importlib.metadata as m; pins="
        + repr(pins)
        + "; actual={k:m.version(k) for k in pins}; "
        + "assert all(actual[k].split('+')[0]==v for k,v in pins.items()),actual; "
        + "from diffusers import AutoencoderKL,AutoencoderKLWan; "
        + "import torch,sentencepiece,ftfy; print(actual)"
    )
    install = ",".join(repr(name + "==" + version) for name, version in pins.items())
    environment = f"""try:
    logged(['git','clone','--filter=blob:none','https://github.com/RICHAAARC/SC-SSTW.git',str(REPO)],'SOURCE_CLONE')
    logged(['git','-C',str(REPO),'fetch','origin',SOURCE_SHA],'SOURCE_FETCH')
    logged(['git','-C',str(REPO),'checkout','--detach',SOURCE_SHA],'SOURCE_CHECKOUT')
    actual=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True)
    if actual!=SOURCE_SHA or dirty:raise RuntimeError('source SHA/clean checkout mismatch')
    probe={probe!r}
    if logged([PYTHON,'-c',probe],'DEPENDENCY_PROBE',check=False):
        logged([PYTHON,'-m','pip','install',{install}],'PINNED_REPAIR')
        logged([PYTHON,'-c',probe],'DEPENDENCY_REPROBE')
    dependency_code=logged([PYTHON,'-m','pip','check'],'DEPENDENCY_REPORT',check=False)
    (OUTPUT/'environment_freeze.txt').write_text(subprocess.check_output([PYTHON,'-m','pip','freeze'],text=True))
    write_json(OUTPUT/'source_receipt.json',dict(source_sha=actual,clean=True))
    write_json(OUTPUT/'setup_receipt.json',dict(status='SETUP_COMPLETE',source_sha=actual,fixed_denominator=FIXED,dependency_check_returncode=dependency_code))
except Exception as exc:
    try:failed('SOURCE_OR_ENVIRONMENT',exc)
    except BaseException as record_error:
        if hasattr(exc,'add_note'):exc.add_note('failure record error: '+repr(record_error))
    raise
"""
    run = """CONFIG_PATH=REPO/'experiments/wan_state_clock/configs/video_trajectory_attribution_uncertainty_v1.json'
CONFIG_SHA=hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
configuration=json.loads(CONFIG_PATH.read_text())
if configuration['name']!='video_trajectory_attribution_uncertainty_v1' or configuration['fixed_denominator']!=FIXED:
    raise RuntimeError('fixed configuration mismatch')
command=[PYTHON,'-u','-m','experiments.wan_state_clock.video_trajectory_attribution_uncertainty_v1_run','--config',str(CONFIG_PATH),'--output',str(RUN_OUTPUT)]
try:
    returncode=logged(command,'FIXED_SAVED_RGB_RUN',cwd=REPO,check=False)
    RESULT_PATH=RUN_OUTPUT/'result.json'
    write_json(OUTPUT/'execution_receipt.json',dict(command=command,returncode=returncode,result_path=str(RESULT_PATH),result_exists=RESULT_PATH.is_file()))
    if not RESULT_PATH.is_file():raise RuntimeError('No runner result; fixed 22-query denominator retained')
except Exception as exc:
    try:failed('RUNNER',exc)
    except BaseException as record_error:
        if hasattr(exc,'add_note'):exc.add_note('failure record error: '+repr(record_error))
    raise
"""
    note = (
        "Published source: " + source_sha
        if source_sha
        else "UNPUBLISHED DRAFT: SOURCE_SHA=None; publish reviewed source then bind before Run all."
    )
    markdown = """# Trajectory attribution uncertainty follow-up V1

This fixed Run-all reuses the six already-audited post-codec received.rgb8 inputs for C1/C2 A_M05, OFF and B_M05 and the exact frozen attribution rule from run 20261008T085244020205Z. It does not generate video, write a new watermark, change strength, invoke a codec, or introduce a new temporal attack.

The preregistered roster has 22 queries over 20 physical observations. Ten PRIMARY_PROBE queries—the same ten for both coverage questions—use A_M05/K0 with SHORT89 starts 0/46/92 or internal H1 b2 deletions k44/k132. Four edge-deletion ALIAS_CONTROL queries k1/k176 and eight existing CROP177 b2 REGRESSION_CONTROL queries are excluded from uncertainty coverage. Same-action path aliases never count as action ambiguity.

Valid sync-unreliability coverage requires complete evidence, M above the frozen threshold, and either more than one top action or one action with local m at or below the frozen threshold. Valid weak-identity coverage requires qualified unique sync, the correct complete action, complete payload evidence, exact I=0, all 32 signed integer vote differences nonnegative and at least one tied bit. Since every frozen tau_I is zero, no positive interval above zero is labeled weak. Technical, missing, interrupted or unsupported UNCERTAIN rows remain unresolved fixed slots and never count as method coverage.

The eight regression controls retain two prior positive and six prior negative decision/reason expectations. Across all 22 rows, a positive ACCEPT with a wrong action or any negative ACCEPT is a false claim; a correct positive probe ACCEPT is simply a non-trigger. No observed scientific trigger is reported as NOT_OBSERVED_FIXED_ROSTER; inputs, thresholds and samples are never replaced or retried. This notebook is a real saved-input receiver validation for the user to run. Static/CPU/fake release checks do not supply its scientific result.

""" + note
    sources = [
        ("code", "from google.colab import drive\ndrive.mount('/content/drive')\n"),
        ("markdown", markdown),
        ("code", f"SOURCE_SHA = {source_sha!r}\nFIXED = {cfg['fixed_denominator']!r}\n" + setup),
        ("code", environment),
        ("code", run),
        ("code", DISPLAY),
    ]
    cells = []
    for index, (kind, source) in enumerate(sources):
        cell = {
            "cell_type": kind,
            "id": f"attribution-uncertainty-{index}",
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
        if kind == "code":
            ast.parse(source)
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "candidate_binding": {
                "candidate": "trajectory-attribution-uncertainty-v1",
                "source_sha": source_sha,
                "status": "UNPUBLISHED_DRAFT" if source_sha is None else "PUBLISHED_SHA_BOUND",
            },
        },
    }
    Path(output).write_text(json.dumps(notebook, indent=1) + "\n")
    return Path(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sha")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(build(args.source_sha, args.output))
