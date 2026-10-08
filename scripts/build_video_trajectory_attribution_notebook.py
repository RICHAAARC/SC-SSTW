"""Build fixed trajectory-attribution-v1 user-run notebook; no agent model execution."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import build_video_trajectory_conditional_joint_notebook as base
CONFIG=ROOT/"experiments/wan_state_clock/configs/video_trajectory_attribution_v1.json"
OUTPUT=ROOT/"notebooks/video_trajectory_attribution_v1_colab.ipynb"
DISPLAY="""result=json.loads(RESULT_PATH.read_text())
if result['source_sha']!=SOURCE_SHA or result['config_sha256']!=CONFIG_SHA:raise RuntimeError('result source/config identity mismatch')
print('Runtime:',result['status'],'Stage:',result['stage'])
print('Rules:',result['rules'])
for source_id,row in result['sources'].items():print('Source:',source_id,row)
counts={}
for qid,row in result['queries'].items():
    decision=row.get('decision') or {}
    key=(row['source_id'],row['video_id'],row['key_label'],row['frames'],decision.get('decision'),decision.get('reason'))
    counts[key]=counts.get(key,0)+1
for key,count in sorted(counts.items()):print(count,key)
print('Seals:',result['seals'])
print('Failures:',result['failures'])
print('Evidence ceiling:',result['evidence_ceiling'])
print('Result:',RESULT_PATH)
"""
def build(source_sha=None,output=OUTPUT):
    if source_sha is not None and not re.fullmatch("[0-9a-f]{40}",source_sha):raise ValueError("immutable source SHA")
    cfg=json.loads(CONFIG.read_text());pins=cfg["environment_pins"]
    setup=base.SETUP_TEMPLATE.replace("Trajectory-Payload-Framewise-Sync-M05","Trajectory-Attribution-V1").replace(
        "SC-SSTW-TRAJECTORY-PAYLOAD-FRAMEWISE-SYNC-M05-","SC-SSTW-TRAJECTORY-ATTRIBUTION-V1-")
    before=base.ENVIRONMENT_TEMPLATE.split("    if logged([PYTHON,'-c','import torch;")[0]
    after="    dependency_code="+base.ENVIRONMENT_TEMPLATE.split("    dependency_code=",1)[1]
    probe="import importlib.metadata as m; pins="+repr(pins)+"; actual={k:m.version(k) for k in pins}; assert all(actual[k].split('+')[0]==v for k,v in pins.items()),actual; from diffusers import AutoencoderKL,AutoencoderKLWan,WanPipeline; import torch,sentencepiece,ftfy; print(actual)"
    env=before+"    probe="+repr(probe)+"\n    if logged([PYTHON,'-c',probe],'DEPENDENCY_PROBE',check=False):\n        logged([PYTHON,'-m','pip','install',"+",".join(repr(k+"=="+v) for k,v in pins.items())+"],'PINNED_REPAIR')\n        logged([PYTHON,'-c',probe],'DEPENDENCY_REPROBE')\n"+after
    run="""CONFIG_PATH=REPO/'experiments/wan_state_clock/configs/video_trajectory_attribution_v1.json'
CONFIG_SHA=hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
configuration=json.loads(CONFIG_PATH.read_text())
if configuration['name']!='video_trajectory_attribution_v1':raise RuntimeError('fixed configuration mismatch')
command=[PYTHON,'-u','-m','experiments.wan_state_clock.video_trajectory_attribution_v1_run','--config',str(CONFIG_PATH),'--output',str(RUN_OUTPUT)]
try:
    returncode=logged(command,'FIXED_RUN',cwd=REPO,check=False)
    RESULT_PATH=RUN_OUTPUT/'result.json'
    write_json(OUTPUT/'execution_receipt.json',dict(command=command,returncode=returncode,result_path=str(RESULT_PATH),result_exists=RESULT_PATH.is_file()))
    if not RESULT_PATH.is_file():raise RuntimeError('No runner result; fixed denominator retained')
except Exception as exc:
    try:failed('RUNNER',exc)
    except BaseException as record_error:
        if hasattr(exc,'add_note'):exc.add_note('failure record error: '+repr(record_error))
    raise
"""
    note="Published source: "+source_sha if source_sha else "UNPUBLISHED DRAFT: SOURCE_SHA=None; publish reviewed source then bind before Run all."
    markdown="""# Trajectory attribution V1

Fixed Run-all for one development source and two preregistered unseen confirmation sources. Each source generates paired OFF/A/B native 50-step trajectories from the same prompt, seed, initial noise and fresh pristine scheduler. OFF has no payload control and no M05; A carries K0/OKOK and yields A_P1 plus A_M05; B is an independent K0/NOPE physical generation and yields B_M05. Every received condition uses the fixed FULL181, three CROP177, three SHORT89 and deletion b2/k88 maps, queried with K0 and K1: 64 logical observations per source, 192 total.

The receiver sees only received RGB, public N, query key and the public OKOK 32-bit claim. It completes singleton/G93/J709 synchronization first, folds top paths by complete correction action, seals sync evidence, applies at most one uniquely resolved correction, then reads the stable payload and seals it. It never performs RAW-payload rejection. Development blind evidence freezes tau_M, tau_m and tau_I once; source roles and expected paths are parsed only after blind seals. Confirmation generation starts only when every N is freezable and development positives are strictly separable. Otherwise all 128 confirmation rows remain NOT_RUN/UNCERTAIN_RULE_NOT_FREEZABLE.

Same-action path ties resolve only the action; absolute source path ambiguity remains reported. A_P1 is a payload-bearing sync-null control, not a no-watermark video. Fixed observation/source aggregation retains ACCEPT/REJECT/UNCERTAIN, failures and any false attribution. This notebook has only CPU/fake/schema validation until the user runs it; no real model, GPU, Colab or Drive result is claimed by publication.

"""+note
    sources=[("code","from google.colab import drive\ndrive.mount('/content/drive')\n"),("markdown",markdown),
        ("code",f"SOURCE_SHA = {source_sha!r}\nFIXED = {cfg['fixed_denominator']!r}\n"+setup),
        ("code",env),("code",run),("code",DISPLAY)]
    cells=[]
    for i,(kind,source) in enumerate(sources):
        cell=dict(cell_type=kind,id="trajectory-attribution-"+str(i),metadata={},source=source.splitlines(keepends=True))
        if kind=="code":ast.parse(source);cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
        kernelspec=dict(display_name="Python 3",language="python",name="python3"),
        language_info=dict(name="python"),candidate_binding=dict(candidate="trajectory-attribution-v1",
            source_sha=source_sha,status="UNPUBLISHED_DRAFT" if source_sha is None else "PUBLISHED_SHA_BOUND")))
    Path(output).write_text(json.dumps(nb,indent=1)+"\n");return Path(output)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source-sha");p.add_argument("--output",type=Path,default=OUTPUT)
    args=p.parse_args();print(build(args.source_sha,args.output))
