"""Build the fixed four-crop Milestone1A user-run notebook."""
from pathlib import Path
import argparse,ast,json,re,sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import build_video_trajectory_payload_framewise_sync_m05_notebook as prior
OUTPUT=ROOT/"notebooks/video_trajectory_receiver_phase23_v1_colab.ipynb"

DISPLAY="""result=json.loads(RESULT_PATH.read_text())
if result['source_sha']!=SOURCE_SHA or result['fixed_denominator']!=FIXED:raise RuntimeError('result identity mismatch')
print('Status:',result['status'],'Counts:',result['counts'],'Calls:',result['calls'])
print('Physical plan:',result['physical_plan']['planned_calls'],'logical aliases are not independent evidence')
print('Actual environment:',result['environment'])
print('Seals:',result.get('blind_sync_sha256'),result.get('physical_plan_sha256'),result.get('blind_payload_sha256'))
for sid,row in sorted(result['estimates'].items()):
    print('Blind estimate:',sid,row)
for sid,row in sorted(result['sync_posthoc'].items()):
    print('Postseal sync:',sid,'key=',row.get('key_role'),'truth_offset=',row.get('truth_offset'),
          'estimated_offset=',row.get('estimated_offset'),'offset_correct=',row.get('offset_correct'),
          'phase_correct=',row.get('phase_correct'),'geometry_only=',row.get('geometry_only'))
for sid,row in sorted(result['payload_posthoc'].items()):
    values=[x['signed_normalized_margin'] for x in row.get('bit_rows',[]) if 'signed_normalized_margin' in x]
    print(sid,row['status'],'key=',row.get('key_role'),'view=',row.get('view'),'phase=',row.get('phase'),
          'alias=',row.get('alias_of'),'bit_errors=',row.get('bit_errors'),'error_bits=',row.get('error_bits'),'minimum signed margin=',min(values,default=None))
    print('  full detailed evidence:',result['payload_reads'][sid]['detail_path'],row.get('path'))
for sid,row in sorted(result['same_run_differences'].items()):
    print('EST_ALIGN minus BASELINE:',sid,row['status'],'bit error delta=',row.get('bit_error_delta'))
print('Preparation:',result.get('source_preparation'))
print('Failures:',result['failures'])
print('Evidence ceiling:',result['evidence_ceiling'])
print('Result:',RESULT_PATH)
"""

def build(source_sha=None,output=OUTPUT):
    if source_sha is not None and not re.fullmatch(r"[0-9a-f]{40}",source_sha):raise ValueError("immutable source SHA required")
    cfg=json.loads((ROOT/"experiments/wan_state_clock/configs/video_trajectory_receiver_phase23_v1.json").read_text())
    prep=json.loads((ROOT/"experiments/wan_state_clock/configs/video_trajectory_receiver_phase23_v1_preparation.json").read_text())
    fixed=cfg["fixed_denominator"]
    setup=prior.SETUP.replace("Trajectory-Payload-Framewise-Sync-M05","Trajectory-Receiver-Phase23-Milestone1A-V1").replace(
        "SC-SSTW-TRAJECTORY-PAYLOAD-FRAMEWISE-SYNC-M05-","SC-SSTW-TRAJECTORY-RECEIVER-PHASE23-MILESTONE1A-V1-")
    before=prior.ENVIRONMENT.split("    import shutil\n")[0]
    after="    dependency_code="+prior.ENVIRONMENT.split("    dependency_code=",1)[1]
    probe=("import importlib.metadata as m; pins="+repr(cfg["environment_pins"])+"; "
           "actual={k:m.version(k) for k in pins}; "
           "assert all(actual[k].split('+')[0]==v for k,v in pins.items()), actual; "
           "from diffusers import AutoencoderKL, AutoencoderKLWan; import torch; print(actual)")
    environment=before+"    probe="+repr(probe)+"\n"
    environment+="    if logged([PYTHON,'-c',probe],'RECEIVER_DEPENDENCY_PROBE',check=False):\n"
    environment+="        logged([PYTHON,'-m','pip','install',"+",".join(repr(k+"=="+v) for k,v in cfg["environment_pins"].items())+"],'RECEIVER_DEPENDENCY_REPAIR')\n"
    environment+="        logged([PYTHON,'-c',probe],'RECEIVER_DEPENDENCY_REPROBE')\n"+after
    run=prior.RUN.replace("experiments.wan_state_clock.video_trajectory_payload_framewise_sync_v1_run",
                          "experiments.wan_state_clock.video_trajectory_receiver_phase23_v1_run")
    note="Published source: "+source_sha if source_sha else "UNPUBLISHED DRAFT: SOURCE_SHA=None; publish reviewed source and bind its immutable SHA before Run all."
    markdown=("# Receiver-Phase23-V1: fixed Milestone1A\n\n"
      "One saved G M05 FULL181 RGB is read and verified once, then cloned into [2:179], [3:180], [38:127], [39:128]. "
      "Lengths remain 177/177/89/89; no FULL/phase1 rerun, codec, generation or writer call. "
      "The fixture preparation knows crop starts; received scoring, estimates and alignment receive only opaque observation RGB, public protocol and the current key. "
      "Preparation records stay separate from blind receiver evidence; message and report labels join after the final payload seal.\n\n"
      "Four fresh framewise encodes (532 frames, 70 batch8 chunks), eight independent key scores, 392 candidates and 9456 local rows. "
      "Only complete finite unique canonical estimates drive p=offset%4; tie tolerance and sync_accepted=False remain unchanged. "
      "No fallback, oracle phase, threshold or BER selection. The original prepend/drop mapping keeps each received length fixed. "
      "Original Wan FP32 posterior-mode R44/R22 reads and Counter semantics are reused directly.\n\n"
      "Sixteen fixed logical payload slots preserve 506880 planned votes, 16896 time-bit rows and 512 final bits. "
      "The sync-sealed physical plan uses observation/phase reuse with independent key reads; physical upper bounds are 12 encodes and 16 reads. "
      "Aligned p0 aliases the same-key baseline; aliases are not independent evidence. Missing, interrupted and unresolved slots remain. "
      "All-time-bit-positive is descriptive, not a scientific PASS gate.\n\n"
      "There is no old same-window record for these four new cuts: no historical comparator is requested, read or used. "
      "Old FULL/Stage2 results are background only. These overlapping crops remain one development source; "
      "1B new-source generation, dynamic paths and rejection thresholds are not implemented.\n\n"
      "The notebook is unexecuted by the agent. After reviewed immutable source publication, the user runs all five code cells. "
      "The eight relevant pins and framewise/Wan dependency probes follow the successful Stage2 path; pip-check warnings persist.\n\n"
      "Fixed FULL181: "+prep["source"]["path"]+"; SHA256 "+prep["source"]["sha256"]+"\n\n"+note)
    rows=[("code","from google.colab import drive\ndrive.mount('/content/drive')\n"),
          ("markdown",markdown),("code",f"SOURCE_SHA = {source_sha!r}\nFIXED = {fixed!r}\n"+setup),
          ("code",environment),("code",run),("code",DISPLAY)]
    cells=[]
    for i,(kind,source) in enumerate(rows):
        cell=dict(cell_type=kind,id=f"receiver-phase23-milestone1a-{i}",metadata={},source=source.splitlines(keepends=True))
        if kind=="code":ast.parse(source);cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    doc=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
        kernelspec=dict(display_name="Python 3",language="python",name="python3"),language_info=dict(name="python"),
        candidate_binding=dict(candidate="trajectory-receiver-phase23-milestone1a-v1",source_sha=source_sha,
            status="UNPUBLISHED_DRAFT" if source_sha is None else "PUBLISHED_SHA_BOUND",
            execution="user-run only; CPU/fake/static checks; no agent real execution")))
    output=Path(output);output.write_text(json.dumps(doc,indent=1)+"\n",encoding="utf-8");return output
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--source-sha");parser.add_argument("--output",type=Path,default=OUTPUT)
    args=parser.parse_args();print(build(args.source_sha,args.output))
