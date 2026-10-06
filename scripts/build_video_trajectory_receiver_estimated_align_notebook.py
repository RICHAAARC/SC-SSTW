"""Build the fixed saved-M05 estimated-alignment Stage2 user-run notebook."""
from pathlib import Path
import argparse,ast,json,re,sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import build_video_trajectory_payload_framewise_sync_m05_notebook as prior
OUTPUT=ROOT/"notebooks/video_trajectory_receiver_estimated_align_v1_colab.ipynb"

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
for sid,row in sorted(result['historical_comparisons'].items()):
    print('Historical postseal comparison:',sid,row)
print('Failures:',result['failures'])
print('Evidence ceiling:',result['evidence_ceiling'])
print('Result:',RESULT_PATH)
"""

def build(source_sha=None,output=OUTPUT):
    if source_sha is not None and not re.fullmatch(r"[0-9a-f]{40}",source_sha):raise ValueError("immutable source SHA required")
    cfg=json.loads((ROOT/"experiments/wan_state_clock/configs/video_trajectory_receiver_estimated_align_v1.json").read_text())
    fixed=cfg["fixed_denominator"]
    setup=prior.SETUP.replace("Trajectory-Payload-Framewise-Sync-M05","Trajectory-Receiver-Estimated-Align-Stage2-V1").replace(
        "SC-SSTW-TRAJECTORY-PAYLOAD-FRAMEWISE-SYNC-M05-","SC-SSTW-TRAJECTORY-RECEIVER-ESTIMATED-ALIGN-STAGE2-V1-")
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
                          "experiments.wan_state_clock.video_trajectory_receiver_estimated_align_v1_run")
    note="Published source: "+source_sha if source_sha else "UNPUBLISHED DRAFT: SOURCE_SHA=None; publish reviewed source and bind its immutable SHA before Run all."
    markdown=("# Receiver-Estimated-Align-V1: fixed Stage2\n\n"
      "Only three saved G M05 RGB observations, with 181/177/89 received frames. "
      "Fresh framewise encoding once per observation and independent K0/K1 scoring use all 198 candidates and 4820 local rows. "
      "Only a complete finite unique canonical argmax estimates the offset; tie tolerance remains 1e-12 and sync_accepted remains false. "
      "An unresolved estimate retains its failed aligned slot without fallback. FULL singleton is geometry only.\n\n"
      "The public receiver operation uses p=estimated_offset%4: prepend p copies of the first received frame, then drop the last p frames; "
      "p0 is identity and length stays fixed. Phases 2/3 are deterministic extensions, not validated by Stage1. "
      "New sync evidence is sealed before a physical phase plan is frozen. Baseline and aligned p0 share the same key read; "
      "same-phase keys share encoding but retain distinct key reads. No cross-observation cache. "
      "12 logical payload slots retain 422400 votes, 14080 time-bit rows and 384 final bits; "
      "physical upper bounds are 7 Wan encodes and 10 reads (aliases are not independent observations). "
      "Wan FP32 posterior mode and the original R44/R44/R22 reader, coordinates, strict >0 and Counter encounter-order ties remain unchanged.\n\n"
      "Actual truth and historical reference reads occur only after the final payload blind seal. "
      "All failures, wrong-key results, ties and full time/channel evidence remain. "
      "Historical comparisons cannot select offsets, phases, retries or a better result. "
      "Receiver latent indices do not assert matched or independent source receptive fields. "
      "No source generation, writer, codec, quality recomputation, strength scan, thresholds, FPR claim or production integration. "
      "This fixed test does not establish generalized synchronization, cache recovery or unique phase causality.\n\n"
      "The notebook is user-run only after reviewed immutable source publication. Local checks are CPU/fake/static only. "
      "Relevant eight package pins reuse the successful receiver environment; actual versions and pip-check warnings are preserved.\n\n"
      "Fixed received RGB identities:\n"+"\n".join("- "+o+": "+s["path"]+"; SHA256 "+s["sha256"] for o,s in cfg["inputs"].items())+"\n\n"+note)
    rows=[("code","from google.colab import drive\ndrive.mount('/content/drive')\n"),
          ("markdown",markdown),("code",f"SOURCE_SHA = {source_sha!r}\nFIXED = {fixed!r}\n"+setup),
          ("code",environment),("code",run),("code",DISPLAY)]
    cells=[]
    for i,(kind,source) in enumerate(rows):
        cell=dict(cell_type=kind,id=f"receiver-estimated-align-stage2-{i}",metadata={},source=source.splitlines(keepends=True))
        if kind=="code":ast.parse(source);cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    doc=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
        kernelspec=dict(display_name="Python 3",language="python",name="python3"),language_info=dict(name="python"),
        candidate_binding=dict(candidate="trajectory-receiver-estimated-align-stage2-v1",source_sha=source_sha,
            status="UNPUBLISHED_DRAFT" if source_sha is None else "PUBLISHED_SHA_BOUND",
            execution="user-run only; CPU/fake/static checks; no agent real execution")))
    output=Path(output);output.write_text(json.dumps(doc,indent=1)+"\n",encoding="utf-8");return output
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--source-sha");parser.add_argument("--output",type=Path,default=OUTPUT)
    args=parser.parse_args();print(build(args.source_sha,args.output))
