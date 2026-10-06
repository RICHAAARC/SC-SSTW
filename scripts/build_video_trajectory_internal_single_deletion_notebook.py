"""Build the adopted single-deletion diagnostic user-run notebook."""
from pathlib import Path
import argparse,ast,json,re,sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import build_video_trajectory_payload_framewise_sync_m05_notebook as prior
OUTPUT=ROOT/"notebooks/video_trajectory_internal_single_deletion_v1_colab.ipynb"

DISPLAY="""result=json.loads(RESULT_PATH.read_text())
if result['source_sha']!=SOURCE_SHA or result['fixed_denominator']!=FIXED:raise RuntimeError('result identity mismatch')
print('Status:',result['status'],'Counts:',result['counts'])
print('Calls:',result['calls'],'Stage calls:',result['stage_calls'])
print('Planned physical:',result['planned_physical_calls'],'Aliases are not independent evidence')
print('Environment:',result['environment'])
for key in ('blind_sync_seal','blind_plan_seal','blind_payload_seal','oracle_plan_seal','oracle_payload_seal'):
    print(key,result.get(key))
for sid,row in sorted(result['estimates'].items()):
    for family,estimate in row.items():
        print('Blind',sid,family,{k:estimate.get(k) for k in ('status','reason','path','best_score','gap')})
for sid,row in sorted(result['path_posthoc'].items()):
    for family in ('H0','joint'):
        e=row.get(family,{})
        print('Path posthoc',sid,family,'key=',row.get('key_role'),'view=',row.get('view'),
              {k:e.get(k) for k in ('family_correct','b_correct','map_correct','map_denominator','k_exact','k_signed_error','k_absolute_error','false_jump_on_this_input')})
for lid,row in sorted(result['payload_posthoc'].items()):
    margins=[x['signed_normalized_margin'] for x in row.get('bit_rows',[]) if 'signed_normalized_margin' in x]
    print(lid,row['status'],'key=',row.get('key_role'),'view=',row.get('view'),'oracle=',row.get('oracle'),
          'bit_errors=',row.get('bit_errors'),'error_bits=',row.get('error_bits'),'minmargin=',min(margins,default=None),'alias=',row.get('alias_of'))
    print('  detail:',result['payload_reads'][lid].get('detail_path'),row.get('path'))
print('Local grid/top evidence:',{sid:row['path'] for sid,row in result['sync_reads'].items()})
print('Failures:',result['failures'])
print('Ceiling:',result['evidence_ceiling'])
print('Result:',RESULT_PATH)
"""

def build(source_sha=None,output=OUTPUT):
    if source_sha is not None and not re.fullmatch(r"[0-9a-f]{40}",source_sha):raise ValueError("immutable source SHA required")
    cfg=json.loads((ROOT/"experiments/wan_state_clock/configs/video_trajectory_internal_single_deletion_v1.json").read_text())
    prep=json.loads((ROOT/"experiments/wan_state_clock/configs/video_trajectory_internal_single_deletion_v1_preparation.json").read_text())
    fixed=cfg["fixed_denominator"]
    setup=prior.SETUP.replace("Trajectory-Payload-Framewise-Sync-M05","Trajectory-Internal-Single-Deletion-V1").replace(
        "SC-SSTW-TRAJECTORY-PAYLOAD-FRAMEWISE-SYNC-M05-","SC-SSTW-TRAJECTORY-INTERNAL-SINGLE-DELETION-V1-")
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
                          "experiments.wan_state_clock.video_trajectory_internal_single_deletion_v1_run")
    note="Published source: "+source_sha if source_sha else "UNPUBLISHED DRAFT: SOURCE_SHA=None; publish reviewed source and bind its immutable SHA before Run all."
    markdown=("# Fixed internal single-frame deletion diagnostic\n\n"
      "Read the saved 1B M05 FULL181 once. Common source [2:180]; C removes tail 179 -> [2:179], "
      "D removes source 90 -> [2:90]+[91:180]. Both 177/R44; D also includes source 179 versus C. No generation/writer/codec.\n\n"
      "Two framewise encodes (354 frames/46 batch8 calls), four independent key searches. "
      "Per read: 885 cached (r,d) cells, H0 b0..4 and H1 b0..3/k1..176, all 709 hypotheses, 225 local-grid cells. "
      "Original full-direction age normalization, including source 180 rho, is retained. "
      "Complete finite unique H0 and joint argmax use original tie_atol, no score gate. H1 has greater complexity; its victory is not detection.\n\n"
      "RAW, GLOBAL_ALIGN and PATH_ALIGN are blind. PATH H1 inserts the previous received frame at estimated k, "
      "then prepends b%4 first frames and truncates to 177. This is a placeholder, not recovery of missing RGB. "
      "Use whole-clip Wan encoding, no truth-driven segmentation or VAE state reset; normal per-input cache clearing retained. Sync/plan/blind payload seals precede oracle-map parsing. "
      "TRUTH_PATH is oracle-only after blind seals. Message parsing occurs only after oracle payload seal.\n\n"
      "Sixteen logical slots retain 675840 votes/22528 time-bit rows/512 final bits: 12 blind plus 4 oracle. "
      "Wan load 1, physical upper bounds 12 encodes/16 reads. Cache by full input index-map within observation; keys read separately. "
      "CPU normalized latents retain exact values and restore to the receiver device. Cached failures never retry. "
      "Oracle may reuse a blind same-map result, but never fills a blind unresolved slot.\n\n"
      "Report family/b, map correct/177 and exact k errors separately from payload; C has no true k. "
      "Local received four-frame bins are display only, not Wan receptive fields. All failures/ties remain. "
      "No presence/FPR/generalization/scientific PASS or new attack scan. Agent only CPU/fake/static; user performs Run all.\n\n"
      "Fixed input: "+prep["source"]["path"]+"; SHA256 "+prep["source"]["sha256"]+"\n\n"+note)
    rows=[("code","from google.colab import drive\ndrive.mount('/content/drive')\n"),
          ("markdown",markdown),("code",f"SOURCE_SHA = {source_sha!r}\nFIXED = {fixed!r}\n"+setup),
          ("code",environment),("code",run),("code",DISPLAY)]
    cells=[]
    for i,(kind,source) in enumerate(rows):
        cell=dict(cell_type=kind,id=f"internal-single-deletion-{i}",metadata={},source=source.splitlines(keepends=True))
        if kind=="code":ast.parse(source);cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    doc=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
        kernelspec=dict(display_name="Python 3",language="python",name="python3"),language_info=dict(name="python"),
        candidate_binding=dict(candidate="trajectory-internal-single-deletion-v1",source_sha=source_sha,
            status="UNPUBLISHED_DRAFT" if source_sha is None else "PUBLISHED_SHA_BOUND",
            execution="user-run only; CPU/fake/static checks; no agent real execution")))
    output=Path(output);output.write_text(json.dumps(doc,indent=1)+"\n",encoding="utf-8");return output
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--source-sha");parser.add_argument("--output",type=Path,default=OUTPUT)
    args=parser.parse_args();print(build(args.source_sha,args.output))
