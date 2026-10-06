"""Build fixed new-source Milestone1B user-run notebook."""
from pathlib import Path
import argparse,ast,json,re,sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import build_video_trajectory_payload_framewise_sync_m05_notebook as prior
OUTPUT=ROOT/"notebooks/video_trajectory_receiver_independent_source_v1_colab.ipynb"

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
print('Preparation:',result.get('source_preparation'))\nprint('Generation/M05:',result.get('generated_source'))\nprint('Preparation workers:',result.get('preparation_workers'))
print('Failures:',result['failures'])
print('Evidence ceiling:',result['evidence_ceiling'])
print('Result:',RESULT_PATH)
"""

def build(source_sha=None,output=OUTPUT):
    if source_sha is not None and not re.fullmatch(r"[0-9a-f]{40}",source_sha):raise ValueError("immutable source SHA required")
    cfg=json.loads((ROOT/"experiments/wan_state_clock/configs/video_trajectory_receiver_independent_source_v1.json").read_text())
    prep=json.loads((ROOT/"experiments/wan_state_clock/configs/video_trajectory_receiver_independent_source_v1_preparation.json").read_text())
    fixed=cfg["fixed_denominator"]
    setup=prior.SETUP.replace("Trajectory-Payload-Framewise-Sync-M05","Trajectory-Receiver-Independent-Source-Milestone1B-V1").replace(
        "SC-SSTW-TRAJECTORY-PAYLOAD-FRAMEWISE-SYNC-M05-","SC-SSTW-TRAJECTORY-RECEIVER-INDEPENDENT-SOURCE-MILESTONE1B-V1-")
    before=prior.ENVIRONMENT.split("    if logged([PYTHON,\'-c\',\'import torch;")[0]
    after="    dependency_code="+prior.ENVIRONMENT.split("    dependency_code=",1)[1]
    probe=("import importlib.metadata as m; pins="+repr({**cfg["environment_pins"],**cfg["generation_dependency_pins"]})+"; "
           "actual={k:m.version(k) for k in pins}; "
           "assert all(actual[k].split('+')[0]==v for k,v in pins.items()), actual; "
           "from diffusers import AutoencoderKL, AutoencoderKLWan, WanPipeline; import torch, sentencepiece, ftfy; print(actual)")
    environment=before+"    probe="+repr(probe)+"\n"
    environment+="    if logged([PYTHON,'-c',probe],'RECEIVER_DEPENDENCY_PROBE',check=False):\n"
    environment+="        logged([PYTHON,'-m','pip','install',"+",".join(repr(k+"=="+v) for k,v in {**cfg["environment_pins"],**cfg["generation_dependency_pins"]}.items())+"],'RECEIVER_DEPENDENCY_REPAIR')\n"
    environment+="        logged([PYTHON,'-c',probe],'RECEIVER_DEPENDENCY_REPROBE')\n"+after
    run=prior.RUN.replace("experiments.wan_state_clock.video_trajectory_payload_framewise_sync_v1_run",
                          "experiments.wan_state_clock.video_trajectory_receiver_independent_source_v1_run")
    note="Published source: "+source_sha if source_sha else "UNPUBLISHED DRAFT: SOURCE_SHA=None; publish reviewed source and bind its immutable SHA before Run all."
    markdown=("# Receiver-Independent-Source-V1: fixed Milestone1B\n\n"
      "One prospectively fixed new prompt and seed, after the receiver rules were frozen. "
      "Generate exactly one PAYLOAD_MULTI 50-step trajectory and one native decode, persist original RGB+SHA, then exit the source process. "
      "A separate process performs one M05 encode/write/decode and one full libx264 CRF18 yuv420p 8fps MP4 save/probe/readback. "
      "No extra P0/P1/M1/quality arms, old saved source, generation retry or selection.\n\n"
      "Fixed prompt: "+prep["generation"]["prompt"]+"; seed="+str(prep["generation"]["seed"])+".\n\n"
      "Writer legally uses message/key/prompt; crop preparation uses starts. The blind receiver receives only current RGB, public protocol and its own key. "
      "Writer evidence, crop truth and message do not select scores/estimates/plans/alignment. Evaluation POSTHOC config first reads after payload seal.\n\n"
      "Clone [2:179], [3:180], [38:127], [39:128] from this run full MP4 readback, with no new codec. "
      "Four fresh framewise encodes; eight scores; 392 candidates / 9456 locals. Complete finite unique estimates drive p=offset%4; no fallback or BER choice. "
      "R44/R22 and original Counter/coordinate/strength/codebook semantics stay fixed. "
      "Sync seal precedes the physical plan; payload seal precedes posthoc.\n\n"
      "Sixteen logical slots retain 506880 votes / 16896 time-bit rows / 512 final bits; Wan physical upper bounds 12 encodes / 16 reads. "
      "Per-observation phase sharing and p0 aliases are preserved. Missing/source/codec/interruption failures remain. "
      "Exact offset, phase, final bits and time/channel margins are separate; all-time-bit-positive is descriptive, not a new PASS criterion.\n\n"
      "A single new-source check does not establish broad generalization/FPR/unique phase causality/scientific PASS. "
      "No dynamic deletion, threshold or enlarged sample is implemented. Agent has not executed the notebook or models. "
      "Eight successful receiver pins plus G-recorded sentencepiece/ftfy, WanPipeline and both VAE imports are probed; pip-check warnings are retained.\n\n"
      "Separate source call budget: "+json.dumps(prep["source_preparation_planned_calls"])+"\n\n"
      "Separate M05/media budget: "+json.dumps(prep["m05_planned_calls"])+"\n\n"+note)
    rows=[("code","from google.colab import drive\ndrive.mount('/content/drive')\n"),
          ("markdown",markdown),("code",f"SOURCE_SHA = {source_sha!r}\nFIXED = {fixed!r}\n"+setup),
          ("code",environment),("code",run),("code",DISPLAY)]
    cells=[]
    for i,(kind,source) in enumerate(rows):
        cell=dict(cell_type=kind,id=f"receiver-independent-source-milestone1b-{i}",metadata={},source=source.splitlines(keepends=True))
        if kind=="code":ast.parse(source);cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    doc=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
        kernelspec=dict(display_name="Python 3",language="python",name="python3"),language_info=dict(name="python"),
        candidate_binding=dict(candidate="trajectory-receiver-independent-source-milestone1b-v1",source_sha=source_sha,
            status="UNPUBLISHED_DRAFT" if source_sha is None else "PUBLISHED_SHA_BOUND",
            execution="user-run only; CPU/fake/static checks; no agent real execution")))
    output=Path(output);output.write_text(json.dumps(doc,indent=1)+"\n",encoding="utf-8");return output
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--source-sha");parser.add_argument("--output",type=Path,default=OUTPUT)
    args=parser.parse_args();print(build(args.source_sha,args.output))
