"""Build the fixed receiver-prepend1-stage1 diagnostic; source binding is publication-only."""
from pathlib import Path
import argparse,ast,json,re,sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import build_video_trajectory_payload_framewise_sync_m05_notebook as prior
OUTPUT=ROOT/"notebooks/video_trajectory_receiver_prepend1_v1_colab.ipynb"

DISPLAY="""import statistics
result=json.loads(RESULT_PATH.read_text())
if result['source_sha']!=SOURCE_SHA or result['fixed_denominator']!=FIXED:raise RuntimeError('result identity mismatch')
print('Status:',result['status'],'Counts:',result['counts'],'Calls:',result['calls'])
print('Actual environment:',result['environment'])
print('Blind seal:',result.get('blind_receiver_sha256'))
for sid,row in sorted(result['payload_posthoc'].items()):
    bits=row.get('bit_rows',[])
    valid=[x for x in bits if 'signed_normalized_margin' in x]
    print(sid,row['status'],'condition=',row.get('condition'),'arm=',row.get('arm'),
          'key=',row.get('key_role'),'errors=',row.get('error_bits'),
          'signed margin min=',min((x['signed_normalized_margin'] for x in valid),default=None))
    print('  per-payload-channel medians:',[
        statistics.median([x['signed_normalized_margin'] for x in valid if x['payload_channel']==ch])
        if any(x['payload_channel']==ch for x in valid) else None for ch in range(4)])
    print('  complete vote/time artifacts:',result['payload_reads'][sid]['detail_path'],row.get('time_bit_path'))
for sid,row in sorted(result['historical_comparisons'].items()):
    print('Historical baseline, postseal only:',sid,row['status'],'decoded_equal=',row.get('decoded_equal'),'votes_equal=',row.get('votes_equal'))
for sid,row in sorted(result['same_run_differences'].items()):
    print('Changed minus ORIGINAL:',sid,row['status'],'key=',row.get('key_role'),
          'bit_error_delta=',row.get('bit_error_delta'))
print('Full 44-time/channel summaries: result.json descriptive_summaries and same_run_differences')
print('Failures:',result['failures'])
print('Historical reference error:',result.get('historical_reference_error'))
print('Evidence ceiling:',result['evidence_ceiling'])
print('Result:',RESULT_PATH)
"""

def build(source_sha=None,output=OUTPUT):
    if source_sha is not None and not re.fullmatch(r"[0-9a-f]{40}",source_sha):
        raise ValueError("published immutable source SHA required")
    cfg=json.loads((ROOT/"experiments/wan_state_clock/configs/video_trajectory_receiver_prepend1_v1.json").read_text())
    fixed=cfg["fixed_denominator"]
    setup=prior.SETUP.replace("Trajectory-Payload-Framewise-Sync-M05","Trajectory-Receiver-Prepend1-Stage1-V1").replace(
        "SC-SSTW-TRAJECTORY-PAYLOAD-FRAMEWISE-SYNC-M05-","SC-SSTW-TRAJECTORY-RECEIVER-PREPEND1-STAGE1-V1-")
    before=prior.ENVIRONMENT.split("    import shutil\n")[0]
    after="    dependency_code="+prior.ENVIRONMENT.split("    dependency_code=",1)[1]
    # Fixed relevant G environment, with one necessary repair only; no media or
    # generation dependencies/probes and no model-name GPU gate.
    probe=("import importlib.metadata as m; pins="+repr(cfg["environment_pins"])+"; "
           "actual={k:m.version(k) for k in pins}; "
           "assert all(actual[k].split('+')[0]==v for k,v in pins.items()), actual; "
           "from diffusers import AutoencoderKLWan; import torch; print(actual)")
    environment=before+"    probe="+repr(probe)+"\n"
    environment+="    if logged([PYTHON,'-c',probe],'RECEIVER_DEPENDENCY_PROBE',check=False):\n"
    environment+="        logged([PYTHON,'-m','pip','install',"+",".join(repr(k+"=="+v) for k,v in cfg["environment_pins"].items())+"],'RECEIVER_DEPENDENCY_REPAIR')\n"
    environment+="        logged([PYTHON,'-c',probe],'RECEIVER_DEPENDENCY_REPROBE')\n"+after
    run=prior.RUN.replace("experiments.wan_state_clock.video_trajectory_payload_framewise_sync_v1_run",
                          "experiments.wan_state_clock.video_trajectory_receiver_prepend1_v1_run")
    note="Published source: "+source_sha if source_sha else "UNPUBLISHED DRAFT: SOURCE_SHA=None; publish reviewed source and bind its immutable SHA before Run all."
    markdown=("# Receiver-Prepend1-V1: fixed Stage1 controlled diagnostic\n\n"
      "Known-start fixture using only saved G FULL181 P0/P1 RGB: each condition has S0_IDENTITY source[0:177], "
      "S1_ORIGINAL clip=source[1:178], S1_PREPEND1_DROP1 concat(clip[:1],clip[:-1]), and "
      "S1_TAIL_REPEAT concat(clip[:-1],clip[-2:-1]). The latter two drop source f177 and duplicate f1/f176 respectively. "
      "S0 old/new rule aliases share one physical observation. No candidate selection.\n\n"
      "Exactly 8 Wan FP32 posterior-mode encodes and 16 K0/K1 reads, all 177 frames, R44, "
      "675840 detailed votes, 22528 time-bit rows, 512 final bits. Failed slots remain. "
      "Original coordinates, Counter order and ties stay unchanged. The receiver operation takes only clip and public operation. "
      "Source-frame maps, condition, truth and eight historical baseline comparisons join only after blind seal. "
      "Historical disagreement never triggers retries. Same-run PREPEND/TAIL minus ORIGINAL differences and full time/channel "
      "summaries are descriptive; they cannot choose a strategy, erase boundaries or select channels.\n\n"
      "This is not blind synchronization, recovery of f0/cache or proof of unique phase causality. "
      "Temporal VAE chunks have cross-chunk causal cache; receiver latent indices do not establish independent four-frame support. "
      "TAIL changes only the last input frame; any early/late differences are reported without hard scientific PASS criteria. "
      "Stage2 is not implemented or executed in this Stage1 notebook; contingent on Stage1 result review. "
      "M05, R22, strength/smoothing choices are outside this Stage1 implementation. "
      "No generation, framewise VAE, writer, codec, new quality or sync search.\n\n"
      "Only CPU/fake/static validation before handoff; real Run all is for the user after immutable source publication. "
      "Eight relevant package pins reuse the successful origin environment; actual versions and pip-check warnings persist.\n\n"
      "Fixed inputs:\n"+"\n".join("- "+c+": "+v["path"]+"; SHA256 "+v["sha256"] for c,v in cfg["inputs"].items())+
      "\n\n"+note)
    rows=[("code","from google.colab import drive\ndrive.mount('/content/drive')\n"),
          ("markdown",markdown),("code",f"SOURCE_SHA = {source_sha!r}\nFIXED = {fixed!r}\n"+setup),
          ("code",environment),("code",run),("code",DISPLAY)]
    cells=[]
    for i,(kind,source) in enumerate(rows):
        cell=dict(cell_type=kind,id=f"receiver-prepend1-stage1-{i}",metadata={},source=source.splitlines(keepends=True))
        if kind=="code":ast.parse(source);cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    doc=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
        kernelspec=dict(display_name="Python 3",language="python",name="python3"),language_info=dict(name="python"),
        candidate_binding=dict(candidate="trajectory-receiver-prepend1-stage1-v1",source_sha=source_sha,
            status="UNPUBLISHED_DRAFT" if source_sha is None else "PUBLISHED_SHA_BOUND",
            execution="user-run only; CPU/fake/static validation, no agent real execution")))
    output=Path(output);output.write_text(json.dumps(doc,indent=1)+"\n",encoding="utf-8");return output
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--source-sha");parser.add_argument("--output",type=Path,default=OUTPUT)
    args=parser.parse_args();print(build(args.source_sha,args.output))

