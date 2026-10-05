"""Build the fixed receiver-origin diagnostic; source binding is publication-only."""
from pathlib import Path
import argparse,ast,json,re,sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import build_video_trajectory_payload_framewise_sync_m05_notebook as prior
OUTPUT=ROOT/"notebooks/video_trajectory_receiver_origin_v1_colab.ipynb"

DISPLAY="""import statistics
result=json.loads(RESULT_PATH.read_text())
if result['source_sha']!=SOURCE_SHA or result['fixed_denominator']!=FIXED:raise RuntimeError('result identity mismatch')
print('Status:',result['status'],'Counts:',result['counts'],'Calls:',result['calls'])
print('Actual environment:',result['environment'])
print('Blind seal:',result.get('blind_receiver_sha256'))
for sid,row in sorted(result['payload_posthoc'].items()):
    bits=row.get('bit_rows',[])
    valid=[x for x in bits if 'signed_normalized_margin' in x]
    print(sid,row['status'],'condition=',row.get('condition'),'start=',row.get('source_start_posthoc_only'),
          'key=',row.get('key_role'),'errors=',row.get('error_bits'),
          'signed margin min=',min((x['signed_normalized_margin'] for x in valid),default=None))
    print('  per-payload-channel medians:',[
        statistics.median([x['signed_normalized_margin'] for x in valid if x['payload_channel']==ch])
        if any(x['payload_channel']==ch for x in valid) else None for ch in range(4)])
    print('  complete vote/time artifacts:',result['payload_reads'][sid]['detail_path'],row.get('time_bit_path'))
for sid,row in sorted(result['historical_comparisons'].items()):
    print('Historical start1, postseal only:',sid,row['status'],'decoded_equal=',row.get('decoded_equal'),'votes_equal=',row.get('votes_equal'))
print('Failures:',result['failures'])
print('Historical reference error:',result.get('historical_reference_error'))
print('Evidence ceiling:',result['evidence_ceiling'])
print('Result:',RESULT_PATH)
"""

def build(source_sha=None,output=OUTPUT):
    if source_sha is not None and not re.fullmatch(r"[0-9a-f]{40}",source_sha):
        raise ValueError("published immutable source SHA required")
    cfg=json.loads((ROOT/"experiments/wan_state_clock/configs/video_trajectory_receiver_origin_v1.json").read_text())
    fixed=cfg["fixed_denominator"]
    setup=prior.SETUP.replace("Trajectory-Payload-Framewise-Sync-M05","Trajectory-Receiver-Origin-V1").replace(
        "SC-SSTW-TRAJECTORY-PAYLOAD-FRAMEWISE-SYNC-M05-","SC-SSTW-TRAJECTORY-RECEIVER-ORIGIN-V1-")
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
                          "experiments.wan_state_clock.video_trajectory_receiver_origin_v1_run")
    note="Published source: "+source_sha if source_sha else "UNPUBLISHED DRAFT: SOURCE_SHA=None; publish reviewed source and bind its immutable SHA before Run all."
    markdown=("# Saved G receiver-origin compatibility diagnostic\n\n"
      "Fixed G run 20261005T140121854219Z: saved FULL181 MP4-readback RGB P0/P1, each cut at [0:177] and [1:178]. "
      "No new video, writer, framewise VAE, codec, sync search, or offset restoration. Exactly four Wan receiver encodes, "
      "eight K0/K1 reads, R44, 1320 votes/bit, 337920 detailed votes, 11264 time-bit rows, 256 final bits. "
      "Missing input or identity mismatch keeps failed slots; there is no codec fallback or retry selection.\n\n"
      "The original reader is unchanged. Signed votes mean 2*int(FFT.real>0)-1; exact coefficient zero gives -1, "
      "not erasure. Original time-major/frequency-subsequence order and Counter first-encounter tie decisions are retained. "
      "Receiver latent indices and nominal stride4 coordinates are indexing labels, not independent four-frame supports or "
      "definite source-frame attribution. Current readouts are sealed before joining condition/start/truth and before reading "
      "historical payload. Historical differences do not trigger reruns. This is a bounded compatibility diagnostic, not proof "
      "of phase causality, an offset-recovery implementation, FPR, or scientific PASS.\n\n"
      "The notebook has only CPU/fake/static validation before handoff. Real Run all is for the user after publication. "
      "Relevant dependency versions follow the saved G environment; pip check warnings remain recorded, with no unrelated repairs.\n\n"
      "Fixed inputs:\n"+ "\n".join("- "+c+": "+s["path"]+"; SHA256 "+s["sha256"] for c,s in cfg["inputs"].items())+
      "\n\n"+note)
    rows=[("code","from google.colab import drive\ndrive.mount('/content/drive')\n"),
          ("markdown",markdown),("code",f"SOURCE_SHA = {source_sha!r}\nFIXED = {fixed!r}\n"+setup),
          ("code",environment),("code",run),("code",DISPLAY)]
    cells=[]
    for i,(kind,source) in enumerate(rows):
        cell=dict(cell_type=kind,id=f"receiver-origin-{i}",metadata={},source=source.splitlines(keepends=True))
        if kind=="code":ast.parse(source);cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    doc=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
        kernelspec=dict(display_name="Python 3",language="python",name="python3"),language_info=dict(name="python"),
        candidate_binding=dict(candidate="trajectory-receiver-origin-v1",source_sha=source_sha,
            status="UNPUBLISHED_DRAFT" if source_sha is None else "PUBLISHED_SHA_BOUND",
            execution="user-run only; CPU/fake/static validation, no agent real execution")))
    output=Path(output);output.write_text(json.dumps(doc,indent=1)+"\n",encoding="utf-8");return output
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--source-sha");parser.add_argument("--output",type=Path,default=OUTPUT)
    args=parser.parse_args();print(build(args.source_sha,args.output))

