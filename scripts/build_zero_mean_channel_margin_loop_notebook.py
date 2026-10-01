"""Build the fixed three-update continuation using the successful setup path."""
from pathlib import Path
import argparse,ast,json
from scripts import build_zero_mean_channel_margin_notebook as previous
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'notebooks/zero_mean_channel_margin_loop_v1_colab.ipynb'

def build(source_sha=None,output=OUTPUT):
    cfg=json.loads((ROOT/'experiments/wan_state_clock/configs/zero_mean_channel_margin_loop_v1.json').read_text())
    path=previous.build(source_sha,output);nb=json.loads(path.read_text())
    setup=''.join(nb['cells'][2]['source'])
    setup=setup.replace('Zero-Mean-Channel-Margin-V1','Zero-Mean-Channel-Margin-Loop-V1').replace('SC-SSTW-CHANNEL-MARGIN-','SC-SSTW-CHANNEL-MARGIN-LOOP-')
    # Parent root is deliberately changed only on its exact assignment, after
    # the output namespace replacement, so the existing parent path stays V1.
    lines=setup.splitlines(keepends=True)
    lines=[f"INPUT_ROOT=Path({cfg['parent_run']['root']!r})\n" if line.startswith('INPUT_ROOT=') else line for line in lines]
    setup=''.join(lines)
    setup=setup.replace("setup=dict(status='SETUP_STARTED',source_sha=SOURCE_SHA)",
        "setup=dict(status='SETUP_STARTED',source_sha=SOURCE_SHA, fixed_denominator="+repr(cfg['fixed_denominator'])+", observations={p:{'status':'NOT_RUN_SETUP'} for p in ('POINT1','POINT2','POINT3','POINT4')})")
    nb['cells'][2]['source']=setup.splitlines(keepends=True)
    run=''.join(nb['cells'][4]['source']).replace('zero_mean_channel_margin_run','zero_mean_channel_margin_loop_run')
    nb['cells'][4]['source']=run.splitlines(keepends=True)
    display="""result=json.loads(RESULT_PATH.read_text())
print('Status:',result['status'],'Counts:',result.get('counts'),'Calls:',result['calls'])
print('Baseline replay:',result.get('baseline_replay'))
for point,row in result['comparisons'].items():print(point,row)
print('Fixed endpoint:',result.get('fixed_endpoint'))
for name,row in result['updates'].items():
    print(name,row['status'],'update=',row.get('receipt'),'sum_new_step_l2=',row.get('sum_new_step_l2'),
          'original_terminal_displacement_l2=',row.get('original_terminal_displacement_l2'))
for name,row in result['message_evaluations'].items():print(name,row)
for point,row in result['observations'].items():
    print(point,'quality_vs_original_marked=',row.get('quality_vs_original_marked'),
          'quality_vs_step1=',row.get('quality_vs_step1'),'quality_vs_previous=',row.get('quality_vs_previous_point'))
print('Resources:',result['resources'])
print('Failures:',result['failures'])
print(result['evidence_ceiling'])
print('Retained result:',RESULT_PATH)
"""
    nb['cells'][5]['source']=display.splitlines(keepends=True)
    intro="""# Fixed channel-margin continuation: three new updates

Run all starts from the actual AFTER/terminal.pt saved in run20261001T133512881485Z. POINT1 replays that existing step1; POINT2–POINT4 apply exactly three additional fresh-gradient updates. The primary endpoint is fixed POINT4. All intermediate successes, regressions, zero-update outcomes and failures remain in the result. No best-iterate selection or stopping when rank1 is first reached.

Unchanged: original code/basis/write support, receiver g0/R44/all174valid candidates, public template-derived margin loss, eta174 and per-step L2cap1. Each update uses its own current terminal→native VAE decode→RGB8→actual materialized YUV444→RGB24→native VAE encode. Encoder and decoder VJPs are recomputed every time. The identity STE supplies a direction only; actual roundtrip observations determine Delta/rank. There is no new total-displacement clipping: new-step L2 sums are bounded by3 in ideal arithmetic; cumulative actual change from the original marked terminal is reported separately.

Four new full observations, eight path/payload reads, sixteen message comparisons and1392valid costs. Historical OFF/PAYLOAD_MULTI/OVERLAP_MULTI controls and the prior step1 are preserved as references, not independent new negatives. Quality is reported against the original marked terminal's roundtrip, the saved step1 and the preceding point; none is an unmarked-source quality certificate.

The previous real run used about13.3minutes for one gradient pair,19.35GiB peak allocated CUDA memory and64.53GiB temporary disk. Three pairs suggest roughly40minutes plus media/model-loading/persistence overhead, not a runtime guarantee. Both gradient phases run separately and clean their temporary storage. No GPU-model gate; use the successful current-Python setup and a fresh child. No Transformer or generation replay, no H264/MP4, no unknown-phase claim.

This fixed same-source development diagnostic has local CPU/static validation. Real pretrained VAE/FFmpeg execution is for the user. Source is immutable when SOURCE_SHA is bound; an unbound draft is not a released notebook.
"""
    nb['cells'][1]['source']=intro.splitlines(keepends=True)
    for i,c in enumerate(nb['cells']):
        c['id']=f'channel-margin-loop-{i}'
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    path.write_text(json.dumps(nb,indent=1)+'\n');return path

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-sha');args=parser.parse_args();print(build(args.source_sha))
