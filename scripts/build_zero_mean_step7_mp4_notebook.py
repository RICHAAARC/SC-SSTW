"""Build the frozen step7 raw444/MP4 user-run comparison."""
from pathlib import Path
import argparse,ast,json
from scripts import build_zero_mean_channel_margin_continue_notebook as previous
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'notebooks/zero_mean_step7_mp4_v1_colab.ipynb'

def build(source_sha=None,output=OUTPUT):
    cfg=json.loads((ROOT/'experiments/wan_state_clock/configs/zero_mean_step7_mp4_v1.json').read_text())
    path=previous.build(source_sha,output);nb=json.loads(path.read_text())
    setup=''.join(nb['cells'][2]['source']).replace('Zero-Mean-Channel-Margin-Continue-V1','Zero-Mean-Step7-MP4-V1').replace('SC-SSTW-CHANNEL-MARGIN-CONTINUE-','SC-SSTW-STEP7-MP4-')
    lines=[]
    for line in setup.splitlines(keepends=True):
        if line.startswith('INPUT_ROOT='):line=f"INPUT_ROOT=Path({cfg['parent_run']['root']!r})\n"
        elif line.startswith('setup=dict('):line="setup=dict(status='SETUP_STARTED',source_sha=SOURCE_SHA,fixed_denominator="+repr(cfg['fixed_denominator'])+",observations={p:{'status':'NOT_RUN_SETUP'} for p in ('RAW444','MP4')})\n"
        lines.append(line)
    nb['cells'][2]['source']=''.join(lines).splitlines(keepends=True)
    nb['cells'][4]['source']=''.join(nb['cells'][4]['source']).replace('zero_mean_channel_margin_continue_run','zero_mean_step7_mp4_run').splitlines(keepends=True)
    intro="""# Frozen step7: actual MP4 save and readback

Click **Run all**. Loads the audited total-step7 POINT4 terminal from run20261001T170749052584Z (true rank1, Delta=5.4846719567056e-7). The fixed primary observation is MP4. No optimization, step6 substitution, new generation or parameter scan.

One native FP32 VAE decode produces a single RGB8 raster for two fixed branches:

- RAW444: materialized RGB8→raw YUV444→reopened RGB24, the previous positive reference channel.
- MP4: the existing project encoder, H264/libx264, CRF18,8fps,yuv420p, followed by independent MP4 readback and native VAE encoding. All181 frames remain320x512; reader stays g0/R44, all174 valid candidates and both keys.

Two normalized observations, four path reads, four payload reads, eight message evaluations,696 valid costs. One VAE decode, two VAE encodes, one MP4 save/read; zero Transformer or gradient calls. This should avoid the previous gradient run's87.7minute workload and disk spool; actual runtime depends on model loading and hardware. No GPU-name gate is added. Existing current-Python dependency repair and fresh-child execution are reused.

The start terminal/result/parent raw444 bytes, normalized observation and both raw readers are checked against their saved identities. Historical OFF/PAYLOAD_MULTI/OVERLAP_MULTI controls remain references, not new FPR samples. Fresh RAW444 replay differences are reported, not hidden. A failed branch remains in the fixed denominator while the other branch can finish.

Inspect both branches' Delta, rank, nearest competition in retained costs, public-margin shortfall, payload and actual RGB quality. Quality is relative to the same marked input raster and raw444, not OFF. RAW444-vs-MP4 changes both chroma subsampling and lossy compression, so a difference does not isolate H264 alone. Exact repeated payload bits remain a separate readout from path recovery.

This is one frozen terminal, one fixed codec configuration, fixed phase and one source. It does not establish generation-time trajectory watermarking, unknown-phase/fragment synchronization, calibrated FPR or broad robustness. Source is immutable when bound below. Local validation is CPU/static with simulated VAE; real pretrained VAE execution is left to the user.
"""
    nb['cells'][1]['source']=intro.splitlines(keepends=True)
    display="""result=json.loads(RESULT_PATH.read_text())
print('Status:',result['status'],'Counts:',result.get('counts'),'Calls:',result['calls'])
print('Decoded raster matches parent:',result.get('decoded_raster_matches_parent'))
print('RAW444 baseline replay:',result.get('baseline_replay'))
for name,row in result['comparisons'].items():print(name,row)
print('Fixed endpoint:',result.get('fixed_endpoint'))
print('MP4:',result['mp4'])
for name,row in result['posthoc'].items():print(name,row)
for name,row in result['message_evaluations'].items():print(name,row)
for name,row in result['observations'].items():
    print(name,'status=',row['status'],'quality_vs_shared_q8=',row.get('quality_vs_shared_q8'),
          'quality_vs_parent_raw444=',row.get('quality_vs_parent_raw444'),
          'quality_vs_current_raw444=',row.get('quality_vs_current_raw444'))
print('Failures:',result['failures'])
print(result['evidence_ceiling'])
print('Retained result:',RESULT_PATH)
"""
    nb['cells'][5]['source']=display.splitlines(keepends=True)
    for i,c in enumerate(nb['cells']):
        c['id']=f'step7-mp4-{i}'
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    path.write_text(json.dumps(nb,indent=1)+'\n');return path

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-sha');args=parser.parse_args();print(build(args.source_sha))
