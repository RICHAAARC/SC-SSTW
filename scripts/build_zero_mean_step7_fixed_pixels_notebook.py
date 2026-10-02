"""Build the frozen step7 raw444/MP4 user-run comparison."""
from pathlib import Path
import argparse,ast,json
from scripts import build_zero_mean_channel_margin_continue_notebook as previous
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'notebooks/zero_mean_step7_fixed_pixels_v1_colab.ipynb'

def build(source_sha=None,output=OUTPUT):
    cfg=json.loads((ROOT/'experiments/wan_state_clock/configs/zero_mean_step7_fixed_pixels_v1.json').read_text())
    path=previous.build(source_sha,output);nb=json.loads(path.read_text())
    setup=''.join(nb['cells'][2]['source']).replace('Zero-Mean-Channel-Margin-Continue-V1','Zero-Mean-Step7-Fixed-Pixels-V1').replace('SC-SSTW-CHANNEL-MARGIN-CONTINUE-','SC-SSTW-STEP7-FIXED-PIXELS-')
    lines=[]
    for line in setup.splitlines(keepends=True):
        if line.startswith('INPUT_ROOT='):line=f"INPUT_ROOT=Path({cfg['parent_run']['root']!r})\n"
        elif line.startswith('setup=dict('):line="setup=dict(status='SETUP_STARTED',source_sha=SOURCE_SHA,fixed_denominator="+repr(cfg['fixed_denominator'])+",observations={p:{'status':'NOT_RUN_SETUP'} for p in ('SAVED_RAW444','RAW420','MP4')})\n"
        lines.append(line)
    nb['cells'][2]['source']=''.join(lines).splitlines(keepends=True)
    nb['cells'][4]['source']=''.join(nb['cells'][4]['source']).replace('zero_mean_channel_margin_continue_run','zero_mean_step7_fixed_pixels_run').splitlines(keepends=True)
    intro="""# Frozen original step7 pixels: encoder replay, raw420 and MP4

Click **Run all**. Reuses original step7 POINT4/rgb8.pt and its saved positive raw444 roundtrip.rgb from run20261001T170749052584Z. The pixel files and raster hashes are fixed. No terminal decode or new watermark update. The later MP4 run's newly decoded pixels are not used.

Three observations use one fixed native FP32 VAE encoder:

- SAVED_RAW444: directly re-encode the exact saved positive raw444 RGB. This isolates encoder/readout replay from decoder and color-conversion replay.
- RAW420: original saved RGB8→materialized raw YUV420→reopened RGB24→VAE encode.
- MP4: the same original saved RGB8→existing libx264 MP4,CRF18,8fps,yuv420p→independent RGB24 readback→VAE encode. MP4 remains the fixed primary endpoint.

Three observations, six path and payload reads, twelve message evaluations,1044 valid costs. Three VAE encodes, zero VAE decodes, zero gradient or Transformer calls. Same g0/R44 reader, all174 valid candidate paths, both keys and both messages. No candidate deletion, threshold change, best-point selection or scan. Each branch preserves failures and the other branches can finish independently.

The original anchor had rank1 and Delta=5.4846719567056e-7. The last fresh-decode run changed its pixel raster and yielded raw444 rank2 and MP4 rank46. This run removes decoder replay variation by holding the original pixels byte-exact. Same input bytes do not force identical VAE outputs: normalized max/RMS errors and correct-key path-cost changes are reported, along with actual package versions. No hard requirement to recreate an older CUDA build or use a particular GPU.

If the anchor changes, encoder execution is implicated without decoder ambiguity. RAW420 measures the color420 transport contrast; MP4 versus RAW420 assesses additional standard codec transport. Internal color conversions are not assumed algebraically identical: commands, verbose raw logs and MP4 stream metadata are retained. This is a diagnostic contrast, not unconditional attribution to a single codec operation.

All181 frames remain320x512. Historical negative controls remain references, not fresh independent negatives. Quality is relative to marked original RGB8, saved raw444 and current raw420, not OFF. Exact repeated payload is a separate readout from path recovery. These fixed-source, fixed-phase results do not establish unknown-phase/fragment synchronization, calibrated FPR or generation-time trajectory watermarking.

Existing current-Python setup and fresh-child execution are reused. Local CPU/static validation uses simulated model execution; real pretrained VAE execution is left to the user. Source is immutable when bound below.
"""
    nb['cells'][1]['source']=intro.splitlines(keepends=True)
    display="""result=json.loads(RESULT_PATH.read_text())
print('Status:',result['status'],'Counts:',result.get('counts'),'Calls:',result['calls'])
print('Saved raster matches parent:',result.get('saved_raster_matches_parent'))
print('Environment package changes:',result.get('environment_package_changes'))
print('SAVED_RAW444 encoder replay:',result.get('baseline_replay'))
for name,row in result['comparisons'].items():print(name,row)
print('Fixed endpoint:',result.get('fixed_endpoint'))
print('MP4:',result['mp4'])
for name,row in result['posthoc'].items():print(name,row)
for name,row in result['message_evaluations'].items():print(name,row)
for name,row in result['observations'].items():
    print(name,'status=',row['status'],'quality_vs_shared_q8=',row.get('quality_vs_shared_q8'),
          'quality_vs_parent_raw444=',row.get('quality_vs_parent_raw444'),
          'quality_vs_current_raw420=',row.get('quality_vs_current_raw420'))
print('Failures:',result['failures'])
print(result['evidence_ceiling'])
print('Retained result:',RESULT_PATH)
"""
    nb['cells'][5]['source']=display.splitlines(keepends=True)
    for i,c in enumerate(nb['cells']):
        c['id']=f'step7-fixed-pixels-{i}'
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    path.write_text(json.dumps(nb,indent=1)+'\n');return path

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-sha');args=parser.parse_args();print(build(args.source_sha))
