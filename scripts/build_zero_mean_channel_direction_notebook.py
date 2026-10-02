"""Build the fixed-direction channel attribution notebook."""
from pathlib import Path
import argparse,ast,json
from scripts import build_zero_mean_channel_margin_notebook as previous
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'notebooks/zero_mean_channel_direction_v1_colab.ipynb'

def build(source_sha=None,output=OUTPUT):
    cfg=json.loads((ROOT/'experiments/wan_state_clock/configs/zero_mean_channel_direction_v1.json').read_text())
    path=previous.build(source_sha,output);nb=json.loads(path.read_text())
    setup=''.join(nb['cells'][2]['source']).replace('Zero-Mean-Channel-Margin-V1','Zero-Mean-Channel-Direction-V1').replace('SC-SSTW-CHANNEL-MARGIN-','SC-SSTW-CHANNEL-DIRECTION-')
    lines=[]
    for line in setup.splitlines(keepends=True):
        if line.startswith('INPUT_ROOT='):line=f"INPUT_ROOT=Path({cfg['input']['root']!r})\n"
        elif line.startswith('REFERENCE_ROOT='):line=f"REFERENCE_ROOT=Path({cfg['reference']['root']!r})\n"
        elif line.startswith('setup=dict('):line="setup=dict(status='SETUP_STARTED',source_sha=SOURCE_SHA,fixed_denominator="+repr(cfg['fixed_denominator'])+",observations={o:{'status':'NOT_RUN_SETUP'} for o in "+repr(cfg['observations'])+"})\n"
        lines.append(line)
    lines.append(f"TERMINAL_ROOT=Path({cfg['terminal']['root']!r})\n")
    nb['cells'][2]['source']=''.join(lines).splitlines(keepends=True)
    nb['cells'][4]['source']=''.join(nb['cells'][4]['source']).replace('zero_mean_channel_margin_run','zero_mean_channel_direction_run').replace("'--reference-root',str(REFERENCE_ROOT)]","'--reference-root',str(REFERENCE_ROOT),'--terminal-root',str(TERMINAL_ROOT)]").splitlines(keepends=True)
    nb['cells'][3]['source']=''.join(nb['cells'][3]['source']).replace("    logged(['ffmpeg','-version'],'FFMPEG_PROBE');logged(['ffprobe','-version'],'FFPROBE_PROBE')\n",'').splitlines(keepends=True)
    nb['cells'][1]['source']=(ROOT/'docs/zero_mean_channel_direction_v1.md').read_text().splitlines(keepends=True)
    display="""result=json.loads(RESULT_PATH.read_text())
if result['source_sha']!=SOURCE_SHA or result['fixed_denominator']!=setup['fixed_denominator']:raise RuntimeError('result source/denominator mismatch')
print('Status:',result['status'],'Counts:',result.get('counts'))
print('Calls:',result['calls'])
print('Raster alignment:',result.get('raster_alignment'))
for name,row in result['input_replay'].items():print('Input replay:',name,row)
for name,row in result['encoder_replays'].items():
    print('Encoder replay:',name,row['status'],'max error=',row.get('normalized_max_error'),'RMSE=',row.get('normalized_rmse'),'cotangent chunks=',len(row.get('cotangent_parts',[])))
for name,row in result['attributions'].items():
    print('Attribution:',name,row['status'],'objective=',row.get('objective'))
    print('Predictions:',{k:row.get(k) for k in ('stored_gradient_dot_step','stored_step_prediction_recomputed','terminal_prediction','actual_objective_change','actual_terminal_change_l2')})
    print('RGB dots:',{k:row.get(k,{}).get('total') for k in ('FLOAT','RGB8','RAW420')})
    print('Differences:',row.get('differences'),'telescoping residual=',row.get('telescoping_residual'))
for name,row in result['reference_observations'].items():
    print('Reference:',name,row['posthoc'],'messages=',row['messages'])
print('Resources:',result['resources'])
print('Failures:',result['failures'])
print(result['evidence_ceiling'])
print('Retained result:',RESULT_PATH)
"""
    nb['cells'][5]['source']=display.splitlines(keepends=True)
    for i,c in enumerate(nb['cells']):
        c['id']=f'channel-direction-{i}'
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    path.write_text(json.dumps(nb,indent=1)+'\n');return path
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-sha');a=p.parse_args();print(build(a.source_sha))
