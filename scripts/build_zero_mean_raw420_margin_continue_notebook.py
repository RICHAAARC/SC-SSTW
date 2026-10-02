"""Build the one-step raw420 channel-margin user-run notebook."""
from pathlib import Path
import argparse,ast,json
from scripts import build_zero_mean_channel_margin_notebook as previous
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'notebooks/zero_mean_raw420_margin_continue_v1_colab.ipynb'

def build(source_sha=None,output=OUTPUT):
    cfg=json.loads((ROOT/'experiments/wan_state_clock/configs/zero_mean_raw420_margin_continue_v1.json').read_text())
    path=previous.build(source_sha,output);nb=json.loads(path.read_text())
    setup=''.join(nb['cells'][2]['source']).replace('Zero-Mean-Channel-Margin-V1','Zero-Mean-Raw420-Margin-Continue-V1').replace('SC-SSTW-CHANNEL-MARGIN-','SC-SSTW-RAW420-MARGIN-CONTINUE-')
    lines=[]
    for line in setup.splitlines(keepends=True):
        if line.startswith('INPUT_ROOT='):line=f"INPUT_ROOT=Path({cfg['input']['root']!r})\n"
        elif line.startswith('setup=dict('):line="setup=dict(status='SETUP_STARTED',source_sha=SOURCE_SHA,fixed_denominator="+repr(cfg['fixed_denominator'])+",observations={o:{'status':'NOT_RUN_SETUP'} for o in "+repr(cfg['observations'])+"})\n"
        lines.append(line)
    nb['cells'][2]['source']=''.join(lines).splitlines(keepends=True)
    nb['cells'][4]['source']=''.join(nb['cells'][4]['source']).replace('zero_mean_channel_margin_run','zero_mean_raw420_margin_continue_run').splitlines(keepends=True)
    nb['cells'][1]['source']=(ROOT/'docs/zero_mean_raw420_margin_continue_v1.md').read_text().splitlines(keepends=True)
    display="""result=json.loads(RESULT_PATH.read_text())
if result['source_sha']!=SOURCE_SHA or result['fixed_denominator']!=setup['fixed_denominator']:raise RuntimeError('result source/denominator mismatch')
print('Status:',result['status'],'Counts:',result.get('counts'))
print('Calls:',result['calls'])
print('Baseline:',result.get('baseline_semantics'))
for name,row in result['comparisons'].items():print(name,row)
for name,row in result['updates'].items():print('Update:',name,row)
print('Baseline replay:',result['baseline_replay'])
print('Fixed endpoint:',result['fixed_endpoint'])
for name,row in result['posthoc'].items():print(name,row)
for name,row in result['message_evaluations'].items():print(name,row)
for name,row in result['observations'].items():print(name,row['status'],'vs baseline=',row.get('quality_vs_baseline'),'vs previous=',row.get('quality_vs_previous_available'))
print('Resources:',result['resources'])
print('Failures:',result['failures'])
print(result['evidence_ceiling'])
print('Retained result:',RESULT_PATH)
"""
    nb['cells'][5]['source']=display.splitlines(keepends=True)
    for i,c in enumerate(nb['cells']):
        c['id']=f'raw420-margin-continue-{i}'
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    path.write_text(json.dumps(nb,indent=1)+'\n');return path
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-sha');a=p.parse_args();print(build(a.source_sha))
