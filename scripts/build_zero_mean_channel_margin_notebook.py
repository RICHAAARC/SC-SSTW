"""Build a draft or source-pinned user-run single-update notebook."""
from pathlib import Path
import argparse,ast,json,re
from scripts import build_zero_mean_c1_yuv444_no_h264_notebook as previous
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'notebooks/zero_mean_channel_margin_v1_colab.ipynb'

def build(source_sha=None,output=OUTPUT):
    if source_sha is not None and not re.fullmatch('[0-9a-f]{40}',source_sha):raise ValueError('immutable source SHA required')
    cfg=json.loads((ROOT/'experiments/wan_state_clock/configs/zero_mean_channel_margin_v1.json').read_text())
    setup=previous.SETUP
    # Preserve logging/probe helpers; replace the old experiment-specific setup.
    setup=setup[:setup.index("ARMS=(")]+"setup=dict(status='SETUP_STARTED',source_sha=SOURCE_SHA)\n"+setup[setup.index('def write_json'):]
    setup=setup.replace('Zero-Mean-C1-YUV444-No-H264-V1','Zero-Mean-Channel-Margin-V1').replace('SC-SSTW-YUV444-NO-H264-','SC-SSTW-CHANNEL-MARGIN-')
    start=setup.index('INPUT_ROOT=');end=setup.index("setup=dict(")
    setup=setup[:start]+f"INPUT_ROOT=Path({cfg['input']['root']!r})\nREFERENCE_ROOT=Path({cfg['reference']['root']!r})\n"+setup[end:]
    setup=setup.replace("print('Fixed input:',INPUT_ROOT,'420 references:',REFERENCE420_ROOT,'New output:',OUTPUT)","print('Full writer terminal:',INPUT_ROOT,'Saved references:',REFERENCE_ROOT,'Output:',OUTPUT)")
    environment=previous.ENVIRONMENT
    run=previous.RUN.replace('zero_mean_c1_yuv444_no_h264_run','zero_mean_channel_margin_run').replace("'--reference420-root',str(REFERENCE420_ROOT)","'--reference-root',str(REFERENCE_ROOT)")
    display="""result=json.loads(RESULT_PATH.read_text())
print('Status:',result['status'],'Counts:',result.get('counts'),'Calls:',result['calls'])
print('Actual channel comparison:',result.get('comparison'))
print('Update:',result['update'].get('receipt'))
for name,row in result['posthoc'].items():print(name,row)
for name,row in result['message_evaluations'].items():print(name,row)
print('Quality:',result['observations']['AFTER'].get('quality'))
print('Resources:',result['resources'])
print('Failures:',result['failures'])
print(result['evidence_ceiling'])
print('Retained result:',RESULT_PATH)
"""
    intro="""# One saved-terminal channel-margin update

Run all loads the saved full OVERLAP_MULTI terminal and performs one update, with eta174/L2cap1 in the original pilot support. It retains all R44 positions and all173 wrong template classes. No diffusion generation or Transformer is invoked. Historical OFF/PAYLOAD_MULTI/OVERLAP_MULTI observations remain reference controls.

Both BEFORE and AFTER use actual materialized RGB8→YUV444→RGB24 and native FP32 VAE encoding. The identity straight-through color derivative supplies direction only. Native causal encoder/decoder VJPs run separately with checkpoint disk storage. Resource peaks are recorded; full-size feasibility is pending. Current Python, fresh child and existing dependency repair are used without a GPU-model gate.

The loss combines a worst-path hinge and all-candidate local contrast hinges; margins are half of public ideal-template separation. This is a proposed writer objective, not a new detector threshold. Exactly one step and no scans. Compare actual Delta, full candidate ranks, repeated payload and RGB quality. A terminal update is not generation-time trajectory watermarking; phase0/R44, unknown phase and MP4 remain separate.

This notebook has static/CPU validation only. Running it executes real VAE/FFmpeg work. Full weights/GPU execution is left to the user. Source binding is pending publication when SOURCE_SHA is None.
"""
    sources=[('code',"from google.colab import drive\ndrive.mount('/content/drive')\n"),('markdown',intro),
      ('code',f'SOURCE_SHA = {source_sha!r}\n'+setup),('code',environment),('code',run),('code',display)]
    cells=[]
    for i,(kind,text) in enumerate(sources):
        cell=dict(cell_type=kind,id=f'channel-margin-{i}',metadata={},source=text.splitlines(keepends=True))
        if kind=='code':ast.parse(text);cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3'),
      language_info=dict(name='python'),candidate_binding=dict(source_sha=source_sha,status='PUBLISHED_SHA_BOUND' if source_sha else 'UNPUBLISHED_DRAFT')))
    Path(output).write_text(json.dumps(nb,indent=1)+'\n');return Path(output)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-sha');args=parser.parse_args();print(build(args.source_sha))
