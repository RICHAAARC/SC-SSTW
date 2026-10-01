"""Build total-step4 to total-step7 continuation on the successful Colab setup."""
from pathlib import Path
import argparse,ast,json
from scripts import build_zero_mean_channel_margin_loop_notebook as previous
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'notebooks/zero_mean_channel_margin_continue_v1_colab.ipynb'

def build(source_sha=None,output=OUTPUT):
    cfg=json.loads((ROOT/'experiments/wan_state_clock/configs/zero_mean_channel_margin_continue_v1.json').read_text())
    path=previous.build(source_sha,output);nb=json.loads(path.read_text())
    setup=''.join(nb['cells'][2]['source']).replace('Zero-Mean-Channel-Margin-Loop-V1','Zero-Mean-Channel-Margin-Continue-V1').replace('SC-SSTW-CHANNEL-MARGIN-LOOP-','SC-SSTW-CHANNEL-MARGIN-CONTINUE-')
    setup=''.join(f"INPUT_ROOT=Path({cfg['parent_run']['root']!r})\n" if line.startswith('INPUT_ROOT=') else line for line in setup.splitlines(keepends=True))
    nb['cells'][2]['source']=setup.splitlines(keepends=True)
    nb['cells'][4]['source']=''.join(nb['cells'][4]['source']).replace('zero_mean_channel_margin_loop_run','zero_mean_channel_margin_continue_run').splitlines(keepends=True)
    display=''.join(nb['cells'][5]['source']).replace('step1','start')
    nb['cells'][5]['source']=display.splitlines(keepends=True)
    intro="""# Fixed continuation: total update 4 to total update 7

Click **Run all**. The source is pinned below. Start from the audited POINT4 terminal of run 20261001T150038391655Z, with Delta=-4.3296350339845635e-7 and true rank2. Local POINT1 replays total-step4; POINT2, POINT3 and POINT4 are total-steps5,6,7. The fixed endpoint is local POINT4 / total-step7, even if an earlier point wins. Exactly three fresh-gradient updates; no best-point selection, early stopping, scan or automatic extension.

Unchanged: eta174, per-update shrink-only L2cap1, original1392 pilot coefficients, existing global-worst plus all-pair loss and public template margins, g0/R44, all174 valid paths and both keys. Each point uses native VAE decode, RGB8, materialized raw YUV444, reopened RGB24 and native VAE encode. Each update recomputes both VJPs. Identity STE is a direction estimate; actual channel readback determines scores. Paths88,90,83 remain ordinary members of the full candidate set; no candidate is removed and no initial position is imposed on the reader.

Four observations / three updates / eight path reads / eight payload reads / sixteen message comparisons /1392 valid costs. Existing negative controls remain historical references. Every point reports top catalog indices, Delta, rank, global target and shortfall, remaining active local hinges, payload and quality. Payload is a separate repeated-bit diagnostic; exact bits do not establish synchronized decoding.

Quality is measured against the original already-marked RGB, the saved total-step4 RGB, and the preceding new point. It is not quality relative to OFF. Per-update cap is1; this batch adds at most3 to the sum of step norms in ideal arithmetic. Prior sum0.10940240292144192 is carried forward. Actual displacement from the original terminal and from this batch start are recorded separately; no new cumulative clipping.

The completed preceding three-update run spent38.54minutes in gradient phases, with19.35GiB peak allocated CUDA memory and64.53GiB peak temporary disk. Expect roughly40minutes plus model loading, media and Drive persistence; no runtime guarantee or GPU-name gate. Current-Python setup and fresh child process follow the successful run. Gradient phases clean their temporary storage separately.

Local CPU/static checks do not execute pretrained weights. This notebook leaves real execution to the user. This remains a one-source terminal-development diagnostic: no Transformer replay, generation-time trajectory embedding, H264/MP4 robustness, unknown-phase synchronization, calibrated FPR or scientific PASS is claimed.
"""
    nb['cells'][1]['source']=intro.splitlines(keepends=True)
    for i,c in enumerate(nb['cells']):
        c['id']=f'channel-margin-continue-{i}'
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    path.write_text(json.dumps(nb,indent=1)+'\n');return path

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-sha');args=parser.parse_args();print(build(args.source_sha))
