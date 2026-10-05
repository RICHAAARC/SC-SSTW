"""Build two fixed user-run G/T notebooks using the proven M05 process wrapper."""
from pathlib import Path
import argparse,ast,json,re
from scripts import build_video_trajectory_payload_framewise_sync_m05_notebook as prior
ROOT=Path(__file__).resolve().parents[1]

DISPLAY = """import statistics
result=json.loads(RESULT_PATH.read_text())
if result['source_sha']!=SOURCE_SHA or result['fixed_denominator']!=FIXED:raise RuntimeError('result identity mismatch')
print('Status:',result['status'],'Counts:',result['counts'])
print('Comparison calls:',result['calls'])
print('Source preparation:',{k:v for k,v in result['source_preparation'].items() if k not in ('steps','trajectory_receipt')})
print('Source steps complete:',sum(row['status']=='COMPLETE' for row in result['source_preparation']['steps']))
print('Source preparation calls:',result['source_calls'])
print('Frozen source identity:',result['source_protocol'])
for condition,row in result['writers'].items():
    print('Writer:',condition,'status=',row['status'],'path=',row.get('path'),
          'constraint/constructed/applied L2=',row.get('constraint_increment_l2',row.get('raw_target_delta_l2')),
          row.get('constructed_delta_l2'),row.get('applied_float32_delta_l2'),
          'full applied residual=',row.get('max_applied_full_residual'),'error=',row.get('error'))
def joined(group):
    for sid,row in sorted(result[group].items()):
        oid,key_id=sid.rsplit('/',1);obs=result['observations'][oid]
        value=dict(row);value.setdefault('condition',obs['condition_truth_join_only'])
        value.setdefault('view',obs['view_truth_join_only'])
        value.setdefault('key_role','REGISTERED' if key_id=='K0' else 'WRONG_KEY')
        yield sid,value
print('=== P1/P0 payload compatibility first ===')
for sid,row in joined('payload_posthoc'):
    if row['condition'] in ('P0_ORIGINAL_RGB','P1_FRAMEWISE_RECON'):
        print(sid,row['status'],row['key_role'],'bit_errors=',row.get('bit_errors'),'error=',row.get('error'))
print('=== All quality pairs: references are explicit ===')
for qid,row in result['quality'].items():
    print(qid,row)
print('=== All keys and views: truth joined after blind seal ===')
for sid,row in joined('sync_posthoc'):
    print('FULL geometry only' if row.get('full_singleton_geometry_only',row['view']=='FULL181') else 'Localization',
          sid,row['status'],row['condition'],row['view'],row['key_role'],
          'truth_score=',row.get('true_score'),'best_other=',row.get('best_other_score'),
          'gap=',row.get('true_score_gap'),'rank=',row.get('true_rank'),
          'unique_truth=',row.get('unique_truth'),'error=',row.get('error'),
          'all-offset readout=',result['sync_reads'][sid]['path'])
print('=== All payload reads; per-bit raw votes/margins/errors retained in result ===')
for sid,row in joined('payload_posthoc'):
    bits=row.get('bit_rows',[])
    margins=[v['margin'] for v in bits];normalized=[v['normalized_margin'] for v in bits]
    print(sid,row['status'],row['condition'],row['view'],row['key_role'],
          'R=',result['payload_reads'][sid].get('R'),'bit_errors=',row.get('bit_errors'),
          'margin_min/median=',(min(margins),statistics.median(margins)) if margins else None,
          'normalized_min/median=',(min(normalized),statistics.median(normalized)) if normalized else None,
          'error=',row.get('error'))
from IPython.display import Video,display
for condition,row in result['transport'].items():
    video=row.get('events',{}).get('mp4',{});path=video.get('path')
    if video.get('status')=='SAVED' and isinstance(path,str) and Path(path).is_file():
        try:print('MP4:',condition);display(Video(str(path),embed=True))
        except Exception as preview_error:print('Preview unavailable:',preview_error)
print('Failures:',result['failures']);print('Evidence ceiling:',result['evidence_ceiling']);print('Result:',RESULT_PATH)
"""
def build(profile,source_sha=None,output=None):
    if profile not in ('G','T'):raise ValueError('fixed notebook required')
    if source_sha is not None and not re.fullmatch(r'[0-9a-f]{40}',source_sha):raise ValueError('published immutable SHA required')
    output=Path(output) if output is not None else ROOT/('notebooks/video_trajectory_payload_'+profile.lower()+'_v1_colab.ipynb')
    cfg=json.loads((ROOT/('experiments/wan_state_clock/configs/video_trajectory_payload_'+profile.lower()+'_v1.json')).read_text())
    fixed=cfg['fixed_denominator']
    intro=(
        'G: one prespecified duck prompt and the original seed 2026092501. Source preparation is exactly one '
        'PAYLOAD_MULTI 50-step native trajectory and one native Wan decode, saved with RGB SHA before comparison. '
        'This is a new prompt case, not a new independent seed sample. No reselection or generation retry. '
        'P0/P1/M1/M05; FULL181, CROP177 [1:178], SHORT89 [37:126]. SHORT89 searches offsets 0..92 and '
        'uses the adopted public length support R22 (660 votes/bit); other views use R44 (1320 votes/bit). '
        'Original coordinates, signs, bit mapping and Counter tie decisions are unchanged.'
        if profile=='G' else
        'T: use the original saved RGB source with SHA '+cfg['source']['sha256']+'. '
        'P0/P1/M05/T05; FULL181 and CROP177 [1:178], R44. T05 starts from the same unmodified encoded '
        'source as M05, never from M05 output. Fixed lambda=1 minimizes ideal L2 squared plus unscaled '
        'adjacent temporal differences over all 181 frames under all original M05 full-tubelet projection '
        'equalities. All 180 noncyclic edges are included. Constructed L2 may increase; it is not a matched '
        'disturbance budget. Receipts separate constraint increments, ideal delta, and actual float32 '
        '(z+delta)-z, including inactive full constraints and partial CROP endpoints.'
    )
    note='Published source: '+source_sha if source_sha else 'UNPUBLISHED DRAFT: SOURCE_SHA=None. Await reviewed source publication and immutable SHA binding.'
    markdown=('# Fixed trajectory payload '+profile+'\n\n'+intro+'\n\n'
        'One complete MP4 save/read per condition, then slices without another codec. '
        'P1/P0 compatibility is shown first. Quality references P0 and P1 are explicit; condition-P1 residual '
        'differences are distinct from video self-difference ratios and are not perceptual flicker measures. '
        'Blind receivers receive only video, key and public protocol. All candidates, ties, missing and '
        'failed slots remain; truth/bit errors join only after sealing blind readouts. FULL has one geometry '
        'candidate; repeated payload success is not synchronization evidence. No threshold, scan or science PASS.\n\n'
        +('Fixed source prompt: '+cfg['generation']['prompt']+'\n\n' if profile=='G' else 'Fixed source path: '+cfg['source']['path']+'\n\n')+
        'Comparison denominator: '+json.dumps(fixed)+'\n\n'
        'Separate source preparation budget: '+json.dumps(cfg['source_preparation_planned_calls'])+'\n\n'+note)
    setup=prior.SETUP.replace('Trajectory-Payload-Framewise-Sync-M05','Trajectory-Payload-'+profile+'-V1').replace(
        'SC-SSTW-TRAJECTORY-PAYLOAD-FRAMEWISE-SYNC-M05-','SC-SSTW-TRAJECTORY-PAYLOAD-'+profile+'-V1-')
    environment=prior.ENVIRONMENT
    if profile=='G':
        environment=environment.replace('from diffusers import AutoencoderKL,AutoencoderKLWan; print(AutoencoderKL.__name__,AutoencoderKLWan.__name__)',
            'from diffusers import AutoencoderKL,AutoencoderKLWan,WanPipeline; import sentencepiece,ftfy; print(WanPipeline.__name__)')
        environment=environment.replace("'safetensors>=0.4.5'","'safetensors>=0.4.5','sentencepiece','ftfy'")
    run=prior.RUN.replace('experiments.wan_state_clock.video_trajectory_payload_framewise_sync_v1_run',
        'experiments.wan_state_clock.video_trajectory_payload_'+profile.lower()+'_v1_run')
    rows=[('code',"from google.colab import drive\ndrive.mount('/content/drive')\n"),
          ('markdown',markdown),('code',f'SOURCE_SHA = {source_sha!r}\nFIXED = {fixed!r}\n'+setup),
          ('code',environment),('code',run),('code',DISPLAY)]
    cells=[]
    for index,(kind,source) in enumerate(rows):
        cell=dict(cell_type=kind,id=f'trajectory-payload-{profile.lower()}-{index}',metadata={},source=source.splitlines(keepends=True))
        if kind=='code':ast.parse(source);cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    notebook=dict(nbformat=4,nbformat_minor=5,cells=cells,
        metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
            language_info=dict(name='python'),
            candidate_binding=dict(candidate='trajectory-payload-'+profile.lower()+'-v1',
                source_sha=source_sha,status='UNPUBLISHED_DRAFT' if source_sha is None else 'PUBLISHED_SHA_BOUND',
                execution='user-run only; no agent model/media execution')))
    output.write_text(json.dumps(notebook,indent=1)+'\n',encoding='utf-8')
    return output
