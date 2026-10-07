"""Fixed adopted Stage1 user-run notebook; no agent execution."""
from pathlib import Path
import argparse,ast,json,re,sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import build_video_trajectory_payload_framewise_sync_m05_notebook as template
CONFIG=ROOT/"experiments/wan_state_clock/configs/video_terminal_sync_path_decision_v1.json"
OUTPUT=ROOT/"notebooks/video_terminal_sync_path_decision_v1_colab.ipynb"
DISPLAY="""result=json.loads(RESULT_PATH.read_text())
if result['source_sha']!=SOURCE_SHA or result['config_sha256']!=CONFIG_SHA or result['fixed_denominator']!=FIXED:raise RuntimeError('result source/config/denominator identity mismatch')
print('Runtime:',result['status'],'Counts:',result['counts'])
print('Thresholds:',result['thresholds'])
for family,row in result['family_summary'].items():
    print('Family:',family,'null false authorization=',row['null_false_authorizations'],'/',row['null_denominator'],'positive authorization=',row['positive_authorized'],'/',row['positive_denominator'],'positive wrong correction=',row['positive_wrong_corrections'],'/',row['positive_denominator'],'technical failures=',row['technical_failures'],'not run=',row['not_run'],'interpretation=',row['fixed_case_interpretation'])
print('Calls:',result['calls'],'Generation gate:',result.get('generation_gate'))
print('Seals:',{k:v for k,v in result.items() if k.endswith('_seal')})
for sid,row in sorted(result['posthoc'].items()):
    d=result['decisions'][sid]
    print(sid,row.get('execution_status'),row.get('key_role'),d['state'],d.get('reason'),'M=',d.get('M'),'m=',d.get('m'),'path=',d.get('path'),'null_false_authorization=',row.get('null_false_authorization'),'positive_wrong_correction=',row.get('positive_wrong_correction'),'absolute_path_correct=',row.get('absolute_path_correct'),'map_correct=',row.get('source_map_correct'),'/',row.get('source_map_denominator'))
    print('CPU operation:',result['operation_plan']['operations'][sid]['GATED'])
print('Source preparation:',result.get('source_preparation_receipt'),'Workers:',result.get('workers'))
print('Environment:',result['environment'],'Failures:',result['failures'])
print('Evidence ceiling:',result['evidence_ceiling'],'Result:',RESULT_PATH)
"""
def build(source_sha=None,output=OUTPUT):
    if source_sha is not None and not re.fullmatch('[0-9a-f]{40}',source_sha):raise ValueError('immutable source SHA')
    cfg=json.loads(CONFIG.read_text());pins={**cfg['environment_pins'],**cfg['generation_dependency_pins']}
    setup=template.SETUP.replace('Trajectory-Payload-Framewise-Sync-M05','Terminal-Sync-Path-Decision-V1').replace('SC-SSTW-TRAJECTORY-PAYLOAD-FRAMEWISE-SYNC-M05-','SC-SSTW-TERMINAL-SYNC-PATH-DECISION-')
    before=template.ENVIRONMENT.split("    if logged([PYTHON,'-c','import torch;")[0]
    after='    dependency_code='+template.ENVIRONMENT.split('    dependency_code=',1)[1]
    probe="import importlib.metadata as m; pins="+repr(pins)+"; actual={k:m.version(k) for k in pins}; assert all(actual[k].split('+')[0]==v for k,v in pins.items()),actual; from diffusers import AutoencoderKL,AutoencoderKLWan,WanPipeline; import torch,sentencepiece,ftfy; print(actual)"
    env=before+'    probe='+repr(probe)+"\n    if logged([PYTHON,'-c',probe],'DEPENDENCY_PROBE',check=False):\n        logged([PYTHON,'-m','pip','install',"+','.join(repr(k+'=='+v) for k,v in pins.items())+"],'PINNED_REPAIR')\n        logged([PYTHON,'-c',probe],'DEPENDENCY_REPROBE')\n"+after
    run="CONFIG_PATH=REPO/'experiments/wan_state_clock/configs/video_terminal_sync_path_decision_v1.json'\nCONFIG_SHA=hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()\nconfiguration=json.loads(CONFIG_PATH.read_text())\nif configuration['name']!='video_terminal_sync_path_decision_v1' or configuration['fixed_denominator']!=FIXED:raise RuntimeError('fixed configuration mismatch')\n"+template.RUN.replace('experiments.wan_state_clock.video_trajectory_payload_framewise_sync_v1_run','experiments.wan_state_clock.video_terminal_sync_path_decision_v1_run')
    note='Published source: '+source_sha if source_sha else 'UNPUBLISHED DRAFT: SOURCE_SHA=None; publish reviewed source then bind before Run all.'
    markdown="""# Terminal-sync path decision: adopted Stage1

Fixed 177 joint709 and 89 global93 original scores, different-frame contrast and independent development-null envelopes (8 / 6). All20 development searches seal before null labels. Only when both thresholds are available does this one Run-all generate the preregistered yellow-sailboat source, seed2026100701; otherwise all16 confirmation slots remain NOT_RUN/UNCERTAIN. No source substitution or result-selected retry.

Separate source/native decode worker then matched P1/M05 worker, sharing the unchanged source latent; one full MP4 codec per condition. Confirm C/D177 and89@38/39 on the same new source, no crop codec. No receiver Wan encode or payload read: RAW/GATED are CPU maps only. Decisions seal, then maps seal, then true construction posthoc. Writer message and preparation maps never enter the blind score/gate. Threshold values do not require a second approval after the adopted formula.

Per development/confirmation fresh stage:8 observations /1064 framewise frames /140 batch8 batches /16 key reads /6416 candidates. Existing4 development177 caches are separately reused;20 total developer reads,16 confirmation decisions=12null+4positive. Every failure/tie/non-run remains. 89 Q uses original local rows; original S never replaced. Operation aliases do not certify absolute paths. Fixed-case empirical evidence only, no FPR/coverage/scientific PASS.

Agent validation is synthetic CPU plus schema/AST; the real notebook has not been executed. User manually runs all after source binding. No stage2 payload/quality execution.

"""+note
    sources=[('code',"from google.colab import drive\ndrive.mount('/content/drive')\n"),('markdown',markdown),('code',f'SOURCE_SHA = {source_sha!r}\nFIXED = {cfg["fixed_denominator"]!r}\n'+setup),('code',env),('code',run),('code',DISPLAY)]
    cells=[]
    for i,(kind,source) in enumerate(sources):
        row=dict(cell_type=kind,id='terminal-path-decision-'+str(i),metadata={},source=source.splitlines(keepends=True))
        if kind=='code':ast.parse(source);row.update(execution_count=None,outputs=[])
        cells.append(row)
    nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python'),candidate_binding=dict(candidate='terminal-sync-path-decision-v1',source_sha=source_sha,status='UNPUBLISHED_DRAFT' if source_sha is None else 'PUBLISHED_SHA_BOUND')))
    Path(output).write_text(json.dumps(nb,indent=1)+'\n');return Path(output)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-sha');p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args();print(build(a.source_sha,a.output))
