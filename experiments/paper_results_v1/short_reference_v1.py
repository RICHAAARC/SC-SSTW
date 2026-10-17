"""Fixed 48-row saved-Y CPU diagnosis; no model, media, or payload dependency."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from experiments.paper_results_v1.resume_state import StateJournal, atomic_bytes, read_state, state_exists, unit_attempt
from main.tube_state import video_trajectory_payload_framewise_sync_v1 as clock
from main.tube_state.receiver_controls_v1 import received_scores, select_score_path
from main.tube_state.video_trajectory_temporal_edit_receiver_v1 import decode_visible_span

CASES=tuple(f'independent_20261015_{i:02d}' for i in range(1,5))
VIEWS=('SHORT9','SHORT9_HOLD','SHORT17','SHORT17_HOLD')
ARMS=('RAW_U','CENTERED_U','REF_U')
KEY='watermark'
SOURCE_DEFAULT='/content/drive/MyDrive/Video-WM/Paper-Results-V1-Independent-Four-V1/resumable-independent-four-v1'
DONE={'SUPPORTED','UNSUPPORTED','UNRESOLVED'}


def write_json(path,value):
    atomic_bytes(path,(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())


def view_indices(view):
    """Fixed attack preparation/posthoc truth; not a selector argument."""
    if view not in VIEWS:raise ValueError('unknown fixed view')
    n=9 if view.startswith('SHORT9') else 17
    return np.array([37+i for i in range(n) for _ in range(n+2 if view.endswith('HOLD') and i==n//2 else 1)])


def public_templates(key=KEY):
    w=np.empty((181,160,4,16),dtype=np.float64)
    for s in range(181):
        tube,age=divmod(s,4)
        sign=clock.sync_sign(key,tube)
        for p,(y,x) in enumerate(clock.spatial_patch_coordinates(40,64)):
            w[s,p]=(sign*clock.patch_direction(key,tube,y,x)[age]).reshape(4,16)
    rho_patch=np.sum(w*w,axis=(-1,-2))
    residual=w-w.mean(axis=-1,keepdims=True)
    residual_energy=np.sum(residual*residual,axis=(-1,-2))
    if not np.all(residual_energy>0):raise ValueError('DC projected direction has zero energy')
    effective=(rho_patch/residual_energy)[...,None,None]*residual
    return w.reshape(181,-1),effective.reshape(181,-1),rho_patch.sum(axis=1)


def observed_scores(y,templates):
    """Received Y + public templates only; framewise, no clip statistics."""
    if y.dtype!=np.float32 or y.ndim!=4 or y.shape[0]<1 or y.shape[1:]!=(4,40,64) or not np.isfinite(y).all():
        raise ValueError('clock Y must be finite float32 [N>=1,4,40,64]')
    patches=y.astype(np.float64).reshape(len(y),4,10,4,16,4).transpose(0,2,4,1,3,5).reshape(len(y),-1)
    w,e,rho=templates
    return patches@w.T/rho[None,:],patches@e.T/rho[None,:]


def blind_select(q):
    estimate=select_score_path(q,'U')
    operation=decode_visible_span(estimate['path']) if estimate['status']=='ESTIMATED' else None
    return dict(estimate=estimate,operation=operation,
                status=operation['status'] if operation else estimate['status'],
                reason=operation.get('reason') if operation else estimate.get('reason'),
                truth_inputs=False,message_inputs=False,attack_inputs=False)


def distinct_reference_scores(q_ref):
    """One column-background vote per exactly equal received REF score row.

    Only the supplied observations enter. First representatives define classes;
    every original row remains in order for the unchanged downstream U search.
    This opt-in candidate does not change ARMS or the existing run() defaults.
    """
    q=received_scores(q_ref)
    representatives=[]
    classes=[]
    for row in q:
        for group,index in enumerate(representatives):
            if np.array_equal(row,q[index]):
                classes.append(group)
                break
        else:
            classes.append(len(representatives))
            representatives.append(len(classes)-1)
    with np.errstate(over='raise',invalid='raise'):
        background=np.median(q[representatives],axis=0)
        corrected=q-background
    if not np.isfinite(corrected).all():
        raise FloatingPointError('distinct REF background subtraction is nonfinite')
    return dict(q=corrected,background=background,
                representative_indices=np.asarray(representatives,dtype=np.int64),
                class_indices=np.asarray(classes,dtype=np.int64))


def planned_rows():
    return [dict(slot_id=f'{case}/{view}/{arm}',case_id=case,view=view,receiver=arm,
                 received_frames=len(view_indices(view)),status='PENDING',reason=None)
            for case in CASES for view in VIEWS for arm in ARMS]


def fixed_method_semantics():
    return dict(key=KEY,source_frames=181,view_indices={v:view_indices(v).tolist() for v in VIEWS},
                arms=list(ARMS),transition='U',reference=dict(patch=[4,4],
                dc='per_patch_per_channel',gain='rho_over_dc_residual_energy',post_center=False))


def check_known_semantics(saved,current,prefix='method'):
    """Compare actual known method choices, never source identity or provenance."""
    for key,value in saved.items():
        if key not in current:continue
        expected=current[key]
        if isinstance(value,dict) and isinstance(expected,dict):
            check_known_semantics(value,expected,prefix+'.'+key)
        elif value!=expected:
            raise ValueError(f'fixed method differs at {prefix}.{key}: saved={value!r}, current={expected!r}')


def read_evidence(row):
    selected=json.loads(Path(row['selection']).read_text())
    with np.load(row['observation'],allow_pickle=False) as z:q=z['q']
    if q.dtype!=np.float64 or q.shape!=(row['received_frames'],181) or not np.isfinite(q).all():
        raise ValueError('saved q must be finite float64 [received_frames,181]')
    if selected['status']!=row['status']:raise ValueError('selection status differs from saved row')
    estimate=selected['estimate'];path=estimate['path'];operation=selected['operation']
    gap=estimate['score_gap']
    if gap is not None and (not isinstance(gap,(int,float)) or not np.isfinite(gap)):
        raise ValueError('invalid selector score gap')
    if path is None:
        if selected['status']!='UNRESOLVED' or operation is not None:
            raise ValueError('missing path for resolved selection')
    else:
        if not isinstance(path,list) or len(path)!=len(q) or any(type(t)!=int or not 0<=t<181 for t in path):
            raise ValueError('invalid saved source path')
        if not isinstance(operation,dict) or operation.get('status')!=selected['status']:
            raise ValueError('missing or inconsistent visible-span operation')
        for key in ('wan_support','output_frames'):
            if key not in operation:raise ValueError('incomplete visible-span operation: '+key)
    return selected,q


class Store:
    def __init__(self,output,source_run):
        self.output=Path(output).resolve();self.output.mkdir(parents=True,exist_ok=True)
        self.path=self.output/'short_reference_state.json'
        self.journal=StateJournal(self.path)
        self.data=read_state(self.path) if state_exists(self.path) else dict(
            phases={},rows=planned_rows(),unit_attempts={},invocations=[],source_run=str(source_run))
        expected=[(r['slot_id'],r['received_frames']) for r in planned_rows()]
        if [(r.get('slot_id'),r.get('received_frames')) for r in self.data['rows']]!=expected:
            raise ValueError('actual fixed 48-row geometry differs')
        semantics=fixed_method_semantics()
        saved=self.data.get('fixed_method_semantics')
        if isinstance(saved,dict):check_known_semantics(saved,semantics)
        else:self.data.setdefault('notes',[]).append('Previous method-choice record unavailable; available saved evidence retained.')
        self.data['fixed_method_semantics']=semantics
        self.journal.seed_cache(self.data)
        self.data['source_run']=str(source_run)
        for r in self.data['rows']:
            if r['status']=='RUNNING':r.update(status='INTERRUPTED',reason='previous CPU process interrupted')
            if r['status'] in DONE:
                try:read_evidence(r)
                except (OSError,ValueError,KeyError,TypeError,IndexError,AttributeError,EOFError,zipfile.BadZipFile) as exc:
                    r.setdefault('history',[]).append(dict(status=r['status'],reason=r.get('reason')))
                    r.update(status='EVIDENCE_INCOMPLETE',reason=f'{type(exc).__name__}: {exc}')
        self.save()

    def save(self):self.journal.save(self.data)


def run(output,source_run=SOURCE_DEFAULT):
    store=Store(output,source_run)
    invocation=dict(started_at=time.time(),selectors=0,files_read=0,status='RUNNING')
    store.data['invocations'].append(invocation);store.save()
    templates=None
    with store.journal.periodic():
        try:
            for case in CASES:
                pending=[r for r in store.data['rows'] if r['case_id']==case and r['status'] not in DONE]
                if not pending:continue
                try:
                    path=Path(source_run)/'joint/run_state/latents/framewise/clean_mp4'/f'{case}.npy'
                    y=np.load(path,allow_pickle=False);invocation['files_read']+=1
                    if templates is None:templates=public_templates()
                    raw,ref=observed_scores(y,templates)
                except Exception as exc:
                    for row in pending:
                        row.setdefault('history',[]).append(dict(status=row['status'],reason=row.get('reason')))
                        row.update(status='MISSING' if isinstance(exc,FileNotFoundError) else 'FAILED',reason=f'{type(exc).__name__}: {exc}')
                    store.save();continue
                for row in pending:
                    row.setdefault('history',[]).append(dict(status=row['status'],reason=row.get('reason')))
                    row.update(status='RUNNING',reason=None);store.save()
                    try:
                        indices=view_indices(row['view'])
                        q_raw,q_ref=raw[indices],ref[indices]
                        background=np.median(q_raw,axis=0)
                        q={'RAW_U':q_raw,'CENTERED_U':q_raw-background,'REF_U':q_ref}[row['receiver']]
                        with unit_attempt(store,row['slot_id'],operation='saved_Y_CPU_U_selector'):
                            invocation['selectors']+=1;store.save()
                            selected=blind_select(q)
                            # Store observations and blind selection before any truth evaluation.
                            path=store.output/'selections'/case/row['view']/f"{row['receiver']}.json"
                            write_json(path,selected)
                            observation=path.with_suffix('.npz')
                            import io
                            buffer=io.BytesIO();np.savez_compressed(buffer,q=q)
                            atomic_bytes(observation,buffer.getvalue())
                            row.update(status=selected['status'],reason=selected['reason'],selection=str(path),observation=str(observation))
                    except BaseException as exc:
                        row.update(status='FAILED' if isinstance(exc,Exception) else 'INTERRUPTED',reason=f'{type(exc).__name__}: {exc}')
                        store.save()
                        if not isinstance(exc,Exception):raise
                    store.save()
            invocation.update(status='COMPLETE',finished_at=time.time())
            store.data['phases']['blind']='COMPLETE'
        except BaseException:
            invocation.update(status='INTERRUPTED',finished_at=time.time());raise
        finally:
            store.save();report(store)
    return store


def report(store):
    """Posthoc evaluator. Never called by or used to configure blind_select."""
    evaluated=[]
    for row in store.data['rows']:
        result={k:row[k] for k in ('slot_id','case_id','view','receiver','received_frames','status','reason')}
        if row['status'] in DONE:
            try:
                selected,q=read_evidence(row)
                truth=view_indices(row['view'])
                result.update(distinct_true_source_frames=len(np.unique(truth)),
                              zero_score_rows=int(np.sum(np.all(q==0,axis=1))),
                              q_rms=float(np.sqrt(np.mean(q*q))))
                correct=q[np.arange(len(truth)),truth];wrong=q.copy();wrong[np.arange(len(truth)),truth]=-np.inf
                result.update(row_top1_correct=int(np.sum(np.argmax(q,axis=1)==truth)),
                              mean_true_wrong_margin=float(np.mean(correct-wrong.max(axis=1))))
                estimate=selected['estimate'];path=estimate['path'];op=selected['operation']
                result['score_gap']=estimate['score_gap']
                if path is not None:
                    error=np.abs(np.asarray(path)-truth)
                    result.update(path_exact=bool(np.array_equal(path,truth)),
                                  interval_exact=bool(path[0]==truth[0] and path[-1]==truth[-1]),
                                  wrong_correspondences=int(np.sum(error!=0)),absolute_error_sum=int(error.sum()),
                                  start_error=int(abs(path[0]-truth[0])),end_error=int(abs(path[-1]-truth[-1])),
                                  R=op['wan_support'],output_frames=op['output_frames'])
            except (OSError,ValueError,KeyError,TypeError,IndexError,AttributeError,EOFError,zipfile.BadZipFile) as exc:
                result['evidence_read_error']=f'{type(exc).__name__}: {exc}'
        evaluated.append(result)
    by_arm={}
    for arm in ARMS:
        rows=[r for r in evaluated if r['receiver']==arm]
        comparable=[r for r in rows if 'absolute_error_sum' in r]
        by_arm[arm]=dict(planned_rows=16,planned_received_positions=320,
            status_counts=dict(Counter(r['status'] for r in rows)),
            exact_paths=sum(r.get('path_exact',False) for r in rows),
            exact_intervals=sum(r.get('interval_exact',False) for r in rows),
            resolved_rows=len(comparable),
            resolved_absolute_error_sum=sum(r['absolute_error_sum'] for r in comparable),
            fixed_denominator_absolute_error_sum=sum(r['absolute_error_sum'] for r in comparable) if len(comparable)==16 else None,
            resolved_wrong_correspondences=sum(r['wrong_correspondences'] for r in comparable))
    summary=dict(status='SAVED_Y_CPU_ANALYSIS',planned_rows=48,independent_sources=4,related_views=16,
        planned_arm_frame_positions=960,payload_reads=0,models_loaded=0,media_calls=0,
        selectors_total=sum(i['selectors'] for i in store.data['invocations']),
        status_counts=dict(Counter(r['status'] for r in evaluated)),by_arm=by_arm,
        evidence_ceiling='Fixed predeclared short/midpoint-majority-repeat stress views; not independent samples, general failure or GPU payload evidence.')
    write_json(store.output/'row_results.json',evaluated);write_json(store.output/'summary.json',summary)
    return summary
