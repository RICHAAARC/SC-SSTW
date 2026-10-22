"""Card101 once-only saved-Q CPU development. No physical backend or old D1 replay."""
from pathlib import Path
import json,time
from main.tube_state import b_absolute_compatibility_v1 as layout
from runtime.wan.b_background_normalized_confirmation_v1 import dump,flush,execution_artifacts,existing_attempt

BASE=Path('/home/richar/projects/Video-WM/diagnostics/b-line-background-normalized-match-package-20261022')
Q_ROOT=BASE/'saved_q'
OUTPUT=BASE/'cpu_development'
OLD=Path('/home/richar/projects/Video-WM/diagnostics/b-line-absolute-confirmation-result-20261022/raw')
HOSTS=('b97-s01','b97-s02','b97-s03','b97-s04')
CAL=('b97-s02','b97-s04')
REPORT_HOSTS=('b97-s01','b97-s03')


def units():
    rows={}
    for h in HOSTS:
        for arm in ('OFF','P0','PM'):
            for oi in range(6):
                for p in range(4):
                    for key in ('key00','key01'):rows[f'Q/{h}/{arm}/r{oi}/p{p}/{key}']='saved_Q_load'
        for q in layout.query_metadata():rows[f'score/{h}/{q}']='normalized_family_score'
    for h in REPORT_HOSTS:
        for q in layout.query_metadata():
            rows[f'bits/{h}/{q}']='saved_bit_cache_load'
            rows[f'decision/{h}/{q}']='development_decision'
    rows.update(background='background_fit',calibration='null_calibration',post='post_truth_evaluation')
    return rows


class Journal:
    def __init__(self,out):
        self.output=Path(out)
        self.rows={k:dict(slot=k,kind=v,status='NOT_ENTERED',attempts=0) for k,v in units().items()}
        self.save()
    def save(self):dump(self.output/'ledger.json',self.rows)
    def call(self,slot,fn):
        row=self.rows[slot]
        if row['status']!='NOT_ENTERED':raise RuntimeError('attempt consumed '+slot)
        row.update(status='STARTED',attempts=1,started_unix=time.time());self.save()
        try:
            value=fn();row['status']='RETURNED';return value
        except BaseException as e:
            row.update(status='FAILED' if isinstance(e,Exception) else 'INTERRUPTED',error=f'{type(e).__name__}: {e}')
            if not isinstance(e,Exception):raise
            return None
        finally:
            row['finished_unix']=time.time();self.save()


def run_saved(q_root=Q_ROOT,output=OUTPUT,old_root=OLD):
    out=Path(output);qroot=Path(q_root);old=Path(old_root)
    found=execution_artifacts(out)
    if found:return existing_attempt(out,found)
    out.mkdir(parents=True,exist_ok=True)
    try:
        with (out/'attempt.json').open('x') as stream:
            json.dump(dict(card=101,scope='one fixed seen-data CPU candidate; any failure consumes attempt',started_unix=time.time()),stream);flush(stream)
    except FileExistsError:return existing_attempt(out)
    journal=Journal(out);banks={};bits={};families={h:{} for h in HOSTS};decisions={h:{} for h in REPORT_HOSTS}
    background=dict(status='BACKGROUND_UNAVAILABLE');thresholds=dict(status='CALIBRATION_UNRESOLVED',tau_N=None)
    result=dict(status='RUNNING',scope='seen four-host Q development',independent_confirmation=False,scientific_PASS=None,
        physical_model_E_FFT_Counter_codec_calls=0,planned=dict(Q_load=576,background_fit=1,family_score=144,hypothesis_rows=19728,
            raw_bit_cache_load=72,decisions=72,members=9864,groups=288,message_slots=2304,post=1))
    dump(out/'result.json',result)
    try:
        from main.tube_state import b_background_normalized_match_v1 as method
        for h in HOSTS:
            for arm in ('OFF','P0','PM'):
                for oi in range(6):
                    for p in range(4):
                        for key in ('key00','key01'):
                            rel=f'hosts/{h}/blind/{arm}/r{oi}/p{p}/{key}/Q.json'
                            def read():
                                with (qroot/rel).open() as stream:q=json.load(stream)
                                method.q_array(q)
                                return q
                            banks[h,arm,oi,p,key]=journal.call(f'Q/{h}/{arm}/r{oi}/p{p}/{key}',read)
        for h in REPORT_HOSTS:
            for q in layout.query_metadata():
                def load_bits():
                    with (old/'hosts'/h/'blind/families'/(q+'.json')).open() as stream:f=json.load(stream)
                    return f.get('phase_local_bits',{})
                bits[h,q]=journal.call(f'bits/{h}/{q}',load_bits) or {}
        def bank(h,arm,oi,key):return {str(p):banks[h,arm,oi,p,key] for p in range(4)}
        def fit():
            value=method.fit_background([[{key:bank(h,'OFF',oi,key) for key in method.KEY_IDS} for oi in range(6)] for h in CAL])
            dump(out/'background.json',value)
            return value
        background=journal.call('background',fit) or background
        if background['status']=='BACKGROUND_FROZEN':
            for h in HOSTS:
                for q,meta in layout.query_metadata().items():
                    def score():
                        value=method.score_family(bank(h,meta['arm'],layout.ORIGINS.index(meta['origin']),meta['key']),background['models'][meta['key']])
                        value['phase_local_bits']=bits.get((h,q),{})
                        dump(out/'numeric'/h/(q+'.json'),value)
                        return value
                    families[h][q]=journal.call(f'score/{h}/{q}',score) or method.empty_family('SCORE_SAVE_FAILED')
            def calibrate():
                value=method.calibrate_null({h:families[h] for h in CAL})
                value['frozen_unix']=time.time() if value['status']=='FROZEN' else None
                dump(out/'thresholds.json',value);return value
            thresholds=journal.call('calibration',calibrate) or thresholds
        else:
            for slot,row in journal.rows.items():
                if row['kind'] in ('normalized_family_score','null_calibration'):row['unavailable_from']=['background']
            dump(out/'background.json',background);dump(out/'thresholds.json',thresholds)
        if thresholds['status']=='FROZEN':
            for h in REPORT_HOSTS:
                for q in layout.query_metadata():
                    def decide():
                        value=method.decide(families[h][q],thresholds)
                        dump(out/'decisions'/h/(q+'.json'),value);return value
                    decisions[h][q]=journal.call(f'decision/{h}/{q}',decide)
            unsaved=[s for s,r in journal.rows.items() if r['kind']=='development_decision' and r['status']!='RETURNED']
            if not unsaved:
                dump(out/'blind_complete.json',dict(saved_unix=time.time(),decisions=72,all_actual_saves=True,truth_opened=False))
                def post():
                    saved={h:{q:json.loads((out/'decisions'/h/(q+'.json')).read_text()) for q in layout.query_metadata()} for h in REPORT_HOSTS}
                    dump(out/'post_truth_access.json',dict(opened_unix=time.time()))
                    with (old/'evaluation.json').open() as stream:previous=json.load(stream)
                    with (old/'protocol.json').open() as stream:protocol=json.load(stream)
                    from runtime.wan.b_background_normalized_evaluation_v1 import evaluate_saved
                    value=evaluate_saved(saved,previous,protocol['config']['writer']['bits'],REPORT_HOSTS)
                    dump(out/'evaluation.json',value);return value['status']
                journal.call('post',post)
            else:journal.rows['post']['unavailable_from']=unsaved
        else:
            for slot,row in journal.rows.items():
                if row['kind'] in ('development_decision','post_truth_evaluation'):row['unavailable_from']=['BACKGROUND_OR_NULL_CALIBRATION_UNAVAILABLE']
        technical=sum(d is None or d['detection'] in ('UNRESOLVED_SCORE','CALIBRATION_UNRESOLVED') or any(b['status']=='TECHNICAL_UNRESOLVED' for b in d['message']) for ds in decisions.values() for d in ds.values())
        complete=all(r['status']=='RETURNED' for r in journal.rows.values()) and technical==0 and all(f['status']=='SCORES_COMPLETE' for fs in families.values() for f in fs.values())
        result.update(status='SEEN_CPU_DEVELOPMENT_COMPLETE' if complete else 'INCOMPLETE_FIXED_DENOMINATORS_RETAINED',technical_queries=technical)
    except BaseException as e:result.update(status='FAILED' if isinstance(e,Exception) else 'INTERRUPTED',error=f'{type(e).__name__}: {e}')
    finally:
        journal.save()
        not_entered=sum(v['kind']=='development_decision' and v['status']=='NOT_ENTERED' for v in journal.rows.values())
        result.update(decision_not_entered=not_entered,message_slots_not_entered=32*not_entered)
        result.update(background_status=background['status'],threshold_status=thresholds['status'],
            returned_by_kind={kind:sum(r['kind']==kind and r['status']=='RETURNED' for r in journal.rows.values()) for kind in set(r['kind'] for r in journal.rows.values())})
        dump(out/'result.json',result)
    return result
