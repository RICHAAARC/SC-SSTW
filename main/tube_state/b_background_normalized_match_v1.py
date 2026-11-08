"""One fixed Card101 Q40 background-normalized matching model. Pure NumPy method."""
import math
import numpy as np
from . import b_absolute_compatibility_v1 as common

H=common.H
ORIGINS=common.ORIGINS
query_metadata=common.query_metadata
DIM=40
KEY_IDS=('key00','key01')


def q_array(observation):
    value=observation.get('roi_q') if isinstance(observation,dict) else observation
    a=np.asarray(value)
    if a.dtype.kind not in 'iuf' or a.shape!=(10,4,44):
        raise ValueError('numeric Q10x4x44 required')
    a=a.astype(np.float64)
    if not np.isfinite(a).all():raise ValueError('nonfinite Q')
    return a


def vectors(phase_bank):
    """All137 equal40 cells; no origin/message/arm/identity arguments."""
    banks={p:q_array(phase_bank[str(p)] if str(p) in phase_bank else phase_bank[p]) for p in range(4)}
    r=np.arange(10)
    return np.stack([banks[p][r,:,s-1+r].reshape(DIM) for p,s in H])


def fit_background(off_hosts):
    """Two host groups; each contains6 crop banks,2 keys,4 phases. No truth."""
    if len(off_hosts)!=2:raise ValueError('two host groups required')
    models={}
    for key in KEY_IDS:
        xs=[]
        for crops in off_hosts:
            if len(crops)!=6:raise ValueError('six OFF crops per host required')
            xs.append(np.concatenate([vectors(crop[key]) for crop in crops],axis=0))
        means=[x.mean(axis=0) for x in xs]
        mu=np.mean(means,axis=0)
        moments=[]
        for x in xs:
            residual=x-mu
            moments.append(residual.T@residual/len(x))
        V=np.mean(moments,axis=0)
        scale=float(np.trace(V)/DIM)
        if not math.isfinite(scale) or scale<=0:raise ValueError('BACKGROUND_SCALE_UNRESOLVED')
        sigma=(V+scale*np.eye(DIM))/2  # Fixed spherical component; never scan.
        factor=np.linalg.cholesky(sigma)  # Failure is technical, no pseudoinverse.
        u=np.linalg.solve(factor,np.ones(DIM))
        if not np.isfinite(factor).all() or not np.isfinite(u).all():raise ValueError('nonfinite fitted background')
        models[key]=dict(status='BACKGROUND_FROZEN',dimension=DIM,host_groups=2,
            correlated_vectors_per_host=[len(x) for x in xs],independent_sample_claim=False,
            feature_order='observed_row then ROI',mean=mu.tolist(),host_means=[x.tolist() for x in means],
            covariance_raw=V.tolist(),covariance_regularized=sigma.tolist(),
            factor=factor.tolist(),template_whitened=u.tolist(),trace_scale=scale,
            covariance_condition=float(np.linalg.cond(sigma)),spherical_weight=0.5,
            working_template='unknown positive common gain across40 features; not guaranteed by media',
            training_null_reuse_optimism=True,CFAR=False)
    return dict(status='BACKGROUND_FROZEN',models=models,role_randomization_rank_guarantee=None)


def compile_model(record):
    if record.get('status')!='BACKGROUND_FROZEN':raise ValueError('background unavailable')
    mu=np.asarray(record['mean'],dtype=np.float64);factor=np.asarray(record['factor'],dtype=np.float64)
    if mu.shape!=(DIM,) or factor.shape!=(DIM,DIM) or not np.isfinite(mu).all() or not np.isfinite(factor).all():
        raise ValueError('finite background mean40/factor40x40 required')
    if np.any(np.diag(factor)<=0) or np.any(np.triu(factor,1)!=0):raise ValueError('lower positive background factor required')
    u=np.linalg.solve(factor,np.ones(DIM))
    return dict(mean=mu,factor=factor,u=u,u2=float(u@u))


def score_family(phase_bank,model):
    """No writer/P0-difference/origin/message inputs. Missing one h keeps137 unresolved."""
    rows=[dict(phase=p,source_start=s,S=None,score_status='NOT_ENTERED') for p,s in H]
    try:
        y=vectors(phase_bank)
        compiled=compile_model(model)
        x=np.linalg.solve(compiled['factor'],(y-compiled['mean']).T).T
        numerator=x@compiled['u'];energy=np.sum(x*x,axis=1);u2=compiled['u2']
        if not np.isfinite(x).all() or not np.isfinite(numerator).all() or not math.isfinite(u2) or u2<=0:
            raise ValueError('NONFINITE_MATCH_UNRESOLVED')
        for row,n,e in zip(rows,numerator,energy):
            if not math.isfinite(float(e)) or e<=0:
                row.update(score_status='ZERO_NORM_UNRESOLVED',whitened_energy=float(e) if math.isfinite(float(e)) else None)
                continue
            raw=float(n/math.sqrt(u2*float(e)))
            if not math.isfinite(raw) or abs(raw)>1+64*np.finfo(np.float64).eps:
                row.update(score_status='NUMERIC_MATCH_UNRESOLVED');continue
            # Only floating Cauchy-Schwarz roundoff is bounded; no result-tuned clipping.
            bounded=min(1.0,max(-1.0,raw))
            alpha=max(0.0,float(n))/u2
            residual=float(e)-max(0.0,float(n))**2/u2
            row.update(S=max(0.0,bounded),score_status='RETURNED',signed_matched_cosine=raw,
                whitened_numerator=float(n),whitened_energy=float(e),template_energy=u2,
                alpha_hat=alpha,profile_residual_energy=max(0.0,residual),
                received_working_scale_squared=float(e)/DIM)
    except (ValueError,KeyError,TypeError,np.linalg.LinAlgError) as error:
        for row in rows:row.update(S=None,score_status='TECHNICAL_UNRESOLVED')
        return dict(status='SCORE_UNRESOLVED',candidates=rows,error=f'{type(error).__name__}: {error}',phase_local_bits={})
    complete=all(r['score_status']=='RETURNED' for r in rows)
    return dict(status='SCORES_COMPLETE' if complete else 'SCORE_UNRESOLVED',
        candidates=rows,phase_local_bits={},background_dimension=DIM,
        truth_inputs=False,message_inputs=False,CFAR=False)


def empty_family(reason='NOT_ENTERED'):
    return dict(status=reason,candidates=[dict(phase=p,source_start=s,S=None,score_status=reason) for p,s in H],phase_local_bits={})


def compatibility_input(family):
    """Legacy neutral score slot T is an internal alias for newS, never oldmin(F,V)."""
    return dict(candidates=[dict(phase=r['phase'],source_start=r['source_start'],T=r.get('S'),score_status=r['score_status']) for r in family['candidates']],
        phase_local_bits=family.get('phase_local_bits',{}))


def calibrate_null(host_families):
    out=dict(status='CALIBRATION_UNRESOLVED',tau_N=None,hosts={},positive_floor_used=False,message_used=False,
        CFAR=False,role_randomization_rank_guarantee=None,training_OFF_reused_in_null=True)
    if len(host_families)!=2:return out
    for h,fs in host_families.items():
        if set(fs)!=set(query_metadata()):return out
        prepared={q:common.prepare_family(compatibility_input(f)) for q,f in fs.items() if not (query_metadata()[q]['arm']=='PM' and query_metadata()[q]['key']=='key00')}
        if any(not p['score_complete'] for p in prepared.values()):return out
        nulls=[dict(query=q,M=p['maximum']) for q,p in prepared.items() if not (query_metadata()[q]['arm']=='PM' and query_metadata()[q]['key']=='key00')]
        out['hosts'][h]=dict(N=max(x['M'] for x in nulls),nulls=nulls)
    out.update(status='FROZEN',tau_N=max(0,*(x['N'] for x in out['hosts'].values())))
    return out


def decide(family,thresholds):
    out=common.decision(common.prepare_family(compatibility_input(family)),
        dict(status='DEVELOPMENT_THRESHOLDS' if thresholds.get('status')=='FROZEN' else 'CALIBRATION_UNRESOLVED',
            tau_D=thresholds.get('tau_N'),L_D=thresholds.get('tau_N')))
    out.update(score_kind='Q40_background_normalized_S',candidate_semantics='detectable evidence set S>=tau_N; not certified confidence region',
        positive_floor_used=False,thresholds=dict(tau_N=thresholds.get('tau_N')),CFAR=False,role_randomization_rank_guarantee=None)
    return out
