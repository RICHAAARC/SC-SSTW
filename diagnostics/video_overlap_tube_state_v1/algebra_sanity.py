"""Stage-A synthesis/projection identity only; no candidate/path scoring."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'experiments/wan_state_clock/configs/video_overlap_tube_state_v1.json'
EXPECTED_CONFIG_SHA='4c45eec080679a8e8c9324a3ea8d2cd37fafbd2c3aad1e8dd250eed8fc444462'


def keyed(domain,key,*indices):
    if '\0' in key:raise ValueError('key contains NUL')
    return hashlib.sha256('\0'.join([domain,key,*map(str,indices)]).encode()).digest()


def state_code(key):
    order=sorted(range(8),key=lambda j:(keyed('VTOS1/state/order',key,j),j))
    signs=np.asarray([2*(keyed('VTOS1/state/sign',key,j)[0]&1)-1 for j in range(8)],dtype=np.int8)
    states=[]
    for t in range(1,46):
        b=[((t-1)>>(5-j))&1 for j in range(6)]
        c=b+[sum(b)%2,(b[0]+b[2]+b[4])%2]
        states.append(signs*np.asarray([2*c[j]-1 for j in order],dtype=np.int8))
    return np.asarray(states,dtype=np.int8)


def block_basis(key,i,dtype):
    H=np.asarray([[(-1)**((r&c).bit_count())/4 for c in range(16)] for r in range(16)],dtype=dtype)
    order=sorted(range(16),key=lambda c:(keyed('VTOS1/basis/order',key,i,c),c))
    signs=np.asarray([2*(keyed('VTOS1/basis/sign',key,i,q)[0]&1)-1 for q in range(8)],dtype=dtype)
    return H[:,order[:8]]*signs[None,:]


def check(cfg,key,dtype):
    states=state_code(key);basis=[block_basis(key,i,dtype) for i in range(4)]
    alpha=dtype(1/np.sqrt(1392))
    target=np.zeros(cfg['public']['normalized_shape'],dtype=dtype)
    coefficient_count=0
    # Every overlapping tube writes the same physical tensor. No hidden tube observations.
    for start in range(1,46):
        for age in range(4):
            u=start+age
            if u>45:continue
            for i,block in enumerate(cfg['public']['blocks']):
                h0,h1=block['h'];w0,w1=block['w']
                for k in range(2):
                    contribution=alpha*basis[i][:,2*age+k]*states[start-1,2*i+k]
                    target[0,4,u,h0:h1,w0:w1]+=contribution.reshape(4,4)
                    coefficient_count+=1
    max_error=0.0;inactive_error=0.0
    for u in range(1,46):
        for i,block in enumerate(cfg['public']['blocks']):
            h0,h1=block['h'];w0,w1=block['w']
            observed=target[0,4,u,h0:h1,w0:w1].reshape(16)
            actual=basis[i].T@observed
            for age in range(4):
                for k in range(2):
                    start=u-age
                    expected=alpha*states[start-1,2*i+k] if start>=1 else 0.0
                    error=abs(float(actual[2*age+k])-float(expected))
                    max_error=max(max_error,error)
                    if start<1:inactive_error=max(inactive_error,error)
    norm=float(np.sqrt(np.sum(target.astype(np.float64)**2)))
    gram=max(float(np.max(np.abs(U.T@U-np.eye(8)))) for U in basis)
    support=np.zeros(target.shape,dtype=bool)
    for block in cfg['public']['blocks']:
        h0,h1=block['h'];w0,w1=block['w'];support[0,4,1:46,h0:h1,w0:w1]=True
    assert coefficient_count==1392 and int(support.sum())==2880
    assert abs(norm-1)<1e-6 and max_error<1e-6 and gram<1e-6
    assert not np.any(target[~support]) and len(set(map(tuple,states.tolist())))==45
    return dict(dtype=np.dtype(dtype).name,alpha=float(alpha),active_coefficients=coefficient_count,
                physical_support_cells=int(support.sum()),actual_target_l2=norm,projection_max_abs_error=max_error,
                inactive_boundary_projection_max_abs_error=inactive_error,basis_gram_max_abs_error=gram,
                state_count=45,distinct_state_codewords=45,
                state_sha256=hashlib.sha256(states.tobytes()).hexdigest(),
                actual_basis_sha256=hashlib.sha256(np.stack(basis).tobytes()).hexdigest(),
                actual_target_sha256=hashlib.sha256(target.tobytes()).hexdigest(),
                target_outside_support_zero=True,first_latent_zero=True)


def main():
    content=CONFIG.read_bytes();assert hashlib.sha256(content).hexdigest()==EXPECTED_CONFIG_SHA
    cfg=json.loads(content)
    results={name:[check(cfg,key,np.float64),check(cfg,key,np.float32)] for name,key in [('correct',cfg['public']['key']),('wrong',cfg['public']['wrong_key'])]}
    for n in range(2):
        assert results['correct'][n]['actual_basis_sha256']!=results['wrong'][n]['actual_basis_sha256']
        assert results['correct'][n]['state_sha256']!=results['wrong'][n]['state_sha256']
    output=dict(status='SYNTHESIS_PROJECTION_IDENTITY_PASS',config_sha256=EXPECTED_CONFIG_SHA,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results=results,
                physical_target_scatter_added=True,projections_read_only_from_actual_target=True,
                candidate_scores_computed=0,path_searches=0,observed_data_scored=False,
                new_model_gpu_vae_fft_media_calls=0,
                evidence_ceiling='Public construction algebra only; no candidate identifiability/noise/survival result')
    path=Path(__file__).with_suffix('.json');path.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
