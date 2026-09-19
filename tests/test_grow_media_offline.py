"""CPU-only integrity/failure checks of saved-tensor diagnostics."""
import hashlib,json
import pytest,torch
from experiments.wan_state_clock import grow_media_offline as audit
from main.tube_state import grow_temporal_difference as method
pytestmark=pytest.mark.unit

def test_wrong_fixed_result_rejected(tmp_path):
    p=tmp_path/'wrong.json';p.write_text('{}')
    with pytest.raises(ValueError,match='SHA'):audit.run(p,tmp_path/'out',tmp_path)

def test_tensor_hash_failure_retains_48_denominator(tmp_path,monkeypatch):
    monkeypatch.setattr(method,'SHAPE',(1,1,46,10,10));torch.set_num_threads(1)
    book=method.codebook(b'test');z=torch.zeros(method.SHAPE);rd=method.read(z,book);source={'cases':{}}
    for case in audit.CASES:
        c={'config':{'key_utf8':'test'},'source_commit':'fixture','file_sha256':{},'videos':{}};source['cases'][case]=c
        p=tmp_path/case/'latents';p.mkdir(parents=True)
        for arm in audit.ARMS:
            c['videos'][arm]={'layers':{}}
            for layer in audit.LAYERS:
                f=p/f'{arm}_{layer}.pt';torch.save(z,f);c['file_sha256']['latents/'+f.name]=hashlib.sha256(f.read_bytes()).hexdigest();c['videos'][arm]['layers'][layer]=rd
    (tmp_path/audit.CASES[0]/'latents/A_mp4.pt').write_bytes(b'corrupt')
    out={'failures':[]};r=audit.tensor_diagnostics(source,tmp_path,out)
    assert len(r['records'])==48 and r['verified_tensors']==47 and len(out['failures'])==1
    assert len(r['transitions'])==36
    assert all(x['pair_denominator']==23 for x in r['records'])
    assert next(x for x in r['records'] if x['case']==audit.CASES[0] and x['arm']=='A' and x['layer']=='mp4')['status']=='MISSING_OR_FAILED'


def test_output_cannot_touch_tensor_source(tmp_path):
    p=tmp_path/'input';p.mkdir()
    with pytest.raises(ValueError,match='outside'):audit.run(tmp_path/'result.json',p/'derived',p)
