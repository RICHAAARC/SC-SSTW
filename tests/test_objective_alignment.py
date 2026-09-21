"""Synthetic objective/gradient/budget checks, not research outcomes."""
import pytest,torch
from main.tube_state import objective_alignment as m,state_clock,projection_margin,terminal_feedback,clipped_margin
from main.tube_state.velocity_coefficients import blocks,scatter,projections
from experiments.wan_state_clock import objective_alignment_run as runner
pytestmark=pytest.mark.unit
torch.set_num_threads(1)

@pytest.fixture(scope='module')
def book():return state_clock.codebook(b'WanProjection-first-validation-key-v1')

def test_original_loss_hard_gap_and_exact_tanh_gradient(book):
    directions=torch.from_numpy(book['directions']).double();codes=torch.from_numpy(book['codes']).double()
    z=torch.randn(projection_margin.SHAPE,generator=torch.Generator().manual_seed(23))*.1
    for message in (0,1):
        assert float(m.loss(z,directions,codes,message,'hinge'))==pytest.approx(terminal_feedback.value(z,book,message),abs=1e-12)
        p=projections(z,directions)
        assert -float(clipped_margin.projection_loss(p,codes,message))==pytest.approx(m.metrics(z,directions,codes,message)['hard_gap'],abs=1e-12)
        leaf=z.double().requires_grad_(True);g,=torch.autograd.grad(m.loss(leaf,directions,codes,message,'tanh'),leaf)
        expected=-(1-p.tanh().square())*(codes[message]-codes[1-message])/1760
        torch.testing.assert_close(blocks(g),expected[:,None]*directions,rtol=1e-12,atol=1e-14)
        assert torch.count_nonzero(g[:,:,0])+torch.count_nonzero(g[:,:,45])==0

def test_same_budget_support_and_saturated_critical_gradient(book):
    directions=torch.from_numpy(book['directions']).double();codes=torch.from_numpy(book['codes']).double()
    p=torch.linspace(-2,2,1760,dtype=torch.float64);z=scatter(p,directions).float()
    records={}
    for name in ('hinge','tanh'):
        r,delta=m.update(z,directions,codes,0,name,.001);records[name]=r
        assert r['actual_control_support_rms']==pytest.approx(.001,rel=1e-5)
        assert r['support_boundary_unchanged'] and not delta.requires_grad
    groups=records['tanh']['gradient_shares']['groups']
    assert groups['noncritical']['squared_l2']==0
    assert groups['critical_saturated']['squared_l2']>0
    assert groups['critical_target_wrong_saturated']['squared_l2']>0
    assert records['tanh']['hard_gap_gain']>0
    with pytest.raises(ValueError,match='gradient'):m.update(z,directions*0,codes,0,'tanh',.001)
    with pytest.raises(ValueError,match='budget'):m.update(z,directions,codes,0,'hinge',0.)
    tiny,_=m.update(z,directions,codes,0,'tanh',1e-300)
    assert tiny['actual_rounded_control_zero'] and tiny['hard_gap_gain']==0

def test_missing_inputs_keep_all_pairs_and_layer_denominators(tmp_path):
    r=runner.run(tmp_path/'missing',tmp_path/'audit')
    assert len(r['rows'])==12 and sum(len(p['updates']) for p in r['rows'])==24
    assert r['terminal_denominator']==6 and r['status']=='WITH_RETAINED_FAILURES'
    assert all(v['pair_denominator']==4 and v['complete']==0 for v in r['summary'].values())
    assert r['updated_MP4_evidence'] is None
