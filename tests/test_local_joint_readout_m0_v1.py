from __future__ import annotations

import json
import types

import numpy as np
import pytest
import torch

from main.tube_state import local_joint_readout_m0_v1 as method
from main.tube_state import local_joint_state_payload_carrier_v1 as carrier
from main.tube_state import local_joint_terminal_bridge_v1 as observer
from runtime.wan import local_joint_readout_m0_v1 as runtime
from runtime.wan.local_joint_readout_checkpoint_v1 import ReplayLedger, checkpoint_decode
from runtime.wan.local_joint_state_payload_provider_v1 import LatentSupport
from runtime.wan import vae as adapter
from runtime.wan import local_joint_readout_checkpoint_v1 as checkpointing

pytestmark = pytest.mark.quick
PROTOCOL = carrier.CarrierProtocol((181,16,16,3),1,8,22,
    ((0,8,0,8),(0,8,8,16),(8,16,0,8),(8,16,8,16)))
SHAPE = (1,3,46,16,16)
SUPPORT = LatentSupport((1,45), ((0,4,0,4),(0,4,12,16),(12,16,0,4),(12,16,12,16)))
KEY = runtime.KEY
MESSAGE = bytes.fromhex("8001a55a")


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


class CausalDecoder(torch.nn.Module):
    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        assert feat_idx[0] == 0 and feat_cache[1] == "Rep"
        previous = torch.zeros_like(x) if feat_cache[0] is None else feat_cache[0]
        alias = torch.zeros_like(x) if feat_cache[2] is None else feat_cache[2]
        if feat_cache[0] is not None:
            assert previous.untyped_storage().data_ptr() == alias.untyped_storage().data_ptr()
            assert alias.stride() != previous.stride()
        y = torch.tanh(.7*x + .31*previous + .13*alias + (.01 if first_chunk else -.02))
        feat_cache[0] = y
        feat_cache[2] = y.transpose(-1,-2)  # Alias storage with different strides.
        feat_idx[0] = 3
        return y


class TinyWan(torch.nn.Module):
    def __init__(self, fail_decode=None, malformed=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.decoder = CausalDecoder()
        self.config = types.SimpleNamespace(latents_mean=[.1,-.2,.3], latents_std=[.5,.8,1.2])
        self.use_tiling = False
        self.fail_decode, self.malformed, self.calls = fail_decode, malformed, 0
        self.clear_cache()

    def clear_cache(self):
        self.cache = [None,"Rep",None]

    def decode(self, z, return_dict=False):
        self.calls += 1
        if self.calls == self.fail_decode:
            raise RuntimeError("injected decoder failure")
        outputs = []
        for t in range(z.shape[2]):
            y = self.decoder(z[:,:,t:t+1], feat_cache=self.cache, feat_idx=[0], first_chunk=t == 0)
            outputs.append(y if t == 0 else y.repeat_interleave(4, dim=2))
        decoded = (1.4*torch.cat(outputs, dim=2)).clamp(-1,1)  # Actual internal clamp participates.
        return (decoded[:,:2] if self.malformed else decoded,)


def test_tensor_reader_original_values_and_directional_finite_difference():
    rgb = (.2+.6*torch.rand(PROTOCOL.video_shape, generator=torch.Generator().manual_seed(9))).double().requires_grad_()
    read = method.tensor_readout(rgb,key=KEY,message=MESSAGE,protocol=PROTOCOL)
    raw = observer.read_rgb(rgb.detach(),key=KEY,rho=.5,protocol=PROTOCOL)
    independent = observer.evaluate_saved(raw,key=KEY,message=MESSAGE,rho=.5,protocol=PROTOCOL)
    values = runtime.metric_values(independent)
    torch.testing.assert_close(read["correlations"],torch.tensor(values["correlations"],dtype=torch.double),rtol=0,atol=1e-15)
    torch.testing.assert_close(read["margins"].flatten(),torch.tensor(values["margins"],dtype=torch.double),rtol=0,atol=1e-15)
    original_q = torch.tensor([[c["q"] for c in row["state_chips"]+row["payload_chips"]] for row in raw],dtype=torch.double)
    torch.testing.assert_close(read["q"].reshape(88,16),original_q,rtol=0,atol=1e-15)
    assert not read["tied"]
    direction = torch.randn(rgb.shape,generator=torch.Generator().manual_seed(8),dtype=torch.double)
    direction /= direction.norm()
    gradient = torch.autograd.grad(read["objectives"][2],rgb)[0]
    h = 1e-4
    plus = method.tensor_readout(rgb.detach()+h*direction,key=KEY,message=MESSAGE,protocol=PROTOCOL)
    minus = method.tensor_readout(rgb.detach()-h*direction,key=KEY,message=MESSAGE,protocol=PROTOCOL)
    assert plus["active"] == minus["active"] == read["active"]
    assert float((plus["objectives"][2]-minus["objectives"][2])/(2*h)) == pytest.approx(float((gradient*direction).sum()),abs=1e-9)
    # Exact pooling derivative, no average of individual tile ratios.
    a = torch.tensor([1.,4.],dtype=torch.double,requires_grad=True)
    b = torch.tensor([2.,3.],dtype=torch.double,requires_grad=True)
    A,B = a.square().sum(),b.square().sum(); q=(A-B)/(A+B)
    da,db=torch.autograd.grad(q,(a,b))
    torch.testing.assert_close(da,4*a.detach()*B.detach()/(A+B).detach().square())
    torch.testing.assert_close(db,-4*b.detach()*A.detach()/(A+B).detach().square())


def test_simplex_degeneracy_kkt_and_no_individual_normalization():
    # All-five strict improvement impossible, but (0,1) weakly improves the last three.
    g = [torch.tensor(x,dtype=torch.float32) for x in [(1,0),(-1,0),(0,1),(0,1),(0,1)]]
    directions,report=method.directions(g)
    assert directions["COMMON"] is None and directions["MEAN"] is not None
    assert report["simplex"]["status"] in ("EXACT_ZERO","NUMERICALLY_ZERO")
    assert "weak" in report["zero_common_ceiling"]
    zero, report = method.directions([torch.zeros(2) for _ in range(5)])
    assert zero == dict(MEAN=None,COMMON=None) and report["mean_status"] == "ZERO_MEAN"
    vectors=[torch.tensor(x,dtype=torch.float32) for x in [(1,0),(2,0),(1,1),(1,-1),(3,2)]]
    answer,report=method.directions(vectors)
    expected=torch.stack(vectors).mean(0); expected/=expected.norm()
    torch.testing.assert_close(answer["MEAN"],expected)
    assert answer["COMMON"] is not None
    assert all(x>0 for x in report["predicted"]["COMMON"])
    assert sum(report["simplex"]["weights"]) == pytest.approx(1)
    assert report["simplex"]["kkt_residual"] < 1e-10
    assert np.all(np.linalg.eigvalsh(report["gram"]) >= -1e-12)
    near=[torch.tensor(x,dtype=torch.float32) for x in [(1,1e-7),(-1,1e-7),(0,1),(0,1),(0,1)]]
    answer,report=method.directions(near)
    assert answer["COMMON"] is None and report["simplex"]["status"] == "NUMERICALLY_ZERO"
    assert all(float(x[1])>0 for x in near)  # A strict common direction really exists.
    assert "does not exclude" in report["zero_common_ceiling"]


def test_checkpoint_matches_cross_chunk_vjp_and_internal_clamp(tmp_path,monkeypatch):
    monkeypatch.setenv("M0_BOUNDARY_SPOOL_ROOT",str(tmp_path))
    z=(torch.randn((1,3,5,8,8),generator=torch.Generator().manual_seed(11))*1.1).requires_grad_()
    model=TinyWan()
    mean,std=adapter._scale_tensors(model,z)
    direct=model.decode(z*std+mean)[0]
    loss=direct[:,:,-1].square().sum()+direct[:,:,-2].sin().sum()
    grad=torch.autograd.grad(loss,z)[0]
    assert grad[:,:,:1].abs().sum() > 0  # Last chunk depends on the first via cache.
    model.clear_cache(); ledger=ReplayLedger()
    replay=checkpoint_decode(model,z*std+mean,ledger)
    actual=torch.autograd.grad(replay[:,:,-1].square().sum()+replay[:,:,-2].sin().sum(),z)[0]
    torch.testing.assert_close(replay,direct,rtol=0,atol=0)
    torch.testing.assert_close(actual,grad,rtol=0,atol=0)
    assert ledger.counts["vae_chunk"]["forward"] == dict(attempted=5,completed=5)
    assert ledger.counts["vae_chunk"]["recompute"] == dict(attempted=5,completed=5)
    assert ledger.summary()["boundary_storage"]["disk_read_bytes"] > 0
    ledger.release_boundary_storage()
    assert ledger.summary()["boundary_storage"]["closed"]
    assert not list(tmp_path.iterdir())


def test_replay_callback_io_failure_releases_restored_alias_storage(tmp_path,monkeypatch):
    monkeypatch.setenv("M0_BOUNDARY_SPOOL_ROOT",str(tmp_path))
    storages=[]
    original=checkpointing.BoundaryStorage
    class InspectedStorage(original):
        def __init__(self,spool):
            super().__init__(spool); storages.append(self)
    monkeypatch.setattr(checkpointing,"BoundaryStorage",InspectedStorage)
    ledger=ReplayLedger()
    model=TinyWan()
    z=torch.randn((1,3,3,8,8),generator=torch.Generator().manual_seed(44),requires_grad=True)
    value=checkpoint_decode(model,z,ledger)
    def fail_callback(summary):
        if summary["counts"]["vae_chunk"]["recompute"]["attempted"]:
            raise IOError("durable replay callback failure")
    ledger.callback=fail_callback
    with pytest.raises(IOError,match="callback failure"):
        torch.autograd.grad(value[:,:,-1].square().sum(),z)
    assert storages and all(not item.restored and not item.copies for item in storages)
    ledger.callback=None
    ledger.release_boundary_storage()
    assert not list(tmp_path.iterdir())


def inputs(tmp_path):
    source=tmp_path/"source"; source.mkdir()
    z=torch.randn(SHAPE,generator=torch.Generator().manual_seed(4))*.3
    delta=runtime.masked(torch.randn(SHAPE,generator=torch.Generator().manual_seed(6)),SUPPORT)
    delta/=delta.double().norm()
    torch.save(z,source/"terminal.pt"); torch.save(delta,source/"capped.pt")
    return dict(carrier=dict(key=KEY,message_hex="8001a55a",rho=.5,cap=1),model={},device="cpu",
                inputs=dict(terminal_latent=str(source/"terminal.pt"),bridge_delta=str(source/"capped.pt")))


def execute(config,tmp_path,model):
    return runtime.run(config,tmp_path/"out",loader=lambda *_a,**_k:model,
        protocol=PROTOCOL,latent_shape=SHAPE,support=SUPPORT)


def test_full_fixed_runner_counts_and_saved_evidence(tmp_path,monkeypatch):
    monkeypatch.setenv("M0_BOUNDARY_SPOOL_ROOT",str(tmp_path))
    config=inputs(tmp_path)
    result=execute(config,tmp_path,TinyWan())
    assert result["status"] == "COMPLETE",result["failures"]
    assert result["counts"]["decode_completed"] == 11
    assert result["counts"]["decoder_vjp_completed"] == 5
    assert result["ordinary_chunks"] == dict(attempted=276,completed=276)
    assert sum(x["counts"]["vae_chunk"]["forward"]["completed"] for x in result["replay"].values()) == 230
    assert sum(x["counts"]["vae_chunk"]["recompute"]["completed"] for x in result["replay"].values()) == 230
    assert result["totals"]["observed_chips"] == 8448
    assert result["totals"]["observed_metrics"] == 330
    assert all(x["status"] == "SCORED" for x in result["views"].values())
    for name in method.VIEWS:
        row=result["views"][name]
        assert row["artifacts"]["quality"] == row["artifacts"]["frames"] == "SAVED"
        if name in ("MEAN","COMMON"): assert row["delta_l2"] == pytest.approx(1.,abs=1e-7)
    report=json.loads((tmp_path/"out/comparison.json").read_text())
    assert len(report["finite_difference"]["central_slope"]) == 5
    for row in result["gradient_readouts"].values():
        assert not row["active_change_from_base"]
        assert max(abs(x) for x in row["objective_change_from_base"]) < 1e-12
    quality=json.loads((tmp_path/"out/COMMON/quality.json").read_text())
    assert quality["max_abs"] > 0 and quality["roi_inside_energy"] > 0
    assert not list(tmp_path.glob("wan-vae-boundary-*"))


def test_gradient_failure_retains_comparators_without_filling_calls(tmp_path,monkeypatch):
    monkeypatch.setenv("M0_BOUNDARY_SPOOL_ROOT",str(tmp_path))
    result=execute(inputs(tmp_path),tmp_path,TinyWan(fail_decode=3))
    assert result["status"] == "ENGINEERING_FAILURE"
    assert result["counts"]["decode_attempted"] == 3 and result["counts"]["decode_completed"] == 2
    assert result["counts"]["decoder_vjp_attempted"] == 0
    assert result["totals"]["expected_metrics"] == 330 and result["totals"]["observed_metrics"] == 110
    assert all(result["views"][x]["direction_status"] == "UNDEFINED" for x in method.VIEWS[2:])
    assert not list(tmp_path.glob("wan-vae-boundary-*"))


@pytest.mark.parametrize("tie_at",[1,2])
def test_active_tie_keeps_independent_views_and_undefined_slots(tmp_path,monkeypatch,tie_at):
    monkeypatch.setenv("M0_BOUNDARY_SPOOL_ROOT",str(tmp_path))
    actual=method.tensor_readout
    calls=0
    def tied(*args,**kwargs):
        nonlocal calls
        calls+=1
        read=actual(*args,**kwargs)
        if calls == tie_at:
            read["tied"]=True
            read["active"]["fragment_bits"][0]=[0,1]
        return read
    monkeypatch.setattr(method,"tensor_readout",tied)
    result=execute(inputs(tmp_path),tmp_path,TinyWan())
    assert result["status"] == "COMPLETE"
    assert result["counts"]["decode_completed"] == (2 if tie_at == 1 else 3)
    assert result["counts"]["decoder_vjp_attempted"] == 0
    assert result["totals"]["observed_metrics"] == 110
    assert result["totals"]["expected_metrics"] == 330
    for name in method.VIEWS[2:]:
        assert result["views"][name]["direction_status"] == "UNDEFINED"
        assert result["views"][name]["status"] == "MISSING"


def test_returned_decode_adapter_failure_counts_return(tmp_path):
    result=execute(inputs(tmp_path),tmp_path,TinyWan(malformed=True))
    assert result["counts"]["decode_completed"] == 2
    assert result["totals"]["observed_metrics"] == 0


def test_interrupt_after_metrics_recovers_counts_without_redecoding(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,"quality",lambda *_a: (_ for _ in ()).throw(KeyboardInterrupt("after saved metrics")))
    model=TinyWan()
    result=execute(inputs(tmp_path),tmp_path,model)
    assert result["status"] == "INTERRUPTED" and model.calls == 1
    assert result["totals"]["observed_metrics"] == 55
    assert result["views"]["BASE"]["artifacts"]["quality"] == "MISSING"
    # Simulate hard kill with stale view counters but durable raw + metrics.
    result["status"]="RUNNING"
    result["views"]["BASE"].update(status="MISSING",observed_metrics=0,observed_chips=0,observed_windows=0)
    runtime.write_json(tmp_path/"out/result.json",result)
    recovered=runtime.finalize_interrupted(tmp_path/"out","reaped child")
    assert recovered["totals"]["observed_metrics"] == 55 and model.calls == 1
    assert len(json.loads((tmp_path/"out/comparison.json").read_text())["metrics"]) == 6
