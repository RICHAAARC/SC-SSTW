"""CPU/static checks for the carrier-agnostic joint-control seam."""
from __future__ import annotations

import inspect
import pytest
import torch

from main.tube_state import local_joint_state_payload_v1 as api
from runtime.wan import grow_video_reference as grow
from runtime.wan import local_joint_state_payload_v1 as adapter


pytestmark=pytest.mark.unit


class Scheduler:
    def __init__(self):
        class Config(dict):
            def __getattr__(self,name):return self[name]
        self.config=Config(
            prediction_type="flow_prediction",thresholding=False,
            lower_order_final=True)
        self.predict_x0=True
        self.step_index=None
        self.timesteps=torch.arange(50,0,-1)
        self.sigmas=torch.linspace(1.0,0.0,51)
        self.history=[]

    def step(self,velocity,timestep,state,return_dict=False):
        index=0 if self.step_index is None else self.step_index
        self.step_index=index+1
        self.history.append(float(velocity.reshape(-1)[0]))
        return (state-.01*velocity,)


class Pipe:
    @staticmethod
    def transformer(hidden_states,timestep,encoder_hidden_states,**kwargs):
        return (hidden_states*.01+encoder_hidden_states,)


def windows():
    return (
        api.VideoWindowCoord("w0",(0,1,2,3),"output_frame","rgb",(0,8,0,8)),
        api.VideoWindowCoord("w1",(4,5,6,7),"output_frame","rgb",(0,8,0,8)),
    )


def count_ledger():
    ledger={}
    def count(kind,complete):
        ledger.setdefault(kind,[0,0])[int(complete)]+=1
    return ledger,count


def test_injected_provider_is_same_callable_on_actual_50_step_path(monkeypatch):
    initial=torch.zeros((1,2,3,2,2))
    calls=[]
    requested={"state_request":.25,"payload_request":.75,"unit":"provider_defined"}
    def provider(request,z,conditional,unconditional):
        calls.append(request)
        state=torch.full_like(z,.01)
        payload=torch.full_like(z,.02)
        joint=torch.full_like(z,.025)
        return api.JointControlResult(
            joint,"external_joint_cap",True,state,payload,True,{"fixture":True})
    control=adapter.make_control_step(
        windows=windows(),state_spec={"kind":"external"},payload_spec={"kind":"external"},
        requested_budget=requested,provider=provider)
    monkeypatch.setattr(grow.method,"build_target",lambda *a,**k: (_ for _ in ()).throw(
        AssertionError("injected path must not build the legacy GROW carrier")))
    ledger,count=count_ledger();rows=[];scheduler=Scheduler()
    final,receipt=grow.run_trajectory(
        Pipe(),initial,scheduler,torch.tensor(.02),torch.tensor(.01),torch.float32,
        "OFF","unused",[],count,rows.append,injected_control=control)
    assert final.shape==initial.shape and scheduler.step_index==50
    assert len(calls)==50 and [x.sampling.index for x in calls]==list(range(50))
    assert all(x.windows==windows() for x in calls)
    assert calls[9].sampling.index==9 and calls[9].windows[0].time_indices==(0,1,2,3)
    assert ledger["injected_joint_control"]==[50,50]
    assert receipt["control_adapter"]=="injected_local_joint_state_payload"
    assert [x["requested_budget"] for x in rows]==[requested]*50
    actual=rows[0]["actual_control_delta"]
    assert actual["coordinate"]=="conditional_clean_estimate"
    assert actual["state_diagnostic"]["rms"]==pytest.approx(.01)
    assert actual["payload_diagnostic"]["rms"]==pytest.approx(.02)
    assert actual["joint"]["rms"]==pytest.approx(.025)
    assert actual["cfg_velocity"]["coordinate"]=="guided_velocity"
    assert rows[0]["composition"]=="external_joint_cap"
    assert rows[0]["provider_declares_component_decomposition"]
    assert rows[0]["joint_minus_components_l2"]==pytest.approx(.005*(initial.numel()**.5))


def test_joint_delta_is_only_applied_value_and_diagnostic_components_are_optional():
    z=torch.zeros((1,1,1,1,1));conditional=torch.ones_like(z);unconditional=torch.zeros_like(z)
    request=api.JointControlRequest(
        api.SamplingStepCoord(4,50,.5),windows(),{}, {}, {"description":"explicit fixture"})
    result=api.JointControlResult(
        joint_delta=torch.full_like(z,.2),composition="provider_defined",enabled=True)
    velocity,row=adapter.apply_joint_control(z,conditional,unconditional,request,result,guidance=5)
    torch.testing.assert_close(velocity,torch.full_like(z,3.0))
    assert row["actual_control_delta"]["state_diagnostic"] is None
    assert row["actual_control_delta"]["payload_diagnostic"] is None
    assert not row["provider_declares_component_decomposition"]
    with pytest.raises(ValueError,match="disabled"):
        adapter.apply_joint_control(
            z,conditional,unconditional,request,
            api.JointControlResult(torch.ones_like(z),"external",False))


def test_default_grow_seam_keeps_legacy_target_and_control(monkeypatch):
    initial=torch.zeros((1,2,3,2,2));target=torch.ones_like(initial);mask=torch.ones_like(initial,dtype=torch.bool)
    build=[];guided=[]
    def build_target(z,key,bits):build.append((key,bits));return target,mask
    def guided_velocity(z,c,u,sigma,t,m,*,enabled,guidance,reference_eta):
        assert t is target and m is mask
        guided.append((enabled,reference_eta))
        return u+guidance*(c-u),dict(enabled=enabled,legacy_fixture=True)
    monkeypatch.setattr(grow.method,"build_target",build_target)
    monkeypatch.setattr(grow.method,"guided_velocity",guided_velocity)
    monkeypatch.setattr(grow.method,"control_enabled",lambda arm,index: index>=25)
    ledger,count=count_ledger();rows=[]
    _,receipt=grow.run_trajectory(
        Pipe(),initial,Scheduler(),torch.tensor(.02),torch.tensor(.01),torch.float32,
        "MULTI","key",[1],count,rows.append,reference_eta=7)
    assert build==[("key",[1])] and len(guided)==50
    assert [x[0] for x in guided]==[False]*25+[True]*25
    assert all(x[1]==7 for x in guided)
    assert ledger["local_gradient"]==[25,25]
    assert "control_adapter" not in receipt


def test_received_only_window_records_preserve_scored_missing_and_failed():
    declared=windows()+(api.VideoWindowCoord(
        "w2",(8,9,10,11),"received_frame","rgb8",(8,16,8,16)),)
    seen=[]
    def observe(received,public,key,window):
        seen.append((received,public,key,window.window_id))
        if window.window_id=="w0":
            return api.WindowObservation(
                window,"SCORED","SCORED",(.125,.25),(.2,-.1),
                {"chips":2},{"coefficients":2})
        if window.window_id=="w1":return None
        raise RuntimeError("fixture observer failure")
    rows=adapter.collect_window_observations("received","public","current-key",declared,observe)
    assert [x.status for x in rows]==["SCORED","MISSING","FAILED"]
    assert [x.window for x in rows]==list(declared) and len(rows)==len(declared)
    assert rows[0].state_soft_q==(.125,.25) and rows[0].payload_soft_evidence==(.2,-.1)
    assert rows[1].state_error=="observer returned no observation"
    assert rows[2].payload_error=="RuntimeError: fixture observer failure"
    assert all(x[:3]==("received","public","current-key") for x in seen)
    assert list(inspect.signature(adapter.collect_window_observations).parameters)==[
        "received","public_protocol","key","windows","observe_window"]


def test_partial_window_preserves_observed_state_vector():
    window=windows()[0]
    row=api.WindowObservation(
        window,"SCORED","MISSING",(.1,.2,.3),None,{"chips":3},{},None,"no payload support")
    assert row.status=="PARTIAL" and row.state_soft_q==(.1,.2,.3)


def test_boundary_path_and_equivalence_are_containers_without_decision_rules():
    boundary=api.BoundaryRef("b0","w0","w1")
    path=api.PathCandidate("p0",("w0","w1"),(boundary.boundary_id,))
    equivalence=api.PathEquivalenceClass("e0",("p0","p1"))
    assert path.status=="UNSCORED" and equivalence.path_ids==("p0","p1")
    for value in (boundary,path,equivalence):
        assert not hasattr(value,"decode") and not hasattr(value,"threshold")
    with pytest.raises(ValueError):api.PathEquivalenceClass("e1",("p0","p0"))
