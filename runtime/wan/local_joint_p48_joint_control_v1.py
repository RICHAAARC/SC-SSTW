"""One fixed output-aware P48 control. No physical retry or extra candidate."""
from __future__ import annotations
import copy
import gc
from pathlib import Path
import time
from main.tube_state import local_joint_p48_joint_control_v1 as method
from main.tube_state import local_joint_state_payload_carrier_v1 as carrier
from main.tube_state.local_joint_p0_input_vjp_v1 import finite_tensor, direction_contract, statistics, difference, LATENT_SHAPE
from runtime.wan.local_joint_p0_input_vjp_v1 import write_json, save_tensor, tensor, prepare_transformer, validate_model_arithmetic
from runtime.wan.local_joint_p0_checkpoint_v1 import checkpointed_branch, DiTSpool
from runtime.wan.local_joint_readout_checkpoint_v1 import ReplayLedger, checkpoint_decode
from runtime.wan.local_joint_p48_journal_v1 import (Journal, initial, read_state, precheck_can_continue,
    needs_model_work, resume_report, save_report, finalize_interrupted)
from runtime.wan.local_joint_p48_native_v1 import coefficients
from runtime.wan.local_joint_state_payload_provider_v1 import PUBLIC_LATENT_SUPPORT
from runtime.wan.local_joint_paired_transfer_native_v1 import restore, controlled_velocity

KEY="local-joint-state-payload-v1-first-mechanism"
MESSAGE=bytes.fromhex("8001a55a")
BRANCHES=("conditional","unconditional")


def validate_config(c):
    if c.get("views")!=list(method.VIEWS) or c.get("objectives")!=list(method.OBJECTIVES) or c.get("guidance")!=5 or c.get("cap")!=.02:
        raise ValueError("fixed ZERO/ANCHOR/CANDIDATE, S/P/W/Q, CFG5 and .02 required")
    if c.get("block_count")!=30 or c.get("transformer_dtype")!="bfloat16":
        raise ValueError("30-block native BF16/FP32 arithmetic required")
    if not isinstance(c.get("source_run"),str) or not c["source_run"]: raise ValueError("saved source path required")


def load_source(c):
    import torch
    p=Path(c["source_run"])
    fork=tensor(p/"native/fork48.pt"); context=tensor(p/"native/context.pt")
    for n in ("z","conditional","unconditional"): finite_tensor(fork[n],LATENT_SHAPE,"fork48/"+n,torch.float32)
    unit=tensor(p/"units/gradient.pt"); old_zero=tensor(p/"ZERO/terminal_latent.pt")
    finite_tensor(unit["input_terminal"],LATENT_SHAPE,"saved gradient input",torch.float32)
    finite_tensor(old_zero,LATENT_SHAPE,"saved ZERO",torch.float32)
    if not torch.equal(unit["input_terminal"],old_zero): raise ValueError("saved d does not belong to original ZERO point")
    d=unit["direction"].detach().clone()
    direction_contract(d)
    for n in ("prompt","negative"):
        e=finite_tensor(context[n],name=n,dtype=torch.bfloat16)
        if e.ndim!=3 or e.shape[0]!=1 or not e.numel(): raise ValueError("invalid saved embedding coordinates")
    # Keep only actual required values; old RGB/gradient buffers are not retained.
    return dict(fork=fork,prompt=context["prompt"],negative=context["negative"],direction=d,
        source_note=dict(path=str(p),saved_dtype_label=context.get("dtype"),direction_dtype=str(d.dtype)))


class VaeLedger(ReplayLedger):
    maximum_recomputes=1
    def __init__(self,callback,root,chunks):
        super().__init__(callback); self.root=root; self.chunks=chunks
    def new_decode(self):
        if self.boundaries: raise RuntimeError("one VAE graph per objective")
        # Same spool transport, with partial-write accounting even on failure.
        self.boundary_spool=DiTSpool(self.root); self.boundary_spool.kind="VAE"
        self.boundaries[0]={p:[] for p in ("forward","recompute")}; self._emit(); return 0
    def start(self,kind,phase):
        if self.counts[kind][phase]["attempted"]>=self.chunks: raise RuntimeError("VAE chunk budget exhausted")
        super().start(kind,phase)


def evidence(directory,row,name,value):
    saved=value.detach().cpu()
    save_tensor(directory/name,saved)
    row.setdefault("evidence",[]).append(dict(file=name,shape=list(saved.shape),dtype=str(saved.dtype).split(".")[-1]))
    finite_tensor(saved,name=name)
    return saved


def json_record(row,name,kind,shape=None):
    row.setdefault("json_evidence",[]).append(dict(file=name,kind=kind,shape=None if shape is None else list(shape)))


def observe(directory,rgb,protocol):
    from main.tube_state import local_joint_terminal_bridge_v1 as observer
    from runtime.wan.local_joint_readout_m0_v1 import metric_values
    raw=observer.read_rgb(rgb,key=KEY,rho=.5,protocol=protocol)
    write_json(directory/"raw.json",raw)  # Blind read committed before target-aware scoring.
    scores=observer.evaluate_saved(raw,key=KEY,message=MESSAGE,rho=.5,protocol=protocol)
    write_json(directory/"scores.json",scores)
    values=metric_values(scores); write_json(directory/"values.json",values); return values


def quality_record(directory,z,rgb,zero_z,zero_rgb,support,protocol):
    values,(_,r,dt,_,_,frames,transitions)=method.quality_data(z,rgb,zero_z,zero_rgb,support,protocol)
    record=dict(energies=dict(zip(method.ENERGIES,values.tolist())),frame_mse=frames.tolist(),transition_mse=transitions.tolist(),
        frame_max_abs=r.abs().flatten(1).amax(1).tolist(),transition_max_abs=dt.abs().flatten(1).amax(1).tolist(),
        residual_clipped=False,reference="same-run ZERO",scientific_pass=False)
    write_json(directory/"quality.json",record); return values


def run(config,output,*,source_loader=None,preparer=None,vae_loader=None,protocol=carrier.PUBLIC,
        shape=LATENT_SHAPE,support=PUBLIC_LATENT_SUPPORT,expected_blocks=30,observer=None,reader=None,solver=None):
    """Test-only dependency injection is Python-only; the delivered CLI has one fixed path."""
    import torch
    from runtime.wan.generation import load_frozen_vae
    from runtime.wan.vae import _clear_cache,_scale_tensors,decode_normalized_latent
    validate_config(config)
    output=Path(output); old=read_state(output)
    if old is not None and not precheck_can_continue(output,old): return resume_report(output,old)
    result=old or initial(config)
    if old:
        result.setdefault("preflight_history",[]).append(copy.deepcopy({k:v for k,v in old.items() if k!="preflight_history"}))
    result["status"]="RUNNING"
    failure_start=len(result["failures"]); attempt=str(time.time_ns())
    result["current_attempt"]=dict(id=attempt,status="RUNNING",failure_start_index=failure_start)
    device=config.get("device","cuda"); journal=Journal(output,result,device)
    model=vae=None; stage="saved_source"
    endpoints={}; gradients={}
    observer=observer or observe; reader=reader or method.tensor_readout; solver=solver or method.solve_correction

    def move(obj,target,label):
        journal.start_stage(label)
        try:
            returned=obj.to(target)
            if returned is not obj: raise RuntimeError("staged offload must retain the same loaded object")
            result.setdefault("object_moves",[]).append(dict(object=label,device=str(target),same_object=True))
        finally: journal.end_stage()
        gc.collect()

    def prediction(x,embedding,timestep,directory,row,name):
        raw=model(hidden_states=x.to(torch.bfloat16),timestep=timestep.to(device),
            encoder_hidden_states=embedding.to(device),attention_kwargs=None,return_dict=False)[0]
        receipt=dict(input_dtype="torch.bfloat16",actual_output_dtype=str(raw.dtype),saved_dtype="torch.float32")
        receipt_name=name.replace(".pt","_arithmetic.json")
        json_record(row,receipt_name,"arithmetic"); write_json(directory/receipt_name,receipt)
        # Retain the actual result even when its arithmetic is invalid; no retry.
        evidence(directory,row,name,raw.float())
        finite_tensor(raw,shape,"actual Transformer output",torch.bfloat16)
        return raw.float()

    def observation(directory,row,rgb):
        for name in ("raw","scores","values"): json_record(row,name+".json",name)
        return observer(directory,rgb,protocol)

    def endpoint_tail(name,theta,source):
        row=result["endpoints"][name]; directory=output/name
        row["status"]="RUNNING"; journal.save()
        scheduler=restore(source["fork"]["scheduler"],device)
        # Validate each restored actual history before its two native updates.
        coefficients(scheduler,shape)
        x=source["fork"]["z"].to(device)
        c,u=(source["fork"][n].to(device) for n in BRANCHES)
        # Reuse the exact FP32 adapter order: conditional - theta/sigma, then CFG.
        velocity,receipt=controlled_velocity(x,c,u,result["native_coefficients"]["sigma48"],theta.to(device))
        receipt.update(provider_composition="fixed P48 conditional-clean control",coordinate_relation="theta conditional-clean; native full-history effect measured",control=statistics(theta))
        json_record(row,"control.json","control")
        write_json(directory/"control.json",receipt)
        evidence(directory,row,"theta.pt",theta)
        def native(v,z,i):
            if scheduler.step_index!=i: raise ValueError("native cursor mismatch")
            with torch.no_grad(): value=scheduler.step(v,scheduler.timesteps[i],z.clone(),return_dict=False)[0]
            if scheduler.step_index!=i+1: raise ValueError("native cursor did not advance")
            finite_tensor(value,shape,"native result",torch.float32); return value.detach()
        x49=journal.call("native_update",name+"/48",lambda:native(velocity,x,48))
        x49=evidence(directory,row,"after48.pt",x49)
        predictions={}
        for b,e in zip(BRANCHES,(source["prompt"],source["negative"])):
            with torch.no_grad(): y=journal.call("dit_ordinary_forward",name+"/"+b,lambda:prediction(x49.to(device),e,scheduler.timesteps[49].reshape(1),directory,row,b+"_output.pt"))
            predictions[b]=y.detach().cpu()
            del y
        c,u=(predictions[b].to(device) for b in BRANCHES)
        z=journal.call("native_update",name+"/49",lambda:native(u+5*(c-u),x49.to(device),49))
        z=evidence(directory,row,"terminal_latent.pt",z)
        row["tail"]="SAVED"; journal.save()
        endpoints[name]=dict(z=z,x49=x49,predictions=predictions)

    def ordinary(name):
        row=result["endpoints"][name]; directory=output/name
        original=vae.decoder.forward
        count=result["ordinary_chunks"][name]=dict(attempted=0,completed=0)
        def chunk(*a,**kw):
            if count["attempted"]>=shape[2]: raise RuntimeError("ordinary VAE chunk budget exhausted")
            count["attempted"]+=1; journal.save()
            v=original(*a,**kw); count["completed"]+=1; journal.save(); return v
        vae.decoder.forward=chunk
        try: rgb=journal.call("ordinary_decode",name,lambda:decode_normalized_latent(vae,endpoints[name]["z"].to(device)))
        finally: vae.decoder.forward=original
        rgb=evidence(directory,row,"rgb.pt",rgb)
        finite_tensor(rgb,protocol.video_shape,"decoded RGB",torch.float32)
        if count["completed"]!=shape[2]: raise ValueError("ordinary chunk count differs from latent frames")
        endpoints[name]["rgb"]=rgb
        row["values"]=observation(directory,row,rgb); row["status"]="COMPLETE"; journal.save()

    def decoder_unit(name,den):
        row=result["objectives"][name]; row["status"]="RUNNING"
        directory=output/"objectives"/name
        def update(v):
            result["vae_replay"][name]=copy.deepcopy(v); journal.save()
        ledger=VaeLedger(update,config.get("spool_root"),shape[2])
        z=rgb=decoded=gz=seed=direct=None
        try:
            _clear_cache(vae)
            z=endpoints["ANCHOR"]["z"].to(device).clone().requires_grad_(True)
            mean,std=_scale_tensors(vae,z)
            decoded=journal.call("gradient_decode",name,lambda:checkpoint_decode(vae,z*std+mean,ledger))
            rgb=(decoded[0].permute(1,2,3,0).float()/2+.5).clamp(0,1)
            saved_rgb=evidence(directory,row,"rgb.pt",rgb)
            finite_tensor(saved_rgb,protocol.video_shape,"gradient RGB",torch.float32)
            row["rgb_comparison"]=difference(saved_rgb,endpoints["ANCHOR"]["rgb"])
            row["rgb_comparison"].update(comparison="same-preparation independent gradient decode versus ordinary ANCHOR",causal_attribution=None)
            json_record(row,"rgb_comparison.json","comparison")
            write_json(directory/"rgb_comparison.json",row["rgb_comparison"])
            row["values"]=observation(directory,row,saved_rgb)
            json_record(row,"objective.json","objective")
            if name=="Q":
                direct,seed,receipt=method.quality_cotangents(z,saved_rgb,endpoints["ZERO"]["z"],endpoints["ZERO"]["rgb"],den,support,protocol)
                residual=saved_rgb.double()-endpoints["ZERO"]["rgb"].double(); temporal=residual[1:]-residual[:-1]
                save_tensor(directory/"pixel_peak_ties.pt",(residual.square()==residual.square().amax()).nonzero())
                save_tensor(directory/"temporal_peak_ties.pt",(temporal.square()==temporal.square().amax()).nonzero())
                del residual,temporal
                evidence(directory,row,"direct_latent_cotangent.pt",direct)
                evidence(directory,row,"rgb_cotangent_fp64.pt",seed)
                seed=seed.to(device=device,dtype=rgb.dtype)
                receipt["cast_rgb_cotangent"]=statistics(seed)
                write_json(directory/"objective.json",receipt)
                gz=journal.call("decoder_vjp",name,lambda:torch.autograd.grad(rgb,z,grad_outputs=seed,retain_graph=False,create_graph=False)[0])
                gz=gz+direct.to(device=device,dtype=z.dtype)
            else:
                read=reader(rgb,key=KEY,message=MESSAGE,protocol=protocol)
                scalar=method.objectives(read)[name]
                m,c=read["margins"].flatten(),read["correlations"]
                receipt=dict(value=float(scalar.detach()),active_other_offsets=(c[1:]==c[1:].amax()).nonzero().flatten().add(1).tolist(),
                    active_weak_bits=(m==m.amin()).nonzero().flatten().tolist(),tie_rule="equal active subgradient")
                write_json(directory/"objective.json",receipt)
                gz=journal.call("decoder_vjp",name,lambda:torch.autograd.grad(scalar,z,retain_graph=False,create_graph=False)[0])
            gradients[name]=evidence(directory,row,"terminal_cotangent.pt",gz)
            row.update(decoder="COMPLETE",terminal_cotangent=statistics(gz),output_support_masked=False)
            journal.save()
        finally:
            z=rgb=decoded=gz=seed=direct=None
            _clear_cache(vae); ledger.release_boundary_storage(); gc.collect()

    def branch_unit(name,b,source):
        row=result["objectives"][name]["branches"][b]; directory=output/"objectives"/name/b
        x=y=g=seed=None
        def update(v): result["dit_replay"][name+"/"+b]=v; journal.save()
        try:
            x=endpoints["ANCHOR"]["x49"].to(device).clone().requires_grad_(True)
            seed=gradients[name].to(device)*((-5 if b=="conditional" else 4)*result["native_coefficients"]["sigma49"])
            evidence(directory,row,"output_cotangent.pt",seed)
            with checkpointed_branch(model,update,root=config.get("spool_root")):
                y=journal.call("dit_gradient_forward",name+"/"+b,lambda:prediction(x,source["prompt" if b=="conditional" else "negative"],source["fork"]["scheduler"]["state"]["timesteps"][49].reshape(1),directory,row,"output.pt"))
                saved=y.detach().cpu()
                row["comparison"]=difference(saved,endpoints["ANCHOR"]["predictions"][b])
                row["comparison"].update(comparison="same-preparation independent gradient forward versus ordinary ANCHOR",causal_attribution=None)
                json_record(row,"comparison.json","comparison")
                write_json(directory/"comparison.json",row["comparison"])
                g=journal.call("dit_input_vjp",name+"/"+b,lambda:torch.autograd.grad(y,x,grad_outputs=seed,retain_graph=False,create_graph=False)[0])
                g=evidence(directory,row,"input_cotangent.pt",g)
                row.update(status="COMPLETE",input_coordinate="FP32 before native BF16 cast",statistics=statistics(g)); journal.save()
            return g
        finally:
            x=y=g=seed=None; gc.collect()

    with journal:
        try:
            journal.save()
            source=(source_loader or load_source)(config)
            anchor=(.02*source["direction"]).float()
            finite_tensor(anchor,shape,"actual anchor",torch.float32)
            if float(anchor.double().norm())>.02 or bool((anchor-method.restrict(anchor,support)).count_nonzero()):
                raise ValueError("actual FP32 anchor outside original support/.02 cap")
            result["native_coefficients"]=coefficients(restore(source["fork"]["scheduler"],"cpu"),shape)
            result["source_note"]=source.get("source_note")
            save_tensor(output/"saved_source.pt",source); save_tensor(output/"anchor.pt",anchor)
            stage="transformer_prepare"
            model=journal.call("transformer_prepare","shared",lambda:(preparer or prepare_transformer)(config,device))
            if len(model.blocks)!=expected_blocks or model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
                raise ValueError("fixed frozen eval Transformer block/arithmetic contract failed")
            result["transformer_arithmetic"]=validate_model_arithmetic(model)
            for n,theta in (("ZERO",torch.zeros_like(anchor)),("ANCHOR",anchor)):
                stage=n; endpoint_tail(n,theta,source)
            move(model,"cpu","Transformer offload")
            stage="vae_load"
            vae=journal.call("vae_load","shared",lambda:(vae_loader or load_frozen_vae)(config,device=device))
            if vae.training or any(p.requires_grad or p.grad is not None or p.dtype!=torch.float32 for p in vae.parameters()):
                raise ValueError("frozen FP32 eval VAE required")
            for n in ("ZERO","ANCHOR"):
                stage=n+"/decode"; ordinary(n)
            json_record(result["endpoints"]["ANCHOR"],"quality.json","quality",protocol.video_shape)
            den=quality_record(output/"ANCHOR",endpoints["ANCHOR"]["z"],endpoints["ANCHOR"]["rgb"],endpoints["ZERO"]["z"],endpoints["ZERO"]["rgb"],support,protocol)
            result["anchor_quality_denominators"]=den.tolist(); journal.save()
            if bool((den==0).any()): raise method.UndefinedConstruction("ZERO_QUALITY_DENOMINATOR; no epsilon or dropped energy")
            for n in method.OBJECTIVES:
                stage="decoder/"+n; decoder_unit(n,den)
            move(vae,"cpu","VAE offload"); move(model,device,"same Transformer reload to device")
            controls=[]
            for n in method.OBJECTIVES:
                stage="DiT/"+n
                gc_,gu=(branch_unit(n,b,source) for b in BRANCHES)
                full=result["native_coefficients"]["k48"]*(gradients[n]+gc_+gu)
                row=result["objectives"][n]; directory=output/"objectives"/n
                evidence(directory,row,"full_control_cotangent.pt",full)
                masked=evidence(directory,row,"control_cotangent.pt",method.restrict(full,support))
                row.update(status="COMPLETE",control_cotangent=statistics(masked),mask_applied="after complete native/DiT transport")
                controls.append(masked); journal.save()
            stage="SOCP"
            candidate,solution=solver(anchor,controls,support)
            json_record(result,"solution.json","solution")
            result["solution"]=solution; write_json(output/"solution.json",solution); journal.save()
            if candidate is None:
                result["status"]="NO_RESOLVED_CANDIDATE"
                result["endpoints"]["CANDIDATE"]["reason"]=solution.get("reason",solution["status"])
            else:
                # Boundary independent of solver implementation: verify actual FP32 control.
                finite_tensor(candidate,shape,"actual candidate",torch.float32)
                delta=candidate.double()-anchor.double()
                if float(candidate.double().norm())>.02 or float(delta.norm())>.02 or bool((candidate-method.restrict(candidate,support)).count_nonzero()):
                    raise ValueError("actual candidate violates fixed control bounds")
                save_tensor(output/"delta.pt",delta)
                stage="CANDIDATE"; endpoint_tail("CANDIDATE",candidate,source)
                move(model,"cpu","Transformer final offload"); move(vae,device,"same VAE reload to device")
                stage="CANDIDATE/decode"; ordinary("CANDIDATE")
                json_record(result["endpoints"]["CANDIDATE"],"quality.json","quality",protocol.video_shape)
                ce=quality_record(output/"CANDIDATE",endpoints["CANDIDATE"]["z"],endpoints["CANDIDATE"]["rgb"],endpoints["ZERO"]["z"],endpoints["ZERO"]["rgb"],support,protocol)
                result["comparison"]=method.finite_comparison(*(result["endpoints"][n]["values"] for n in method.VIEWS),den.tolist(),ce.tolist())
                result["status"]="COMPLETE"
            result.setdefault("comparison",method.finite_comparison(None,None,None,None,None))
        except method.UndefinedConstruction as exc:
            result.update(status="CONSTRUCTION_UNDEFINED",construction_reason=str(exc),comparison=dict(status="UNDEFINED",reason=str(exc),scientific_pass=False))
        except BaseException as exc:
            result["status"]="ENGINEERING_FAILURE" if isinstance(exc,Exception) else "INTERRUPTED"
            result["failures"].append(dict(stage=stage,reason=repr(exc),attempt_id=attempt,time_unix=time.time()))
            if not isinstance(exc,Exception): raise
        finally:
            for group in (result["endpoints"],result["objectives"]):
                for row in group.values():
                    if row["status"]=="RUNNING": row.update(status="INCOMPLETE",reason="stopped at "+stage)
                    if row["status"]=="MISSING" and row.get("reason")=="not_entered": row["reason"]="not entered before stop at "+stage
            result["finished_at_unix"]=time.time()
            result["current_attempt"].update(status=result["status"],failure_end_index=len(result["failures"]),failures=copy.deepcopy(result["failures"][failure_start:]))
            model=vae=None; gc.collect()
            journal.save(); save_report(output,result)
    return result
