"""Fresh ZERO terminal gradient paired with a full-history native short tail."""
from __future__ import annotations
import copy
import gc
from pathlib import Path
import time
from main.tube_state import local_joint_paired_transfer_v1 as method
from main.tube_state import local_joint_state_payload_carrier_v1 as carrier
from main.tube_state import local_joint_terminal_bridge_v1 as observer
from runtime.wan import generation,trajectory,vae as adapter
from runtime.wan import local_joint_paired_transfer_native_v1 as native
from runtime.wan import local_joint_all_bit_resume_v1 as resume
from runtime.wan import local_joint_same_point_dual_v1 as previous
from runtime.wan.grow_video_reference import predict_branches
from runtime.wan.local_joint_readout_m0_v1 import metric_values
from runtime.wan.local_joint_readout_checkpoint_v1 import ReplayLedger,checkpoint_decode
from runtime.wan.local_joint_state_payload_provider_v1 import PUBLIC_LATENT_SUPPORT
from runtime.wan.local_joint_terminal_bridge_v1 import KEY,LATENT_SHAPE,tensor_contract

EXPECTED=dict(generation_prepare=1,vae_load=2,gradient_decode=1,decoder_vjp=1,ordinary_decode=2,
    decode=3,gradient_forward_chunks=46,ordinary_chunks=92,total_forward_chunks=138,
    nominal_replay_chunks=46,baseline_dit=100,tail_dit=2,dit=102,gradient_dit=0,dit_vjp=0,
    dit_recompute=0,native=52,scheduler=52,encode=0,codec=0,independent_samples=0,
    existing_source_count=1,new_source_endpoints=3,endpoint_windows=264,endpoint_chips=4224,
    endpoint_metrics=165,gradient_windows=88,gradient_chips=1408,gradient_metrics=55,
    gradient_observations_reused_as_ZERO=True,quality_pairs=2)
FINISHED=previous.FINISHED
UNDEFINED=previous.UNDEFINED
KINDS=("generation_prepare","vae_load","decode","decoder_vjp","transformer_conditional",
       "transformer_unconditional","native_step")
tensor=previous.tensor
write_json=previous.write_json

def initial_result(config):
    return dict(schema="local-joint-paired-transfer-v1",config=copy.deepcopy(config),status="RUNNING",
        views={n:previous.slot() for n in method.VIEWS},units={"gradient":previous.slot()},
        steps={n:dict(status="NOT_RUN") for n in ("baseline","tail")},
        quality={n+"_vs_ZERO":dict(status="MISSING",reason="not_run") for n in method.VIEWS[1:]},
        model_calls={},call_history=[],counts={},replay={},failures=[],failure_history=[],cleanup_errors=[],
        ordinary_chunks=dict(attempted=0,completed=0),transport={},expected_cost=EXPECTED,
        scientific_pass=False,actual_model_execution="NOT_STARTED",automatic_retry=False,
        chunk_outcome_policy="Chunks report attempted/completed only. The enclosing decode receipt classifies forward/ordinary failure; the same-attempt decoder_vjp receipt classifies recompute failure or completion-unknown interruption. Their difference alone is not labeled failed.",
        ceiling=method.CEILING,selection_status="ADOPTED_BY_USER_20261015")

def read_state(output):
    output=Path(output);result=None;problems=[]
    for rel in ("result.json","result.json.previous","checkpoints/latest.json","checkpoints/previous.json"):
        path=output/rel
        if not path.exists():continue
        try:result=resume._valid_state(path,method.VIEWS,("baseline","tail"));break
        except (OSError,ValueError,TypeError) as exc:problems.append(dict(path=str(path),reason=repr(exc)))
    if result is None:
        if problems:raise OSError("no readable paired-transfer state: "+str(problems))
        return None
    if problems:result.setdefault("recovery_events",[]).extend(problems)
    for folder,field in (("calls","call_history"),("failures","failure_history")):
        rows={r["id"]:r for r in result.get(field,[])}
        for path in (output/folder).glob("*.json"):
            try:
                r=resume.read_json(path)
                if folder=="calls" and r["kind"] not in KINDS:raise ValueError("unknown physical call kind")
                rows[r["id"]]=r
            except (OSError,ValueError,TypeError,KeyError) as exc:
                result.setdefault("recovery_events",[]).append(dict(path=str(path),reason=repr(exc)))
        result[field]=sorted(rows.values(),key=lambda r:r["id"])
    refresh_counts(result);return result

def refresh_counts(result):
    calls=result["call_history"];own=[r for r in calls if r["attempt_id"]==result.get("attempt_id")]
    statuses={"completed":"RETURNED","failed":"FAILED","interrupted_completion_unknown":"INTERRUPTED_COMPLETION_UNKNOWN","running":"RUNNING"}
    phases=("attempted",)+tuple(statuses)
    for field,rows in (("counts",calls),("current_attempt_counts",own)):
        result[field]={kind+"_"+phase:sum(r["kind"]==kind and (phase=="attempted" or r["status"]==statuses[phase]) for r in rows)
            for kind in KINDS for phase in phases}
        for phase in phases:
            result[field]["dit_"+phase]=sum(result[field][k+"_"+phase] for k in KINDS if k.startswith("transformer_"))
        result[field].update(encode=0,codec=0,gradient_dit=0,dit_vjp=0,dit_recompute=0)
    for field,rows in (("physical_replay_chunks",calls),("current_attempt_replay_chunks",own)):
        result[field]={phase:{outcome:sum(r.get("replay",{}).get("counts",{}).get("vae_chunk",{}).get(phase,{}).get(outcome,0) for r in rows)
            for outcome in ("attempted","completed")} for phase in ("forward","recompute")}
    for field,rows in (("ordinary_chunks",calls),("current_attempt_ordinary_chunks",own)):
        result[field]={phase:sum(r.get("ordinary_chunks",{}).get(phase,0) for r in rows) for phase in ("attempted","completed")}
    for row in calls:result["model_calls"][row["unit"]]={k:v for k,v in row.items() if k!="id"}
    result["observation_accounting"]={"endpoints":{field:sum(r.get(field,0) for r in result["views"].values())
        for field in ("expected_windows","expected_chips","expected_metrics","observed_windows","observed_chips","observed_metrics")},
        "gradient_forward":"same physical ZERO RGB, not an additional independent endpoint"}

def committed(output,name):
    return (Path(output)/"commits"/(name+".json")).exists()

def model_work_remaining(output):
    output=Path(output);result=read_state(output)
    if result is None:
        if output.exists() and any(output.iterdir()):raise OSError("nonempty output without recoverable task state")
        return True
    for name,path in (("baseline",output/"ZERO/terminal_latent.pt"),("gradient",output/"units/gradient.pt")):
        if not path.exists():return not committed(output,name)
    if tensor(output/"units/gradient.pt")["status"] in UNDEFINED:return False
    for name,path in (("TERMINAL_rgb",output/"TERMINAL/float_rgb.pt"),
                      ("tail",output/"NATIVE/terminal_latent.pt"),("NATIVE_rgb",output/"NATIVE/float_rgb.pt")):
        if not path.exists():return not committed(output,name)
    return False

def finalize_interrupted(output,reason):
    result=read_state(output)
    if result is None or result["status"]!="RUNNING":return result
    result["status"]="INTERRUPTED";previous.record_failure(output,result,"external_exit",reason)
    for r in result["call_history"]:
        if r["status"]=="RUNNING":r.update(status="INTERRUPTED_COMPLETION_UNKNOWN",reason=reason)
    refresh_counts(result)
    journal=resume.Journal(output,views=method.VIEWS,steps=("baseline","tail"))
    journal.save(result);journal.checkpoint("report-only child cleanup");return result

def refresh_report(output,result):
    values={n:result["views"][n].get("values") if result["views"][n]["status"]=="SCORED" else None for n in method.VIEWS}
    report=method.comparison(values)
    for n,row in report["views"].items():
        row.update(status=result["views"][n]["status"],reason=result["views"][n].get("reason"))
        if row["status"]=="FAILED":
            for item in row["payload_rows"]+row["state_rows"]:item["status"]="FAILED"
    report.update(transport=result["transport"],quality=result["quality"],selection_status=result["selection_status"])
    refresh_counts(result);report["observation_accounting"]=result["observation_accounting"]
    write_json(Path(output)/"comparison.json",report);return report

def validate_config(config):
    c,g=config["carrier"],config["generation"]
    if (config["schema"],c["key"],c["message_hex"],c["rho"],g["steps"],g["guidance_scale"],
        config["objective"],config["amplitude"],config["control_index"],config["views"]) != (
        "local-joint-paired-transfer-v1",KEY,"8001a55a",.5,50,5.,method.OBJECTIVE,.1,48,list(method.VIEWS)):
        raise ValueError("fixed paired-transfer objective/coordinate/control/key/reader semantics required")
    if g["seed"]!=2026100701 or (g["height"],g["width"],g["frames"])!=(320,512,181):
        raise ValueError("single original yellow-sailboat seed and geometry required")
    if (g["prompt"],g["negative_prompt"],g["max_sequence_length"],config["model"]["id"]) != (
        "locked camera, a small yellow ceramic sailboat gently drifting across a shallow clear glass tank on a stone tabletop, steady soft daylight, no people, no cuts",
        "text, watermark, logo, camera motion, cuts, multiple objects, flicker",512,
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"):
        raise ValueError("original source prompt/negative prompt/token limit and Wan model required")

def run(config,output,*,preparer=None,loader=None,protocol=carrier.PUBLIC,latent_shape=LATENT_SHAPE,
        support=PUBLIC_LATENT_SUPPORT,source_identity=None):
    import torch
    output=Path(output);result=read_state(output)
    if result is None and output.exists() and any(output.iterdir()):raise OSError("nonempty output without recoverable task state")
    requested_config=copy.deepcopy(config)
    # A durable logical task uses its saved effective parameters. New requested
    # provenance/environment values never relabel a saved numerical context.
    config=copy.deepcopy(result["config"]) if result is not None else config
    validate_config(config)
    output.mkdir(parents=True,exist_ok=True);result=result or initial_result(config)
    attempt=str(time.time_ns())
    for r in result["call_history"]:
        if r["status"]=="RUNNING":r.update(status="INTERRUPTED_COMPLETION_UNKNOWN",reason="prior process ended before durable return")
    result.update(status="RUNNING",attempt_id=attempt,failures=[],source=source_identity or {},
                  requested_config=requested_config,effective_config=config,started_at_unix=time.time())
    journal=resume.Journal(output,views=method.VIEWS,steps=("baseline","tail"))
    device=config.get("device","cuda");pipe=None;vae=None;context=None
    stage="setup";active={};load_failed=set()
    def flush():
        refresh_counts(result);journal.save(result)
    def failure(where,exc):
        previous.record_failure(output,result,where,repr(exc));flush()
    def commit(name):
        write_json(output/"commits"/(name+".json"),dict(attempt_id=attempt,time=time.time()));flush()
    def begin(name,kind):
        row=dict(id=attempt+"_"+str(len(result["call_history"])).zfill(5),attempt_id=attempt,
                 unit=name,kind=kind,status="RUNNING",started_at_unix=time.time())
        result["call_history"].append(row);flush();return row
    def call(name,kind,fn):
        row=begin(name,kind)
        try:value=fn()
        except BaseException as exc:
            row.update(status="FAILED" if isinstance(exc,Exception) else "INTERRUPTED_COMPLETION_UNKNOWN",reason=repr(exc));flush();raise
        row.update(status="RETURNED",finished_at_unix=time.time());flush();return value
    def counter(prefix):
        def count(kind,returned):
            key=prefix+"_"+kind
            if not returned:active[key]=begin(key,kind)
            else:active.pop(key).update(status="RETURNED",finished_at_unix=time.time());flush()
        return count
    def clear():
        gc.collect()
        if str(device).startswith("cuda") and torch.cuda.is_initialized():torch.cuda.empty_cache()
    def release_vae():
        nonlocal vae
        if vae is not None:adapter._clear_cache(vae)
        vae=None;clear()
    def require_pipe():
        nonlocal pipe,context
        release_vae()
        if pipe is None:
            if "generation" in load_failed:raise RuntimeError("generation load already failed; no automatic retry")
            # A missing committed context must fail before any replacement load.
            if committed(output,"context") and not (output/"native/context.pt").exists():raise OSError("committed native context missing")
            try:
                pipe,z,p,n,dtype=call("prepare","generation_prepare",lambda:(preparer or generation.prepare_generation)(
                    config,load_vae=False,device=device,model_dtype=getattr(torch,config.get("transformer_dtype","bfloat16"))))
                path=output/"native/context.pt"
                if path.exists():context=tensor(path)
                else:
                    context=dict(initial=z.cpu(),prompt=p.cpu(),negative=n.cpu(),dtype=str(dtype).split(".")[-1],scheduler=native.snapshot(pipe.scheduler))
                    resume.atomic_tensor(path,context);commit("context")
                result["actual_model_execution"]="REAL" if preparer is None and loader is None else "INJECTED_TEST_DOUBLE"
            except BaseException:load_failed.add("generation");raise
        pipe.transformer.to(device);return pipe
    def require_vae():
        nonlocal vae
        if vae is None:
            if pipe is not None:
                pipe.transformer.to("cpu")
                pipe.scheduler=native.restore(native.snapshot(pipe.scheduler),"cpu")
            clear()
            if "vae" in load_failed:raise RuntimeError("VAE load already failed; no automatic retry")
            try:vae=call("load_"+str(len(result["call_history"])),"vae_load",
                lambda:(loader or generation.load_frozen_vae)(config,device=device))
            except BaseException:load_failed.add("vae");raise
            result["actual_model_execution"]="REAL" if preparer is None and loader is None else "INJECTED_TEST_DOUBLE"
        return vae
    def existing(path,name):
        if path.exists():return tensor(path)
        if committed(output,name):raise OSError("committed "+name+" missing; no replacement physical work")
        return None
    def save_progress(name,z,scheduler,next_index,**extra):
        path=output/"native"/(name+".pt")
        if path.exists():resume.atomic_bytes(path.with_suffix(".previous.pt"),path.read_bytes())
        resume.atomic_tensor(path,dict(z=z.detach().cpu(),scheduler=native.snapshot(scheduler),next_index=next_index,**extra))
        result["steps"][name]=dict(status="COMPLETE" if next_index==50 else "RUNNING",next_index=next_index);flush()
    def progress(name):
        for suffix in (".pt",".previous.pt"):
            path=output/"native"/(name+suffix)
            if not path.exists():continue
            try:
                value=tensor(path);tensor_contract(value["z"],latent_shape);s=native.restore(value["scheduler"],"cpu")
                if value["next_index"] not in range(1,51) or s.step_index!=value["next_index"]:raise ValueError("saved native cursor/state disagree")
                return value
            except (OSError,ValueError,RuntimeError,KeyError) as exc:
                result.setdefault("recovery_events",[]).append(dict(path=str(path),reason=repr(exc)))
        if result["steps"][name].get("next_index",0)>0:raise OSError("saved native progress unreadable; no restart from noise")
        return None
    def baseline():
        path=output/"ZERO/terminal_latent.pt";z=existing(path,"baseline")
        if z is not None:tensor_contract(z,latent_shape);commit("baseline");return z
        old=progress("baseline");require_pipe()
        scheduler=native.restore(context["scheduler"] if old is None else old["scheduler"],device)
        z=(context["initial"] if old is None else old["z"]).to(device);start=0 if old is None else old["next_index"]
        for i in range(start,50):
            count=counter("ZERO_"+str(i))
            c,u=predict_branches(pipe,z,scheduler,context["prompt"].to(device),context["negative"].to(device),getattr(torch,context["dtype"]),i,count)
            if i==48:
                resume.atomic_tensor(output/"native/fork48.pt",dict(z=z.cpu(),conditional=c.cpu(),unconditional=u.cpu(),scheduler=native.snapshot(scheduler)));commit("fork48")
            z=trajectory.native_step(scheduler,z,u+5.*(c-u),i,count,kind="native_step")
            if i==48:resume.atomic_tensor(output/"native/zero_after48.pt",z.cpu());commit("zero_after48")
            save_progress("baseline",z,scheduler,i+1)
        z=z.cpu();tensor_contract(z,latent_shape);resume.atomic_tensor(path,z);commit("baseline");return z
    def gradient(zero):
        model=require_vae()
        def attach(summary):
            result["replay"]["ZERO"]=summary
            for r in reversed(result["call_history"]):
                if r["unit"]=="ZERO_gradient_decode" and r["attempt_id"]==attempt:r["replay"]=copy.deepcopy(summary);break
        ledger=ReplayLedger(callback=lambda s:(attach(s),flush()))
        z=zero.to(device).clone().requires_grad_(True);original=model.decode;unit=None
        try:
            adapter._clear_cache(model);model.decode=lambda *a,**k:call("ZERO_gradient_decode","decode",lambda:original(*a,**k))
            mean,std=adapter._scale_tensors(model,z);decoded=checkpoint_decode(model,z*std+mean,ledger)
            rgb=(decoded[0].permute(1,2,3,0).float()/2+.5).clamp(0,1);tensor_contract(rgb,protocol.video_shape,bounded_rgb=True)
            read=method.tensor_readout(rgb,key=KEY,message=bytes.fromhex("8001a55a"),protocol=protocol)
            unit=dict(status="COMPLETE",input_terminal=zero,rgb=rgb.detach().cpu(),q=read["q"].detach().cpu(),
                      readout=previous.method.read_report("NEXT_ALLBIT",read))
            try:objective=method.scalar(read)
            except method.UndefinedActiveTie as exc:unit.update(status="UNDEFINED_ACTIVE_TIE",reason=str(exc));return unit
            g=call("ZERO_vjp","decoder_vjp",lambda:torch.autograd.grad(objective,z)[0]).detach().cpu()
            tensor_contract(g,latent_shape);d,receipt=method.direction(g,support)
            unit.update(raw_gradient=g,direction=d,direction_receipt=receipt)
            if d is None:unit["status"]="ZERO_GRADIENT"
            return unit
        except BaseException as exc:
            # A failed VJP must not erase its completed forward observation.
            if unit is not None:
                unit.update(status="FORWARD_ONLY_VJP_FAILED" if isinstance(exc,Exception) else "FORWARD_ONLY_VJP_INTERRUPTED",gradient_complete=False,
                            reason="forward completed; VJP/gradient unit did not complete",failure_reason=repr(exc))
                resume.atomic_tensor(output/"failures"/(attempt+"_ZERO_forward.pt"),unit)
                result["units"]["gradient"].update(status="FAILED",reason="VJP/gradient failed after saved forward",
                    observed_windows=88,observed_chips=1408,observed_metrics=55)
            raise
        finally:
            model.decode=original
            try:adapter._clear_cache(model)
            finally:ledger.callback=None;ledger.release_boundary_storage();attach(ledger.summary());flush()
    def ordinary(name,z):
        model=require_vae();original=model.decoder.forward;decode=model.decode
        def counted(*a,**k):
            row=next(r for r in reversed(result["call_history"]) if r["unit"]==name+"_ordinary_decode" and r["attempt_id"]==attempt)
            chunks=row.setdefault("ordinary_chunks",dict(attempted=0,completed=0));chunks["attempted"]+=1;flush()
            value=original(*a,**k);chunks["completed"]+=1;flush();return value
        model.decoder.forward=counted;model.decode=lambda *a,**k:call(name+"_ordinary_decode","decode",lambda:decode(*a,**k))
        try:return adapter.decode_normalized_latent(model,z.to(device)).cpu()
        finally:model.decoder.forward,model.decode=original,decode
    def observe(name,z,zero_rgb=None):
        directory=output/name
        if committed(output,name+"_scoring"):
            try:
                raw=resume.read_json(directory/"raw.json");previous.validate_raw(raw);q=tensor(directory/"q.pt")
                if not torch.equal(q.double(),previous.prior.raw_q(raw)):raise ValueError("q/raw numerical mismatch")
                metrics=resume.read_json(directory/"metrics.json")
                if not previous.evaluation_complete(metrics):raise ValueError("fixed receiver rows incomplete")
                values=metric_values(metrics);stored=resume.read_json(directory/"readout.json")["values"]
                if values!=stored:raise ValueError("metrics/readout disagreement")
            except (OSError,ValueError,KeyError,TypeError,RuntimeError) as exc:raise OSError("committed scoring missing/invalid: "+str(exc)) from exc
        else:
            rgb=zero_rgb
            if rgb is None:
                path=directory/"float_rgb.pt";rgb=existing(path,name+"_rgb")
                if rgb is None:rgb=ordinary(name,z);tensor_contract(rgb,protocol.video_shape,bounded_rgb=True);resume.atomic_tensor(path,rgb)
                commit(name+"_rgb")
            tensor_contract(rgb,protocol.video_shape,bounded_rgb=True)
            raw=observer.read_rgb(rgb,key=KEY,rho=.5,protocol=protocol);write_json(directory/"raw.json",raw)
            q=previous.prior.raw_q(raw);resume.atomic_tensor(directory/"q.pt",q)
            metrics=observer.evaluate_saved(raw,key=KEY,message=bytes.fromhex("8001a55a"),rho=.5,protocol=protocol)
            write_json(directory/"metrics.json",metrics);values=metric_values(metrics)
            write_json(directory/"readout.json",dict(values=values,source="ZERO gradient forward reused" if name=="ZERO" else "saved ordinary RGB"));commit(name+"_scoring")
        result["views"][name].update(status="SCORED",reason=None,values=values,summary=method.summarize(values),
            observed_windows=88,observed_chips=1408,observed_metrics=55);flush()
    def tail(zero,unit,conditional_delta):
        path=output/"NATIVE/terminal_latent.pt";done=existing(path,"tail")
        if done is not None:
            tensor_contract(done,latent_shape);result["transport"].update(resume.read_json(output/"native/transport.json"));return done
        fork=tensor(output/"native/fork48.pt");after=tensor(output/"native/zero_after48.pt")
        for key in ("z","conditional","unconditional"):tensor_contract(fork[key],latent_shape)
        tensor_contract(after,latent_shape);old=progress("tail")
        # Release VAE before creating GPU tail history. The final step needs the transformer.
        if old is None or old["next_index"]<50:require_pipe()
        tail_device=device if old is None or old["next_index"]<50 else "cpu"
        if old is None:
            scheduler=native.restore(fork["scheduler"],device);z=fork["z"].to(device)
            velocity,receipt=native.controlled_velocity(z,fork["conditional"].to(device),fork["unconditional"].to(device),
                float(scheduler.sigmas[48]),conditional_delta.to(device))
            z=trajectory.native_step(scheduler,z,velocity,48,counter("NATIVE_48"),kind="native_step")
            receipt["paired_step48_effect"]=method.displacement(z.cpu().double()-after.double(),unit["direction"],support)
            save_progress("tail",z,scheduler,49,transport=receipt)
        else:scheduler=native.restore(old["scheduler"],tail_device);z=old["z"].to(tail_device);receipt=old["transport"]
        if scheduler.step_index==49:
            count=counter("NATIVE_49")
            c,u=predict_branches(pipe,z,scheduler,context["prompt"].to(device),context["negative"].to(device),getattr(torch,context["dtype"]),49,count)
            z=trajectory.native_step(scheduler,z,u+5.*(c-u),49,count,kind="native_step");save_progress("tail",z,scheduler,50,transport=receipt)
        receipt["final_native_effect"]=method.displacement(z.cpu().double()-zero.double(),unit["direction"],support)
        result["transport"].update(receipt);write_json(output/"native/transport.json",result["transport"])
        resume.atomic_tensor(path,z.cpu());commit("tail");return z.cpu()
    def quality(name,unit):
        from PIL import Image
        directory=output/"quality"/(name+"_vs_ZERO");rgb=tensor(output/name/"float_rgb.pt");reference=unit["rgb"]
        for value in (rgb,reference):tensor_contract(value,protocol.video_shape,bounded_rgb=True)
        visual=previous.visual;row=None;index={};reused=False
        was_committed=committed(output,name+"_quality") or result["quality"][name+"_vs_ZERO"]["status"]=="SCORED"
        try:
            row=resume.read_json(directory/"quality.json")
            if not previous.saved.quality_complete(row,protocol.video_shape[0]):raise ValueError("incomplete saved quality rows")
            reused=True
        except (OSError,ValueError,TypeError,KeyError) as exc:
            if was_committed:raise OSError("committed quality missing/invalid; no replacement scoring: "+str(exc)) from exc
            row=visual.quality(rgb,reference,protocol);row.update(reference="ZERO",target=name,no_quality_threshold=True)
            previous.semantics.add_rgb_semantics(row);write_json(directory/"quality.json",row)
        frames,crops=visual.selection({name:row},len(rgb),views=("ZERO",name))
        try:
            index=resume.read_json(directory/"frames.json")
            if not previous.saved.visual_index_complete(index,frames,crops):index={}
        except (OSError,ValueError,TypeError,KeyError):index={}
        saved_items={r["path"]:r for r in index.get("frames",[])+index.get("crops",[])}
        rendered=[]
        def picture(relative,i,box=None):
            if relative in saved_items:
                try:
                    with Image.open(directory/relative) as image:image.verify()
                    return saved_items[relative]
                except (OSError,ValueError):pass
            rendered.append(relative)
            return visual.render_pair(directory/relative,name+"_vs_ZERO",i,rgb[i],reference[i],box=box)
        images=[picture(f"frames/frame_{i:03d}.png",i) for i in frames]
        crop_images=[]
        for n,r in enumerate(crops):
            i,y,x=r["frame"],r["y"],r["x"];radius=visual.CROP_RADIUS
            box=(max(0,y-radius),min(rgb.shape[1],y+radius+1),max(0,x-radius),min(rgb.shape[2],x+radius+1))
            crop_images.append(picture(f"crops/crop_{n:02d}_{i:03d}.png",i,box))
        write_json(directory/"frames.json",dict(status="SAVED",frame_indices=frames,crop_requests=crops,frames=images,crops=crop_images))
        result["quality"][name+"_vs_ZERO"]=dict(status="SCORED",reused=reused,rendered_images=rendered);commit(name+"_quality")
    with journal.periodic():
        try:
            flush();stage="baseline";zero=baseline()
            stage="gradient";path=output/"units/gradient.pt";unit=existing(path,"gradient")
            if unit is None:unit=gradient(zero);resume.atomic_tensor(path,unit)
            if not torch.equal(unit["input_terminal"],zero):raise ValueError("fresh gradient input differs from this ZERO terminal")
            tensor_contract(unit["rgb"],protocol.video_shape,bounded_rgb=True)
            if unit["status"] not in ("COMPLETE",)+UNDEFINED:raise ValueError("invalid gradient unit status")
            if unit["status"]!="UNDEFINED_ACTIVE_TIE":
                d,receipt=method.direction(unit["raw_gradient"],support)
                if (d is None)!=(unit["status"]=="ZERO_GRADIENT") or (d is not None and not torch.equal(d,unit["direction"])):
                    raise ValueError("saved direction violates masked negative normalized gradient")
            result["units"]["gradient"].update(status=unit["status"],reason=unit.get("reason"),
                observed_windows=88,observed_chips=1408,observed_metrics=55)
            commit("gradient");stage="ZERO_scoring";observe("ZERO",zero,unit["rgb"])
            if unit["status"]=="COMPLETE":
                terminal,conditional=method.operands(zero,unit["direction"])
                result["transport"]["direct_terminal_effect"]=method.displacement(terminal.double()-zero.double(),unit["direction"],support)
                result["transport"]["conditional_clean_operand"]=method.measure(conditional)
                resume.atomic_tensor(output/"TERMINAL/terminal_latent.pt",terminal)
                for name in ("TERMINAL","NATIVE"):
                    stage=name
                    try:
                        z=terminal if name=="TERMINAL" else tail(zero,unit,conditional);observe(name,z)
                    except Exception as exc:result["views"][name].update(status="FAILED",reason=repr(exc));failure(stage,exc)
                for name in ("TERMINAL","NATIVE"):
                    if result["views"][name]["status"]!="SCORED":continue
                    try:quality(name,unit)
                    except Exception as exc:
                        result["quality"][name+"_vs_ZERO"]=dict(status="MISSING",reason=repr(exc),replacement_decode=False);failure("quality_"+name,exc)
            else:
                for name in method.VIEWS[1:]:result["views"][name].update(status="MISSING",reason=unit["status"])
            result["status"]="ENGINEERING_FAILURE" if result["failures"] else "COMPLETE_WITH_UNDEFINED_ARMS" if unit["status"] in UNDEFINED else "COMPLETE"
        except BaseException as exc:
            result["status"]="ENGINEERING_FAILURE" if isinstance(exc,Exception) else "INTERRUPTED";failure(stage,exc)
            forward_path=output/"failures"/(attempt+"_ZERO_forward.pt")
            if stage=="gradient" and forward_path.exists():
                # Read-only CPU scoring of already completed raw forward; no replacement model call.
                try:
                    forward=tensor(forward_path);observe("ZERO",forward["input_terminal"],forward["rgb"])
                except Exception as scoring_error:failure("failed_ZERO_forward_scoring",scoring_error)
            if not isinstance(exc,Exception):raise
        finally:
            for r in active.values():r.update(status="FAILED" if result["status"]=="ENGINEERING_FAILURE" else "INTERRUPTED_COMPLETION_UNKNOWN",reason="call did not return")
            try:release_vae()
            except Exception as exc:result["cleanup_errors"].append(repr(exc))
            pipe=None;clear();result["storage_bytes"]=sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
            result["finished_at_unix"]=time.time();refresh_report(output,result);flush()
    return result
