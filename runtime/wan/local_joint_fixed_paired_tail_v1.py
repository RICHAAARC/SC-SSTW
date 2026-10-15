"""Two independently restored saved-fork tails, fixed control and zero control, no VJP."""
from __future__ import annotations
import copy
import gc
from pathlib import Path
import time
from main.tube_state import local_joint_fixed_paired_tail_v1 as method
from main.tube_state import local_joint_state_payload_carrier_v1 as carrier
from main.tube_state import local_joint_terminal_bridge_v1 as observer
from runtime.wan import generation,trajectory,vae as adapter
from runtime.wan import local_joint_paired_transfer_native_v1 as native
from runtime.wan import local_joint_all_bit_resume_v1 as resume
from runtime.wan import local_joint_same_point_dual_v1 as previous
from runtime.wan.grow_video_reference import predict_branches
from runtime.wan.local_joint_readout_m0_v1 import metric_values
from runtime.wan.local_joint_state_payload_provider_v1 import PUBLIC_LATENT_SUPPORT
from runtime.wan.local_joint_terminal_bridge_v1 import KEY,LATENT_SHAPE,tensor_contract

EXPECTED=dict(generation_prepare=1,vae_load=1,gradient_decode=0,decoder_vjp=0,ordinary_decode=2,
    decode=2,gradient_forward_chunks=0,ordinary_chunks=92,total_forward_chunks=92,
    nominal_replay_chunks=0,baseline_dit=0,tail_dit=4,dit=4,gradient_dit=0,dit_vjp=0,
    dit_recompute=0,native=4,scheduler=4,encode=0,codec=0,independent_samples=0,
    existing_source_count=1,new_source_endpoints=2,endpoint_windows=176,endpoint_chips=2816,
    endpoint_metrics=110,gradient_windows=0,gradient_chips=0,gradient_metrics=0,
    saved_reference_windows=88,saved_reference_chips=1408,saved_reference_metrics=55,quality_pairs=2)
FINISHED=previous.FINISHED
UNDEFINED=previous.UNDEFINED
KINDS=("generation_prepare","vae_load","decode","decoder_vjp","transformer_conditional",
       "transformer_unconditional","native_step")
tensor=previous.tensor
write_json=previous.write_json

def initial_result(config):
    return dict(schema="local-joint-fixed-paired-tail-v1",config=copy.deepcopy(config),status="RUNNING",
        views={n:previous.slot() for n in method.VIEWS},units={"saved_source":dict(status="NOT_READ")},
        steps={n:dict(status="NOT_RUN") for n in method.VIEWS},
        quality={pair:dict(status="MISSING",reason="not_run") for pair in method.PAIRS},
        model_calls={},call_history=[],counts={},replay={},failures=[],failure_history=[],cleanup_errors=[],
        ordinary_chunks=dict(attempted=0,completed=0),transport={},expected_cost=EXPECTED,
        scientific_pass=False,actual_model_execution="NOT_STARTED",automatic_retry=False,
        chunk_outcome_policy="Ordinary chunks report attempted/completed only. The enclosing decode receipt classifies failure or completion-unknown interruption; an unfinished chunk is not assumed failed. No gradient/replay calls exist in this path.",
        ceiling=method.CEILING,selection_status="ADOPTED_BY_USER_20261016")

def read_state(output):
    output=Path(output);result=None;problems=[]
    for rel in ("result.json","result.json.previous","checkpoints/latest.json","checkpoints/previous.json"):
        path=output/rel
        if not path.exists():continue
        try:result=resume._valid_state(path,method.VIEWS,method.VIEWS);break
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
        "saved_reference":"existing ZERO RGB/readouts reused; no new physical measurement"}

def committed(output,name):
    return (Path(output)/"commits"/(name+".json")).exists()

def model_work_remaining(output):
    output=Path(output);result=read_state(output)
    if result is None:
        if output.exists() and any(output.iterdir()):raise OSError("nonempty output without recoverable task state")
        return True
    for name in method.VIEWS:
        for suffix,unit in (("terminal_latent.pt",name+"_tail"),("float_rgb.pt",name+"_rgb")):
            if not (output/name/suffix).exists() and not committed(output,unit):return True
    return False

def finalize_interrupted(output,reason):
    result=read_state(output)
    if result is None or result["status"]!="RUNNING":return result
    result["status"]="INTERRUPTED";previous.record_failure(output,result,"external_exit",reason)
    for r in result["call_history"]:
        if r["status"]=="RUNNING":r.update(status="INTERRUPTED_COMPLETION_UNKNOWN",reason=reason)
    refresh_counts(result)
    journal=resume.Journal(output,views=method.VIEWS,steps=method.VIEWS)
    journal.save(result);journal.checkpoint("report-only child cleanup");return result

def refresh_report(output,result):
    values={n:result["views"][n].get("values") if result["views"][n]["status"]=="SCORED" else None for n in method.VIEWS}
    report=method.comparison(values,result.get("saved_reference",{}).get("values"))
    for n,row in report["views"].items():
        row.update(status=result["views"][n]["status"],reason=result["views"][n].get("reason"))
        if row["status"]=="FAILED":
            for item in row["payload_rows"]+row["state_rows"]:item["status"]="FAILED"
    report.update(transport=result["transport"],quality=result["quality"],selection_status=result["selection_status"],pair_execution_context=result.get("pair_execution_context"))
    refresh_counts(result);report["observation_accounting"]=result["observation_accounting"]
    write_json(Path(output)/"comparison.json",report);return report

def validate_config(config):
    c,g=config["carrier"],config["generation"]
    if (config["schema"],config["views"],c["key"],c["message_hex"],c["rho"],g["steps"],
        g["guidance_scale"],config["objective"],config["amplitude"],config["control_index"]) != (
        "local-joint-fixed-paired-tail-v1",list(method.VIEWS),KEY,"8001a55a",.5,50,5.,"historical_allbit_v1",.1,48):
        raise ValueError("fixed paired-tail reader/coordinate/control semantics required")
    if (g["height"],g["width"],g["frames"])!=(320,512,181):raise ValueError("fixed reader geometry required")
    if not isinstance(config.get("source_run"),str) or not config["source_run"]:
        raise ValueError("readable saved paired-transfer run path required")
    # Unused seed/prompt/revision labels are location notes, never source admission.


def prepare_saved_transformer(config,*,device,model_dtype):
    """One preparation: only frozen DiT weights; no text encoding/noise/baseline."""
    from types import SimpleNamespace
    from diffusers import WanTransformer3DModel
    model=config["model"]
    transformer=WanTransformer3DModel.from_pretrained(model["id"],
        **generation._load_args(model,torch_dtype=model_dtype,subfolder="transformer")).eval()
    disable=getattr(transformer,"disable_gradient_checkpointing",None)
    if callable(disable):disable()
    for parameter in transformer.parameters():parameter.requires_grad_(False)
    return SimpleNamespace(transformer=transformer.to(device))


def saved_source(source,protocol,latent_shape,support):
    import torch
    source=Path(source)
    fork=tensor(source/"native/fork48.pt");context=tensor(source/"native/context.pt")
    zero=tensor(source/"ZERO/terminal_latent.pt");unit=tensor(source/"units/gradient.pt")
    tensor_contract(zero,latent_shape)
    for key in ("z","conditional","unconditional"):tensor_contract(fork[key],latent_shape)
    state=fork["scheduler"]["state"]
    if state.get("_step_index")!=48 or len(state["timesteps"])!=50:raise ValueError("saved fork must be before update48 of50 with full history")
    for field in ("sigmas","model_outputs","last_sample","this_order","lower_order_nums"):
        if field not in state:raise ValueError("saved fork missing native history field: "+field)
    for value in state["model_outputs"]+[state["last_sample"]]:tensor_contract(value,latent_shape)
    for field in ("timesteps","sigmas"):
        if not torch.is_tensor(state[field]) or not bool(state[field].isfinite().all()):raise ValueError("nonfinite saved scheduler "+field)
    if unit.get("status")!="COMPLETE":raise ValueError("saved terminal gradient unit is not COMPLETE")
    if not torch.equal(unit["input_terminal"],zero):raise ValueError("saved gradient input differs numerically from saved ZERO")
    direction=unit["direction"].detach().cpu()
    method.validate_direction(direction,latent_shape,support)
    tensor_contract(unit["rgb"],protocol.video_shape,bounded_rgb=True)
    if context["dtype"] not in ("float32","bfloat16","float16"):raise ValueError("unsupported saved transformer arithmetic dtype")
    for key in ("prompt","negative"):
        if not torch.is_tensor(context[key]) or not context[key].numel() or not bool(context[key].isfinite().all()):raise ValueError("invalid saved "+key+" embedding")
    raw=resume.read_json(source/"ZERO/raw.json");previous.validate_raw(raw)
    metrics=resume.read_json(source/"ZERO/metrics.json")
    if not previous.evaluation_complete(metrics):raise ValueError("saved ZERO receiver rows incomplete")
    values=metric_values(metrics)
    if values!=resume.read_json(source/"ZERO/readout.json")["values"]:raise ValueError("saved ZERO metrics/readout numerical disagreement")
    return fork,context,zero,direction,unit["rgb"],values

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
    source=Path(config["source_run"])
    if source.resolve()==output.resolve() or source.resolve() in output.resolve().parents or output.resolve() in source.resolve().parents:
        raise ValueError("new output task must be separate from saved source run")
    output.mkdir(parents=True,exist_ok=True);result=result or initial_result(config)
    attempt=str(time.time_ns())
    for r in result["call_history"]:
        if r["status"]=="RUNNING":r.update(status="INTERRUPTED_COMPLETION_UNKNOWN",reason="prior process ended before durable return")
    result.update(status="RUNNING",attempt_id=attempt,failures=[],source=source_identity or {},
                  requested_config=requested_config,effective_config=config,started_at_unix=time.time())
    journal=resume.Journal(output,views=method.VIEWS,steps=method.VIEWS)
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
        nonlocal pipe
        release_vae()
        if pipe is None:
            if "generation" in load_failed:raise RuntimeError("transformer preparation already failed; no automatic retry")
            try:
                pipe=call("prepare","generation_prepare",lambda:(preparer or prepare_saved_transformer)(
                    config,device=device,model_dtype=getattr(torch,context["dtype"])))
                result["actual_model_execution"]="REAL" if preparer is None and loader is None else "INJECTED_TEST_DOUBLE"
            except BaseException:load_failed.add("generation");raise
        # Identical device round trip for each arm, before its independent history restore.
        pipe.transformer.to("cpu");clear();pipe.transformer.to(device)
        return pipe
    def require_vae():
        nonlocal vae
        if vae is None:
            if pipe is not None:
                pipe.transformer.to("cpu")
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
    def execution_context():
        return dict(attempt_id=attempt,model_preparation_call_id=next((r["id"] for r in reversed(result["call_history"])
            if r["kind"]=="generation_prepare" and r["attempt_id"]==attempt and r["status"]=="RETURNED"),None),
            device=str(device),input_dtype=context["dtype"])
    def saved_predictions(name,*,required=False,expected_input=None):
        saved=existing(output/"native"/(name+"_step49_predictions.pt"),name+"_step49_predictions")
        if saved is None:
            if required:raise OSError("completed "+name+" missing step49 predictions; no replacement physical work")
            return None
        for field in ("input","conditional","unconditional"):tensor_contract(saved[field],latent_shape)
        if expected_input is not None and not torch.equal(saved["input"],expected_input.cpu()):
            raise ValueError("saved final predictions have a different numerical input")
        return saved
    def tail(name,fork,direction):
        path=output/name/"terminal_latent.pt";done=existing(path,name+"_tail")
        if done is not None:
            tensor_contract(done,latent_shape)
            saved_predictions(name,required=True)
            result["transport"][name]=resume.read_json(output/"native"/(name+"_transport.json"));return done
        old=progress(name)
        saved=saved_predictions(name,required=old is not None and old["next_index"]==50,
            expected_input=old["z"] if old is not None and old["next_index"]==49 else None)
        if old is None or old["next_index"]<50:require_pipe()
        receipt=dict(arm=name,execution_context={},device_roundtrip=["cpu",str(device)],input_dtype=context["dtype"],
            branch_dtype="float32",cfg_dtype="float32",source="saved fork48 and exact saved terminal direction")
        if old is None:
            scheduler=native.restore(fork["scheduler"],device);z=fork["z"].to(device)
            conditional=(.02*direction).float() if name=="CONTROL_TAIL" else torch.zeros_like(direction,dtype=torch.float32)
            velocity,control=native.controlled_velocity(z,fork["conditional"].to(device),fork["unconditional"].to(device),
                float(scheduler.sigmas[48]),conditional.to(device))
            control["requested_budget"]={"nominal_CFG_clean":.1 if name=="CONTROL_TAIL" else 0.,"conditional_clean":.02 if name=="CONTROL_TAIL" else 0.}
            if name=="ZERO_TAIL":
                control.update(provider_composition="zero_conditional_clean",provider_composition_unverified=False,
                    coordinate_relation="delta_c=0; delta_v_CFG=0; native update uses unchanged saved c/u")
            receipt.update(control=control,conditional_clean_operand=method.measure(conditional))
            z=trajectory.native_step(scheduler,z,velocity,48,counter(name+"_48"),kind="native_step")
            receipt["execution_context"]["update48"]=execution_context()
            save_progress(name,z,scheduler,49,transport=receipt)
        else:
            scheduler=native.restore(old["scheduler"],device if old["next_index"]<50 else "cpu")
            z=old["z"].to(device if old["next_index"]<50 else "cpu");receipt=old["transport"]
        if scheduler.step_index==49:
            after=output/"native"/(name+"_after48.pt")
            if not after.exists():resume.atomic_tensor(after,z.cpu())
            predictions=output/"native"/(name+"_step49_predictions.pt")
            if saved is not None:
                if not torch.equal(saved["input"],z.cpu()):raise ValueError("saved final predictions have a different numerical input")
                c,u=saved["conditional"].to(device),saved["unconditional"].to(device)
            else:
                c,u=predict_branches(pipe,z,scheduler,context["prompt"].to(device),context["negative"].to(device),getattr(torch,context["dtype"]),49,counter(name+"_49"))
                saved=dict(input=z.cpu(),conditional=c.cpu(),unconditional=u.cpu(),input_dtype=context["dtype"],execution_context=execution_context())
                resume.atomic_tensor(predictions,saved)
            commit(name+"_step49_predictions")
            contexts=receipt.setdefault("execution_context",{})
            contexts["prediction49"]=saved.get("execution_context",{})
            receipt["step49_predictions_reused"]=contexts["prediction49"].get("attempt_id")!=attempt
            z=trajectory.native_step(scheduler,z,u+5.*(c-u),49,counter(name+"_49"),kind="native_step")
            contexts["update49"]=execution_context()
            save_progress(name,z,scheduler,50,transport=receipt)
        result["transport"][name]=receipt
        write_json(output/"native"/(name+"_transport.json"),receipt)
        resume.atomic_tensor(path,z.cpu());commit(name+"_tail");return z.cpu()
    def quality(pair,reference_name,name,reference):
        from PIL import Image
        directory=output/"quality"/pair;rgb=tensor(output/name/"float_rgb.pt")
        for value in (rgb,reference):tensor_contract(value,protocol.video_shape,bounded_rgb=True)
        visual=previous.visual;row=None;index={};reused=False
        was_committed=committed(output,pair+"_quality") or result["quality"][pair]["status"]=="SCORED"
        try:
            row=resume.read_json(directory/"quality.json")
            if not previous.saved.quality_complete(row,protocol.video_shape[0]):raise ValueError("incomplete saved quality rows")
            reused=True
        except (OSError,ValueError,TypeError,KeyError) as exc:
            if was_committed:raise OSError("committed quality missing/invalid; no replacement scoring: "+str(exc)) from exc
            row=visual.quality(rgb,reference,protocol);row.update(reference=reference_name,target=name,no_quality_threshold=True)
            previous.semantics.add_rgb_semantics(row);write_json(directory/"quality.json",row)
        frames,crops=visual.selection({name:row},len(rgb),views=(reference_name,name))
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
            return visual.render_pair(directory/relative,pair,i,rgb[i],reference[i],box=box)
        images=[picture(f"frames/frame_{i:03d}.png",i) for i in frames]
        crop_images=[]
        for n,r in enumerate(crops):
            i,y,x=r["frame"],r["y"],r["x"];radius=visual.CROP_RADIUS
            box=(max(0,y-radius),min(rgb.shape[1],y+radius+1),max(0,x-radius),min(rgb.shape[2],x+radius+1))
            crop_images.append(picture(f"crops/crop_{n:02d}_{i:03d}.png",i,box))
        write_json(directory/"frames.json",dict(status="SAVED",frame_indices=frames,crop_requests=crops,frames=images,crops=crop_images))
        result["quality"][pair]=dict(status="SCORED",reused=reused,rendered_images=rendered);commit(pair+"_quality")
    with journal.periodic():
        try:
            flush();stage="saved_source"
            fork,context,zero,direction,zero_rgb,values=saved_source(source,protocol,latent_shape,support)
            result["saved_reference"]=dict(status="SCORED",values=values,source=str(source),reused=True)
            result["units"]["saved_source"]=dict(status="READ",exact_saved_direction=True,gradient_recomputed=False)
            endpoints={}
            # Finish both DiT tails before loading the one shared VAE.
            for name in method.VIEWS:
                stage=name
                try:endpoints[name]=tail(name,fork,direction)
                except Exception as exc:
                    result["views"][name].update(status="FAILED",reason=repr(exc));failure(name+"_tail",exc)
                    for row in active.values():row.update(status="FAILED",reason=repr(exc))
                    active.clear();flush()
            for name,z in endpoints.items():
                try:observe(name,z)
                except Exception as exc:
                    result["views"][name].update(status="FAILED",reason=repr(exc));failure(name+"_readout",exc)
            for pair,ref,name in method.PAIR_SPECS:
                if result["views"][name]["status"]!="SCORED" or (ref!="SAVED_ZERO" and result["views"][ref]["status"]!="SCORED"):continue
                try:quality(pair,ref,name,zero_rgb if ref=="SAVED_ZERO" else tensor(output/ref/"float_rgb.pt"))
                except Exception as exc:
                    result["quality"][pair]=dict(status="MISSING",reason=repr(exc),replacement_decode=False);failure("quality_"+pair,exc)
            for pair,ref,name in method.PAIR_SPECS:
                a=zero if ref=="SAVED_ZERO" else endpoints.get(ref);b=endpoints.get(name)
                if a is not None and b is not None:result["transport"][pair]=method.displacement(b.double()-a.double(),direction,support)
            if all((output/"native"/(n+"_after48.pt")).exists() for n in method.VIEWS):
                a,b=[tensor(output/"native"/(n+"_after48.pt")) for n in method.VIEWS]
                result["transport"]["paired_step48_effect"]=method.displacement(b.double()-a.double(),direction,support)
                result["transport"]["step49_input_bf16_difference"]=method.measure(b.bfloat16().double()-a.bfloat16().double()) if context["dtype"]=="bfloat16" else None
            arm_contexts={n:result["transport"].get(n,{}).get("execution_context",{}) for n in method.VIEWS}
            segments=[arm_contexts[n].get(phase,{}) for n in method.VIEWS for phase in ("update48","prediction49","update49")]
            attempts=sorted({x["attempt_id"] for x in segments if x.get("attempt_id")})
            preparations=sorted({x["model_preparation_call_id"] for x in segments if x.get("model_preparation_call_id")})
            known=all(x.get("attempt_id") and x.get("model_preparation_call_id") for x in segments)
            result["pair_execution_context"]=dict(arm_contexts=arm_contexts,attempt_ids=attempts,model_preparation_call_ids=preparations,
                complete_execution_contexts=known,spans_attempts=len(attempts)>1,
                same_preparation=known and len(preparations)==1 and len(attempts)==1,
                note="Resume retains completed arms; cross-attempt pairs remain visible, without automatic rerun or equivalence claim.")
            result["status"]="ENGINEERING_FAILURE" if result["failures"] else "COMPLETE"
        except BaseException as exc:
            result["status"]="ENGINEERING_FAILURE" if isinstance(exc,Exception) else "INTERRUPTED";failure(stage,exc)
            if stage=="saved_source":
                result["units"]["saved_source"]=dict(status="FAILED",reason=repr(exc))
                for row in result["views"].values():row.update(status="MISSING",reason="saved source unavailable: "+repr(exc))
            if not isinstance(exc,Exception):raise
        finally:
            for row in active.values():row.update(status="FAILED" if result["status"]=="ENGINEERING_FAILURE" else "INTERRUPTED_COMPLETION_UNKNOWN",reason="call did not return")
            try:release_vae()
            except Exception as exc:result["cleanup_errors"].append(repr(exc))
            pipe=None;clear();result["storage_bytes"]=sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
            result["finished_at_unix"]=time.time();refresh_report(output,result);flush()
    return result
