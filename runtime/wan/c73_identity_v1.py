"""Card74 fixed C73 L1/L2 user run: fresh D3/local reset E18/FFT0."""
from __future__ import annotations
import errno
import json
import os
import resource
import time
from pathlib import Path
import torch
from main.tube_state import c73_identity_v1 as method
from main.tube_state import local_segmented_terminal_sync_v1 as legacy
from runtime.wan import local_joint_positive_projection_margin_v1 as shared
from runtime.wan.c73_identity_quality_v1 import distortion, preclip_record
from runtime.wan.c73_identity_evaluation_v1 import evaluate

DEFAULT_MODEL_ID = shared.DEFAULT_MODEL_ID
KEY_IDS = ("key00", "key01")
# Public preparation only. These roles/maps are not passed to observations.
PROBES = (("clean", 61), ("clean", 60), ("clean", 62),
          ("clean", 109), ("delete", 61), ("delete", 64))
PAIRS = (("P0", "S_base"), ("P0", "S_ID"), ("S_base", "S_ID"))


SYNC_UNAVAILABLE_ERRNOS = set()


def flush_file(stream):
    stream.flush()
    try:
        os.fsync(stream.fileno())
    except OSError as exc:
        unsupported = {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}
        if exc.errno not in unsupported:
            raise
        SYNC_UNAVAILABLE_ERRNOS.add(exc.errno)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        flush_file(stream)
    temporary.replace(path)


def execution_artifacts(output):
    output = Path(output)
    records = ("attempt.json", "result.json", "checkpoint.json", "ledger.json", "protocol.json",
               "construction.json", "numeric_completion.json", "evaluation.json", "native_audit.json")
    found = [name+suffix for name in records for suffix in ("", ".tmp")
             if (output/(name+suffix)).exists() or (output/(name+suffix)).is_symlink()]
    for folder in ("blind", "features", "media", "quality", "books"):
        root = output/folder
        if root.is_dir():
            found.extend(str(p.relative_to(output)) for p in root.rglob("*")
                         if p.is_file() or p.is_symlink())
    return sorted(found)


def fixed_denominator():
    return dict(hosts=1, arms=3, keys=2, identity_rows=46, dimensions=64,
                native_identity_vectors=276, native_pooled_coordinates=17664,
                native_roi_dots=70656, native_identity_checks=12696, legacy22_native_roi_dots=11616,
                legacy22_native_pooled_coordinates=2904, media_roi_dots=9216, media_pooled_coordinates=2304,
                combined_identity_roi_dots=79872, combined_physical_identity_checks=14352,
                physical_local_E=18, physical_local_vectors=36, logical_local_vectors=42,
                physical_identity_slots=1656, geometric_positive_slots=3, geometric_negative_slots=1653,
                logical_identity_slots=1932, reused_identity_slots=276, negative_control_vectors=30,
                complete_target_vectors=18, noncomplete_target_vectors=18,
                continuous_context_vectors=30, discontinuous_context_vectors=6,
                D=3, quantizations=3, mp4_pairs=3, probes=3, model_loads=1,
                FFT=0, fullclip_E=0, DiT=0, VJP=0, quality_tables=6,
                quality_frame_rows=1086, quality_adjacent_rows=1080, preclip_records=3)


def protocol():
    return dict(method="C73 additional orthogonal cell identity on saved S_base", adopted_card=74,
                A_ID=method.A_ID, rho_threshold=method.TAU, comparison="strict <; equality unresolved",
                net_user_bits=32, default_receiver_changed=False, source_generation=0,
                identity_basis_positions=[22,85], guards=[0,45], mask_domain=method.MASK_DOMAIN,
                local_operator="nine frames: five context plus four real target; cache reset each E; last latent",
                no_phase_certificate=True, no_path_search=True, no_FPR_guarantee=True,
                payload_effect="UNMEASURED; FFT0", nominal_encoder_chunks=54,
                actual_internal_calls="recorded by hooks, not assumed from nominal chunks",
                fixed_denominator=fixed_denominator(), resource_hard_caps=None, scientific_pass=None)


def units():
    result = {"source/P0":"terminal_load", "source/S_base":"terminal_load", "construct":"identity_construct",
              "native_audit":"native_audit", "model":"vae_load", "evaluation":"post_numeric_evaluation"}
    for key in KEY_IDS:
        result[f"book/{key}"] = "identity_book"
        result[f"legacy_book/{key}"] = "legacy22_book"
    for arm in method.ARMS:
        for key in KEY_IDS:
            result[f"native/{arm}/{key}"] = "native_identity_observation"
            result[f"legacy/{arm}/{key}"] = "legacy22_native_observation"
        for name, kind in (("decode","ordinary_decode"), ("preclip","preclip_record"),
                           ("quantize","rgb8_quantize"), ("save_mp4","mp4_save"),
                           ("probe","ffprobe"), ("read_mp4","mp4_read")):
            result[f"{arm}/{name}"] = kind
    for n in range(18):
        result[f"prepare/{n:02d}"] = "nine_frame_preparation"
        result[f"encode/{n:02d}"] = "posterior_encode"
        for key in KEY_IDS:
            result[f"local/{n:02d}/{key}"] = "local_identity_observation"
    for left, right in PAIRS:
        for layer in ("rgb8", "mp4"):
            result[f"quality/{left}_to_{right}/{layer}"] = "paired_quality"
    return result


class Journal(shared.Journal):
    def __init__(self, output, backend):
        self.output, self.backend = output, backend
        self.data = {name:dict(kind=kind, status="NOT_ENTERED") for name,kind in units().items()}
        self.save()

    def save(self):
        dump(self.output/"ledger.json", self.data)


class WanBackend(shared.WanBackend):
    def load_terminal(self, path):
        value = torch.load(path, map_location="cpu", weights_only=True)
        legacy.validate_latent(value)
        if value.dtype != torch.float32:
            raise ValueError("saved P0/S_base terminal must be FP32")
        return value.detach().cpu()

    def encode_local(self, rgb):
        from runtime.wan.vae import reencode_rgb24_readback
        if tuple(rgb.shape) != (9,320,512,3) or rgb.dtype != torch.uint8:
            raise ValueError("fixed local E requires nine actual RGB24 frames")
        z = reencode_rgb24_readback(self.vae, rgb.float()/255).detach().cpu().float()
        if tuple(z.shape) != (1,16,3,40,64) or not bool(torch.isfinite(z).all()):
            raise ValueError("local posterior must be finite normalized [1,16,3,40,64]")
        return z


def existing_attempt(output, artifacts):
    result = dict(status="EXISTING_ATTEMPT_NOT_REEXECUTED", fixed_denominator=fixed_denominator(),
                  artifacts=artifacts, saved={}, read_errors={}, physical_reentry=False)
    for name in ("result.json", "checkpoint.json", "ledger.json", "numeric_completion.json", "evaluation.json"):
        path = output/name
        if path.is_file():
            try:
                result["saved"][name] = json.loads(path.read_text())
            except (OSError,ValueError) as exc:
                result["read_errors"][name] = f"{type(exc).__name__}: {exc}"
    return result


def run(source_p, source_s, output, model_id=DEFAULT_MODEL_ID, *, backend=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = execution_artifacts(output)
    if artifacts:
        return existing_attempt(output, artifacts)
    # Atomic exclusive attempt before any codebook, scientific load or call.
    try:
        with (output/"attempt.json").open("x") as stream:
            json.dump(dict(started_unix=time.time(), source_p=str(source_p), source_s=str(source_s),
                           scope="fixed Card74 L1/L2; failure/interruption spends attempt"), stream)
            flush_file(stream)
    except FileExistsError:
        return existing_attempt(output, ["attempt.json"])
    backend = backend or WanBackend(model_id)
    ledger = Journal(output, backend)
    result = dict(status="RUNNING", fixed_denominator=fixed_denominator(), protocol=protocol(),
                  scientific_pass=None, construction={"status":"NOT_ENTERED"}, native_audit={"status":"NOT_ENTERED"})
    native = {f"native/{arm}/{key}":method.unavailable("NOT_ENTERED") for arm in method.ARMS for key in KEY_IDS}
    old = {f"legacy/{arm}/{key}":legacy.unavailable_state("NOT_ENTERED") for arm in method.ARMS for key in KEY_IDS}
    local = {f"local/{n:02d}/{key}":method.unavailable("NOT_ENTERED", local=True) for n in range(18) for key in KEY_IDS}
    quality = {f"{a}_to_{b}/{layer}":shared.unavailable_quality("NOT_ENTERED") for a,b in PAIRS for layer in ("rgb8","mp4")}
    preclip = {arm:dict(status="NOT_ENTERED", frame_denominator=181, rows=[dict(frame=t,status="NOT_ENTERED") for t in range(181)]) for arm in method.ARMS}
    books, old_books, features, media = {}, {}, {}, {}
    started = time.perf_counter()

    def checkpoint():
        result.update(counts=ledger.counts())
        dump(output/"checkpoint.json", result)

    def save_tensor(path, value):
        path = output/path
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(value, path)
        return value

    def save_record(slot, operation):
        def call():
            value = operation()
            dump(output/"blind"/(slot+".json"), value)
            return value
        return ledger.call(slot, call)

    def native_reads(arm):
        for key in KEY_IDS:
            slot = f"native/{arm}/{key}"
            if key in books:
                value = save_record(slot, lambda key=key:method.observe(features[arm], books[key]))
                if value is not None:
                    native[slot] = value
            slot = f"legacy/{arm}/{key}"
            if key in old_books:
                value = save_record(slot, lambda key=key:legacy.observe_state(features[arm], method.KEYS[KEY_IDS.index(key)], old_books[key]))
                if value is not None:
                    old[slot] = value
        checkpoint()

    dump(output/"protocol.json", protocol())
    for slot, value in {**native, **old, **local}.items():
        dump(output/"blind"/(slot+".json"), value)
    checkpoint()
    try:
        for key_id, key in zip(KEY_IDS, method.KEYS):
            made = ledger.call(f"book/{key_id}", lambda key=key:method.codebook(key))
            if made is not None:
                books[key_id] = made
                save_tensor(Path("books")/(key_id+".pt"), made)
            made = ledger.call(f"legacy_book/{key_id}", lambda key=key:legacy.public_codebook(key))
            if made is not None:
                old_books[key_id] = made
        for arm, source in (("P0",source_p), ("S_base",source_s)):
            value = ledger.call(f"source/{arm}", lambda source=source,arm=arm:save_tensor(Path("features")/arm/"terminal.pt", backend.load_terminal(source)))
            if value is not None:
                features[arm] = value
                native_reads(arm)
        if "S_base" in features and "key00" in books and native["native/S_base/key00"]["status"] == "OBSERVED":
            def construct():
                value, receipt = method.construct(features["S_base"], native["native/S_base/key00"], books["key00"])
                save_tensor(Path("features/S_ID/terminal.pt"), value)
                dump(output/"construction.json", receipt)
                return value,receipt
            made = ledger.call("construct", construct)
            if made is not None:
                features["S_ID"], result["construction"] = made
                native_reads("S_ID")
        if (native["native/S_ID/key00"]["status"] == "OBSERVED" and
                all(old[f"legacy/{arm}/{key}"]["status"] == "READ" for arm in ("S_base","S_ID") for key in KEY_IDS)):
            audit = ledger.call("native_audit", lambda:method.native_audit(native["native/S_base/key00"], native["native/S_ID/key00"], books["key00"],
                {key:old[f"legacy/S_base/{key}"] for key in KEY_IDS}, {key:old[f"legacy/S_ID/{key}"] for key in KEY_IDS}))
            if audit is not None:
                result["native_audit"] = audit
                dump(output/"native_audit.json", audit)
        loaded = ledger.call("model", backend.load) if features else None
        if loaded is not None:
            result["model"] = loaded
            for arm_index, arm in enumerate(method.ARMS):
                if arm not in features:
                    continue
                decoded = ledger.call(f"{arm}/decode", lambda arm=arm:backend.decode(features[arm]))
                if decoded is None:
                    continue
                row = ledger.call(f"{arm}/preclip", lambda:preclip_record(shared.validate_rgb(decoded["preclip"],preclip=True)))
                if row is not None:
                    preclip[arm] = row
                pixels = ledger.call(f"{arm}/quantize", lambda arm=arm:save_tensor(Path("media")/arm/"rgb8.pt", backend.quantize(shared.validate_rgb(decoded["float_rgb"]))))
                del decoded
                if pixels is None:
                    continue
                media[arm,"rgb8"] = pixels
                path = output/"media"/(arm+".mp4")
                saved = ledger.call(f"{arm}/save_mp4", lambda:backend.save_mp4(pixels,path))
                if saved is None:
                    continue
                probe = ledger.call(f"{arm}/probe", lambda:backend.probe(path))
                if probe is None:
                    continue
                dump(output/"media"/(arm+"_probe.json"), probe)
                rgb = ledger.call(f"{arm}/read_mp4", lambda arm=arm:save_tensor(Path("media")/arm/"readback.pt", shared.validate_rgb(backend.read_mp4(path,probe),pixels=True)))
                if rgb is None:
                    continue
                media[arm,"mp4"] = rgb
                deleted = torch.cat((rgb[:63], rgb[64:]), dim=0)
                for case, (view,a) in enumerate(PROBES):
                    n = arm_index*6+case
                    def prepare(view=view,a=a,n=n):
                        clip, receipt = method.prepare_local(rgb if view == "clean" else deleted, a)
                        dump(output/"blind"/f"preparation/{n:02d}.json", receipt)
                        return clip
                    clip = ledger.call(f"prepare/{n:02d}", prepare)
                    if clip is None:
                        continue
                    def encode(clip=clip,n=n):
                        return save_tensor(Path("features/local")/(f"{n:02d}.pt"), backend.encode_local(clip))
                    z = ledger.call(f"encode/{n:02d}", encode)
                    if z is None:
                        continue
                    for key in KEY_IDS:
                        slot = f"local/{n:02d}/{key}"
                        if key not in books:
                            continue
                        value = save_record(slot, lambda key=key,z=z:method.observe(z,books[key],local=True))
                        if value is not None:
                            local[slot] = value
                    checkpoint()
                del deleted
            for left,right in PAIRS:
                for layer in ("rgb8","mp4"):
                    name = f"{left}_to_{right}/{layer}"
                    if (left,layer) in media and (right,layer) in media:
                        value = ledger.call("quality/"+name, lambda left=left,right=right,layer=layer:distortion(media[left,layer],media[right,layer],includes_guard=right=="S_ID",meaning=f"same fresh D chain {left} to {right}; observed increment, no quality PASS"))
                        if value is not None:
                            quality[name] = value
    except BaseException as exc:
        result["fatal_error"] = f"{type(exc).__name__}: {exc}"
    # Persist every fixed row, including missing dependencies and interrupted calls.
    for table, missing in ((native,lambda s:method.unavailable(s)), (old,legacy.unavailable_state), (local,lambda s:method.unavailable(s,local=True))):
        for slot,value in table.items():
            status = ledger.data[slot]["status"]
            if status != "RETURNED":
                table[slot] = missing(status)
            dump(output/"blind"/(slot+".json"), table[slot])
    aliases = []
    for arm_index in range(3):
        for key in KEY_IDS:
            slot = f"local/{arm_index*6+3:02d}/{key}"
            # G's input is exactly D's nine source frames104..112, same arm.
            # No new E, projection or classification, including when D failed.
            alias = dict(reuse_of=slot, new_physical_observation=False, observation=local[slot])
            aliases.append(alias)
            dump(output/"blind/logical_reuse"/f"arm{arm_index}_{key}.json", alias)
    for name in quality:
        status = ledger.data["quality/"+name]["status"]
        if status != "RETURNED":
            quality[name] = shared.unavailable_quality(status)
        dump(output/"quality"/(name+".json"),quality[name])
    for arm in method.ARMS:
        status = ledger.data[f"{arm}/preclip"]["status"]
        if status != "RETURNED":
            preclip[arm] = dict(status=status, frame_denominator=181, rows=[dict(frame=t,status=status) for t in range(181)])
        dump(output/"quality/preclip"/(arm+".json"),preclip[arm])
    for name in ("construct","native_audit"):
        if ledger.data[name]["status"] != "RETURNED":
            result["construction" if name == "construct" else name] = dict(ledger.data[name])
    checkpoint()
    # No semantic roles/source-map evaluator has been entered before this point.
    dump(output/"numeric_completion.json", dict(completed_unix=time.time(), physical_vectors=36,
        physical_identity_slots=1656, logical_vectors=42, reused_vectors=6,
        all_fixed_rows_persisted=True, post_truth_roles_read=False, payload_read=False))
    def post_evaluation():
        persisted = {slot:json.loads((output/"blind"/(slot+".json")).read_text()) for slot in local}
        value = evaluate(persisted)
        dump(output/"evaluation.json",value)
        dump(output/"post_only_roles.json",value["roles"])
        return value
    evaluation = ledger.call("evaluation", post_evaluation)
    result.update(evaluation_summary=None if evaluation is None else {k:v for k,v in evaluation.items() if k not in ("rows","roles")},
                  calls=ledger.data, counts=ledger.counts(), seconds=time.perf_counter()-started,
                  quality={name:{k:v for k,v in q.items() if k not in ("frames","adjacent")} for name,q in quality.items()},
                  preclip={arm:{k:v for k,v in q.items() if k != "rows"} for arm,q in preclip.items()},
                  process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  optional_file_fsync_unsupported_errnos=sorted(SYNC_UNAVAILABLE_ERRNOS))
    if backend.device == "cuda":
        result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    result["status"] = "EXECUTION_COMPLETE" if all(v["status"] == "RETURNED" for v in ledger.data.values()) and not result.get("fatal_error") else "INCOMPLETE"
    dump(output/"result.json", result)
    checkpoint()
    return result
