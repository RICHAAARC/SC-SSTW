"""Bounded native-baseline runner. Real execution is for the user's Colab only."""
from collections import Counter
import csv
import json
from pathlib import Path
import random
import time

from main.baselines.native_paper_v1 import (METHODS, crop_rgb, denominators, exact_message_metrics,
                                           operations, read_tasks, validate_config)
from runtime.wan.native_baseline_storage_v1 import (Journal, now, read_json, recover, reserve_local, write_json)

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "experiments/wan_state_clock/configs/native_baselines_v1.json"


def writer_messages(config):
    """Writer/evaluator fixture only. This function is never called by blind readers."""
    rng = random.Random(config["message_fixture_seed"])
    return {m: {s["id"]: [rng.randrange(2) for _ in range(METHODS[m])] for s in config["sources"]}
            for m in config["methods"]}


def _append(path, row):
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def blind_read(adapter, received_rgb, task, journal, output):
    """Native read boundary: task has scheduling identity, never semantic truth."""
    import numpy as np
    folder = Path(output) / "blind" / task["id"]
    folder.mkdir(parents=True, exist_ok=True)
    if task["method"] == "wam":
        for frame in range(task["frames"]):
            pred = journal.call(f"detect/{task['id']}/{frame}", lambda f=frame: adapter.detect(received_rgb[f:f+1]))
            # Mask bytes and sufficient selected-logit statistics precede any evaluation.
            mask_file = folder / f"mask_{frame:03d}.npy"
            np.save(mask_file, pred["mask"], allow_pickle=False)
            _append(folder / "frames.jsonl", dict(frame=frame, bits=pred["bits"],
                    selected_pixels=pred["selected_pixels"], selected_logit_sums=pred["sums"].tolist(),
                    selected_mean_logits=pred["selected_means"],
                    mask_file=mask_file.name, mask_shape=pred["mask_shape"], native_threshold=0.5))
    else:
        pred = journal.call(f"detect/{task['id']}", lambda: adapter.detect(received_rgb))
        np.save(folder / "frame_logits.npy", pred["logits"], allow_pickle=False)
        write_json(folder / "video.json", dict(bits=pred["video_bits"], frames=len(received_rgb),
                                               reduction="unweighted frame-mean logits then >0"))
        for frame, bits in enumerate(pred["bits"]):
            _append(folder / "frames.jsonl", dict(frame=frame, bits=bits))
    write_json(folder / "complete.json", dict(status="BLIND_SAVED", frame_rows=task["frames"]))


def evaluate_saved(output):
    """Only saved observations. Never imports a model or invokes a physical call."""
    import numpy as np
    output = Path(output)
    marker = read_json(output / "blind_terminated.json")
    if marker["status"] != "BLIND_TERMINATED":
        raise ValueError("Blind outputs must be closed before evaluation")
    write_json(output / "post_truth_access.json", dict(started_at=now(), after_blind_terminated=True))
    config = read_json(output / "protocol.json")
    truth = read_json(output / "writer_truth.json")
    tasks = read_tasks(config)
    ledger = {r["id"]: r for r in read_json(output / "ledger.json")}
    rows, videos, read_errors = [], [], []
    for task in tasks:
        folder = output / "blind" / task["id"]
        saved = {}
        if (folder / "frames.jsonl").is_file():
            for line, text in enumerate((folder / "frames.jsonl").read_text().splitlines(), 1):
                try:
                    row = json.loads(text)
                    if row["frame"] in saved or row["frame"] not in range(task["frames"]):
                        raise ValueError("duplicate/out-of-range received frame")
                    saved[row["frame"]] = row
                except (KeyError, ValueError) as error:
                    read_errors.append(dict(task=task["id"], line=line, error=str(error)))
        owner = truth[task["method"]][task["source"]]
        roster = truth[task["method"]]
        for frame in range(task["frames"]):
            pred = saved.get(frame)
            call = "detect/" + task["id"] + (f"/{frame}" if task["method"] == "wam" else "")
            status = ledger[call]["status"]
            if pred is None and status == "RETURNED":
                status = "OUTPUT_MISSING"
            metric = exact_message_metrics(None if pred is None else pred["bits"], owner, roster,
                                           positive=task["arm"] == "WM")
            row = dict(task=task["id"], method=task["method"], source=task["source"], group="seen-development",
                       arm=task["arm"], view=task["view"], received_frame=frame, status=status, **metric)
            if pred and task["method"] == "wam":
                try:
                    mask = np.unpackbits(np.load(folder / pred["mask_file"], allow_pickle=False))[:256*256].reshape(256, 256).astype(bool)
                    target = np.zeros((256, 256), dtype=bool)
                    if task["arm"] == "WM":
                        if task["view"] == "FULL":
                            target[64:192, 64:192] = True
                        else:
                            target[:] = True
                    intersection, union = int((mask & target).sum()), int((mask | target).sum())
                    row.update(selected_pixels=pred["selected_pixels"], localization_iou=intersection/union if union else 1.0,
                               predicted_mask_fraction=float(mask.mean()))
                except (OSError, ValueError, KeyError) as error:
                    read_errors.append(dict(task=task["id"], frame=frame, error=str(error)))
            rows.append(row)
        if task["method"] == "pixelseal":
            bits = read_json(folder / "video.json")["bits"] if (folder / "video.json").is_file() else None
            videos.append(dict(task=task["id"], arm=task["arm"], **exact_message_metrics(bits, owner, roster,
                                                                                       positive=task["arm"] == "WM")))
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with (output / "frame_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for method in METHODS:
        for arm in ("WM", "OFF"):
            for view in ("FULL", "CENTER_RGB"):
                part = [r for r in rows if (r["method"], r["arm"], r["view"]) == (method, arm, view)]
                counts = {k: sum(r[k] for r in part) for k in ("correct", "wrong", "unknown")}
                available = counts["correct"] + counts["wrong"]
                summaries.append(dict(method=method, arm=arm, view=view, planned_frames=len(part),
                                      planned_bits=len(part)*METHODS[method], **counts,
                                      ber_on_available_bits=counts["wrong"]/available if available else None,
                                      exact_owner_frames=sum(r["exact_owner"] is True for r in part),
                                      exact_other_frames=sum(bool(r["exact_other"]) for r in part),
                                      statuses=dict(Counter(r["status"] for r in part))))
    result = dict(status="SAVED_OUTPUT_EVALUATION", evidence_level="seen-development native baseline only",
                  denominators=denominators(config), frame_summaries=summaries, video_messages=videos,
                  read_errors=read_errors, scientific_pass=None,
                  caveats=["32/256 native capacities and coverage differ", "No calibrated FPR/identity threshold",
                           "No WAM video vote", "No temporal or joint identity-time recovery"])
    write_json(output / "evaluation.json", result)
    return result


def execute_reserved(config, output, scratch, vendor_root, *, adapter_factory=None, media=None):
    """Internal worker for the one reservation created by bootstrap or CLI."""
    import numpy as np
    from runtime.wan.native_baseline_media_v1 import Media, quality
    from runtime.wan.native_baseline_models_v1 import NativeAdapter
    output, scratch = Path(output), Path(scratch)
    attempt = read_json(output / "attempt.json")
    if attempt.get("status") != "RESERVED" or (output / "plan.json").exists():
        return recover(output)
    config = validate_config(config)
    attempt.update(status="RUNNING", started_at=now(), consumed=True)
    write_json(output / "attempt.json", attempt)
    write_json(output / "protocol.json", config)
    truth = writer_messages(config)
    write_json(output / "writer_truth.json", truth)
    journal = Journal(output, operations(config))
    scratch.mkdir(parents=True, exist_ok=True)
    adapter_factory = adapter_factory or (lambda m: NativeAdapter(m, vendor_root))
    media = media or Media()
    shape, fps, rectangle = config["shape"], config["fps"], config["crop_yxyx"]
    quality_rows, model_rows, internal_counts = [], {}, {}
    failure, adapter = None, None
    started = time.monotonic()
    try:
        # Fixed clean controls prepared once and shared by the two model readers.
        for source in config["sources"]:
            host = source["id"]
            rgb = journal.call("input/"+host, lambda s=source: media.read_source(config["input_root"], s, shape, fps))
            np.save(scratch / f"{host}.raw.npy", rgb, allow_pickle=False)
            path = output / "media" / host / "OFF.mp4"
            journal.call("encode/OFF/"+host, lambda: media.encode(rgb, path, fps))
            journal.call("probe/OFF/"+host, lambda: media.probe(path, shape, fps))
            decoded = journal.call("decode/OFF/"+host, lambda: media.decode(path, shape))
            np.save(scratch / f"{host}.off.npy", decoded, allow_pickle=False)
            del rgb, decoded
        for method in config["methods"]:
            adapter = adapter_factory(method)
            model_rows[method] = journal.call("load/"+method, adapter.load)
            write_json(output / "models.json", model_rows)
            for source in config["sources"]:
                host = source["id"]
                rgb = np.load(scratch / f"{host}.raw.npy", mmap_mode="r", allow_pickle=False)
                off = np.load(scratch / f"{host}.off.npy", mmap_mode="r", allow_pickle=False)
                message = truth[method][host]
                if method == "wam":
                    wm = np.empty_like(rgb)
                    for frame in range(shape[0]):
                        wm[frame:frame+1] = journal.call(f"embed/{method}/{host}/{frame}",
                            lambda f=frame: adapter.embed(rgb[f:f+1], message, rectangle))
                else:
                    wm = journal.call(f"embed/{method}/{host}", lambda: adapter.embed(rgb, message, rectangle))
                del message
                path = output / "media" / host / (method+".mp4")
                journal.call(f"encode/{method}/{host}", lambda: media.encode(wm, path, fps))
                journal.call(f"probe/{method}/{host}", lambda: media.probe(path, shape, fps))
                decoded = journal.call(f"decode/{method}/{host}", lambda: media.decode(path, shape))
                for stage, left, right in (("pre", rgb, wm), ("decoded", off, decoded)):
                    metrics = journal.call(f"quality_{stage}/{method}/{host}", lambda a=left, b=right: quality(a, b))
                    quality_rows.append(dict(method=method, source=host, stage=stage, **metrics))
                write_json(output / "quality.json", quality_rows)
                del wm, rgb
                for task in read_tasks(config):
                    if task["method"] == method and task["source"] == host:
                        received = crop_rgb(decoded if task["arm"] == "WM" else off, task["view"], rectangle)
                        blind_read(adapter, received, task, journal, output)
                del decoded, off, received
                journal.snapshot()
                write_json(output / "progress.json", dict(method=method, source=host, state="SOURCE_READS_SAVED"))
                print(method, host, "blind outputs saved", flush=True)
            internal_counts[method] = dict(adapter.counts)
            if hasattr(adapter, "resources"):
                internal_counts[method]["resources"] = adapter.resources()
            adapter.close()
            adapter = None
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error))
    finally:
        if adapter is not None:
            internal_counts[adapter.method] = dict(adapter.counts)
            if hasattr(adapter, "resources"):
                internal_counts[adapter.method]["resources"] = adapter.resources()
            try:
                adapter.close()
            except Exception as error:
                if failure is None:
                    failure = dict(type=type(error).__name__, message="Resource release failed: " + str(error))
        snapshot = journal.snapshot()
        write_json(output / "blind_terminated.json", dict(status="BLIND_TERMINATED", closed_at=now(),
                                                          failure=failure, **snapshot))
        result = dict(status="BLIND_SAVED_EVALUATION_PENDING", failure=failure,
                      physical_status="COMPLETED" if failure is None else "STOPPED_FIXED_ATTEMPT",
                      denominators=denominators(config), internal_counts=internal_counts, **snapshot,
                      seconds=time.monotonic()-started, scientific_pass=None)
        write_json(output / "result.json", result)
        attempt.update(status=result["status"], consumed=True)
        write_json(output / "attempt.json", attempt)
    try:
        evaluate_saved(output)
    except BaseException as error:
        result["evaluation_failure"] = dict(error=f"{type(error).__name__}: {error}",
                                            denominators=denominators(config))
        result["status"] = "STOPPED_FIXED_ATTEMPT"
        # Publish the non-success terminal state before attempting an additional
        # diagnostic write: a failed diagnostic must not restore a success bit.
        write_json(output / "result.json", result)
        attempt.update(status=result["status"], finished_at=now(), consumed=True)
        write_json(output / "attempt.json", attempt)
        write_json(output / "evaluation_failure.json", result["evaluation_failure"])
    else:
        result["status"] = result["physical_status"]
        write_json(output / "result.json", result)
        attempt.update(status=result["status"], finished_at=now(), consumed=True)
        write_json(output / "attempt.json", attempt)
    return result


def run(config, output, scratch, vendor_root, **kwargs):
    if not reserve_local(output):
        return recover(output)
    return execute_reserved(config, output, scratch, vendor_root, **kwargs)
