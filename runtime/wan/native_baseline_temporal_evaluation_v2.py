"""Post-evaluation of closed temporal blind outputs. No model or codec imports."""
from collections import Counter
import csv
import json
from pathlib import Path

from main.baselines.native_paper_temporal_v2 import (ARMS, CHANNELS, CONDITIONS, METHODS,
    denominators, exact_message_metrics, read_tasks)
from runtime.wan.native_baseline_storage_v1 import now, read_json, write_json


def _metrics(rows, nbits):
    counts = {k: sum(r[k] for r in rows) for k in ("correct", "wrong", "unknown")}
    readable = counts["correct"] + counts["wrong"]
    planned = len(rows)*nbits
    return dict(planned_frames=len(rows), planned_bits=planned, **counts,
                readable_bits=readable, readable_ber=counts["wrong"]/readable if readable else None,
                coverage=readable/planned if planned else None,
                complete_message_correct=sum(r["exact_owner"] is True for r in rows),
                complete_message_error=sum(r["exact_owner"] is False for r in rows),
                complete_message_unknown=sum(r["exact_owner"] is None for r in rows),
                literal_other_matches=sum(bool(r["exact_other"]) for r in rows),
                statuses=dict(Counter(r["status"] for r in rows)))


def _csv(path, rows):
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_saved(output):
    import numpy as np
    output = Path(output)
    if read_json(output / "blind_terminated.json")["status"] != "BLIND_TERMINATED":
        raise ValueError("Blind outputs must be closed before truth access")
    write_json(output / "post_truth_access.json", dict(started_at=now(), after_blind_terminated=True))
    config = read_json(output / "protocol.json")
    truth = read_json(output / "writer_truth.json")
    maps = read_json(output / "edit_truth.json")
    ledger = {r["id"]: r for r in read_json(output / "ledger.json")}
    rows, videos, details, read_errors = [], [], [], []
    retention = []
    for source in config["sources"]:
        for condition, mapping in maps[source["id"]]["conditions"].items():
            retention.append(dict(source=source["id"], condition=condition,
                **{k: mapping[k] for k in ("original_source_frames", "unique_retained_source_frames",
                    "received_instances", "duplicate_instances", "zero_deltas", "backward_deltas")},
                not_received_source_frames=len(mapping["not_received"]),
                not_received_indices=mapping["not_received"], not_received_state="NOT_RECEIVED",
                interpretation="editing ground truth; not a receiver recovery"))
    for task in read_tasks(config):
        folder = output / "blind" / task["id"]
        saved = {}
        owner, roster = truth[task["method"]][task["source"]], truth[task["method"]]
        if (folder / "frames.jsonl").is_file():
            for line, text in enumerate((folder / "frames.jsonl").read_text().splitlines(), 1):
                try:
                    pred = json.loads(text)
                    f = pred["frame"]
                    if f in saved or f not in range(task["frames"]):
                        raise ValueError("duplicate/out-of-range received frame")
                    exact_message_metrics(pred["bits"], owner, roster, positive=task["arm"] == "WM")
                    saved[f] = pred
                except (KeyError, TypeError, ValueError) as error:
                    read_errors.append(dict(task=task["id"], line=line, error=str(error)))
        part = []
        for frame in range(task["frames"]):
            pred = saved.get(frame)
            call = "detect/" + task["id"] + (f"/{frame}" if task["method"] == "wam" else "")
            status = ledger[call]["status"]
            if pred is None and status == "RETURNED":
                status = "OUTPUT_MISSING"
            bits = pred["bits"] if pred else None
            metric = exact_message_metrics(bits, owner, roster, positive=task["arm"] == "WM")
            mapping = maps[task["source"]]["conditions"][task["condition"]]["instances"][frame]
            row = dict(task=task["id"], method=task["method"], source=task["source"], arm=task["arm"],
                       condition=task["condition"], channel=task["channel"], group="seen-development",
                       status=status, received_state="UNKNOWN" if bits is None else "READABLE",
                       **mapping, **metric)
            if pred and task["method"] == "wam":
                try:
                    mask = np.unpackbits(np.load(folder / pred["mask_file"], allow_pickle=False))[:256*256].reshape(256, 256).astype(bool)
                    target = np.zeros((256, 256), dtype=bool)
                    if task["arm"] == "WM":
                        target[64:192, 64:192] = True
                    intersection, union = int((mask & target).sum()), int((mask | target).sum())
                    row.update(selected_pixels=pred["selected_pixels"], predicted_mask_fraction=float(mask.mean()),
                               localization_iou=intersection/union if union else 1.0)
                except (OSError, ValueError, KeyError) as error:
                    read_errors.append(dict(task=task["id"], frame=frame, error=str(error)))
            rows.append(row)
            part.append(row)
        detail = {k: task[k] for k in ("id", "method", "source", "arm", "condition", "channel")}
        details.append(dict(detail, **_metrics(part, task["nbits"])))
        if task["method"] == "pixelseal":
            bits = None
            try:
                if (folder / "video.json").is_file():
                    bits = read_json(folder / "video.json")["bits"]
                    exact_message_metrics(bits, owner, roster, positive=task["arm"] == "WM")
            except (OSError, KeyError, TypeError, ValueError) as error:
                read_errors.append(dict(task=task["id"], file="video.json", error=str(error)))
                bits = None
            status = ledger["detect/" + task["id"]]["status"]
            videos.append(dict(detail, status="OUTPUT_MISSING" if bits is None and status == "RETURNED" else status,
                               **exact_message_metrics(bits, owner, roster, positive=task["arm"] == "WM")))
    _csv(output / "frame_results.csv", rows)
    _csv(output / "source_condition_channel_results.csv", details)
    _csv(output / "source_retention.csv", retention)
    _csv(output / "video_results.csv", videos)
    summaries = []
    video_summaries = []
    for method in METHODS:
        for arm in ARMS:
            for channel in CHANNELS:
                selected = [d for d in details if (d["method"], d["arm"], d["channel"]) == (method, arm, channel)]
                available = [d["readable_ber"] for d in selected if d["readable_ber"] is not None]
                # Every source-condition cell remains in the table and coverage mean.
                # Readable BER is undefined for wholly unknown cells; count them explicitly.
                summaries.append(dict(method=method, arm=arm, channel=channel, planned_cells=len(selected),
                    equal_source_equal_condition_coverage=sum(d["coverage"] for d in selected)/len(selected),
                    equal_cell_readable_ber=sum(available)/len(available) if available else None,
                    readable_ber_cells=len(available), undefined_ber_cells=len(selected)-len(available),
                    equal_cell_complete_message_accuracy=sum(d["complete_message_correct"]/d["planned_frames"] for d in selected)/len(selected),
                    **{k: sum(d[k] for d in selected) for k in ("correct", "wrong", "unknown", "planned_bits")}))
    for arm in ARMS:
        for channel in CHANNELS:
            part = [v for v in videos if (v["arm"], v["channel"]) == (arm, channel)]
            video_summaries.append(dict(arm=arm, channel=channel, **_metrics(part, 256)))
    by_id = {d["id"]: d for d in details}
    comparisons = []
    for d in details:
        identity = by_id[f"{d['method']}/{d['source']}/{d['arm']}/{d['channel']}/IDENTITY"]
        g1 = by_id[f"{d['method']}/{d['source']}/{d['arm']}/{CHANNELS[0]}/{d['condition']}"]
        diff = lambda a, b: None if a is None or b is None else a-b
        comparisons.append(dict(task=d["id"], within_generation_identity=identity["id"],
            readable_ber_minus_same_generation_identity=diff(d["readable_ber"], identity["readable_ber"]),
            coverage_minus_same_generation_identity=d["coverage"]-identity["coverage"],
            g2_minus_g1_readable_ber=diff(d["readable_ber"], g1["readable_ber"]) if d["channel"] == CHANNELS[1] else None,
            g2_interpretation="compression plus edit-context interaction"))
    result = dict(status="SAVED_OUTPUT_EVALUATION", evidence_level="four seen-development sources; no independent test",
        denominators=denominators(config), source_condition_channel_details=details, equal_cell_summaries=summaries,
        source_retention=retention, video_messages=videos, video_summaries=video_summaries, comparisons=comparisons,
        read_errors=read_errors, scientific_pass=None,
        unsupported=dict(wam_video_message="N/A", source_position="N/A", gap="N/A", repeat="N/A",
            replay_backward="N/A", joint_identity_time="N/A"),
        not_in_batch={m: "NOT_IN_BATCH/ADAPTATION_UNVERIFIED" for m in ("SIGMark", "VideoShield", "VideoMark")},
        caveats=["WAM32 quarter-frame support versus PixelSeal256 full-frame support",
            "OFF matches are literal matches, not calibrated FPR or authentication",
            "Repeated received instances retain native weight; no deduplication or best-copy selection",
            "TRIM starts26..29 are overlapping conditions, not independent sources or all startup cases"])
    write_json(output / "evaluation.json", result)
    return result
