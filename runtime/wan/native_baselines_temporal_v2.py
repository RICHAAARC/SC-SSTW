"""One consumed temporal development batch. Real execution belongs in user Colab."""
from pathlib import Path
import time

from main.baselines.native_paper_temporal_v2 import (CHANNELS, CONDITIONS, denominators, edit_maps,
    media_key, operations, physical_budget, read_tasks, validate_config)
from runtime.wan.native_baseline_storage_v1 import Journal, now, read_json, recover, reserve_local, write_json
from runtime.wan.native_baselines_v1 import blind_read, writer_messages
from runtime.wan.native_baseline_temporal_evaluation_v2 import evaluate_saved

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "experiments/wan_state_clock/configs/native_baselines_temporal_v2.json"


def execute_reserved(config, output, scratch, vendor_root, *, adapter_factory=None, media=None):
    import numpy as np
    from runtime.wan.native_baseline_media_v1 import Media, quality
    from runtime.wan.native_baseline_models_v1 import NativeAdapter
    output, scratch = Path(output), Path(scratch)
    attempt = read_json(output / "attempt.json")
    if attempt.get("status") != "RESERVED" or (output / "plan.json").exists():
        return recover(output)
    attempt.update(status="RUNNING", started_at=now(), consumed=True)
    write_json(output / "attempt.json", attempt)
    journal = Journal(output, operations(config))
    quality_rows, model_rows, internal_counts = [], {}, {}
    failure, adapter = None, None
    started = time.monotonic()
    try:
        validate_config(config)
        write_json(output / "protocol.json", config)
        write_json(output / "planned_read_tasks.json", read_tasks(config))
        write_json(output / "physical_budget.json", physical_budget(config))
        # The editor gets maps; readers and native postprocessing never do.
        maps = edit_maps(config)
        write_json(output / "edit_truth.json", maps)
        truth = writer_messages(config)
        write_json(output / "writer_truth.json", truth)
        scratch.mkdir(parents=True, exist_ok=True)
        adapter_factory = adapter_factory or (lambda m: NativeAdapter(m, vendor_root))
        media = media or Media()
        shape, fps, rectangle = config["shape"], config["fps"], config["crop_yxyx"]
        tasks = read_tasks(config)

        def channel(rgb, host, carrier, generation, condition="IDENTITY"):
            key = media_key(host, carrier, generation, condition)
            path = output / "media" / (key + ".mp4")
            received_shape = list(rgb.shape)
            encoded = journal.call("encode/" + key, lambda: media.encode(rgb, path, fps))
            probe = journal.call("probe/" + key, lambda: media.probe(path, received_shape, fps))
            write_json(path.with_suffix(".json"), dict(shape=received_shape, fps=fps, encode=encoded, probe=probe))
            return journal.call("decode/" + key, lambda: media.decode(path, received_shape))

        def cache(host, suffix):
            return scratch / (host + "." + suffix + ".npy")

        def measure(method, host, stage, left, right):
            metrics = journal.call(f"quality/{method}/{host}/{stage}", lambda: quality(left, right))
            quality_rows.append(dict(method=method, source=host, stage=stage, **metrics))
            write_json(output / "quality.json", quality_rows)

        # OFF media (including every G2 condition) is created once and shared.
        for source in config["sources"]:
            host = source["id"]
            rgb = journal.call("input/"+host, lambda s=source: media.read_source(config["input_root"], s, shape, fps))
            np.save(cache(host, "raw"), rgb, allow_pickle=False)
            off = channel(rgb, host, "OFF", CHANNELS[0])
            np.save(cache(host, "off_G1"), off, allow_pickle=False)
            for condition in CONDITIONS:
                edited = off[maps[host]["conditions"][condition]["indices"]]
                decoded = channel(edited, host, "OFF", CHANNELS[1], condition)
                np.save(cache(host, "off_G2_" + condition), decoded, allow_pickle=False)
            del rgb, off, edited, decoded
        for method in config["methods"]:
            adapter = adapter_factory(method)
            model_rows[method] = journal.call("load/"+method, adapter.load)
            write_json(output / "models.json", model_rows)
            for source in config["sources"]:
                host = source["id"]
                rgb = np.load(cache(host, "raw"), mmap_mode="r", allow_pickle=False)
                off = np.load(cache(host, "off_G1"), mmap_mode="r", allow_pickle=False)
                message = truth[method][host]
                if method == "wam":
                    wm = np.empty_like(rgb)
                    for frame in range(shape[0]):
                        wm[frame:frame+1] = journal.call(f"embed/{method}/{host}/{frame}",
                            lambda f=frame: adapter.embed(rgb[f:f+1], message, rectangle))
                else:
                    wm = journal.call(f"embed/{method}/{host}", lambda: adapter.embed(rgb, message, rectangle))
                del message
                decoded = channel(wm, host, method, CHANNELS[0])
                measure(method, host, "precodec", rgb, wm)
                measure(method, host, "G1_IDENTITY", off, decoded)
                del wm, rgb
                for condition in CONDITIONS:
                    indices = maps[host]["conditions"][condition]["indices"]
                    wm_g1, off_g1 = decoded[indices], off[indices]
                    wm_g2 = channel(wm_g1, host, method, CHANNELS[1], condition)
                    off_g2 = np.load(cache(host, "off_G2_" + condition), mmap_mode="r", allow_pickle=False)
                    measure(method, host, "G2/" + condition, off_g2, wm_g2)
                    for task in tasks:
                        if (task["method"], task["source"], task["condition"]) != (method, host, condition):
                            continue
                        pixels = {("WM", CHANNELS[0]): wm_g1, ("WM", CHANNELS[1]): wm_g2,
                                  ("OFF", CHANNELS[0]): off_g1, ("OFF", CHANNELS[1]): off_g2}[task["arm"], task["channel"]]
                        if len(pixels) != task["frames"]:
                            raise ValueError("Received frame count differs from this fixed task")
                        # Only pixels reach native inference; task labels only locate saved rows.
                        blind_read(adapter, pixels, task, journal, output)
                    del wm_g1, wm_g2, off_g1, off_g2, pixels
                    write_json(output / "progress.json", dict(method=method, source=host, condition=condition,
                                                              state="CONDITION_BLIND_SAVED"))
                    print(method, host, condition, "blind outputs saved", flush=True)
                del decoded, off
                journal.snapshot()
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
                      denominators=denominators(config), planned_physical_budget=physical_budget(config),
                      internal_counts=internal_counts, **snapshot, seconds=time.monotonic()-started, scientific_pass=None)
        write_json(output / "result.json", result)
        attempt.update(status=result["status"], consumed=True)
        write_json(output / "attempt.json", attempt)
    try:
        evaluate_saved(output)
    except BaseException as error:
        result["evaluation_failure"] = dict(error=f"{type(error).__name__}: {error}", denominators=denominators(config))
        result["status"] = "STOPPED_FIXED_ATTEMPT"
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
