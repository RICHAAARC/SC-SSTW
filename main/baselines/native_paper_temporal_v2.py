"""Adopted temporal development protocol; no model, filesystem or truth lookup."""
from collections import Counter
from itertools import product

from main.baselines.native_paper_v1 import METHODS, exact_message_metrics

EXPERIMENT = "native-wam-pixelseal-temporal-dev-v2"
CHANNELS = ("G1_EDIT_RGB", "G2_EDIT_H264")
ARMS = ("WM", "OFF")
CONDITIONS = {
    "IDENTITY": 181,
    **{f"TRIM_S{s}_L129": 129 for s in range(26, 30)},
    "DROP18": 163, "DROP54": 127,
    "BURST18": 163, "BURST54": 127,
    "REPEAT_FRAME18": 199, "REPLAY_BLOCK18": 199,
}


def validate_config(config):
    if config["experiment"] != EXPERIMENT:
        raise ValueError("Wrong temporal experiment identifier")
    if config["shape"] != [181, 320, 512, 3] or config["fps"] != 8:
        raise ValueError("This development batch requires uint8 181x320x512 RGB at 8 fps")
    if config["crop_yxyx"] != [80, 128, 240, 384]:
        raise ValueError("WAM writer support must retain the adopted central rectangle")
    if config["methods"] != list(METHODS) or config["conditions"] != list(CONDITIONS) or config["channels"] != list(CHANNELS):
        raise ValueError("Keep both native methods and all eleven conditions in both channels")
    if config["message_fixture_seed"] != 20261105 or config["edit_seed_base"] != 20261105:
        raise ValueError("Keep the adopted message and edit fixture rules")
    sources = config["sources"]
    if [s["id"] for s in sources] != [f"b-rank3r-m{i:02d}" for i in range(1, 5)]:
        raise ValueError("Keep the four declared development sources")
    if any(s["group"] != "seen-development" for s in sources):
        raise ValueError("These previously viewed sources remain development inputs")
    return config


def edit_maps(config):
    """Editor/evaluator truth only. Generate once, before any physical calls."""
    import numpy as np
    result = {}
    for ordinal, source in enumerate(config["sources"], 1):
        permutation = np.random.Generator(np.random.PCG64(config["edit_seed_base"] + 100 * ordinal)).permutation(181).tolist()
        d18, d54 = set(permutation[:18]), set(permutation[:54])
        maps = {"IDENTITY": list(range(181)),
                **{f"TRIM_S{s}_L129": list(range(s, s+129)) for s in range(26, 30)},
                "DROP18": [i for i in range(181) if i not in d18],
                "DROP54": [i for i in range(181) if i not in d54],
                "BURST18": list(range(81)) + list(range(99, 181)),
                "BURST54": list(range(63)) + list(range(117, 181)),
                "REPEAT_FRAME18": [i for i in range(181) for _ in range(2 if i in d18 else 1)],
                "REPLAY_BLOCK18": list(range(99)) + list(range(81, 99)) + list(range(99, 181))}
        conditions = {}
        for name, indices in maps.items():
            seen = Counter()
            instances = []
            for received, index in enumerate(indices):
                instances.append(dict(received_frame=received, source_id=source["id"], source_frame_index=index,
                                      source_time_seconds=index/config["fps"], received_time_seconds=received/config["fps"],
                                      copy_ordinal=seen[index], edit_kind=name))
                seen[index] += 1
            conditions[name] = dict(indices=indices, instances=instances, original_source_frames=181,
                                    unique_retained_source_frames=len(seen), received_instances=len(indices),
                                    duplicate_instances=len(indices)-len(seen),
                                    not_received=[i for i in range(181) if i not in seen],
                                    zero_deltas=sum(b == a for a, b in zip(indices, indices[1:])),
                                    backward_deltas=sum(b < a for a, b in zip(indices, indices[1:])))
        result[source["id"]] = dict(seed=config["edit_seed_base"] + 100 * ordinal,
                                    permutation=permutation, conditions=conditions)
    return result


def read_tasks(config):
    """Scheduling labels only: no indices, edit seed, message or true phase."""
    return [dict(id=f"{m}/{s['id']}/{a}/{c}/{v}", method=m, source=s["id"], arm=a,
                 condition=v, channel=c, frames=CONDITIONS[v], nbits=METHODS[m])
            for m, s, a, c, v in product(config["methods"], config["sources"], ARMS, CHANNELS, CONDITIONS)]


def media_key(source, carrier, channel, condition="IDENTITY"):
    return f"{source}/{carrier}/{channel}/{condition}"


def operations(config):
    rows = []
    def add(key, kind):
        rows.append(dict(id=key, kind=kind, status="NOT_STARTED"))
    for source in config["sources"]:
        host = source["id"]
        add("input/" + host, "input")
        for carrier in ("OFF", *config["methods"]):
            for channel, condition in [(CHANNELS[0], "IDENTITY")] + [(CHANNELS[1], v) for v in CONDITIONS]:
                for op in ("encode", "probe", "decode"):
                    add(op + "/" + media_key(host, carrier, channel, condition), op)
    for method in config["methods"]:
        add("load/" + method, "load")
        for source in config["sources"]:
            host = source["id"]
            if method == "wam":
                for frame in range(181):
                    add(f"embed/{method}/{host}/{frame}", "embed")
            else:
                add(f"embed/{method}/{host}", "embed")
            for stage in ("precodec", "G1_IDENTITY"):
                add(f"quality/{method}/{host}/{stage}", "quality_" + stage)
            for condition in CONDITIONS:
                add(f"quality/{method}/{host}/G2/{condition}", "quality_G2")
    for task in read_tasks(config):
        if task["method"] == "wam":
            for frame in range(task["frames"]):
                add(f"detect/{task['id']}/{frame}", "detect")
        else:
            add("detect/" + task["id"], "detect")
    return rows


def denominators(config):
    tasks = read_tasks(config)
    frames = {m: sum(t["frames"] for t in tasks if t["method"] == m) for m in METHODS}
    videos = sum(t["method"] == "pixelseal" for t in tasks)
    return dict(read_tasks=len(tasks), frame_rows=sum(frames.values()), frames_by_method=frames,
                frame_bits={m: frames[m]*METHODS[m] for m in METHODS},
                positive_frame_bits={m: frames[m]*METHODS[m]//2 for m in METHODS},
                pixelseal_video_messages=videos, pixelseal_video_bits=videos*256,
                wam_video_aggregation="N/A", time_recovery="N/A", joint_identity_time="N/A")


def physical_budget(config):
    tasks = read_tasks(config)
    s = len(config["sources"])
    media = s * 3 * (1 + len(CONDITIONS))
    return dict(model_loads=2, wam_embed_api=s*181, wam_embedder_forward=s*181,
                wam_detect_api=sum(t["frames"] for t in tasks if t["method"] == "wam"),
                wam_detector_forward=sum(t["frames"] for t in tasks if t["method"] == "wam"),
                pixelseal_embed_api=s, pixelseal_embedder_forward=s, pixelseal_embedder_frames=s*23,
                pixelseal_detect_api=sum(t["method"] == "pixelseal" for t in tasks),
                pixelseal_detector_forward=sum((t["frames"]+31)//32 for t in tasks if t["method"] == "pixelseal"),
                pixelseal_detector_frames=sum(t["frames"] for t in tasks if t["method"] == "pixelseal"),
                encode=media, probe=media, decode=media, quality_precodec=2*s,
                quality_G1_IDENTITY=2*s, quality_G2=2*s*len(CONDITIONS))
