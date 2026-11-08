"""Native development-baseline protocol; no models, filesystem or truth lookup."""
from itertools import product

METHODS = {"wam": 32, "pixelseal": 256}
ARMS = ("WM", "OFF")
VIEWS = ("FULL", "CENTER_RGB")


def validate_config(config):
    if config["shape"] != [181, 320, 512, 3] or config["fps"] != 8:
        raise ValueError("This development batch requires 181x320x512 RGB at 8 fps")
    if config["crop_yxyx"] != [80, 128, 240, 384]:
        raise ValueError("This batch uses the stated central RGB rectangle")
    if config["methods"] != list(METHODS):
        raise ValueError("This batch has WAM and PixelSeal; no implicit fallback")
    ids = [s["id"] for s in config["sources"]]
    if ids != [f"b-rank3r-m{i:02d}" for i in range(1, 5)]:
        raise ValueError("Keep the four declared development sources")
    if any(s["group"] != "seen-development" for s in config["sources"]):
        raise ValueError("These previously viewed sources are development inputs")
    return config


def read_tasks(config):
    return [dict(id=f"{m}/{s['id']}/{a}/{v}", method=m, source=s["id"],
                 arm=a, view=v, frames=config["shape"][0], nbits=METHODS[m])
            for m, s, a, v in product(config["methods"], config["sources"], ARMS, VIEWS)]


def operations(config):
    """All call slots exist before execution, including never-started slots."""
    n = config["shape"][0]
    rows = []
    def add(key, kind):
        rows.append(dict(id=key, kind=kind, status="NOT_STARTED"))
    for s in config["sources"]:
        h = s["id"]
        add(f"input/{h}", "input")
        for op in ("encode", "probe", "decode"):
            add(f"{op}/OFF/{h}", op)
    for m in config["methods"]:
        add(f"load/{m}", "load")
        for s in config["sources"]:
            h = s["id"]
            if m == "wam":
                for frame in range(n):
                    add(f"embed/{m}/{h}/{frame}", "embed")
            else:
                add(f"embed/{m}/{h}", "embed")
            for op in ("encode", "probe", "decode", "quality_pre", "quality_decoded"):
                add(f"{op}/{m}/{h}", op)
            for a, v in product(ARMS, VIEWS):
                key = f"{m}/{h}/{a}/{v}"
                if m == "wam":
                    for frame in range(n):
                        add(f"detect/{key}/{frame}", "detect")
                else:
                    add(f"detect/{key}", "detect")
    return rows


def denominators(config):
    tasks = read_tasks(config)
    return dict(read_tasks=len(tasks), frame_rows=sum(t["frames"] for t in tasks),
                frame_bits={m: sum(t["frames"] * t["nbits"] for t in tasks if t["method"] == m)
                            for m in METHODS},
                positive_frame_bits={m: sum(t["frames"] * t["nbits"] for t in tasks
                                            if t["method"] == m and t["arm"] == "WM")
                                     for m in METHODS},
                pixelseal_video_messages=16, pixelseal_video_bits=4096,
                wam_video_aggregation="UNSUPPORTED", time_recovery="UNSUPPORTED",
                joint_identity_time="UNSUPPORTED")


def crop_rgb(rgb, view, rectangle):
    """Crop received pixels, before the native detector's own resize."""
    if view == "FULL":
        return rgb
    if view != "CENTER_RGB":
        raise ValueError("unknown received view")
    y0, x0, y1, x1 = rectangle
    return rgb[:, y0:y1, x0:x1, :].copy()


def exact_message_metrics(bits, owner, roster, *, positive):
    """Posterior descriptive equality, not an identification decision rule."""
    n = len(owner)
    if bits is None:
        return dict(correct=0, wrong=0, unknown=n, exact_owner=None, exact_other=[])
    if len(bits) != n or any(b not in (0, 1) for b in bits):
        raise ValueError("invalid native message")
    correct = sum(a == b for a, b in zip(bits, owner))
    return dict(correct=correct, wrong=n-correct, unknown=0,
                exact_owner=bits == owner,
                exact_other=[h for h, msg in roster.items() if msg != owner and msg == bits],
                interpretation="payload" if positive else "clean_literal_matches_not_calibrated_FPR")
