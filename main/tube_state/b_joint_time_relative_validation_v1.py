"""Fixed single-new-content temporal confirmation protocol; no execution or IO."""
EXPERIMENT = "joint-time-relative-consistency-fresh-r01-v1"
SOURCE = "joint-relative-r01"
VIEWS = ("FULL", "CROP", "DELETE1", "REPEAT1", "DELETE3")
LENGTHS = (181, 101, 180, 182, 178)
ARMS = ("NEW", "BASE")
INNER_BUDGET = dict(transformer_conditional=50, transformer_unconditional=50, native_step=50,
                    payload_gradient=25, D181=1, baseline_head=46, head_replay=46, K=4201, KT=4200)


def construction_maps():
    return dict(FULL=list(range(181)), CROP=list(range(38, 139)),
                DELETE1=[i for i in range(181) if i != 44],
                REPEAT1=list(range(45))+list(range(44, 181)),
                DELETE3=[i for i in range(181) if i not in (43, 44, 45)])


def units(config):
    return [dict(id=f"{config['source']['id']}/{arm}/{view}", source=config['source']['id'],
                 arm=arm, view=view, frames=n, edges=n-1, status="NOT_STARTED")
            for arm in ARMS for view, n in zip(VIEWS, LENGTHS)]


def validate(config):
    if config["experiment"] != EXPERIMENT or config["source"]["id"] != SOURCE or config["source"]["seed"] != 2026110501:
        raise ValueError("Fixed single-source content roster required")
    if config["generation"] != dict(height=320, width=512, frames=181, fps=8, steps=50,
                                      guidance_scale=5.0, max_sequence_length=512):
        raise ValueError("Original native generation geometry/schedule required")
    if config["views"] != list(VIEWS) or config["arms"] != list(ARMS):
        raise ValueError("Keep all five received views and paired NEW/BASE")
    if config["writer"]["key"] != "watermark" or len(config["writer"]["bits"]) != 32:
        raise ValueError("Original 32-bit DCT2 writer required")
    return config


def denominators(config):
    us = units(config)
    return dict(units=len(us), nodes=sum(u["frames"] for u in us), edges=sum(u["edges"] for u in us),
                physical_hard_bit_slots=sum(u["frames"]*630 for u in us), independent_content_sources=1,
                per_arm_nodes=sum(LENGTHS), per_arm_edges=sum(n-1 for n in LENGTHS))


def operation_plan(config):
    rows = []
    def add(key, kind):
        rows.append(dict(id=key, kind=kind, status="NOT_STARTED"))
    for key in ("generation_load", "generate_P0", "capture_vae_load", "capture", "head_prepare", "q0",
                "target", "solver", "prediction", "handoff", "replay_vae_load", "native_replay"):
        add(key, key)
    for arm in ARMS:
        for op in ("encode", "probe", "decode_RGB"):
            add(f"{arm}/first/{op}", op)
    for u in units(config):
        for op in ("encode", "probe", "decode_RGB", "decode_Y", "hard_receive", "relative_compare"):
            add(u["id"] + "/" + op, op)
    return rows


STAGES = ("prepare", "solve", "replay", "receive")


def stage_operations(config, stage):
    names = dict(prepare={"generation_load", "generate_P0", "capture_vae_load", "capture", "head_prepare"},
                 solve={"q0", "target", "solver", "prediction", "handoff"},
                 replay={"replay_vae_load", "native_replay"})
    assigned = set().union(*names.values())
    return [row for row in operation_plan(config) if row["id"] in names.get(stage, set())
            or (stage == "receive" and row["id"] not in assigned)]
