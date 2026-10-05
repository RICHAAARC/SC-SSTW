"""Adopted G/T writer helpers; unchanged public codebook and blind receiver."""
from __future__ import annotations
from dataclasses import replace
import math
import numpy as np
from . import video_trajectory_payload_framewise_sync_v1 as sync
from . import video_trajectory_payload_temporal_constraint_v1 as temporal


def public_protocol(profile):
    if profile not in ("G", "T"):
        raise ValueError("fixed G or T entrypoint required")
    return replace(sync.PUBLIC, received_frame_counts=(181, 177, 89) if profile == "G" else (181, 177))


def _m05_rows(source, key, public):
    rows = []
    for b in range(46):
        start, stop = 4*b, min(4*b+4, 181)
        sign = sync.sync_sign(key, b, public)
        for y, x in sync.spatial_patch_coordinates(public.latent_height, public.latent_width, public):
            direction = sync.patch_direction(key, b, y, x, public).astype(np.float64)
            before = float(np.sum(source[start:stop, :, y:y+4, x:x+4].astype(np.float64)*direction)*sign)
            rows.append(dict(source_tubelet=b, source_start=start, source_stop=stop,
                patch_y=y, patch_x=x, sync_sign=sign, signed_projection_before=before,
                raw_target_delta_l2=max(0.0, 0.5-before), target_margin=0.5))
    return rows


def apply_temporal_writer(source, key, public=sync.PUBLIC):
    source = sync.validate_source_latent(source, public)
    rows = _m05_rows(source, key, public)
    delta0, constructed, construction = temporal.construct_temporal_constraint_delta(rows, key, public)
    written = (source + constructed.astype(np.float32)).astype(np.float32)
    actual = written.astype(np.float64) - source.astype(np.float64)
    full_rows, partial_rows = [], []
    for row in rows:
        b, y, x = row["source_tubelet"], row["patch_y"], row["patch_x"]
        start, stop = row["source_start"], row["source_stop"]
        direction = sync.sync_sign(key,b,public)*sync.patch_direction(key,b,y,x,public).astype(np.float64)
        block = (slice(start,stop),slice(None),slice(y,y+4),slice(x,x+4))
        target = float(np.sum(delta0[block]*direction))
        ideal = float(np.sum(constructed[block]*direction))
        applied = float(np.sum(actual[block]*direction))
        full_rows.append(dict(row, constraint_increment=target, constructed_increment=ideal,
            applied_float32_increment=applied, constructed_constraint_residual=ideal-target,
            applied_constraint_residual=applied-target, applied_minus_constructed=applied-ideal,
            signed_projection_after=float(np.sum(written[block].astype(np.float64)*direction))))
        if b in (0,44):
            left,right = max(start,1),min(stop,178)
            ages = slice(left-start,right-start)
            visible = direction[ages]
            support = (slice(left,right),slice(None),slice(y,y+4),slice(x,x+4))
            rho = float(np.sum(visible**2))
            baseline_q = float(np.sum(delta0[support]*visible))
            ideal_q = float(np.sum(constructed[support]*visible))
            actual_q = float(np.sum(actual[support]*visible))
            partial_rows.append(dict(source_tubelet=b,patch_y=y,patch_x=x,source_ages=list(range(left-start,right-start)),
                rho=rho,m05_ideal_increment=baseline_q,constructed_increment=ideal_q,
                applied_float32_increment=actual_q,constructed_minus_m05=ideal_q-baseline_q,
                applied_minus_m05=actual_q-baseline_q,applied_minus_constructed=actual_q-ideal_q,
                constructed_q_over_rho=ideal_q/rho,applied_q_over_rho=actual_q/rho))
    return written, dict(status="COMPLETE",method_version=public.method_version,key_id=sync.key_identifier(key),
        target_margin=0.5,temporal_lambda=1.0,source_shape=list(source.shape),projection_rows=len(rows),
        active_projection_rows=sum(row["raw_target_delta_l2"]>0 for row in rows),
        constraint_increment_l2=math.sqrt(sum(row["raw_target_delta_l2"]**2 for row in rows)),
        constructed_delta_l2=float(np.linalg.norm(constructed.ravel())),
        applied_float32_delta_l2=float(np.linalg.norm(actual.ravel())),
        constraint_increment_is_not_constructed_or_applied_l2=True,
        constructed_structure=temporal.perturbation_structure(constructed),
        applied_structure=temporal.perturbation_structure(actual),
        construction_summary={k:v for k,v in construction.items() if k!="projection_rows"},
        max_constructed_full_residual=max(abs(row["constructed_constraint_residual"]) for row in full_rows),
        max_applied_full_residual=max(abs(row["applied_constraint_residual"]) for row in full_rows),
        rows=full_rows,crop177_partial_rows=partial_rows,
        partial_projection_preservation_guaranteed=False,truth_used=False,message_used=False)


def write_condition(source, key, condition, public=sync.PUBLIC):
    """Every invocation receives the original source, never another arm's output."""
    if condition == "T05_TEMPORAL_CONSTRAINT":
        return apply_temporal_writer(source,key,public)
    targets = {"M1_FRAMEWISE_SYNC":1.0,"M05_FRAMEWISE_SYNC":0.5}
    written, receipt = sync.apply_projection_margin(source,key,public,target_margin=targets[condition])
    actual = written.astype(np.float64)-source.astype(np.float64)
    receipt["applied_structure"] = temporal.perturbation_structure(actual)
    return written, receipt
