"""Truth-free known-grid reads and descriptive saved-terminal comparisons.

No new detector, selector, threshold or carrier is defined here.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

from main.tube_state import local_joint_state_payload_carrier_v1 as carrier
from main.tube_state import local_joint_state_payload_posthoc_v1 as posthoc

STAGES = ("base", "candidate", "posterior_reconstruction", "raw_reinjection",
          "masked_reinjection", "capped_reinjection")
VIEWS = ("postclip", "preclip")


def known_catalog(protocol=carrier.PUBLIC):
    return tuple(s for s in carrier.phase_window_catalog(protocol)
                 if s.phase == 1 and 0 <= s.slot < 22)


def read_rgb(rgb: Any, *, key: str, rho: float, protocol=carrier.PUBLIC) -> list[dict]:
    """Original 88-window readout, with truth-free tile-error sufficient statistics."""
    import torch

    rows = json.loads(json.dumps([r.to_dict() for r in carrier.observe_catalog(
        rgb, key, protocol=protocol, catalog=known_catalog(protocol))]))
    for row in rows:
        spec = row["spec"]
        chips = row["state_chips"] + row["payload_chips"]
        for chip in chips:
            ep, em = chip["energy_plus"], chip["energy_minus"]
            chip["energy_total"] = None if ep is None or em is None else ep + em
        try:
            y0, y1, x0, x1 = protocol.rois[spec["roi_index"]]
            pixels = rgb[spec["received_frames"], y0:y1, x0:x1].double()
            y = pixels[..., 0] * .299 + pixels[..., 1] * .587 + pixels[..., 2] * .114
            tiles = y.reshape(len(y), (y1-y0)//8, 8, (x1-x0)//8, 8).permute(0, 1, 3, 2, 4).reshape(-1, 8, 8)
            coefficients = carrier.dct2_ortho(tiles)
            for chip in chips:
                p, m = chip["plus_coord"], chip["minus_coord"]
                ep = coefficients[:, p[0], p[1]].square()
                em = coefficients[:, m[0], m[1]].square()
                energy = ep + em
                valid = torch.isfinite(energy) & (energy > 0)
                q = (ep[valid] - em[valid]) / energy[valid]
                weights = energy[valid]
                chip["tile_ratio_statistics"] = dict(
                    supported_count=int(valid.sum()),
                    hypotheses={str(sign): dict(
                        absolute_error_sum=float((q - sign*rho).abs().sum()),
                        energy_weighted_absolute_error_sum=float(((q-sign*rho).abs()*weights).sum()),
                    ) for sign in (-1, 1)},
                )
        except Exception as exc:
            row["tile_ratio_statistics_error"] = f"{type(exc).__name__}: {exc}"
    return rows


def evaluate_saved(rows: list[dict], *, key: str, message: bytes, rho: float,
                   protocol=carrier.PUBLIC) -> dict:
    """Adapter to the unchanged original reducer; no extra pixels are read.

    Its catalog validator expects 768 specs but only consumes 88 known-grid
    rows. The other 680 specs are explicitly unobserved in-memory adapters,
    never persisted or counted as measurements.
    """
    expected = {s.observation_id for s in known_catalog(protocol)}
    indexed = {r["spec"]["observation_id"]: r for r in rows}
    if len(rows) != 88 or set(indexed) != expected:
        raise ValueError("exactly 88 distinct known-grid observations required")
    full = []
    for spec in carrier.phase_window_catalog(protocol):
        full.append(indexed.get(spec.observation_id, dict(
            spec=json.loads(json.dumps(asdict(spec))), truth_used=False,
            adapter_only_unobserved=True)))
    result = posthoc.evaluate_observations(full, key=key, message=message, protocol=protocol)
    result["scope"].update(raw_windows=88, raw_chips=1408, unused_specs_are_not_observations=True)
    targets = carrier.state_matrix(key, 22)
    fragments = carrier.message_fragments(message)
    errors = []
    for row in rows:
        spec = row["spec"]
        for chip in row["state_chips"] + row["payload_chips"]:
            sign = (targets[spec["slot"]][chip["component_index"]] if chip["component"] == "state"
                    else 2 * fragments[spec["slot"] % 4][chip["component_index"]] - 1)
            stats = chip.get("tile_ratio_statistics", {})
            hypothesis = stats.get("hypotheses", {}).get(str(sign), {})
            count, energy = stats.get("supported_count", 0), chip["energy_total"]
            errors.append(dict(
                observation_id=spec["observation_id"], pair_index=chip["pair_index"], target_sign=sign,
                target_ratio=rho*sign, status=chip["status"], supported_samples=chip["supported_samples"],
                aggregate_q_absolute_error=None if chip["q"] is None else abs(chip["q"]-rho*sign),
                unweighted_tile_absolute_error=None if not count else hypothesis["absolute_error_sum"]/count,
                energy_weighted_tile_absolute_error=None if not energy or not hypothesis else
                    hypothesis["energy_weighted_absolute_error_sum"]/energy,
            ))
    result["target_errors"] = errors
    result["descriptive_condition"] = posthoc.descriptive_condition(result)
    result["weakest_bits"] = []
    for fragment in range(4):
        metrics = [m for m in result["payload"]["metrics"] if m["fragment"] == fragment]
        valid = [m for m in metrics if m["signed_mean"] is not None]
        minimum = min((m["signed_mean"] for m in valid), default=None)
        result["weakest_bits"].append(dict(fragment=fragment, minimum=minimum,
            tied_bits=[m["bit"] for m in valid if m["signed_mean"] == minimum],
            missing_bits=[m["bit"] for m in metrics if m["signed_mean"] is None]))
    return result


def paired(left: dict, right: dict, left_rows: list[dict], right_rows: list[dict], *,
           key: str, message: bytes, protocol=carrier.PUBLIC) -> dict:
    """Right minus left, retaining every expected chip and metric slot."""
    def index(rows):
        return {(r["spec"]["observation_id"], c["pair_index"]): c
                for r in rows for c in r["state_chips"] + r["payload_chips"]}
    a, b = index(left_rows), index(right_rows)
    targets = carrier.state_matrix(key, 22)
    fragments = carrier.message_fragments(message)
    result = posthoc.compare_arms(left, right)
    result["chips"] = []
    for spec in known_catalog(protocol):
        for pair in range(16):
            sign = (targets[spec.slot][spec.roi_index*8+pair] if pair < 8
                    else 2*fragments[spec.slot % 4][pair-8]-1)
            x, y = a.get((spec.observation_id, pair), {}), b.get((spec.observation_id, pair), {})
            deltas = {field: None if x.get(field) is None or y.get(field) is None else y[field]-x[field]
                      for field in ("q", "energy_plus", "energy_minus", "energy_total")}
            result["chips"].append(dict(observation_id=spec.observation_id, pair_index=pair,
                left_status=x.get("status", "MISSING"), right_status=y.get("status", "MISSING"),
                target_sign=sign, deltas=deltas,
                signed_q_delta=None if deltas["q"] is None else sign*deltas["q"]))
    return result
