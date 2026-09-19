"""Actual same-history native-step control energy, not terminal distortion."""
import math

def cumulative_energy(step_rms):
    values=[float(x) for x in step_rms]
    if any(not math.isfinite(x) or x<0 for x in values):raise ValueError('finite nonnegative response RMS required')
    return sum(x*x for x in values)

def matching_scale(target_energy,unit_response_energy):
    target=float(target_energy);unit=float(unit_response_energy)
    if not all(math.isfinite(x) and x>=0 for x in (target,unit)):raise ValueError('invalid response energy')
    if unit==0:
        return dict(status='ZERO_BUDGET' if target==0 else 'UNMATCHABLE_ZERO_RESPONSE',scale=0. if target==0 else None,target_energy=target,unit_energy=unit)
    scale=math.sqrt(target/unit)
    if not math.isfinite(scale):return dict(status='UNMATCHABLE_NONFINITE_SCALE',scale=None,target_energy=target,unit_energy=unit)
    return dict(status='MATCHABLE',scale=scale,target_energy=target,unit_energy=unit)
