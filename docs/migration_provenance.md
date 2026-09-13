# Feasibility Migration Provenance

## Imported scope

The synthetic package and feasibility runners were imported from the local
SC-SSTW-Feasibility working tree on 2026-08-30:

    archive/SC-SSTW-Feasibility/src/sc_sstw_feasibility/
    archive/SC-SSTW-Feasibility/experiments/run_aisb_probe.py
    archive/SC-SSTW-Feasibility/experiments/run_temporal_robustness_probe.py

The imported method boundary is public affine-invariant AISB acquisition,
public-only affine calibration, and state-constrained synchronization.

## Evidence ceiling

The source C2, C3 and C4 results are controlled synthetic mechanism evidence.
They are retained as rationale for the imported protocol, not as an SC-SSTW
real-video claim. No Drive package, MP4, GPU implementation, Wan adapter,
Flow-guidance carrier, structured-noise carrier, or learned observer was
imported.

## Deliberate adaptation

The initial package name sc_sstw_synthetic was subsequently split: generic
method interfaces are now main/sc_sstw, while constructed-channel-only support
is experiments/feasibility/synthetic. The two runner import roots were adjusted
to the SC-SSTW layout. The fast native scoring helper was excluded; the
migrated package is pure Python.
