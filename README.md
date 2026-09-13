# SC-SSTW

SC-SSTW is the governed main repository for research on affine-invariant,
public-only calibrated state-space synchronization watermarking.

## Current feasible starting point

main/sc_sstw/ holds the carrier-independent method interfaces;
experiments/feasibility/synthetic/ holds the constructed-channel bridge used
to validate them under declared temporal edits. It is explicitly not a video
watermark runtime or a formal detection claim.

The next scientific boundary is a real-video motion-subject observer and
frozen public-candidate acquisition. That work is not implemented here.

## Layout

- main/: minimal method interfaces and the synthetic mechanism package.
- runtime/: reserved for future real-model adapters.
- experiments/: protocols and reproducible synthetic runners.
- paper_artifacts/: future record-to-artifact rebuild layer.
- governance/: research-state and validation controls.

Run the lightweight suite with python -m pytest -q.
