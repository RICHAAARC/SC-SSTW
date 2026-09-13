# Synthetic Feasibility Runners

These runners reproduce only the controlled synthetic AISB mechanism:

- public affine-invariant burst acquisition;
- frozen ambiguity candidates before owner/wrong-key scoring;
- public-only affine calibration;
- state-constrained temporal synchronization;
- deletion, duplication and synthetic clock-distortion diagnostics.

They do not accept or generate video, invoke a GPU, calibrate a fixed false
positive rate, or establish a publication claim.
