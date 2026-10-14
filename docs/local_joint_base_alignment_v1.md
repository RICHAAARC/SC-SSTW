# Local Joint BASE Alignment V1

This fixed diagnostic measures two independent decoder VJPs at the same saved
JOINT BASE terminal. `aggregate_payload` is the mean message-aligned margin over
the historical roster `[0,1,4,5,7,9,12,13,16,17,18,20,27,29,30,31]`.
`state_G16` is `correlation[0]-correlation[16]`. A fresh readout can have a
different nonpositive roster or active wrong offset; those differences are saved
without changing either scalar or failing the run.

The CPU report projects both unnormalized masked gradients onto three fixed unit
directions: `M(raw)`, `M(raw-reconstruction_residual)`, and saved
`CONT_FINAL-BASE`. It separately records masked direction norms, pairwise unit
direction cosines, raw scalar slopes per unit direction, gradient norms, and
dimensionless gradient/direction cosines, plus the two-gradient Gram and cosine.
A zero masked gradient is complete:
its slopes are zero and its cosines are null.
If a saved direction itself is zero, its unit direction and slope are null and
reported as undefined rather than fabricated as zero.

The nominal new budget is one VAE load, two serial gradient decodes, two VJPs,
92 forward and 92 nominal recompute chunks, 176 windows, 2,816 chips, and 110
metrics at one source point. Ordinary decode, encode, DiT, native/scheduler and
codec calls are zero. Each complete unit is committed immediately; 60-second
snapshots, attempts, failures, observed replay counts, and interrupted RUNNING
calls are retained. A repeated Run all skips complete units and needs no model
after both are saved.

The notebook uses the existing Six-Bit-Final `fixed_inputs.pt` fields `base` and
`final`, plus the completed Terminal-Bridge `tensors/raw.pt` and
`tensors/reconstruction_residual.pt`. These paths are editable locators. Source
versions and environment details are optional location notes, not execution
gates. Return the whole stable task directory.

Completion means that the two fixed observations and CPU projections were saved.
There is no sign, margin, protection, scientific-success, or candidate gate. The
result cannot determine per-bit common improvement, sacrifice of an individual
bit, protection of the other 16 bits, ROI-specific gradient conflict, finite-step
wrong-branch switching, blind recovery, media survival, or generalization.
