# Local Joint Posterior Bridge V1

This user-run Colab evaluates one fixed candidate from the original saved
JOINT/BASE latent:

`FP32(BASE + 0.99998 * normalize(M(raw - reconstruction_residual)))`.

`lambda=1`, the original public mask, message, key and `rho=0.5` are fixed. The
run does not continue from BALL and performs no scan, solver, endpoint
projection, encode or VJP. BASE and the completed terminal-bridge CAPPED
readouts are the primary reused controls. Only POSTERIOR_BRIDGE is newly
decoded.

The nominal and maximum planned new model work is one VAE load and one ordinary
decode with 46 chunks. Encode, VJP, DiT, native/scheduler steps and codec calls
are zero. The candidate latent is committed before model loading. Candidate RGB
and scoring are committed immediately, and task state is checkpointed every 60
seconds. A completed rerun loads no model; an interrupted incomplete decode may
repeat while cumulative calls, chunks and failures remain recorded.

Saved BASE and candidate RGB use the existing decoder-output clamped float
`[0,1]` semantics. Numeric residual metrics use the full difference range after
that RGB clamp. The `[-0.25,0.25]` residual range is only for rendered displays.
All 32 payload margins with four ROI contributions, the correct and all 21 wrong
state correlations and G, plus aggregate, framewise, temporal and peak quality
records are retained. The old CAPPED control has its saved aggregate quality
summary and is not re-decoded.

The fixed Drive output is
`MyDrive/Video-WM/Local-Joint-State-Payload-V1-Posterior-Bridge/resumable-posterior-bridge-v1`.
The Run-all notebook is
`notebooks/local_joint_posterior_bridge_v1_colab.ipynb`; its portable source is
`notebooks/local_joint_posterior_bridge_v1_portable_source.zip`.

Local CPU construction on the saved tensors measured an actual FP32 BASE radius
of `0.9999800114596782`, zero change outside the mask, and finite output. Stub,
resume, notebook and isolated portable-import tests passed. No real GPU, model,
Colab, Drive-write or codec execution was performed locally.
