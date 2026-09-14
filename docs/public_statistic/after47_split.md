# One bounded split VJP audit

Reuse the existing after47 snapshot. Keep current mixed arithmetic. No prefix rebuild, no precision sweep, no new delta. Run the remaining solver once to get terminal latent u and decode once to RGB. Compute the loss-to-RGB cotangent once. Hold u and this cotangent fixed for two VAE VJPs. Hold the first VAE VJP fixed for two solver/Transformer VJPs from the same original state. This separates variation within each backward stage; it does not characterize every joint variation or prove acceptance stability.

Budget: three two-step solver forwards x2 CFG =12 transformer calls,3 VAE calls. Five autograd calls: one readout-only and four model VJPs. Replays at most240 blocks and26 chunks. No MP4,1800 seconds,zero retry. Preserve original settings and failures; no model/version/memory gate. One CPU test checks split vs coupled gradient and workload; no expansion of CPU matrices.

Stop after this audit and report concrete errors or the practical significance/limits of observed variation. Do not pursue bitwise equality across devices, auto-expand diagnostics, tune a direction or rewrite run08. If no actionable defect is identified, keep the fixed construction's result and return to the smallest method test; any changed update rule is a new explicitly described method construction.

## split02 residency repair

split01 on L4 (22.03 GiB available capacity) hit real CUDA OOM in the first VAE backward; allocator_cap_gib was null. The split entry unnecessarily retained both models on GPU. After the initial no-grad solver forward, move the idle Transformer to CPU. After both VAE backward graphs have been consumed and local references deleted, move VAE to CPU before restoring Transformer to its original device. Empty allocator cache only at those boundaries after releasing idle weights. Dtypes, model values, inputs, calls and scientific semantics are unchanged. This applies only to split VJPs, not the coupled full-graph control runner. Real L4 completion remains unverified.
