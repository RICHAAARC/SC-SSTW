# Current formal method

The formal method writes a projection margin before the first Wan VAE decode
into the normalized terminal latent. It encodes a finite circular state-clock,
then performs a fixed bounded blind read over persisted receiver observations.
The core is in `main/tube_state`; shared Wan adapters are in `runtime/wan`; the
sole formal orchestration is `experiments/wan_state_clock`.

The runner accepts only `state_clock_v1`. It holds fixed arm and receiver-encode
denominators, persists running/failure/missing states, and applies truth only
after blind ranking. Message 0 and message 1 use identical condition-specific
save/readback and candidate budgets; NORMAL is saved once, while RESAVED, DELETE,
and REPEAT receive a matched second lossy save before readback.

Historical AISB, C2A, C2T1, projection-only fallback, and compatibility
interfaces are intentionally absent from authoritative main. Existing real Wan records are
fixed-condition diagnostics; this source adds no GPU rerun or scientific
conclusion.
