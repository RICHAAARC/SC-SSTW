# SC-SSTW main project contract

## Authorization and status

`main` is the sole authoritative implementation branch. Historical branches
remain reproducibility records and must not be imported into main. Method
execution authority remains scope-specific. Implementation completion and
evidence completion are separate: architecture/test PASS cannot turn recorded
diagnostics into FPR, generalization, paper, or Flow-writing PASS.

## Method safeguards

- Use the fixed state-clock arm denominator and persist every success, failure,
  and missing observation.
- Keep receiver search blind to message truth and writer evidence; join truth
  only for post-search reporting.
- Message 0 and message 1 use the same condition-specific persisted
  save/readback chain and candidate budget. NORMAL has one lossy save; RESAVED,
  DELETE, and REPEAT each have a matched second lossy save before readback.
  Record calls, seed, config, source, and output paths.
- A future notebook must mount Drive in cell zero using the exact two lines,
  bind a published immutable GitHub source SHA, and persist progress during its
  authorized run.

## Architecture

- `main/tube_state` is pure method code and imports no runtime, experiments, or
  governance modules.
- `runtime/wan` contains shared VAE/generation/media adapters and may not
  import `experiments.wan_state_clock` or any historical runner.
- `experiments/wan_state_clock` is the only formal orchestration layer. The
  legacy `run.py` and integrated `integrated_payload_run.py` are separate,
  explicit entrypoints; neither imports a historical experiment runner.
- Governance is a local check layer, never a method-runtime dependency.
