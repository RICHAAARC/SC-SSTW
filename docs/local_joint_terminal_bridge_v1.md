# Fixed saved-JOINT terminal bridge

This is a user-run diagnostic of the JOINT terminal point from run
`20261009T132316545817Z`. It is not a reconstruction of the original step25–49
conditional-clean states. The implementation is on the B development branch;
publication and independent review belong to the parent audit.

Open `notebooks/local_joint_state_payload_v1_terminal_bridge_colab.ipynb`, select
a CUDA runtime, and Run all. The notebook embeds its runtime closure. No Git
checkout, immutable identity, exact dependency-version match or GPU-name gate
is needed. The original B dependency repair and process-group logging/cleanup
are reused; codec installation and pipeline import are removed. No automatic
retry, scan or experimental mode switch is provided.

The fixed inputs are the original `run/joint/terminal_latent.pt` and
`run/joint/float_rgb.pt` under the source run's Drive directory. Both are loaded
with `torch.load(..., map_location='cpu', weights_only=True)` and checked for
the actual finite FP32 tensor contract: `[1,16,46,40,64]` normalized latent and
`[181,320,512,3]` clamped RGB. Missing or incompatible float input stops the
dependent run as an engineering failure; there is no extra decode fallback.
Input files are read-only and outputs must be outside their directories.

The fixed sequence is:

1. Read the saved base RGB using the original known-grid reader. Apply the
   original carrier once, with the original key, message `8001a55a`, rho `.5`,
   four ROIs and 22 eight-frame segments, then read the candidate.
2. Load one frozen FP32 Wan VAE at the original model revision. Compute
   `Ebase = E(base)` and `Ecandidate = E(candidate)` using posterior mode and
   the original mean/std normalization. Set `raw = Ecandidate - Ebase` and
   retain `residual = z_T - Ebase`.
3. Perform exactly four decodes, in order: `D(Ecandidate)`, `D(z_T + raw)`,
   `D(z_T + mask(raw))`, `D(z_T + cap(mask(raw)))`. The mask is the original
   latent support and the shrink-only FP64 global cap is fixed at `1`.

Complete cost is **2 encode + 4 decode + one VAE load**, with **0 DiT, native
scheduler, codec and backward calls**. Negative or missing-support statistics
do not gate later executable stages. A failed decode retains its failed slot
and the later independent declared decodes are still attempted; there is no
retry. An encode failure leaves dependent decodes missing. Runtime/load/input
failures are not valid scientific negatives.

Every layer retains 88 known-grid windows × 16 pairs, including zero support,
failed and missing evidence. Each raw chip carries Eplus/Eminus/total energy,
q, support and failure counts. The unchanged original posthoc reducer supplies
all 22 offset correlations, c0/max-other/gap and 32 fragment-bit signed means
with fixed evidence counts `[24,24,20,20]`. Its legacy full-directory validator
is adapted with 680 explicitly unobserved specs in memory; these are not read,
persisted as observations or counted in the denominator. Truth is applied only
after the 88 raw rows have been saved.

Default-off observer hooks capture the actual original carrier's FP32 preclip
ROI outputs and each decode's actual preclip RGB. They receive detached copies
and cannot change the normal output. Thus candidate and all four decodes have
pre/post-clamp known-grid reads, unweighted and energy-weighted tile errors,
clipping counts and clamp-adjustment L2, with **no additional model call**.
The saved base has only postclip pixels; its preclip record is explicitly
`saved_base_preclip_unavailable`. Norms and RGB differences are auxiliary and
do not stand in for carrier support or signal retention.

`paired_comparisons.json` contains right-minus-left metric and per-chip
increments for adjacent bridge stages, every decoded stage versus base, and
each available pre/post-clamp pair. All 1,408 chip slots and 55 summary metric
slots are retained per comparison, including missing deltas. Per-fragment
weakest-bit reports retain ties and missing bits. There is no OFF or wrong-key
search, parameter scan, selective seed, changed budget or new acceptance gate.

Output is a fresh UTC directory under
`MyDrive/Video-WM/Local-Joint-State-Payload-V1-Terminal-Bridge/`. Return the whole
directory even if the notebook stops. It contains setup/source/environment and
execution receipts, log/failure records, `run/result.json`, per-layer raw and
metric JSON, paired comparisons, and saved encodings/differences/residual.
`COMPLETE` means the engineering sequence and records completed, not scientific
PASS. `actual_model_execution=INJECTED_TEST_DOUBLE` is reserved for CPU tests.

Model-call intent and attempts are durably saved immediately before each VAE
method entry; returned calls are saved before adapter/reader processing. They
are distinct from valid normalized tensors or completed observations. The
notebook reaps the child/process group before taking over a leftover RUNNING
record. It retains completed observations, marks in-flight completion unknown,
and never fabricates counts. Abrupt loss of the entire Colab kernel cannot run
this takeover; the last durable RUNNING record remains available for audit.

The standalone command requires only the embedded/source tree and dependencies:

```bash
python -m experiments.wan_state_clock.local_joint_terminal_bridge_v1_run \
  --config experiments/wan_state_clock/configs/local_joint_terminal_bridge_v1.json \
  --output /path/to/new-terminal-bridge-output
```

The already-executed original B notebook and companion ZIP remain byte-for-byte
unchanged. Their test now validates the historical artifact against its own
embedded manifest and companion, while the new notebook is checked against the
current bridge builder and source closure. Default carrier/VAE numerical paths
are unchanged; the two hooks are diagnostic additions only.

Local validation uses CPU arithmetic, model doubles, a no-Git source copy and
sequential notebook stubs. The previously downloaded actual terminal latent can
be read on CPU to validate its input contract. The 355,862,092-byte source float
has not been downloaded or fully inspected locally; the notebook performs that
contract check when the user runs it. No real model, GPU, dependency installation
or Colab execution is part of this delivery. Real memory/latency and scientific
bridge behavior remain unmeasured.

Interpretation must remain bounded to this one terminal point. Read candidate,
posterior roundtrip, raw reinjection, mask and cap in order, but retain all
non-monotonic or jointly negative outcomes. Even a complete positive bridge
would not locate the original conditional-clean, CFG/native or trajectory
failure, establish blind message recovery/FPR, or authorize retuning.
