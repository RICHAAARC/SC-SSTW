# Current method status

Updated 2026-10-04. The research mainline is **local tube carriers + state-space
temporal synchronization**, written through multiple genuine generation steps.
Generation sampling time and output-video time are distinct axes.

## What main currently provides

- Shared Wan runtime and the independent GROW fixed reference entrypoint.
- The retained early four-bit / 44-46 engineering integration and terminal
  state-clock entrypoint for reproduction.
- No imports of the unclosed local-state development experiments.

The verified GROW run demonstrates native MULTI generation control, real
VAE/media persistence and saved-MP4 blind repeated-payload recovery for one fixed
source. MULTI and LAST each recovered all 32 bits. MULTI includes the last step;
necessity or superiority of MULTI is not claimed and is not a prerequisite for
the proposal. This full-video geometry-specific receiver provides no temporal
alignment. [Run details and evidence](grow_video_reference_v1.md).

## What remains open

| Mechanism | Current boundary |
|---|---|
| Local spatiotemporal support | OLD8 has a defined local construction in dev; GROW's global repeated carrier does not replace it. |
| MULTI + local time states | OLD8 has same-version terminal and DIRECT positive path evidence. |
| MP4 blind time correspondence | Unclosed: OLD8 correct-key ABS/DIFF ranks terminal 1/1, DIRECT 1/1, RAW420 3/2, MP4 22/30. |
| State-space synchronization | Historical finite-path/observer results are carrier-specific; a general same-chain receiver with declared edit bounds, ambiguity and rejection remains open. |
| Time-dependent segment payload | Repeated payload does not establish that synchronization helps attribution. |
| Segment/sequence aggregation | Missing support, overlapping evidence, message conflict and rejection require same-chain validation. |
| Temporal edits | Crop, deletion, repetition and declared speed changes need current-version evidence; arbitrary inserted content and interpolation are separate models. |
| Affine-invariant synchronization | Optional extension when justified by observations; not a mandatory prerequisite. |

The two-spatial-copy candidate did not fix this gap: terminal 10/3, DIRECT
19/7, RAW420 27/40 and MP4 34/30. Its negative begins before media saving.
DWELL4 has only CPU construction checks: 174 ideal paths distinguishable, but
minimum template separation is one third of OLD8; no media result exists.
These candidates stay in development, with no success assembled across branches.

## Early integration retained as engineering history

The four-bit RM payload, 44/46 trajectory controller, saved edits, four-phase
VAE receiver and calibration/aggregation code are retained at their existing
entrypoints. Their local/fake tests check implementation and interfaces. The
historical two-message writer/receiver positives do not validate that complete
four-bit integrated chain. See [technical record](integrated_payload_v1.md)
and [immutable delivery record](integrated_payload_delivery.md).

## Completion criterion

The same real version must connect multi-step generation writing → VAE/media
survival → independent blind reception → local state/path inference →
time-dependent segment evidence → payload attribution, sequence aggregation and
necessary rejection. Local peaks, path ranks, payload recovery and execution
counts are reported separately. A calibrated population FPR, broader quality
claims and generalization require evidence beyond these fixed mechanism examples.

This main integration adds no GPU/Colab run, new threshold or new candidate.
[Development evidence index](development_evidence_index.md).
