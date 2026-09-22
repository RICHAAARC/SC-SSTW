# Window-State-MSE-V1 paired run specification

Status: frozen implementation specification for the isolated
`dev/window-state-mse-v1` candidate.  It does not authorize or report a GPU,
model, Colab, or real-media run, and it does not change `main`.

## Fixed experiment

- Reuse the existing four prompt/seed cases, key, model revision, 320x512
  resolution, 181 frames, 8 fps, 50 steps, guidance, and negative prompt.
- The two calibration cases generate only `OFF`.  Each of the two evaluation
  cases generates, from one new initial noise and one common state-44 latent
  plus complete scheduler snapshot, exactly five forks:
  `OFF`, `LEGACY_SINGLE46`, `MSE_SINGLE46`, `LEGACY_MULTI44_46`, and
  `MSE_MULTI44_46`.
- `OFF` is physically generated once per evaluation source and shared by the
  old/new writer comparisons.  A shared OFF reference is never presented as
  two independent samples.
- Keep control indices 44/46 and the current cumulative native-response budget:
  each SINGLE uses R* at step 46; each MULTI uses R*/2 at steps 44 and 46.
  Reuse the existing `_marker_branch`, direction preparation, controlled step,
  and cumulative accounting.  Do not implement a second scheduler loop.
- The LEGACY forks use the existing fixed-marker tanh-mean objective.  The MSE
  forks use `window-state-mse-v1`; their live clean leaf and objective gradient
  are recomputed independently at every controlled step (44 and/or 46).
- Keep `FULL`, `DELETE90`, and `SPEED5_4` unchanged.

The fixed denominator is 4 sources, 12 physical source-arm instances, 36 saved
arm-views, and 144 phase encodes.  A complete run has these exact calls:

| call | expected completed |
| --- | ---: |
| generation | 4 |
| transformer | 496 |
| scheduler step | 248 |
| zero shadow step | 12 |
| unit response probe step | 12 |
| clean leaf backward | 12 |
| VAE decode | 12 |
| MP4 save | 36 |
| MP4 read | 36 |
| VAE encode | 144 |

The scheduler count is `2*50 + 2*(44 + 5*6) = 248`; classifier-free guidance
makes the transformer count `2*248 = 496`.  The controlled forks contribute
`2*(1 + 1 + 2 + 2) = 12` shadow, probe, and backward calls.  Attempted and
completed counts are both persisted.  Any shortfall remains a failed/unrun row
and cannot shrink the denominator.

## One observation, three blind receivers

For each of the 36 saved arm-views, read the MP4 once and call
`encode_four_phases` once.  Persist the four phase tensors once, then pass that
same in-memory observation mapping to these parallel receivers:

1. `ORIGINAL`, using the unchanged `main.tube_state.fixed_key` search and score.
2. `C1_MATCHED_CONFIRM`.
3. `C2_STATE_CONFIRM`.

Port the accepted diagnostic implementation from
`diagnostics/project-status-20260921/flow_fixed_key_v1_20260922T014420975188Z/receiver_split_v1/split_receiver.py`
to the pure method module `main/tube_state/fixed_key_split_receiver.py`.  Remove
the absolute repository path and `sys.path` mutation only.  Preserve the fixed
checkerboard A/B partition, A-only 4,284-path selection, one selected B
confirmation path, fixed post-search reference diagnostics, FULL/PARTIAL3
rules, clipping, observer recurrence, event cost, ranking, and tie breaks.
Neither split-receiver failure may remove the ORIGINAL result or any receiver
slot.  Do not change `main/tube_state/fixed_key.py` or the original receiver.

Selection receives only the four observations, key-derived codebook, and the
fixed receiver specification.  Writer objective, writer records, truth,
source-arm label, and reference view/path never enter path selection or score.
Declared source IDs may be used only by the outer calibration aggregation.
Truth and fixed-path alignment are joined after blind scoring for reporting.

Freeze three independent thresholds, one per receiver, using only this run's
two calibration OFF sources.  For each calibration source, take the maximum
over its three fixed views; the threshold is the maximum of those two source
statistics plus `1e-6`.  Do not copy an old threshold, use evaluation OFF in
calibration, combine receiver scores with OR, or select the best receiver.

Each receiver has 36 view decisions and 12 source-arm decisions; across three
receivers the fixed decision slots are 108 and 36.  Per receiver these split as:

- calibration OFF: 6 views / 2 source-arms;
- evaluation OFF: 6 views / 2 source-arms;
- each writer family (LEGACY or MSE): 12 marked views / 4 source-arms.

Thus the unique evaluation marked denominator is 24 views / 8 source-arms.
The shared evaluation OFF has 6 views / 2 source-arms and is referenced by both
paired writer comparisons without duplication.

## Persisted result and claim boundary

Preallocate every source, arm, view, phase, receiver, view-decision, and
source-arm-decision slot as `NOT_RUN`; preserve every failure and missing value.
For every receiver report blind scores, frozen threshold, signed margins,
decisions, winning view, and old/new paired differences separately for SINGLE
and MULTI on the same source.  Keep fixed-reference/window alignment strictly
post hoc.  Also retain source/noise/state-44/scheduler-snapshot identities,
writer objective identity, control and cumulative records, nominal terminal
evidence, saved-media paths/hashes/frame counts, phase paths/statuses, call
counts, logs, source SHA/file hashes, and resource records.

The result may establish only same-run paired development evidence after the
user executes it.  It must not claim quality, FPR, generalization, payload
capacity, or scientific success.  Process completion alone is not a method
PASS.

## Runner and notebook handoff

Implement a new paired config/runner and leave the old runner/config usable.
The runner keeps the current fresh child-process generation/media split,
visible progress logs, offload/resource handling, and a new output directory.
CPU/fake/static validation may check topology, exact call accounting, shared
snapshots, receiver blindness, preallocated failures, and report schema; it
does not execute the model or media path.

Build one fixed Run-all Colab notebook from the successful dependency and
fresh-subprocess path in `scripts/build_flow_fixed_key_notebook.py`.  Its first
cell is exactly:

```python
from google.colab import drive
drive.mount('/content/drive')
```

The notebook has no mode control or GPU whitelist, uses a fresh directory under
`/content/drive/MyDrive/Video-WM/Window-State-MSE-V1`, binds an immutable source
commit, and runs the paired config in a fresh subprocess with visible argv,
installation, checkout, progress, and result paths.  Validate only notebook
JSON/AST/schema, fixed source binding, empty outputs, and a stubbed non-GPU
workflow locally.  The user performs the Colab/GPU run.

Publication order is: approve this specification; implement and CPU/static
validate; review one frozen tree; commit source as S; obtain push approval;
publish S; bind the notebook to S and commit it as N; then verify remote S/N and
that remote `main` is unchanged before returning the immutable Colab link.
