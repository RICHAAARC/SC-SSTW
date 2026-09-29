# Video-Temporal-Sync-Bridge-V1 review closure

Candidate v2 has passed the local engineering milestone. A2, A3, A4, the main
review and A5 have closed their scoped reviews. This report is outside the
frozen 12 files. It does not change either version's snapshot, patch or test
log, and it supersedes only their historical pending-review status.

## Identity and version boundaries

- Branch: `dev/video-trajectory-blind-validation-v1`.
- Base: `91afbc71aee9f999ff6f7c1738c5ac0be1249a52`.
- Current snapshot: `snapshot_v2.json`, SHA256
  `2f37c6b2f0b43aec1d267254784b6368ed12ea30f098bd5e5124330ea4c761e5`.
- Current patch: `candidate_v2.patch`, SHA256
  `6a409be23d997f304a2c1cdbad91468f00b5569e811d02b04211a2c277a58e02`.
- Current 12-file aggregate:
  `8a7ffa6ef0cf83468eb349912d9b6fc65b5747b795424904ab1cc2ea55211ce6`.
  The snapshot defines this path/NUL/file-SHA/LF aggregate; it is not a Git tree.
- v1 snapshot/patch remain preserved. Its aggregate was
  `92d9ea3e3229740475ffc913abb8f0654402866404190110b32888845c2e0fcf`.

The v2 repair changed only the notebook dependency reference, generated
notebook, canonical checkpoint ordering, targeted tests and implementation
records. Pure method, Wan runtime, configuration and protocol document retain
v1 bytes. Earlier numerical evidence is inherited on that exact-byte basis;
it is not relabeled as a v2 full rerun. All candidate files are new relative
to the base; previous successful GROW implementations and evidence are intact.
The authorized local delivery is 17 files: 12 frozen files, four historical
snapshot/patch artifacts and this closure report.

## Reviews and repaired findings

| Role | Evidence and conclusion |
|---|---|
| A1, sole writer | v1 full target: 12 passed in 19.24s, zero skips. v2 scoped regressions: 4 passed, 10 deselected in 1.60s. CPU fixtures only; AST, notebook, whitespace and patch checks passed. |
| A2, independent method/numerical review | v1 PASS, inherited into v2 after identity/delta verification. Original payload update was bitwise identical; joint writer's original four-channel maximum difference was 2.9802e-8; independent NumPy adjoint maximum error was 2.6844e-8. |
| A3, independent runtime review | v1 identified the canonical/projection failure window. v2 P1/P2 closed; both canonical-before and canonical-after interruption boundaries were independently checked. No additional runtime blocker remained. |
| A4/root, scope and evidence synthesis | Verified all 12 v2 hashes, exact patch/tree and narrow delta; previous tracked files were unchanged. Review scope and evidence attribution passed. |
| Main review | Independently verified v1 identity and all 12 tests in 18.02s; v2 identity and four targeted tests in 1.60s passed. Canonical-first event/recovery ordering was checked. |
| A5, final milestone audit | Independent combined-failure exercise passed; exact preserved denominators and outcomes are recorded below. |

Independent reviewer outcomes were relayed by the coordinating session; they
are not additional A1 executions. Main-review receipts are in the parent
project under `diagnostics/temporal-sync-bridge-main-review-20260929/`:
`main_candidate_v1_review.json` and `main_candidate_v2_review.json`.

P1 was the nonexistent notebook requirements path. v2 uses the existing
`experiments/wan_state_clock/requirements-grow-video-reference.txt`; the static
regression extracts actual pip `-r` paths and verifies repository existence.

P2 allowed a hard exit after blind-file replacement but before result.json
committed, so recovery could overwrite a saved READ with stale PENDING data.
v2 makes result.json canonical, saves it before the blind projection, and
reconstructs the projection from that canonical state on recovery. A phase
interrupted before canonical commit is not falsely reported as READ. A phase
committed before projection interruption retains its 32 decoded bits. Existing
candidate-batch interruption coverage also preserves committed phase records.

## A5 combined failure evidence

The CPU/fixture scenario combined a PAYLOAD_LAST crop-save failure and a
PILOT_LAST crop g2/WRONG feature failure. All fixed 30 phase, 12 search,
324 candidate and 24 posthoc rows remained present with no PENDING rows.
Completed counts were:

| Artifact/operation | Completed / fixed |
|---|---:|
| Derived MP4s | 5 / 6 |
| Normalized phase artifacts | 11 / 15 |
| Phase payload observations | 21 / 30 |
| Pilot candidates | 205 / 324 |
| Searches | 9 / 12 |
| Posthoc evaluations | 18 / 24 |

The nine completed searches were normal NO_PILOT rejections, with no accepted
payload. Three SEARCH_INCOMPLETE results remained incomplete; oracle
observations did not promote them to success. Evaluation added zero FFT,
VAE or candidate-score computations. Blind-file bytes remained unchanged
through posthoc evaluation and quality reporting. Correct-key observations
survived reopening the disk records despite the separate wrong-key failure.
Quality performed five additional MP4 reads, counted separately and without
a threshold. Six terminal/key diagnostics remained a separate denominator.
These are failure-retention checks, not actual generated-video outcomes.

## Claim ceiling and handoff

The fixed source and OFF/PAYLOAD_LAST/PILOT_LAST construction remain unchanged.
The pilot threshold is public and uncalibrated; full-view localization is
trivial. The two positive and ten negative pilot controls are dependent.
PAYLOAD_LAST is payload-only, not unwatermarked; repeated payload recovery
on the wrong grid is not localization success. Terminal/oracle/fixed-g0
results cannot replace primary MP4 evidence. Quality metrics do not create a
quality PASS.

This closure establishes CPU/fixture engineering readiness only. There is no
new real-video, GPU, Colab, scientific, FPR or quality PASS. The notebook still
has `SOURCE_SHA=None`. Current authorization permits A1's single local source
commit only. The main session handles publication, verification of the
published immutable source SHA and notebook binding. A1 does not push, merge,
load models, execute GPU/Colab or run a new experiment for this closure.
