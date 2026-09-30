# State-path development diagnostic implementation V1

The complete fixed CPU run is available, but this emission model cannot uniquely
recover the full source correspondence tau. All30 conditions have one numerical
best equivalence class with multiple structurally indistinguishable paths.
This is a development limitation, not synchronization or edit-detection success.

## Scope and preserved mechanism

Only new pure path code, an offline runner/config, tests and documentation were
added. The MULTI writer, payload, pilot, generator, runtime and historical files
were not changed. Base is `9d37f9d250f1afc25b9f4ae06ad104b9fb8ee7f6`. The config
and mechanism document hashes were sent before path scoring and remain unchanged.
PREREGISTERED_DEVELOPMENT means fixed before this scoring operation only: the
whole source run had already been seen. It is not unseen-data preregistration
or independent confirmation.

The path retains one phase, R45/full or R31/crop, finite tau1..45, and at most
one stride-window repeat/skip. There is no edit penalty, advance preference or
path detection threshold. Numeric equality tolerance remains1e-12. The exact
(g, emitted-sign-sequence) grouping yields8 full and499 crop classes,12024
overall; every class has multiple paths. This pilot therefore cannot identify
a unique complete tau even under ideal observations in this path family.

Canonical lexicographic representation does not privilege the zero-edit path.
Every summary separately records canonical event, whether a top class contains
zero edit, whether all top paths require an event, and path-minus-zero-best.
All path results remain UNCALIBRATED_DIAGNOSTIC/accepted_payload=False. Fixed
phase/nominal stride4 edits do not validate arbitrary single-RGB-frame edits,
VAE receptive fields or dynamic phase changes.

## Complete own-run evidence

102/102 artifact hashes/public schemas,30/30 conditions and60/60 posthoc rows
completed. The full78126-entry catalog includes77862 scored entries and264
finite-support exclusions;1278 zero-edit paths remain. Every condition retains
its full equivalence membership/tau feasible sets and optimal-predecessor DAG.
The30 gzip raw files total4252784 bytes and are externally retained with all33
output files hashed in offline_result_manifest.json. No large results are
duplicated in the source patch. Raw hashes are identical before/after truth joins.

The DP maximum differs from enumeration by at most1.1102230246251565e-16;
all numerical top-path sets match. Zero-edit scores differ from the unchanged
global cosine by at most2.220446049250313e-16. Prior global0.5 decisions are
retained only as a separate legacy diagnostic, never reused for path acceptance.

Canonical paths show events in30/30 conditions, but four top classes contain
zero-edit paths: these are representation choices, not edit evidence. The other
26 have strictly higher fitted scores than the best zero-edit hypothesis.
The input views contain only full/crop observations, not actual repeat/delete
edits, so these fits cannot be reported as detected edits or FPR measurements.

For the five PILOT_MULTI/correct-key conditions:

|View|Canonical b / event / edge i|Path score|Path minus best zero-edit|Top paths|Top class has zero-edit|
|---|---|---:|---:|---:|---|
|FULL_RESAVED181|0 / REPEAT / 44|0.615707398|0.000000000|3|True|
|CROP4_129|0 / SKIP / 21|0.667696255|0.008059954|9|False|
|CROP5_129|0 / SKIP / 21|0.648358305|0.007998357|9|False|
|CROP6_129|2 / SKIP / 20|0.671752823|0.004496235|9|False|
|CROP7_129|3 / SKIP / 20|0.672604554|0.009380715|9|False|

FULL's repeat representative is equivalent to its true zero-edit hypothesis.
All four unedited crops instead favor SKIP-containing classes; their known
true zero-edit paths are not top. This exposes no-edit fitting bias/structural
ambiguity in this development model. It does not validate real edited-MP4
recovery. There is no tuning or rescore with adjusted penalties/thresholds.

Posthoc payload diagnostics aggregate the same saved primary R windows in
received order. They neither remap payload via fitted tau nor enter the path
API. Repeated payload cannot establish synchronization necessity. Actual
time-dependent payload and real edited-MP4 validation remain future work.

## Verification and handoff

15 targeted CPU tests passed in18.87s with zero skips. They cover independent
enumeration, old zero-edit equivalence, finite repeat/skip boundaries and
backtraces, near numerical ties with final catalog filtering, constant-code
and same-sign nonidentifiability, explicit zero-energy/missing/NaN handling,
truth/path metadata exclusion, j32 isolation, fixed failure rosters, entire
source failure, raw-before-posthoc isolation and DP disagreement fail-closed.

The offline run finished in8.49s. The borrowed environment was not modified.
No model, GPU, VAE, FFT, video read/encode or Colab call was made. The main
session's independent pure-code review receipt is linked by path/hash in
local_validation.json; formal review of this complete frozen delivery is pending.
No commit or push is performed at this freeze.
