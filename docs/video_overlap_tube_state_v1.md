# Overlapping spatial tube state V1 — stage A

This freezes a concrete real-tensor construction before candidate scoring. No
state/path scoring, new model, GPU, VAE or media experiment has run. The old
MULTI/bridge/path implementations and their negative results remain unchanged.

## What the archive actually implements

SyncTube.zip and SyncTube_results.zip were inspected with zipfile.read, never
extracted over a workspace. The archived real candidate declares length4 and
4x4 spatial patches, but no temporal stride override: embedding.py:52 and
tubelet_partition.py:51 therefore use stride4, not time-overlapping stride1.
Its synthetic configuration enables scale search; the real candidate does not.

codebook.py:102–135 generates sync signs by temporal index, payload signs and
full normalized random spatial-temporal directions by tubelet index. All three
include sample_id through _build_integer_seed:162–175. payload_bits=64 is a
config field; executed per-tube payload signs are keyed pseudorandom values,
not evidence of user-bit recovery. Directions are multidimensional, although
combined_codes is one payload sign times one temporal sync sign per tubelet.

embedding.py:121–148 sequentially mutates the actual latent tensor to enforce
projection margins. This is useful evidence of a real tensor path, but those
old nonoverlapping defaults do not prove overlap normalization or identifiability.
evidence.py:449–474 reconstructs the codebook using sample_id and a reference
shape from mechanism_trace. The receiver reads before/after mean projections
(:990–999) and adds their support term (:1045–1075); the real candidate weight
is0.09. Attack clip metadata changes search-rule selection (:849–854), and
reference/original frame count expands offset range (:1161–1198). Ground-truth
offset itself is used for reporting, not the direct ranking expression in
synchronization.py:119–159; this distinction does not remove the other inputs.
The archived PASS concerns its recorded protocol and is not evidence for the
current stricter blind API. Archive hashes/source-entry hashes are separately
recorded; old results are preserved without reinterpretation as this method.

## State, blocks and genuine overlap

Time t=1..45 denotes Wan latent regular windows, nominally four newly supplied
RGB frames per window. Latent index0 is excluded. This is not a single-RGB-frame
clock and not a VAE receptive-field statement.

Let b(t) be the six-bit MSB-first binary representation of t-1. Append two
nonconstant check bits: XOR of all six and XOR of b0,b2,b4. Public-key-derived
permutation/sign masks turn these eight bits into s_t in{-1,+1}^8. The exact
domain-separated SHA256 byte rules are in the config; no sample identifier
enters them. The public block relation is h_t,i[k]=s_t[2i+k]. These are two-
component spatial partitions plus joint parity constraints, not four independent
complete state copies. Missing one block is tested and may cause ambiguity.

Use channel4 and four fixed, disjoint4x4 latent spatial patches. Each patch has
a keyed orthonormal Hadamard16 basis. Its first eight selected columns are
U_i,a,k, with age a=0..3 and component k=0..1. A tube starting at t writes its
h_t,i into physical times u=t+a, on different age subspaces at each time. All
tubes are scatter-added into ONE actual tensor before any receiver projection.

At physical u the compound state is [s_u,s_(u-1),s_(u-2),s_(u-3)]. Active
groups are a=0..min(3,u-1). Thus the same voxels contain a sum of different
state/age directions, not repeated identical targets averaged into one write:

    W[u,4,S_i] = alpha * sum_(active a,k) U_i,a,k * h_(u-a),i[k].

No writes occur after u45; source states below1 give boundary zeros, and states
after45 are unsupported. State t appears min(4,46-t) times, so tail states do
not have four complete copies. Temporal repetition supplies redundancy of each
component across physical times/age subspaces. The simultaneous age lanes are
not separate independent temporal samples or independent noise realizations.

In tube-index notation, let I=(start a, spatial block i) and r be its age.
The physical contribution is h_(a+r),I=alpha*sum_k U_i,r,k*s_a[2i+k].
Different overlapping tube identities therefore carry known delayed states,
not four copies of the current global s_u. Their sum is the augmented state
x_u=[s_u,s_(u-1),s_(u-2),s_(u-3)]. Source-clock inference compares actual
received composite coordinates with the public x_tau family; it never treats
a delayed component as an independently supplied clean observation.

## Same-tensor observation and budget

Hadamard columns are orthonormal within each actual patch. There are
8*(45+44+43+42)=1392 active coefficients, each magnitude alpha=1/sqrt1392.
The synthesized full-source tensor therefore has L2 norm1. The unique physical
support is45*4*16=2880 cells; it is not1392 independently stored observations.
There is no local tube/block normalization and no extra strength per coverage.
Crop inputs retain this SAME full-source alpha and are never renormalized by
received length. Direct real spatial synthesis avoids an unrealizable FFT-real
target or unaccounted conjugate completion.

The receiver obtains r[j,i,a,k]=dot(U_i,a,k,Y[j,4,S_i]) from the actual received
tensor. Its local model is alpha*h_(tau_j-a),i[k] plus projected host/noise,
with zeros at source boundaries. Age a refers to the original compound state
at tau_j, not necessarily the previous received row j-a after an edit. Reader
arguments are received tensor, public key/protocol and received-coordinate
availability only. sample_id, source offset, attack metadata, writer pre/post
statistics, payload truth and separately invented clean tube vectors are banned.

The CPU additive fixture writes Y=host+W, so its actual pilot delta norm is1.
This is distinct from C's generator control budget. C keeps MULTI25..49, fresh
conditional clean autograd, Flow delta/sigma, FP32 CFG5 and native history.
Its pilot loss is mean squared residual of1392 active basis coefficients, with
raw eta174; the actual combined pilot gradient delta is capped at L2 norm1,
never amplified. The existing payload branch on channels0..3 keeps its separate
FFT-real mean loss/eta5520. Add both actual tensors; measure total norm/RMS and
cross term. Disjoint channels give sum of squared norms up to numeric precision.
There is no claim of matched cross-arm total budgets, no old joint47104 mask
reuse, and no automatic strength increase from extra overlapping tubes.
Record target norm, every actual control delta, and25-step accumulated norms
separately. CPU target norm1 is not predicted real-video signal strength or
survival against real latent host interference. C remains unexecuted.

## Fixed score and incomplete observations

Candidate costs are float64 mean squared residuals across the available
projected coordinates r[j,i,age,k]. The numerator sums
(r-alpha*s_(tau_j-age)[2i+k])^2; boundary states below1 have mean0 and still
count as observed coordinates. M=8*sum availability[j,i] is fixed per condition
and shared by every candidate. Neither templates nor observed crops are
renormalized. No edit penalty, advance prior or decision threshold is added.

Structural equivalence is the exact integer predicted-template tuple in
received j/i/age/k order after applying the same availability mask, with values
-1,0,+1. Equal templates share one numerical cost. Floating synthesis roundoff
must not split mathematically equal templates. Absolute cost-minus-min<=1e-12
defines numerical ties between classes, not confidence or a detection threshold.
Keep every tied class/member and per-index feasible tau; canonical lexicographic
tau/event representation never breaks the scientific ambiguity.

Missing (j,i) removes its8 age/component coordinates from score and signature,
but preserves received index j and hidden tau_j. It is not a deleted window.
If M=0, retain all slots/classes/exclusions with costs unavailable and report
INCOMPLETE/NO_OBSERVATIONS, no canonical path. Nonfinite available ROI, projection
or reduction similarly reports INCOMPLETE/INVALID_OBSERVATION; do not score a
remaining subset. Explicitly unavailable coordinates need not contain numbers.
If available projected energy is exactly zero, retain finite SSE costs/top cost
ties for audit but report NO_ENERGY and no canonical/accepted path.

Every condition remains UNCALIBRATED_DIAGNOSTIC, accepted_payload=False and
state_path_accepted=False. Even a unique finite-model argmin is neither real
synchronization acceptance nor confidence. Negative/no-edit fitted paths and
the unchanged zero-edit baseline remain visible.

## Fixed next CPU boundary; not yet scored

The config fixes the public code, four tensor clips, zero/delete/repeat edit
positions, five ideal Gaussian-noise seed/strength cases and three missing
patterns (including loss of a spatial block). The primary1680 conditions and
all failures remain. All noise is applied to the shared physical tensor before
projection; these noise levels are not a model of real VAE/host noise.

Before noise, enumerate all bounded zero/one-edit paths for lengths30,31,32,44,
45,46 and retain all exact observed-template equivalences:20250 catalog rows,
3049 structurally valid paths and17201 finite-boundary exclusions. Canonical
choice/prior is never an identifiability proof. No edit penalty, offset repair
or post-score parameter scan is permitted. Zero-edit baselines and no-edit
false-event fitting are separate reported outcomes.

Unknown insert is not repeat. An exact copy inserted as unknown content is
observationally identical to a repeated window; preserve that label ambiguity.
Zero-watermark null insertion and midpoint interpolation are separate explicit
counterexamples/interfaces, not free wildcard emissions or hidden deletion of
observations. The24 such records are outside the primary repeat/skip family.
Their known generating family belongs only in posthoc reporting. No arbitrary
insert or dynamic-phase single-RGB edit recovery is promised.

Reserve unused basis columns for future position-dependent block payload and
CRC-8 checks; currently disabled with no energy allocation. Estimated, oracle-
correct (posthoc only), fixed-wrong and no-alignment payload paths stay distinct.
Payload/checks never select this state path. Existing constant32-bit recovery
cannot show that synchronization is necessary.

Only after CPU structural review can a new namespace adapt the existing
whole-video VAE/media runtime. Real crops/edits require actual RGB save/readback
and complete-frame phase VAE encodes, not old latent/window reordering or
independent ROI encodes. No notebook binding, GPU execution or push is authorized
in this stage.
