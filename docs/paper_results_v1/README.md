# Paper Results V1

The original `cli` entry turns an explicit predeclared slot manifest plus saved
experiment outputs into deterministic JSON, CSV, and Markdown reports. That
reporting entry does not import or change generation, receiver, runtime, or
core mathematics. The older manifest-driven `workflow_cli` calls only
explicitly injected generation, framewise, codec, and native-baseline backends.

## Run

From a repository root or a source copy with no `.git` directory:

```bash
python -m experiments.paper_results_v1.cli \
  --manifest experiments/paper_results_v1/manifest.template.json \
  --result RESULT_ID=/absolute/path/to/result.json \
  --output-dir /absolute/new/output/directory
```

The template intentionally has an empty source roster. Before a real run, the
user must adopt the evaluation sources, arm definitions, run budget, and any
decision thresholds. Each planned payload slot then names a `result_id` and an
exact saved-output locator. Saved observations fill those rows by left join;
they never define or enlarge the denominator.

The runnable example is explicitly synthetic:

```bash
python -m experiments.paper_results_v1.cli \
  --manifest experiments/paper_results_v1/synthetic_fixture.manifest.json \
  --result fixture=experiments/paper_results_v1/synthetic_fixture.result.json \
  --output-dir /tmp/paper-results-v1-synthetic
```

The outer workflow has a separate empty-roster template and an executable CPU
fixture:

```bash
python -m experiments.paper_results_v1.workflow_cli \
  --manifest experiments/paper_results_v1/workflow_fixture.manifest.json \
  --fixture-backends \
  --output-dir /tmp/paper-results-v1-workflow
```

For an explicitly supplied real integration module, the CLI accepts exactly
one named zero-argument factory and performs no discovery, download, or model
selection:

```bash
python -m experiments.paper_results_v1.workflow_cli \
  --manifest /absolute/path/to/adopted-workflow.json \
  --backend-factory my_package.paper_backends:build_bundle \
  --output-dir /absolute/new/workflow-output
```

For that older workflow entry, the factory returns exactly `operations` and
`native_adapters` mappings.
`--backend-factory` and `--fixture-backends` are mutually exclusive. Loading a
factory is an explicit caller action. Its callback path contains no default
real-model factory; the separate staged `real_cli` below now provides explicit
local-only loaders.

The fixture uses the production plan expansion, dependency handling, artifact
registry, codec callback, native adapter, quality, cost, and strict-main-report
link paths. `--fixture-backends` is rejected unless the manifest says
`SYNTHETIC_FIXTURE_ONLY`. Without injected callbacks, planned steps remain
`SKIPPED_BACKEND_UNAVAILABLE` or `BLOCKED_DEPENDENCY`; the CLI does not wrap a
missing model or source as a successful run. `workflow.template.json` has no
source roster, seeds, budget, or thresholds; running it reports
`PENDING_EMPTY_ROSTER`, not a completed result.

## Staged real execution entry

`real_cli` is the concrete repository entry for a future user-selected local
configuration. It contains direct lazy loaders for the frozen Wan generation
and receiver paths, the frozen framewise VAE, VideoSeal local card/checkpoint,
and the RivaGAN local community checkpoint. It never downloads a model and it
has no default config. The checked
`real_eval.proposal.json` is explicitly unadopted; its ten cases are two
excluded pilots plus eight confirmation candidates.
`real_eval.adopted.json` freezes the user-adopted local method definition,
including the 2+8 roster, `A6D39C5E`, original `watermark`/`watermark-wrong`
keys, native-tie reducers, and paper-facing M05/K0 comparison rows. Its
`real_execution_authorized=false` field records that local method adoption is
not permission to run models or media. Usable local model/source paths remain
required execution inputs; digests and source/version differences are optional
recorded provenance and do not block solely because they differ or are absent.

The standard-library-only phases are safe to run in a source copy without
`.git` and do not import torch or load media:

```bash
python -m experiments.paper_results_v1.real_cli \
  --config /absolute/selected-real-config.json \
  --output /absolute/output \
  --phase preflight

python -m experiments.paper_results_v1.real_cli \
  --config /absolute/selected-real-config.json \
  --output /absolute/output \
  --phase plan

python -m experiments.paper_results_v1.real_cli \
  --config /absolute/selected-real-config.json \
  --output /absolute/new/output \
  --phase init
```

After the user supplies real local paths and parameters, each heavyweight
phase is a separate invocation with `--case-id`: `generate`, `decode`,
`framewise`, `baseline-embed-videoseal`, `baseline-embed-rivagan`, `codec`,
`quality`,
`baseline-extract-videoseal`, `baseline-extract-rivagan`, `receiver-sync`, and
`receiver-read`. The final `evaluate` phase aggregates all cases and takes no
case ID. Separate processes allow each model family to release memory before
the next phase. Failures update the fixed `run_state.json`; they do not remove
planned artifacts, receiver slots, or cost rows.
Each cost-bearing phase/case can enter `RUNNING` only once in a run; a failed,
complete, or already-running phase requires a new explicit output run rather
than overwriting its evidence. The report-only `evaluate` phase may be
regenerated and carries prior phase failures into the report.

Wan and framewise entries require usable local snapshot directories. Wan is
passed to the existing diffusers loader by local directory path; framewise uses
`local_files_only=True`. VideoSeal requires an explicit usable local source
tree, model card, and checkpoint; RivaGAN requires an explicit usable local
source tree and the declared community-checkpoint object. Actual digests,
declared digests, source commits, and dirty state are recorded when available,
but differences or missing optional identity metadata do not block a usable
source. Preflight checks module/path availability and records digest differences
without importing or deserializing models.

The real runner saves full RGB8 media with byte SHA-256, Wan/framewise latent
receipts, codec commands and readbacks, seven fixed POST quality pairs, full
RivaGAN frame logits, and complete VideoSeal detector arrays in a forced
lossless NPZ sidecar. Each baseline retains nine logical edit rows over eight
unique index maps. Blind receiver plans/readouts are persisted incrementally
before truth evaluation. Pilot and confirmation summaries remain separate.
See [evaluation_method_v1.md](evaluation_method_v1.md) for the adopted local
method, fixed denominator, effective-32 rules, cost accounting, and claim
ceiling. The bounded roster-history evidence is in
[source_identity_audit.md](source_identity_audit.md). The older
[evaluation_proposal_v1.md](evaluation_proposal_v1.md) and strict config remain
unchanged compatibility records.

`historical_conditional_joint.manifest.json` is a static 44-slot import map for
the already-audited conditional-joint development result. It was declared from
the frozen protocol layout, not generated from observed successful rows. It
retains 40 blind and 4 oracle rows and separates correct-key, wrong-key, and
oracle summaries. The historical run as a whole has one seen development
source, nine logical views over eight physical input files, 21 blind receiver
encodes, and 32 unique physical key-read identities. The eight primary K0
synchronization cases use seven of those physical inputs. These counts are
different objects and must not be pooled as independent sources.

The checked example outputs are under `historical_import_example/`. Its result
binding checks the original source SHA, content identity, and config identity;
the generated report also records the imported file path and byte SHA-256.

## Outputs and interpretation

- `report.json` contains the manifest denominator, blind rows, truth rows,
  conditional recovery summaries, predeclared pairs, measurements, input
  identities, and any unplanned saved observations.
- `blind_rows.csv` excludes posthoc truth and message-recovery fields.
- `truth_rows.csv` contains postseal recovery evaluation.
- `paired_sync_effect.csv` includes every predeclared pair, including incomplete
  ones. A RAW=0/SYNC=0 pair is `NO_BER_GAIN_OBSERVED`.
- `measurements.csv` retains missing quality and cost fields.
- `report.md` is a compact human-readable rendering of the same records.

The workflow CLI writes `workflow_report.json` and Markdown plus `plan.csv`,
`artifacts.csv`, `quality.csv`, `cost.csv`, and full `native_records.json`. When
declared, it also writes the unchanged strict 32-bit reporter under
`main_report/`. Every planned transform, codec call, native job, quality pair,
and cost row stays in its fixed denominator when its source or backend is
missing.

The staged real evaluator additionally writes `comparison_rows.csv` and
`comparison_source_summaries.csv`. These left-join every predeclared
source/view/baseline row to the single selected M05/K0 main readout and retain
missing or failed evidence. Confirmation summaries use eight sources as the
independence denominator; their eight non-FULL view rows are clustered within
each source. A partially observed pair contributes the strict range implied by
the known side rather than a blanket plus/minus one; cohort bounds sum source
bounds. FULL is reported separately as a geometry control. Main Counter-vote
ties and external reduced-soft-zero ties have separate count, policy, evidence
status, and semantics fields.

The six slot states are `OBSERVED`, `FAILED`, `MISSING`, `EXCLUDED`,
`UNSUPPORTED`, and `CONFLICT`. The full fixed denominator is every manifest
slot, including excluded and unsupported rows. Conditional tables separately
show the full fixed, eligible, and evaluable denominators. Failed, missing, and
conflicting rows remain in the full and eligible denominators and cannot become
success; excluded and unsupported rows remain in the full denominator with
their reasons. A pair-level exclusion, such as FULL181, does not remove its
payload rows from the 44-slot denominator.

`global_00` / FULL181 has only one public offset and is a geometry/payload
control. The historical manifest retains its rows but excludes that comparison
from the eight-case synchronization pair denominator. Historical P0 already
contains generation payload, so P1/P0 and M05/P0 quality rows are not an OFF
watermark distortion measurement.

The historical 32-bit rows recover the fixed configured `OKOK` message. They
do not establish arbitrary 32-bit message capacity. The separately adopted
new evaluation message and roster do not randomize, replace, or retroactively
reinterpret the payload when importing that old evidence.

## Implemented paired-control workflow boundary

The workflow manifest now declares source/content identity, noise ID and seed,
one shared codec callback, and PRE/POST artifact IDs. It expands each case into
`OFF_NATIVE` and `PAYLOAD_NATIVE` (P0). P0 PRE enters one explicit shared
framewise encode step. P1 decodes an independent clone of that frozen latent;
M05 writes and decodes a separate clone of the same latent. The scheduler does
not feed P1 pixels into M05 or encode P0 twice. Every POST artifact used in a
comparison traverses that case's same explicit codec callback, while PRE,
latent, and POST remain distinct. Native baselines likewise run as native
embed, the shared codec callback, then native extract. Matching noise/codec
fields and callback receipts verify the declaration and code path; they are
not a claim that two media files were physically identical without inspecting
the files.

Quality rows report absolute paired MSE, RMSE, and PSNR. They never subtract
PSNR or present pairwise metrics as an additive decomposition. Identical arrays
use `psnr_db=null` with `IDENTICAL_INFINITE`, avoiding non-finite JSON. P0 still
contains payload. Historical P0 cannot fill OFF, and its old POST references do
not establish the new same-codec comparison. CPU quality calculation accepts
numeric lists and tensor/ndarray values; tensor values are detached and moved
to CPU, and array-backed values are checked and reduced without converting a
full video to nested Python lists. Quality pairs may reference main POST or
native-baseline NATIVE_POST artifacts from the same declared case.

The current main prepare path already creates P0 and uses one shared framewise
encode for separate P1/M05 decodes and codecs; its fixed runner has no OFF arm.
This outer workflow supplies callback and saved-report binding points without
changing that runner or claiming that a new-source full configuration has run.
## External baseline inventory status

The local archive contains historical RivaGAN, VideoSeal, and framewise HiDDeN
adapters plus 7,000 score records per method. This first implementation does
not import those JSONL records into the main 32-bit recovery table. The raw
record decisions remain `pending_threshold_calibration`, while a separate
historical aggregation table contains thresholded summaries; neither artifact
has been rebound to the new protocol or re-audited here.

The inventory locations are:

- `SyncTube.zip::SyncTube/configs/baselines/external_{rivagan,videoseal}_source.json`
- `SyncTube.zip::SyncTube/experiments/baseline_comparison_gate/{rivagan,videoseal}_adapter.py`
- `SyncTube_results.zip::SyncTube_results/numeric_conclusion_data/source_evidence/baseline_comparison_gate/external_*/records/baseline_formal_score_records.jsonl`
- the sibling `tables/baseline_comparison_table.csv` files

RivaGAN uses the community `Peachypie98/RivaGAN` 32-bit checkpoint; its source
pin is
`efffa72a4ca46d4d5051f6970c96424c2cdab441`. VideoSeal, pinned at
`e00b98c7ca77eb1fb5b9b68260e7c6c8fc207a84`, expands its message to the
model-native length (commonly 256) while historical scoring compares an
expected prefix, so a 32-bit/native-capacity mapping requires a user decision.
Official commit `870ca7fb33578b90f14c602016b6c2788096226e` differs from that
historical pin only in four copyright headers; its card, `cfg.py`, and
algorithm are unchanged. This resolves the source/API comparison, while the
historical adapter-to-score-record identity remains separate.
The archived source adapter metadata says `formal`, while the score records say
`real_smoke_adapter`; exact same-version binding is therefore not established.
The HiDDeN record path has a known 30 predicted bits versus a recorded
`compared_length=32` inconsistency and stays a fallback inventory item. No
wrapper is copied and no historical TPR/FPR is promoted into this report. This
historical-record import still waits for adopted capacity, roster, codec, and
score semantics.

`native_adapters.py` now provides real call-through interfaces for explicitly
loaded backends. VideoSeal calls `embed(..., msgs=..., is_video=True)` and
`detect(..., is_video=True)`. Before embed it verifies that the caller's
message builder preserved every declared bit exactly at shape `[1,K]`, and it
records the exact submitted bits, shape, and count. Embedded media are retained
as workflow artifacts and represented in JSON only by shape, dtype, count, URI,
and a real file/backend identity when one exists; `imgs_w` is never expanded
into report JSON. Detect output retains the complete raw values, nested shapes,
and element counts. Large outputs require an explicit lossless sidecar writer
and URI rather than truncation. The adapter applies no spatial, temporal,
capacity, or 32-bit reduction. The interface references are pinned
[`videoseal.py`](https://github.com/facebookresearch/videoseal/blob/870ca7fb33578b90f14c602016b6c2788096226e/videoseal/models/videoseal.py),
[`cfg.py`](https://github.com/facebookresearch/videoseal/blob/870ca7fb33578b90f14c602016b6c2788096226e/videoseal/utils/cfg.py),
and the [256-bit card](https://github.com/facebookresearch/videoseal/blob/870ca7fb33578b90f14c602016b6c2788096226e/videoseal/cards/videoseal_1.0.yaml).
The card names `y_256b_img.pth`, but this work did not download it or invent a
digest; every real backend records the actual local file digest when available.
A declared digest difference is disclosed rather than used as a load gate.
VideoSeal's documented and code-comment layouts have varied; an output such as
`T,1+K,H,W` is preserved rather than silently reduced. Construction records
source version, model version, weight identity, and detected-output layout
metadata when available; unknowns stay explicit rather than being inferred
from the adapter name.

RivaGAN retains every decoded frame's soft logits and its native per-frame
zero-threshold bits and requires each frame to have strict shape `[32]`. The
[pinned upstream implementation](https://github.com/DAI-Lab/RivaGAN/blob/efffa72a4ca46d4d5051f6970c96424c2cdab441/rivagan/rivagan.py)
writes OpenCV `mp4v` at 20 fps in path `encode` and decodes BGR frames with
`value/127.5-1`; the path wrapper exposes that hidden transport and marks it
incompatible with the shared-codec claim. `RivaGANLoadedTensorBackend` wires an
explicitly loaded encoder/decoder to BGR uint8 frames, `[1,3,1,H,W]` normalized
model tensors, and `[1,32]` messages. It clamps encoder output to `[-1,1]` and
uses the pinned implementation's `(x+1)*127.5` uint8 truncation, without a
hidden file codec. Extraction detaches and returns each frame immediately so a
video does not retain every autograd graph. Neither adapter receives truth
during extract, and the adapter output itself remains unreduced; the staged
evaluator applies the separately selected native-tie rule afterward. RivaGAN
construction requires explicit source, model, weight, color, normalization,
transport, and codec-comparability metadata.

`archive/SSTW/external_baseline/source_registry.json` and its
`official_eval_adapters` directory also declare VidSig, VideoShield, VideoMark,
REVMark, and WAM-frame entries. This pass checked those inventory entry points
and declarations only; it does not claim that their historical evaluations ran
successfully. `videoseal_official_runtime.py` is a CWD layout helper and cannot
by itself establish a complete VideoSeal baseline. Cross-model generation and
inversion cost and their intended use remain pending, so these entries are not
added to the default minimal comparison set.

## Adopted local evaluation definition

The earlier inventory stage identified eight confirmation sources and 48
shared-codec roundtrips as a possible engineering batch, while leaving the
capacity mapping unresolved. It remains historical planning context, not an
active default.

The adopted definition is specified in
[`evaluation_method_v1.md`](evaluation_method_v1.md): two excluded pilots,
eight separately summarized confirmation sources, VideoSeal effective-32
`j mod 32` channel repetition, and RivaGAN all-frame soft means. Exact zero
keeps each baseline's native decision and records `tie_count`; the old strict
`UNEVALUABLE_ZERO_TIE` rules remain available only through the historical
proposal config and do not change old saved-run meaning. The paper-facing main
row is M05/K0: GLOBAL for non-FULL GLOBAL views, PATH for SINGLE_JUMP, and RAW
FULL in a separate table, giving 64 non-FULL pairs per external method and
eight FULL controls over the eight confirmation sources. The complete 160 rows
per source remain controls and ablations.

This local method adoption does not authorize execution. VideoSeal native-K,
threshold/FPR work, strength or reducer scans, extra attacks/baselines, and
automatic confirmation execution remain outside the adopted method. The
self-contained [`paper_results_v1_two_pilot_colab.ipynb`](../../notebooks/paper_results_v1_two_pilot_colab.ipynb)
is a fixed Run-all handoff for the two excluded pilots only. It embeds the
actual evaluator/runtime source closure and all ten adopted cases, but attempts
only `pilot_01` and `pilot_02`. The eight confirmation cases are not executed
by this notebook; full-denominator evaluation projects their absent evidence
for receiver/baseline/comparison to disclosed failure/unevaluable rows, while
quality retains its recorded state. None are treated as attempted model
failures. See
[`colab_two_pilot_handoff.md`](colab_two_pilot_handoff.md). This repository pass
did not execute the notebook, models, media, codec, GPU, Colab, or Drive.
