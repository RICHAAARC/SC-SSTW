# Paper Results V1 validation

## 2026-10-10 companion-source notebook revision

The two Run-all notebooks now fetch the ordinary public
`notebooks/paper_results_v1_companion.zip` selected by editable `SOURCE_REF`.
They contain no embedded B64 source, expected ZIP digest, per-file manifest, or
Git-clean admission flow. A usable edited source directory is reused. Corrupt
ZIPs, path traversal, missing executable source entries, unreadable model/media
inputs, method violations, and fixed-denominator violations still fail.

The final scoped command was:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. <project-python> -m pytest -q \
  -p no:cacheprovider \
  tests/test_paper_results_v1_two_pilot_notebook.py \
  tests/test_paper_results_v1_attack_notebook.py \
  tests/test_paper_results_v1_real_eval.py
```

It completed with `46 passed, 2 deselected in 2.29s`. The tests parse every
generated code cell, require empty outputs and the exact two-line Drive mount,
inspect and execute source from a no-manifest companion ZIP, reject corrupt and
path-traversing archives, reuse an edited source directory, exercise the real
RGB8 reader with a differing optional digest, and run both
generated notebooks through their isolated no-model boundary stubs to final
evaluation/handoff. A final separate static pass parsed all Python entries in
the 56-file companion ZIP and confirmed GPU notebook metadata. No real Colab,
Drive, GPU, model, VAE, codec, media, dependency installation, or model/source
download was executed.

Validation is limited to CPU/static reporting behavior. No model, GPU, Colab,
Drive, media encode/decode, generation, or new scientific experiment ran.

## Focused tests

Command:

```bash
PYTHONHOME=/home/richar/projects/Video-WM/framework/.conda \
PYTHONDONTWRITEBYTECODE=1 \
/lib64/ld-linux-x86-64.so.2 \
/home/richar/projects/Video-WM/framework/.conda/bin/python3.13 \
  -m pytest -q --capture=sys -p no:cacheprovider \
  tests/test_paper_results_v1.py
```

Result: `12 passed in 0.08s`.

The cases cover retained failed/missing/excluded/unsupported slots, complete
manifest denominators, unevaluable planned pairs, duplicate slot and input
conflicts, unplanned saved observations, conditional-joint schema adaptation,
blind/truth separation, the static 44-slot historical manifest, and CLI use
from a copied source directory with no `.git`. They also exercise the strict
integer-32 manifest and saved-result contract, explicit key labels, pair
identity mismatches, malformed result objects and rows, non-finite JSON input,
overflowing finite-looking JSON numbers such as `1e400`, and report writing
after those failures are retained.

## Same-version review repairs

Independent A2/A3 review requested a narrow engineering repair pass. The
reporter now binds V1 to integer 32 in the manifest and saved
`planned_final_bits`, requires manifest `key_label` and enumerated `key_role`,
checks both against their saved evidence, and prevents RAW/SYNC
comparisons across different result, view, input, or key identities. Malformed
top-level data, payload/posthoc containers, individual rows, and non-finite JSON
are retained as failed or conflicting evidence rather than aborting report
generation. Recovery summaries separately expose full fixed, eligible, and
evaluable bit denominators. Unplanned observations are the union of saved
payload and posthoc locators, with both sides' presence and status recorded.
Physical counts are labeled as unique physical key-read identities rather than
media counts or receiver call counts. Recovery rows group on both key label and
key role, so two keys with the same declared role cannot be pooled.

## Native adapter and workflow tests

Command:

```bash
PYTHONHOME=/home/richar/projects/Video-WM/framework/.conda \
PYTHONDONTWRITEBYTECODE=1 \
/lib64/ld-linux-x86-64.so.2 \
/home/richar/projects/Video-WM/framework/.conda/bin/python3.13 \
  -m pytest -q --capture=sys -p no:cacheprovider \
  tests/test_paper_results_v1.py \
  tests/test_paper_results_v1_workflow.py
```

Result: `27 passed in 0.14s` (the 12 strict-report regressions plus 15 workflow
and native-adapter tests).

The workflow fixture ran two predeclared cases through the real scheduler: one
available inline CPU array and one missing file. The scheduler inserts one
explicit shared framewise encode row per case, so its fixed denominators are 18
main plan steps, 4 native jobs, 10 quality
pairs, and 30 cost rows. The available case produced 9 successful main steps,
2 successful native jobs, and 5 observed quality pairs. The missing case
retained 9 blocked main steps, 2 blocked native jobs, and 5 blocked quality
rows. VideoSeal retained a complete synthetic
`[2,5,1,1]` raw output without reduction; RivaGAN retained two complete 32-logit
frames and their native zero-threshold bits without a sequence reducer. The
same CLI linked the six-row synthetic strict 32-bit report under `main_report/`
and also ran from a copied source directory with no `.git`.

Additional tests verify dependency validation, exact native messages, callback
failure retention, absent-backend behavior, RivaGAN path mp4v/20fps disclosure,
truth-free extraction, declared pairing conflicts, and identical-array
`psnr_db=null` serialization. This repair pass also checks a single shared
latent encode with two independent deep-cloned decode inputs, a concrete
pinned-layout RivaGAN loaded-tensor backend, strict 32-logit frames, immediate
per-frame detach, VideoSeal message-builder tamper rejection before embed,
non-inlined embedded media, explicit lossless sidecars for large detect output,
tensor/array quality reduction with non-finite rejection, NATIVE_POST quality
pairs, adapter/method conflicts with retained rows, and explicit CLI factory
loading from a copied source directory. These are deterministic engineering
fixtures; they are not model, codec, quality, timing, or scientific results.

## Historical saved-result import

Input:

`/home/richar/projects/Video-WM/diagnostics/trajectory-conditional-joint-real-run-audit-20261008/raw/result.json`

Command:

```bash
python3 -m experiments.paper_results_v1.cli \
  --manifest experiments/paper_results_v1/historical_conditional_joint.manifest.json \
  --result historical_joint=/home/richar/projects/Video-WM/diagnostics/trajectory-conditional-joint-real-run-audit-20261008/raw/result.json \
  --output-dir docs/paper_results_v1/historical_import_example
```

Observed import: 44/44 logical slots were present, all 22 correct-key K0 slots
had zero final bit errors for the fixed `OKOK` message, and eight predeclared
non-FULL RAW/SYNC pairs were evaluable. Every one had RAW=0 and SYNC=0, so each
is `NO_BER_GAIN_OBSERVED`. The FULL181 pair is retained as an excluded geometry
control. The report distinguishes one seen source label, 44 logical slots, and
32 unique physical key-read identities. The whole run contains nine logical
views over eight physical input files; the eight primary synchronization cases
use seven of those inputs. The saved call receipts separately record 21 blind
Wan receiver encodes. All 15 available quality/call fields were imported;
the planned wall-clock cost remains `MISSING`, producing
`COMPLETE_WITH_RETAINED_ISSUES` without changing the 44-slot result denominator.

## Staged real-evaluation entry

Command:

```bash
PYTHONHOME=/home/richar/projects/Video-WM/framework/.conda \
PYTHONDONTWRITEBYTECODE=1 \
/lib64/ld-linux-x86-64.so.2 \
/home/richar/projects/Video-WM/framework/.conda/bin/python3.13 \
  -m pytest -q --capture=sys -p no:cacheprovider \
  tests/test_paper_results_v1.py \
  tests/test_paper_results_v1_workflow.py \
  tests/test_paper_results_v1_real_eval.py
```

Result: `37 passed in 0.76s` (the previous 27 checks plus 10 staged-runner
checks).

The unadopted ten-case proposal expands to 690 artifact rows, 1,600 main
receiver rows (51,200 bits), 180 baseline edit rows, 70 quality rows, and 110
phase-cost rows. Pilot rows remain separate from the eight confirmation
candidates. The manifest-derived receiver plan is 360 framewise sync encodes,
1,600 reads, 320 RAW `(case, arm, expanded map)` encode identities, and an
upper bound of 1,200 Wan receiver encodes after adding 880 non-RAW rows.

Focused checks cover no-`.git` preflight/plan/init, local-only VideoSeal model
construction with forced lossless detector sidecars, strict RivaGAN sequence
failure semantics, the exact 50-step no-payload sibling arithmetic, retained
phase failures, nine baseline logical views with eight deduplicated expanded
maps, and the full 36-observation blind-sync construction from declared
`map_id` values. Syntax compilation also passed for `real_backends.py`,
`real_eval.py`, and `real_cli.py`.

The current lightweight interpreter reports the real preflight as blocked: it
does not expose torch, NumPy, diffusers, transformers, safetensors, accelerate,
or OmegaConf, and proposal local snapshots/source/checkpoints are placeholders.
No dependency was installed, no model or checkpoint was imported or
deserialized, and no generation, VAE, baseline, media codec, GPU, Colab, or
Drive operation ran. `ffmpeg`/`ffprobe` path discovery is static evidence only.

## Same-version staged-runner review repair

Targeted command:

```bash
PYTHONHOME=/home/richar/projects/Video-WM/framework/.conda \
PYTHONDONTWRITEBYTECODE=1 \
/lib64/ld-linux-x86-64.so.2 \
/home/richar/projects/Video-WM/framework/.conda/bin/python3.13 \
  -m pytest -q --capture=sys -p no:cacheprovider \
  tests/test_paper_results_v1_real_eval.py
```

Result: `17 passed in 0.73s`.

The repair isolates invalid receiver receipts, baseline receipts, and native
sidecars to their fixed rows so evaluation still writes the full JSON and CSV
reports. VideoSeal now requires the detector's leading frame dimension to
match each declared edit. Expensive case phases accept exactly one attempt per
run, while the report-only evaluation phase can be regenerated and includes
the retained phase history. The proposed effective-32 reducers preserve the
VideoSeal `>0` and RivaGAN `>=0` native decisions, but an exact reduced zero is
reported as `UNEVALUABLE_ZERO_TIE` and never as exact recovery. These tests use
synthetic receipts and fake arrays only; no model, media, or codec ran.
The final focused case also derives each multi-case phase summary after every
transition: active work takes `RUNNING`, any terminal failure takes `FAILED`, a
completed subset with remaining planned cases takes `PARTIAL`, and per-case
failure histories remain unchanged.

## Proposal metadata closure

This docs-only pass pins the reviewed VideoSeal source/card/API identity,
separates the recommended native-bit-plus-tie presentation from the currently
implemented unadopted zero-tie rejection candidate, and predeclares the M05/K0
paper comparison readout. It changes no execution code or adopted setting.

`python -m json.tool` accepted `real_eval.proposal.json`. A standard-library
`real_cli --phase plan` expansion retained exactly 10 cases, 690 artifacts,
1,600 receiver rows (51,200 bits), 180 baseline rows, 70 quality rows, and 110
cost rows. No model test or model/media operation ran.

## Adopted method definition and fixed comparison tables

Affected CPU/fake command:

```bash
PYTHONHOME=/home/richar/projects/Video-WM/framework/.conda \
PYTHONDONTWRITEBYTECODE=1 \
/lib64/ld-linux-x86-64.so.2 \
/home/richar/projects/Video-WM/framework/.conda/bin/python3.13 \
  -m pytest -q --capture=sys -p no:cacheprovider \
  tests/test_paper_results_v1_real_eval.py
```

Result: `21 passed in 0.77s`. This is the prior 17 staged-runner checks plus
four affected checks for the adopted manifest, native-tie versus strict-tie
semantics, fixed comparison left joins/source denominators, and reopening a
legacy strict run state with no comparison manifest. The older 27 reporter and
workflow tests were not rerun because their paths were unchanged.

The adopted plan expands to the existing 690 artifacts, 1,600 receiver rows,
180 baseline rows, 70 quality rows, and 110 costs, plus 180 explicit paper
comparison rows. For each external method, confirmation contains 64 non-FULL
rows and eight separately labeled FULL controls; pilots contain 16 non-FULL
rows and two FULL controls and remain outside confirmation summaries. The
source summary keeps a fixed denominator of eight clustered non-FULL views and
separately reports observed exact successes, observed errors, unavailable
main/baseline rows, evaluable pairs, and the paired-difference range compatible
with unavailable evidence. A missing side never becomes a completed pair.

Static compilation passed for `real_eval.py`, `real_cli.py`, and the affected
test file. Standard-library JSON/config validation accepted both the new
adopted config and the historical proposal config. The adopted plan has 180
comparison rows; the historical proposal has zero, so a legacy strict run is
not silently reinterpreted as the native-tie paper comparison. The supplied
source-identity metadata parses with 9,971 enumerated plain-text files, 401 zip
text entries, and `NO_MATCH_IN_SEARCHED_LOCAL_HISTORY`; its status remains
unproven. No dependency was installed and no model, weight, source media, VAE,
codec, GPU, Colab, or Drive operation ran.

## Vote-tie evidence and partial-pair bound repair

Affected command:

```bash
PYTHONHOME=/home/richar/projects/Video-WM/framework/.conda \
PYTHONDONTWRITEBYTECODE=1 \
/lib64/ld-linux-x86-64.so.2 \
/home/richar/projects/Video-WM/framework/.conda/bin/python3.13 \
  -m pytest -q --capture=sys -p no:cacheprovider \
  tests/test_paper_results_v1_real_eval.py
```

Result: `27 passed in 0.83s`. These are the affected staged-runner tests; the
older reporter/workflow 27-test suite was not rerun. Saved 32-row
`original_readout.votes` now yields a separate main Counter vote-tie count in
receiver and comparison JSON/CSV. Legacy missing or malformed vote evidence
stays null with an explicit evidence status/reason and does not invalidate a
valid decoded result. Baseline reduced-soft-zero ties remain separately named.

Per-row exact-success difference bounds now use every observed side:
`[m-b,m-b]` when both sides are known, `[m-1,m]` when only main is known,
`[-b,1-b]` when only baseline is known, and `[-1,1]` when neither is known.
Source bounds sum their eight fixed non-FULL rows and cohort bounds sum source
bounds. The focused regression with one complete `+1`, one known-main failure,
and six double-missing rows is `[-6,7]`, replacing the loose and incorrect
`[-6,8]` result. No new inference, threshold, model, or media execution was
introduced.

## Fixed two-pilot Colab handoff

Targeted command:

```bash
PYTHONHOME=/home/richar/projects/Video-WM/framework/.conda \
PYTHONDONTWRITEBYTECODE=1 \
/lib64/ld-linux-x86-64.so.2 \
/home/richar/projects/Video-WM/framework/.conda/bin/python3.13 \
  -m pytest -q --capture=sys -p no:cacheprovider \
  tests/test_paper_results_v1_two_pilot_notebook.py
```

Result after the same-version notebook orchestration repair: `9 passed in
0.81s`.

The checks parse every notebook code cell with Python AST, require empty
outputs and the exact two-line first Drive cell, and freeze the two attempted
pilots plus eight non-executed confirmation IDs. They decode the embedded ZIP,
verify its ZIP and per-file SHA-256 values, reject `.git`, parse every embedded
Python source file, and run `real_cli --phase plan` from the extracted no-`.git`
copy. The expanded adopted denominator is unchanged: 10 cases, 690 artifacts,
1,600 receiver rows/51,200 bits, 180 baseline rows, 180 comparison rows, 70
quality rows, and 110 costs. Static checks also cover the fixed phase order,
one-pass failure retention, report execution after failed stages, official
source/model pins, isolated baseline dependencies, and the explicit historical
evidence boundary.

The added behavior stubs force a real `OSError` launch failure, verify that the
next independent phase and final report-only evaluation still run, and inspect
the immediately persisted attempt history. A separate `KeyboardInterrupt`
stub verifies that later expensive phases stop, the interrupted attempt and
fixed scope are saved, report-only evaluation is attempted in `finally`, and
the interrupt is re-raised. Summary stubs verify that pilot receiver, baseline,
comparison, and quality counts come from final `evaluation_report` rows when
available; fallback state is explicitly labeled unevaluated. Confirmation plan
counts and the report's missing-evidence projections are separate, and pilot
source-level comparison summaries are carried into the handoff.
The final regression constructs the actual adopted ten-case manifest, evaluates
an empty-evidence RunStore using the real standard-library report path, and
confirms four pilot source summaries plus both method-specific cohort summaries
(`videoseal` and `rivagan`). It also verifies that all 36 pilot comparison rows
are unevaluable and that the eight unexecuted confirmation cases retain their
immutable 1,280/144/144/56 receiver/baseline/comparison/quality plan counts.

The generated notebook code cells were also executed in order under a fresh
isolated `/usr/bin/python3 -I` kernel boundary stub, with an empty
`PYTHONPATH`, a non-repository working directory, and fake Drive, network, pip,
Hugging Face, model, and media boundaries. Embedded-source identity
observations, the standard-library plan/init/evaluate path, and the handoff
summary ran. The test confirms that the extracted portable root is activated
in the live kernel,
all portable CLI children execute from that root, exactly two pilot IDs reach
the phase scheduler, and Torch is never imported by the stub run.

The baseline-environment stub separately verifies dependency-resolution,
entry-import, and real-model-compatibility receipts for success, resolver
failure, and import failure. Notebook AST checks require the dedicated venv pip
commands to receive the recorded core constraints file, reject `--no-deps`,
and require VideoSeal's explicit `antlr4-python3-runtime==4.9.*` and
`PyYAML>=5.1.0` OmegaConf closure. They also require import-only probes for
`videoseal.utils.cfg.setup_model` and the RivaGAN pickle-compatibility classes.
No pip resolver, package installation, import probe, model, or checkpoint was
run locally; those operations occur only when the user runs the notebook.

## Optional reproduction metadata repair

This later local repair makes source commits, dirty state, manifests,
file/config digests, model revisions, and installed-version differences
best-effort observations throughout the two-pilot notebook and its real runner.
Usable cached source is preserved. Existing imports are tried before any
historical known-stack dependency repair. Corrupt ZIPs, unsafe extraction
paths, missing/unreadable runtime inputs, actual import/model-load failures,
invalid media/native-output shapes, and fixed method/roster/truth constraints
remain blocking.

Focused CPU/fake validation produced `30 passed in 0.80s` for
`tests/test_paper_results_v1_real_eval.py`, followed after the final notebook
rebuild by `13 passed in 0.97s` for the notebook file plus the two affected
native-adapter tests. Coverage includes changed/missing declared digests,
unavailable digest observation, altered Wan/framewise revision metadata,
config-hash difference on reopen, dirty cached source, absent portable
manifest, corrupt/path-traversing ZIPs, and direct native-adapter construction
with unknown source and weight identity. Digest-observer failure and optional
adapter metadata were tested separately; this is not a complete no-hash
loader-to-adapter model-load test. Static code review confirms that receiver
cache identity uses the observed content plus the actual edit map, with a
source-specific fallback when digest observation is unavailable; the tests do
not directly assert cross-source cache non-aliasing. Every generated
`python -c` probe payload is compiled in the isolated ordered-cell boundary
stub. No dependency installation, download, Drive access, codec, media, model,
VAE, or GPU execution occurred.

An already-running Colab job continues under the exact notebook/source version
with which it started. This repair is for later runs and does not mutate,
restart, or reinterpret that existing run directory.

## Same-run environment and failure-retention repair

The read-only audit at
`diagnostics/a-line-20261009T123350716100Z-f5e1f880-audit/` records two concrete
engineering faults in the completed user run. Both default
`python -m venv --system-site-packages` commands failed inside `ensurepip` but
left a `bin/python`; later baseline processes reported `No module named numpy`.
Both baseline-extract phases then reported `NameError: name 'self' is not
defined`, while the saved RunStore left those phases `RUNNING` and their costs
`PLANNED`. The original run is not changed or retried by this repair.

The generated notebook now creates a dedicated interpreter with
`--without-pip --system-site-packages`, verifies that its resolved prefix is
the requested dedicated venv and that it can actually import NumPy and Torch,
and rebuilds a stale partial venv when that probe fails or cannot start.
Dependency repair remains conditional on entry-import
failure and is targeted with the working main interpreter's
`pip --python <dedicated-python>` command; it does not fall back to running a
baseline under the main interpreter. `RunStore._phase_artifacts` is now an
instance method. A root phase error passes through `_run_timed`, preserving the
original exception while marking the phase, cost, every nine-row baseline
extract denominator, or all 160 receiver slots plus the 36 planned observation
artifacts as failed without deleting rows.

Focused CPU/stub validation produced `15 passed in 1.02s` for the generated
notebook tests plus both parameterized failure-chain cases, and the complete
`tests/test_paper_results_v1_real_eval.py` file produced `32 passed in 0.83s`.
The venv stub starts with a misleading partial `bin/python`, forces its runtime
probe to fail, verifies removal and a single `--without-pip` plus
`--system-site-packages` rebuild, compiles the actual NumPy/Torch probe, and
checks validated reuse. The ordered notebook boundary stub verifies both final
creation commands and compiles every generated `python -c` payload. No venv,
package, model, codec, media, GPU, or Drive operation was executed locally.

The real Colab/model workflow was not executed because this task forbids Drive,
GPU, model, VAE, codec, media, package installation, and network execution. The
ordered code-cell boundary stub above is engineering evidence only; it does not
claim a real Colab or model run. The unclosed validation gap is the real
user-run environment: snapshot download, current
VideoSeal/RivaGAN compatibility with the recorded modern main stack, actual
GPU peak memory, wall time, and resulting media/readouts. The notebook records
those operations and failures instead of claiming they passed. No dependency,
source checkout, checkpoint, or model snapshot was downloaded during this
validation.

## Fixed two-pilot temporal-attack handoff

The adopted attack manifest expands to 15 instances in nine condition groups.
The complete plan contains 2,400 main receiver rows, 300 external-baseline
rows, and 70 original-POST quality rows. The notebook attempt scope is 480
main rows/15,360 bits, 60 baseline rows, and 14 quality rows; the eight
confirmation sources retain 1,920 main and 240 baseline rows as
`NOT_EXECUTED_BY_NOTEBOOK`. The CPU plan check confirmed edit lengths
181/89/241/145/181/181/163×3/199×3/199×3 and all materialized random
positions.

Focused validation produced 9 passing stdlib tests and 4 passing tests under
the already-present archive NumPy runtime. It covers top-two monotone DP against
exhaustive enumeration, an exact unresolved tie, crop anchor/front-copy/tail
semantics, fixed denominators, pilot/confirmation rate separation, missing-row
retention, RAW survival when clock-model setup fails, variable codec command
parameters, weighted-truth spread, exact temporal-edit arithmetic, RivaGAN's
real phase-level ndarray conversion, and phase-level per-attack VideoSeal
sidecar uniqueness. The reference-flow fixture uses coordinate-aware integer
sampling to prove the adopted backward-flow sign, zero stable residual,
nonzero wrong-direction residual, an exact +20 flicker transition, and the
no-valid-coverage result.

The generated notebook has empty outputs and all code cells parse. Its code
cells ran in order in a fresh `/usr/bin/python3 -I` process with Drive,
network, pip, Hugging Face, model, and media boundaries replaced. That test
uses the notebook's real `logged` implementation with fake subprocess return
codes: VideoSeal's first entry probe fails, repair and reprobe succeed;
RivaGAN's repair reprobe remains failed but does not stop the 20 independent
pilot phase attempts or final fixed-denominator evaluation. It also verifies
both `--without-pip --system-site-packages` venv creations, constrained
`pip --python` repairs, quality import repair, Wan VAE-only snapshot patterns,
portable source activation, final report creation, and confirmation isolation.

No real model, VAE, GPU, codec, media, Colab, Drive, dependency installation,
network download, or external-baseline checkpoint load was performed. The
real compatibility, GPU peak, wall time, actual physical Wan-encode aliases,
MP4 sizes, LPIPS/flow receipts, and scientific recovery/quality results remain
execution evidence to be returned from the unique notebook run directory.

### Temporal-attack review delta

The strict two-pilot physical Wan-encode upper bound is 360: each of 120
observations can require one shared RAW map and two independently estimated
key-specific BLIND_PATH maps. The logical denominator remains 480 rows. Clock
scoring now isolates K0 and K1 after their shared framewise encode, so a
score/DP/sidecar failure for one key does not manufacture a failure for the
other. External failure retention closes both `PLANNED` and `RUNNING` owned
work while preserving completed rows and an already-recorded, more specific
internal failure.

Framewise and physical Wan call records are persisted at call start and again
on return or ordinary exception. The focused fixtures verify a completed first
call and an unfinished second call survive interruption for both layers. A failure before the
Wan VAE call is excluded from physical-call counts; a failure while moving an
already-returned result to CPU leaves the encoding call `COMPLETE`. The
generated notebook carries Colab `accelerator=GPU` and Python language
metadata, but this is a runtime hint rather than a GPU model gate.

The read-only saved-run audit at
`diagnostics/a-line-20261009T123350716100Z-f5e1f880-audit/` is the concrete
input-reuse evidence for both pilot PRE/POST artifacts. It reports 320 evaluated
legacy receiver rows and retained external-baseline environment failures. It
does not show that the new temporal-attack path ran. Delta validation produced
10 passing tests for the evaluator and generated-notebook boundary suite,
including an isolated fresh-kernel ordered-cell stub. No model, VAE, GPU,
codec, media, dependency installation, network, Colab, or Drive operation was
performed.
