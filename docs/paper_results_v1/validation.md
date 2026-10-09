# Paper Results V1 validation

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

Result: `3 passed in 0.15s`.

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

The notebook was not executed top to bottom because this task forbids Colab,
Drive, GPU, model, VAE, codec, and media execution. The unclosed validation gap
is therefore the real user-run environment: snapshot download, current
VideoSeal/RivaGAN compatibility with the recorded modern main stack, actual
GPU peak memory, wall time, and resulting media/readouts. The notebook records
those operations and failures instead of claiming they passed. No dependency,
source checkout, checkpoint, or model snapshot was downloaded during this
validation.
