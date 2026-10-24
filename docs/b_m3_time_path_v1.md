# B2 M3-151 component implementation

2026-10-24, B-THREE-IMPLEMENT-01. The current user adopted COMMON r1 / integrated
c0 and B2 r1. Their older proposed/read-only labels are historical. This component
implements the adopted B2 geometry and state decisions; it does not execute the
new DCT2 writer, a model, VAE, media, or the physical scientific experiment.

## Files and ownership

- `main/tube_state/b_m3_time_path_v1.py`: pure Q40 wrapper, calibration, local
  geometry/candidates, E2 and M3 compatibility.
- `runtime/wan/b_m3_observation_v1.py`: the12 received-only view plans and41-frame
  uint8 extraction. It neither imports nor calls an encoder/codec.
- `tests/test_b_m3_time_path_v1.py`: small synthetic CPU and static checks.

No shared module, `__init__`, existing runner/notebook/builder, original worktree,
or another group's file is changed. No new shared edit is required by this
component. The total implementer owns the DCT2/shared E/Q adapter, roster,
calibration/evaluation sequencing, c0 budgets, opaque namespaces, persistence,
failure journal, B3 binding, notebook and builder.

## Thin API: exact names match IMPLEMENTATION_HANDOFF

`score_family(phase_bank, background_model)`

- `phase_bank` is a mapping with exactly one int or string key for each p0..3;
  each value has `roi_q[10][4][44]`. Missing/duplicated phase representations
  return an unresolved full137-slot family.
- Delegates the arithmetic to the existing
  `b_background_normalized_match_v1.score_family`. No background fit, Cholesky,
  threshold, FFT or payload computation is added. NumPy is imported lazily via
  that pure method. The output drops the old empty `phase_local_bits` field.
- The common runner supplies the fixed Card102 mean/factor and **new DCT2-chain
  Q**. Historical tau values and the B2 paper's old FFT payload component budget
  are not used by this module. c0 governs the actual integrated budget.

`calibrate(null_families)`

- Accepts `{opaque_family_id: raw_score_family}` or a list of exactly90
  `{"family_id": opaque_id, "family": raw_score_family}` entries. IDs must be
  nonempty and unique within the submitted schedule; they are ordinary slot
  references, not hashes or provenance certification.
- The **outer scheduler** supplies the predeclared 2cal×3fixture×5null×3window
  roster. This function has no source/role/fixture/arm or message parameters and
  cannot independently establish that the caller selected the proper nulls.
- Returns `status, tau, tau_rule, maxima[90], missing_members, cause_set`, planned
  family/score counts90/12330, submitted count and extra slot indices. Missing or
  additional families, duplicated IDs, missing/duplicate/outside H members,
  nonfinite/out-of-range scores or upstream failures prevent freezing. No
  positive-floor, fallback or reduction of the90-slot denominator occurs.
- `path_calculations=0`. Calibration does not import/call the path domain.
  The caller keeps the raw supplied families/Q; extra unplanned family indices
  are diagnostic errors, not additional null scores to silently consume.

`prepare_local(family, threshold, query_id, received_query_start)`

- `query_id` is an opaque logical key/window ID. `received_query_start` is exactly
  0/53/106. Optional anonymous `input_id` and `key_id` fields from the raw family
  pass through; all other arbitrary fields, including payload, are excluded.
- `threshold` is `{status: "FROZEN", tau: <finite0..1>}` from this run's90-family
  calibration. The numerical record is not used as an identity gate; the outer
  scheduler is responsible for calling calibration once and persisting tau
  before evaluation enters.
- Returns all `members[137]`, `C`, `D` (`detection` alias), `maximum`, completeness,
  `domain_count=137`, `missing_members`, `unexpected_members`, upstream failure
  labels and `cause_set`. The output is in canonical H order even if inputs
  arrive reordered. Strict `maximum>tau` and inclusive `S>=tau` are distinct.
- On numeric or calibration uncertainty, **C/D/maximum are not collapsed into
  an empty-set rejection**: C and D are null; maximum remains available when
  only calibration is unavailable. Complete boundary equality produces a
  defined C and D=false.
- Each member has `candidate_id, phase, source_start, s, O_h,
  query_source_origin_hypothesis, received_query_start` and
  `local_received_to_source_map[10]`. Each map entry has `local_row,
  canonical_cell, group, query_local_interval, nominal_received_cell,
  source_frames_under_h[4], source_support_under_h`. The latter is explicitly
  nominal `FULL_UNDER_H`, not verified source truth or a VAE receptive field.
- There is no physical view/evidence ID fabricated here. B3 receives the actual
  views separately and binds `(phase, row, actual region)` to shared physical
  evidence. O_h=4(s−1)−p does not receive an extra w offset.

`paths(locals_by_window, mode)`

- The outer runner calls this twice per **evaluation** input/key, mode E2/M3,
  for36 tasks. It never calls it for calibration or fragment-pair bundles.
  Therefore per method the planned denominator is36×961=34596 classes and
 36×8401=302436 seam hypotheses. The API deliberately has no private role field.
- Input mapping keys are L/M/R, with prepared starts0/53/106. E2 accesses only L/R;
  missing or failed M leaves E2 available. M3 accesses all three.
- Checks full H137, prepared D/C consistency, origin and received coordinates,
  matching tau, and matching anonymous input/key metadata when supplied. No
  payload field or private map is accessed. It does not modify any local C.
- `paths[961]` always retains the planned classes. Each class contains
  `(a,d,gap)`, the three hypothetical query origins/states, conditional mapping,
  valid received domain, seam interval/multiplicity, `observed_windows`, required
  `member_candidate_ids`, compatibility mask/status and cause set.
- CONT has no seam and multiplicity1; a positive class retains all9 aliases.
  Class IDs are local to the current input/key task; identical class strings
  across tasks do not identify a common source. E2 also records M's *hypothetical*
  origin in the model, but `observed_windows` is only L/R: it does not pretend to
  have consumed the actual M observation.
- `compatible_path_ids` and compatible counts are null for unresolved paths;
  a defined reject/incompatible result has an empty list and zero count.
  A nonempty result is `CONDITIONAL_PATH_SET`; its location label distinguishes
  retained continuity from positive jump **within the declared model**.
- Output `planned/returned/failed/not_entered` counts describe class decision
  slots in this function invocation, not physical attempts. `predicate_evaluations`
  separately reports the numeric membership work after gates. The common
  scheduler records tasks/physical units that were never entered; it must not
  call this API to create calibration path placeholders.

## View plan and real adapter handoff

`view_plan(input_id)` returns12 planned view records, L/M/R × p0..3. View IDs
do not contain a key. Full actual E input support, expected reset/posterior mode,
one-/255 preprocessing contract and expected latent shape are included as
**planned requirements**, not a claim that the encoder already ran.

`extract_e41(received, received_query_start, phase)` accepts the array/tensor
protocol with shape `[151,320,512,3]`, uint8 dtype and clone/copy support. It
returns a separate41-frame uint8 input; it does not normalize, import torch,
run VAE or apply a private edit. The common encoder adapter performs exactly one
/255 and its actual reset/posterior-mode operation, then records returned status.
For the last phase the received frames are109..149; frame150 is not encoded.

The common runtime should reuse each E view for both keys and for B1's DCT2
regions. This component provides no FFT fallback and no model executor. It
cannot turn saved Card104 endpoint Q into new middle-window observations.

## Validation and limits

The targeted test file covers origin0/136 and invalid/negative coordinates,
explicit nominal mappings,961/8401 enumeration and safe boundary seams,
CONT/single-gap paths, conditional18→9 alias behavior including1-frame model
hypotheses, preservation of C, E2 independence from missing/failed M, strict vs
inclusive threshold boundary,90-family max and bad denominators, nonfinite and
partial/unreturned scores, payload isolation, actual existing Q40 on tiny
synthetic arrays, and12-view scheduling with a broadcast uint8 extraction fixture.
It does not replay any saved scientific evidence or instantiate a model/VAE.

Use the available CPU interpreter, for example the currently available
`diagnostics/trajectory-main-integration-20261007/validation-venv/bin/python`, and:

```text
python -m pytest -q tests/test_b_m3_time_path_v1.py
```

That environment is a convenience, not a version requirement or execution gate.
On2026-10-24 the targeted file completed with **44 passed** using that CPU
interpreter; no other suite or physical scientific execution was run.
Tests establish engineering behavior only. New DCT2-chain state distributions,
threshold value, middle-window retention, quality, real latency/memory, physical
minimum deletion size, general edits and scientific performance remain untested.
The integrated c0 experiment still requires user execution; this component has
not changed the method into a general deletion or source-authentication claim.
