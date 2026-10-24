# B3 CDC32 implementation handoff

2026-10-24, B-THREE-IMPLEMENT-01. Implements adopted COMMON_INTERFACE r1,
INTEGRATED_PROPOSAL c0 and B3 SPEC r2. This is a component engineering delivery;
no model/media/scientific result, publication or method PASS is claimed.

## Files and API

- `main/tube_state/b_cdc32_aggregation_v1.py`: stdlib-only method.
- `runtime/wan/b_cdc32_bundle_v1.py`: local JSON persistence, opaque bundle
  adaptation and lazy cached-bit access; no physical operations.
- Two matching test files: small synthetic input cases only.

`plan_local(prepared_local, views, required_regions=('R0','R1'))` accepts the
actual B2 prepared-local shape: ordered 137 `members` with phase/source_start,
S/score_status/in_C, full C, D, tau, completeness/score_complete, query_id,
received_query_start. The common wrapper supplies input_id and key_id. An
explicit `local_received_to_source_map` is a length45 list of source integers
indexed by query-local offset (or a dict keyed by received indices). It is not
the ten nominal-cell detail records. Query source origin is unchanged by w.
Absent explicit maps use exactly the public H137 local-contiguous hypothesis;
an explicitly unavailable/malformed map remains unresolved, with no fallback.

`views` is a list (or dict of records), one per planned phase. Each record has
view_id, phase, status=`RETURNED` for an available E, encoder_input_support as
the full41 received-frame list (also accepts a dict with received_frames), and
encoder_context. Missing/failed views keep their planned references. Optional
`units[str(row)][region]` gives evidence_id, observed_region_id,
carrier_region_id, status and diagnostic_only. Without an explicit unit ID,
use the exported `evidence_id(view_id,row,region)` when wrapping B1 bits.
Rc/renamed regions are never admissible production references.

The planner returns fixed four `groups`, 137 members, complete C, all under-h
nominal row/source-cell support, and no payload values. `references` retain
all candidate/replica relations; duplicate physical addresses are consumed
only once. Call `reconcile_plans(plans)` for a source hypothesis before saving
the plans to mark cross-query contradictory evidence-to-cell assignments.
Within-query contradictions are checked by plan_local itself.

`read_local(plan, cache, memo=None)` reads a saved/read-back plan. Cache is
`cache[evidence_id][key_id][str(bit)] -> {value, status, count=8, ...}` or a
callable `(evidence_id,key_id,bit) -> same record`. AVAILABLE/RETURNED/READ are
accepted cache statuses; 0/1 must be integer, not bool. Cache access exceptions
become technical failures at their original slots. A shared memo deduplicates
addresses across logical reads; it never reruns B1 Counter.

`aggregate(local_decisions, source_relation, path_context=...)` implements
CDC conflict veto and partial-evidence behavior. `union_reference` uses the
same local decisions but ignores already-erased local conflict, as required
for UNION_M. Values, detection, source relation and path context are separate;
source_authenticated is always false. Counts use planned=returned+failed+
not_entered; returned means a decision was formed, not a bit was recovered.
Use recovered_slot_count for message coverage. Per-query records, cause sets,
source-position buckets and overlap dependencies remain inspectable.

`plan_comparisons({'L':planL,'M':planM,'R':planR})` constructs CDC_E, CDC_M,
UNION_M, ALIAS_M_X10 and REVERSE_ORDER without bits. The common runtime saves
and reads back the whole returned pack, then calls `read_comparisons(pack,
cache,path_context={'E2':..., 'M3':...})`. No path context is used to shrink C
or choose references. The result includes semantic invariance comparisons and
unique cache-access accounting. Single-region diagnostics use the same
planner with ('R0',) or ('R1',); do not insert them into CDC missing slots.

## Bundle isolation and fixed c0 denominators

`RawFragment(prepared_local, views, cache)` is a private runtime envelope for
one actual45-frame endpoint query. `run_bundle([A,B], output_directory)`:

1. Allowlist only the two local families and their actual views. Create a new
   opaque namespace for each call, rebase received coordinates by old w,
   preserve query source origin and actual E/reset, and retain within-call
   actual aliases. Parent paths, prior query bit decisions, global aggregates,
   source/arm/fixture labels and cross-bundle links are not copied.
2. Produce SEP_A, SEP_B and conditional JOIN plans. Save `plan.json` and load
   it back before the first lazy cache callback. Only then compute the local
   reads and three32-slot outputs. Save/read back `decision.json`.
3. Use path_context=NOT_APPLICABLE_FRAGMENT_PAIR and retain both SEP outputs
   even when all JOIN bits agree. No physical same-source inference exists.

`prepare_bundle` and `read_bundle_plan` are exposed for a common runtime that
owns persistence. The former does not consume caches; its private lazy reader
must not be exposed to the planner or serialized into blind records. The
common runtime must complete the same save/read-back boundary before calling
the latter. Raw E/Q and per-unit caches can pre-exist; query-level decisions
cannot replace the bundle's independent plan/read.

`run_fixed_bundles(requests, output_directory)` requires exactly72 entries,
each a pair of envelopes or None for upstream NOT_ENTERED. The common private
scheduler creates the c0 fixture/arm/key/source-pair roster; these labels are
not blind method arguments. Every entry retains 64 local slots, 64 SEP slots
and32 conditional JOIN slots: 4608 local,4608 SEP and2304 JOIN total. The
adapter records zero additional physical calls; logical costs are nonzero.

The common runtime still owns physical attempt accounting, the36 evaluation
input/key roster and global blind-before-post barrier. These components do
not access private truth or implement the GPU runner/notebook/builder.

## Verification and scope

Targeted command (environment is a location record, not a version gate):

```text
PYTHONPATH=.:tests /home/richar/projects/Video-WM/diagnostics/trajectory-main-integration-20261007/validation-venv/bin/python -m pytest -q tests/test_b_cdc32_aggregation_v1.py tests/test_b_cdc32_bundle_v1.py
```

The targeted suite passed 20 tests. Checks cover full-C earliest support, strict equality rejection, incomplete
families, two-region conflict plus missingness, partial recovery, mapping
conflicts, mixed/partial/absent/unobserved distinctions, alias/reverse
invariance, cache-access deduplication, per-bundle privacy/order and all72
missing requests. A direct small constructed B2 actual prepare_local→B3
plan_local check also passed after B2 adopted the45-integer map representation
(w53, p3/s6, query origin17, selectedcell6). No saved scientific Q/gradient/
votes/path/media was replayed. One fixture expectation was corrected: origin3
still contains three frames of cell12, so structural absence starts at cell13.
This was a test-case geometry correction, not a change to the method.

Only the listed B3 files are authored here. Shared __init__, original B tree,
other group modules, public design files and common delivery entrypoints are
unchanged by B3. Independent review and common runtime integration remain for
the coordinator; CPU results do not establish new-carrier physical success.
