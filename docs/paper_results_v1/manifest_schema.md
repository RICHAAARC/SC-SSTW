# Paper Results V1 manifest fields

The manifest is the sole denominator authority.

`method` records the method name, bounded claim scope, and evidence ceiling.
`comparability` records message length, redundancy, codec, and auxiliary inputs
as descriptions. `matching_rule_status` must be
`DESCRIPTIVE_ONLY_NOT_ADOPTED`; the reporter does not infer a fairness rule.
`result_bindings` records the evidence role and any required exact top-level
identity fields for each result input. Identity mismatch becomes `CONFLICT`;
the input file path and SHA-256 remain in the report.

Each `slots[]` row declares one logical payload read with a stable `slot_id`,
source, arm, receiver mode (`RAW`, `GLOBAL`, `PATH`, or `ORACLE`), result input,
adapter, exact result locator, explicit `key_label`, and planned bit count. V1
requires integer 32 for both `comparability.message_length_bits` and every
`planned_bits`; the saved payload row must independently contain integer
`planned_final_bits=32`. Every slot also requires `key_role` equal to
`CORRECT_KEY` or `WRONG_KEY`; the saved posthoc row must match it. Recovery
summaries group by both `key_label` and `key_role`. `included=false` and
`supported=false` remain explicit rows. Duplicate IDs, duplicate locators, or
multiple files for one `result_id` become `CONFLICT` rows.

Each `pairs[]` row explicitly identifies a RAW slot and a GLOBAL or PATH slot.
The reporter never constructs pairs from available results. A pair is evaluable
only when both planned members have valid postseal truth rows. Missing or failed
members remain in the pair denominator.

Each `measurements[]` row uses a list-valued `locator` to address an exact
nested saved-result field. Quality and cost values are copied without a
numerical pass threshold; absent fields remain `MISSING`.

The current concrete adapter is `conditional_joint_v1`. It reads the frozen
runner's top-level `payload_reads` for blind bookkeeping and `posthoc` for
truth-only recovery. A successful row requires `posthoc.status` equal to
`EVALUATED_TRUTH`, a valid integer `bit_errors`, and a corresponding blind
payload status of `READ`. Unplanned records are retained from the union of
`payload_reads` and `posthoc`, with presence and status recorded for both sides.

## Workflow manifest

`workflow.template.json` uses schema `paper-workflow-v1` and deliberately has
empty `cases`, `native_jobs`, and `quality_pairs`. Its backend policy is
`EXPLICIT_INJECTION_ONLY`; no model, codec, source roster, or budget is inferred.

Each `cases[]` entry declares one source artifact and content ID, one noise ID
and integer seed, and one codec operation with explicit parameters. It must
declare exactly four stages. OFF and P0 consume the source; P1 consumes the P0
PRE artifact; M05 consumes the P1 PRE artifact. Every stage declares its input,
PRE, and POST artifact IDs plus its transform callback. Plan expansion creates
one transform and one codec row per stage before any callback runs.

Each `native_jobs[]` entry declares VideoSeal or RivaGAN, an input OFF artifact,
native PRE and POST artifact IDs, and the complete native message. RivaGAN
requires exactly 32 integer bits. VideoSeal requires the injected adapter's
declared native length; no repeat, truncation, or 32-bit mapping occurs. The
workflow calls native embed, the case's shared codec callback, then native
extract. Missing source, transform, codec, or native backend remains in plan,
artifact, native, quality, and cost records as appropriate.

Each `quality_pairs[]` entry references two POST artifacts from the same case
and supplies a finite positive data range. Matching source/content, noise,
seed, and codec receipts are checked before CPU array metrics run. This is
declaration and callback-receipt verification, not physical media identity
proof. MSE, RMSE, and PSNR are absolute pairwise metrics; the reporter does not
subtract them into an additive VAE or M05 decomposition.

`main_report` either stays explicitly pending or names a strict
`paper-results-v1` manifest and its saved inputs. The workflow CLI writes that
unchanged report under `main_report/`, keeping native-capacity records outside
the main method's strict 32-bit recovery table.
