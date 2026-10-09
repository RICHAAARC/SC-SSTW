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
