# Trajectory attribution uncertainty follow-up V1

This research-branch follow-up asks whether the already frozen trajectory-attribution V1 decision rule produces two scientific uncertainty outcomes on a small preregistered set of real saved observations:

1. synchronization is strong enough to continue, but the action is unreliable; or
2. synchronization resolves to the correct action, but identity evidence is exactly weak.

It does not change the writer, payload, M05 strength, codec, receiver mathematics, thresholds or keys. It does not generate or re-encode media. The user-run notebook reads six exact post-codec received.rgb8 files from the audited C1/C2 run 20261008T085244020205Z and verifies their frozen SHA-256 identities.

## Frozen rule and inputs

The runner reads the original frozen_rules.json bytes and requires:

- file SHA-256 9b4449569daabfee6d572fb849afa61d80175433b0585fbf3eaac4d27e84ed7c;
- canonical rule SHA-256 4c0a14c1493ff642405d13111866b53d9373477668e572e964bb414239b796fa;
- formula trajectory-attribution-v1-action-folded-development-maxima;
- tau_I = 0 for N=181, 177 and 89.

The six inputs are C1/C2 × A_M05/OFF/B_M05. Their paths, shapes, byte counts and SHA-256 values are fixed in the bundled config from the audited receipts. The runner uses the saved RGB bytes directly. It does not invoke FFmpeg.

## Fixed roster and budget

The 22 queries use 20 physical temporal observations:

| Role | Fixed rows | Coverage |
|---|---:|---|
| PRIMARY_PROBE | 10 | The same ten rows form both uncertainty denominators |
| ALIAS_CONTROL | 4 | Excluded |
| REGRESSION_CONTROL | 8 | Excluded |

For each source, primary probes are A_M05/K0 SHORT89 starts 0, 46 and 92 plus H1 b2 deletions k44 and k132. Edge deletions k1 and k176 are alias controls. Regression controls reuse CROP177 b2 for A_M05/K0, A_M05/K1, OFF/K0 and B_M05/K0.

The fixed maximum budget is 11,902 synchronization candidates, 3,012 framewise frames, 394 batch-eight framewise calls, 22 payload reads, 802,560 votes, 26,752 time-bit rows and 704 final-bit rows. Paths and aliases are not independent queries.

These time selections are inside the existing H0/H1 single-deletion and N=181/177/89 grammar. Their use here does not establish robustness beyond this roster. Spatial edits, multiple deletions, insertion, concatenation, splicing, new codecs and unsupported lengths are outside scope.

## Blind order and decisions

The outer runner first seals all synchronization evidence under opaque observation/query identifiers. It then applies at most one qualified unique action, seals payload evidence, and seals all 22 decisions. Only after that decision seal does it join source, condition, construction, intended role and true action.

A scientific sync-unreliability trigger requires complete receiver evidence, M > tau_M, decision reason UNCERTAIN_SYNC, and either more than one distinct top action or a unique action with local m <= tau_m. Multiple top paths folded to one action remain absolute-path ambiguity and never count as action ambiguity.

A valid weak-identity trigger requires qualified unique synchronization, the correct complete action, complete payload evidence, decision reason UNCERTAIN_IDENTITY_WEAK, exact I == 0, all 32 integer signed vote differences nonnegative, and at least one exact tied bit. exact32 is not an extra weak-coverage requirement because weak identity is decided before the exact-bit acceptance branch. With frozen tau_I = 0, no positive interval above zero is weak.

Technical, missing, interrupted, nonfinite, unsupported or rule-read failures retain their fixed slots as unresolved. They never count as scientific uncertainty coverage. If a branch has no valid trigger and all ten primary rows are technically complete, its status is NOT_OBSERVED_FIXED_ROSTER; if no valid trigger exists and any primary row is technical, it is UNRESOLVED_TECHNICAL. The roster is not extended or replaced after observing results.

## Controls and false attribution

The eight regression controls retain the prior run's two positive and six negative decision/reason expectations. A positive regression match additionally requires the correct action and complete exact-32 identity evidence. The four alias controls check reporting boundaries and remain outside both coverage numerators and denominators.

Across all 22 rows, an ACCEPT on a negative row or an ACCEPT on a positive row with a wrong action is a false claim. A correct positive probe ACCEPT is not a false claim; it is simply a non-trigger. A technical row leaves false-attribution aggregation unresolved unless another row already proves a false claim.

## Evidence ceiling

The source candidate is validated with static, CPU and fake fixtures only. The notebook is delivered for user execution and contains no agent-run real result. It reuses two previously seen confirmation sources, so even a successful uncertainty trigger would establish only the behavior of the frozen rule on this fixed roster. It would not establish population false-positive rate, unknown-attack robustness, new-source generalization, or a guaranteed uncertainty-triggering construction.
