# A M05 receiver-only fixed four-arm definition V1

Status: ACCEPTED before execution by root (review_root_definition.md) and independent A5 (review_method_definition.md), 2026-10-10. This file fixes the method before any new four-arm saved-score execution. Current authorization covers implementation and the complete CPU saved-score comparison. No GPU/model/media generation, VAE, codec, writer/codebook change, payload experiment or main-branch integration is included. The implementation writer may create the local review commit but does not push; root retains the user-authorized development-branch publication role.

The testable question is whether received-score column centering and a finite residence-time path grammar separately or jointly reduce the already observed temporal-path collapse. This is development on the same two pilots, not confirmation or independent generalization.

## 1. Fixed inputs and denominator

Each selector receives only one finite FP64 q matrix of shape [N,181], N >= 1, where q = saved signed_projection / saved rho and rho is finite positive. The columns are the public 181 source frame indices. These matrices already encode the fixed correct key K0; no new key or latent evidence is computed. The key/public source clock defines columns, not a selector-side true path.

The runner must retain 2 pilots x the same 15 attacks x 4 arms = 120 condition-arm slots. The 15 names are full, crop37_126, speed075, speed125, mean3, mean5, delete_g0/g1/g2, repeat_g0/g1/g2, interp_g0/g1/g2. This list locates and labels input/output artifacts only. Selectors never receive the attack label, recipe, original run readout, source video, RECON, writer state, message bits, or weighted truth. There is no input from another pilot or another condition when estimating a background.

Input locations (existing files, read-only):

- FULL: diagnostics/a-line-temporal-attack-20261010T031617441784Z-2536172d-audit/records/pilot_01 or pilot_02/clock/PAYLOAD_FRAMEWISE_M05/full/K0.npz.
- Other 14: diagnostics/a-line-m05-sync-root-cause-20261010/evidence/p1 or p2/PAYLOAD_FRAMEWISE_M05/<attack>/K0.npz.
- Associated JSON is read only after selection for legacy-path comparison and provenance, never as a selector input. Recipe truth is expanded only in the posthoc reporting stage.

All inputs are below /home/richar/projects/Video-WM. All full diagnostic outputs go to diagnostics/a-line-m05-receiver-controls-20261010. No failed/missing condition is replaced or removed, and none of the eight unexecuted confirmation cases is introduced.

## 2. One fixed received-only background estimate

For every column s independently, define

    b[s] = median over all received rows t of q[t,s]
    q_centered[t,s] = q[t,s] - b[s].

Use NumPy's ordinary FP64 median: if N is even, the median is the arithmetic mean of the two middle ordered values. No trimming, learned weight, scale normalization, clipping, background subtraction from RECON, cross-condition averaging, selected-path masking, iterative refitting, or attack-specific rule is used. Persist the entire 181-vector b and [N,181] q_centered beside q.

The statistical hypothesis is that a candidate column's persistent location-dependent score across the received clip is nuisance background, while the true source correspondence occupies comparatively few received rows in that column. This is a hypothesis, not a guaranteed decomposition. A short clip, genuine long repetition, or stationary true correspondence may occupy much of a column and be subtracted. N=1 produces all-zero q_centered and hence ambiguity, not an input exclusion or automatic success. N=2 and N=3 remain valid inputs. The same formula applies to every condition; a poor result never changes the estimator.

## 3. Two fixed path grammars

Both optimize the sum of the supplied local matrix a[t,s_t] over one source index for every received frame. Source indices are 0..180. Both have free start and free end. Neither requires the clip to begin at source 0, end at 180, cover the whole source, match a predetermined speed, or derive a coverage span from N/181. No transition, endpoint, deletion or repetition soft penalty is added.

### Original grammar U

    s_t >= s_(t-1).

Unlimited stay and arbitrary positive forward jumps are permitted. This is the existing solve_monotone grammar. Reuse the existing top-two distinct-complete-path selector by passing a as its numerator and all-ones denominator. The raw/U arm is the original method reference; historical path agreement is a reported engineering comparison, not a hash/version admission condition.

### Fixed finite-duration grammar D4

    s_t >= s_(t-1), and every maximal equal-source run has length <= 4.

The maximum residence is fixed at 4 received frames per source index for this candidate. It is an explicit modeling assumption, not a physical bound implied by the writer's four-frame tubelet. That public temporal scale supplies a fixed heuristic reference only. Do not claim arbitrary repetitions or arbitrary slow motion are supported. A fifth consecutive true repeat is outside this grammar even when the received data are perfectly valid. This limitation must be reported without changing truth, calling the input invalid, dropping the condition, or adjusting 4 afterward.

Positive forward jumps remain unrestricted, permitting deletion and faster progression. Stays of up to 4 received frames permit genuine repetition and slower progression inside this declared domain. Short crops keep free endpoints, and N < 4 has no minimum-length exclusion. N > 4*181 = 724 cannot have a complete D4 path; retain the arm as UNSUPPORTED with NO_COMPLETE_DURATION_PATH. The U arms still run on that input. Within the current fixed 15-attack roster, no label-specific adjustment is permitted. Posthoc truth-path feasibility is an explanatory diagnostic, not a selector input or a filter, and does not establish general attack support beyond D4.

Explicit DP state and recurrence for D4:

    state (t, s, d), d in {1,2,3,4}; keep top two distinct prefix paths.
    t=0: score(s,1)=a[0,s]; other d are unreachable.
    d>1: predecessor is (t-1,s,d-1).
    d=1, t>0: predecessors are all (t-1,p,e) with p<s and e in {1,2,3,4}.
    Add a[t,s] at every transition.
    Final candidates: every (N-1,s,d), both retained ranks.

Use explicit backpointers. The two best finals must be different complete source-index sequences; a duplicate of one path reached through bookkeeping cannot serve as its runner-up. One path uniquely determines its duration states. Prefix top-two optimization is allowed, with a small exhaustive-path check establishing equivalence before running saved data.

## 4. Fixed four arms: score x grammar

| Arm | Local matrix | Path grammar |
|---|---|---|
| RAW_U | q | Original U |
| CENTERED_U | q_centered | Original U |
| RAW_D4 | q | D4 |
| CENTERED_D4 | q_centered | D4 |

The phrase original path refers to the original path-selection rule: CENTERED_U must select its own optimum under centered scores. It does not reuse RAW_U's selected index sequence. The original reference is always retained; do not pick the best arm separately per attack or form a best-of-four ensemble. Centering is calculated once per input and is identical for the two centered arms. D4 is identical for raw and centered scores.

## 5. Ambiguity, nonexistence and failure

For both grammars retain the best and second-best distinct complete paths, both objective sums, and gap = best - second. Exactly equal complete-path scores are UNRESOLVED / EXACT_BEST_PATH_TIE, with path=null. Store both witness paths as diagnostic_best_path and diagnostic_runner_up_path. Do not silently treat a tie-broken witness as an accepted temporal estimate. Near-zero nonzero gaps are reported as measured values without adding a confidence threshold, epsilon rejection or success rule. A single legal complete path has runner_up=null and gap=null.

- ESTIMATED: unique best complete path; preserve its full path and all diagnostics, even if badly wrong posthoc.
- UNRESOLVED: exact complete-path tie; no accepted path. Witness-only metrics may be separately retained and clearly labeled, never mixed into accepted-path means.
- UNSUPPORTED: valid received matrix but no complete path under the chosen duration grammar; keep the input and denominator.
- MISSING: input NPZ absent/unreadable as a missing artifact; every corresponding arm row remains.
- FAILED: malformed shape, nonfinite score/rho, nonpositive rho, or actual computation/I/O error; record concrete reason and keep the affected rows. An unavailable or malformed posthoc truth/evaluation artifact is an evaluation failure separate from a successfully persisted selector result.

Continue independent conditions and arms after an individual failure. Planned slots are initialized before work; persist completed selector rows before posthoc truth/reporting. A process interruption must leave completed records and explicit remaining/in-progress state so a partial result can be inspected; no automatic repeated model/media action exists because none is called.

## 6. Required outputs and separated posthoc metrics

Persist per input q, b and q_centered as compact NPZ. For each of the 120 rows persist the score/grammar identity, received length, status/reason, accepted full path if any, distinct top-two diagnostic paths, sums, gap, source start/end, number of distinct source positions, transition counts and maximum dwell. Also store raw-q and centered-q sums of the same accepted path as descriptive cross-score quantities; do not compare optimization sums across differently centered arms as if they had a common zero.

Only after every selector result for that input is stored may reporting load the fixed recipe truth. For an accepted path p and positive-weight source support W_t:

    start error = min_{s in W_0} |p_0 - s|
    per-frame support error e_t = min_{s in W_t} |p_t - s|
    in-support fraction = count(e_t == 0) / N
    weighted error w_t = sum_{s in W_t} weight_(t,s) * |p_t-s|.

Report support-error sum/mean/max over the entire received path, weighted-error sum/mean/max separately, and exact all-frames-in-support as a descriptive flag. Mixed frames allow every positive-weight contributing source for support membership; weighted error need not be zero for a support-valid choice. No weighted average or center is substituted for the support set. Start/end, first-offset accuracy and full-path recovery are separate quantities. Retain per-frame received index, selected source, support, minimum error and weighted error. Reporting may determine whether any support-valid D4 path exists, but may never feed that conclusion or an oracle score back to the selector.

Summary: four arms with 30 planned conditions each, all status counts, received-frame denominators, accepted-path coverage, and conditional error distributions with their actual evaluated denominators displayed. A missing/unresolved arm has undefined accuracy, not a fabricated zero error; status and planned denominator remain alongside valid measurements. Preserve all 120 rows. Present all four outcomes without posthoc arm selection. This is same-batch development evidence, never confirmation or a scientific PASS.

## 7. Implementation and minimal pre-run verification

Proposed implementation, after definition review:

- experiments/paper_results_v1/receiver_controls.py: pure NumPy received-score centering / U wrapper / D4 selector with no truth or attack input.
- experiments/paper_results_v1/receiver_controls_cli.py: fixed 30-input loader, four-arm run, persistence and separate posthoc reporter. Runnable from a no-.git source copy with no notebook, GPU, installation or model import.
- tests: targeted CPU tests only; tiny exhaustive best/runner-up equivalence, exact ties, constant-column-shift centering, duration-boundary sequences, valid one-to-three-frame inputs, N>724 no-path handling, and per-condition failure/denominator persistence. Known synthetics test code and declared limitations; they do not tune max dwell or choose another estimator.
- docs/paper_results_v1: copy of this fixed definition, compact run summary and all 120 case-arm rows after execution. Large matrices/full paths stay in diagnostics, with no private download indices committed. No GPU notebook is needed.

The duration tests must include a real repeated source of length 4 (legal) and length 5 (outside D4), unrestricted forward deletion, a free-start short clip, and an impossible N>724 case. They must not weaken the grammar to make an out-of-domain synthetic pass. Mathematical tests can use a private small-width DP helper, while the public selector always requires 181 columns.

Root and A5 accepted this definition before implementation/real saved-score execution. This estimator, max dwell 4, four arms and all 30 conditions are fixed for the full comparison. Do not alter parameters or select conditions after seeing any new real control results. Engineering corrections that restore this stated method are documented and independently reviewed; they do not authorize result-driven changes.

Source main, original writer, original nine-view notebook, A recovery artifacts and B method are unchanged. Root will organize independent implementation review and any later commit/publication. No source digest, manifest, B64, exact dependency version, Git state or hardware model is an execution gate.
