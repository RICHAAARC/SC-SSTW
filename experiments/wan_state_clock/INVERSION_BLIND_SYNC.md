# Fixed nominal-latent blind synchronization

Frozen before examining this run's search outcomes. Reuse only recovered
tensors from `inversion_crop_observation_20260918T171553283563Z`; no new MP4,
VAE, Transformer, inversion, training or parameter tuning.

The search takes a recovered [1,16,33,40,64] tensor and the fixed public key/book,
never the source start, message truth, attacker manifest or oracle scores.
Nominal latent shifts are all integers 0–13. These cover the latent-index range
induced by every feasible 129-frame crop of an original 181-frame video, but
do not identify the within-latent RGB phase. No 53-frame-offset estimator is
claimed or simulated.

All candidates use exactly local j=1,...,31: exclude j=0 (causal reset) and
j=32 (shift13 would reach the unencoded source boundary45). For each shift h,
source latent index is t=j+h, always 1–44. Demodulate channel0 spatially sorted
coordinates with public pad[t], then average even/odd coordinate amplitudes
to raw two-axis q[j,h]. Candidate state is state_clock trajectory[m][(t-1)//4].
The sole primary score is mean(q[j,h,axis]*state[m,n,axis]) over 31 slices and
two axes. No direction normalization, fitted weight, missing-window penalty,
amplitude threshold, or outcome-dependent support selection enters this score.

Persist all 14 shifts x two messages = 28 scores per clip, all raw q and
unchanged boundary observations. Rank all pairs, retaining ties within 1e-12.
Report top/runner-up gap and each message's maximum over shifts, with its
message margin. Unique ranking is not detection. Zero evidence remains tied.

The unchanged state observer on/off is auxiliary only: form core windows from
the same 31 slices, keeping partial/missing rows; every shift has seven complete
windows. Only complete nonzero windows are valid for the existing observer.
Its score never reranks the primary raw-amplitude candidates.

Finish and persist all 18 blind search outputs before loading any attacker
metadata or source result for posthoc scoring/hash verification. For aligned
starts, report nominal latent-shift and message correctness separately. For a
nonaligned start (including17), report floor/ceil candidate scores/ranks and
neighboring shifts separately; do not select the more favorable map as success.
Report nominal adjacent-shift score gaps rather than inventing near-tie cutoffs.
Within-latent phase remains unresolved even when a shift rank is unique.

OFF has all scores/ranks and no true message, detection count or FPR claim.
Fixed denominators are 18 related clips from six source videos, 12 marked clips,
6 OFF clips, 504 candidate records and 558 scored local slices per shift across
the run. Preserve missing/failed clip and candidate records. Multiple crops of
one source and adjacent mappings are not independent evidence.

Output is a new `MyDrive/Video-WM/InversionBlindSync/inversion_blind_sync_<UTC>`
directory, outside the existing input run. A CPU Run all notebook performs only
offline search/reporting. The fixed carrier, state dynamics and observer are not
changed. Engineering checks do not establish blind detection or robustness.
