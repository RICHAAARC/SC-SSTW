# M05 receiver payload follow-up V1

This delivery connects the accepted received-column median + duration-four
selector to the actual temporal operation and unchanged Wan payload reader.
The local checks are CPU/stub engineering evidence. No real model, media,
codec or Colab execution has been performed for this follow-up.

## Fixed experiment

Reuse the two pilots and all 15 saved M05 observations from temporal recovery
`20261010T063205774076Z-74da28d6`. For K0 compare RAW_U, CENTERED_U and
CENTERED_D4: **30 conditions × three receivers = 90 rows / 2,880 planned bits**.
RAW_U means the original uncentered monotone selector; its saved reference is
the old BLIND_PATH reader, not the old RAW reader. RAW_D4 remains the previous
CPU path control and adds no physical payload read here.

The original 30 RAW_U reads are reused, including their five non-exact results
(25/30 exact; 8/960 bit errors). Each centered receiver also reuses the two FULL
RAW reads. This gives 34 reused logical rows and 56 outstanding logical reads.
The two centered input maps differ only on the two mean5 observations: there
are **30 new physical Wan inputs**, shared across the other cases. These are
planned counts, separate from actual call records. New output lengths/support
are (89,22) × 2, (181,44) × 26 and (177,44) × 2.

The score estimator, column median, D4=4, exact-tie/top-two rule, nearest/earlier
mapping, 1+4k length, R=min(44,k), FFT coordinates, strict zero and Counter vote
semantics are unchanged. D4 is a fixed candidate assumption, not a physical
upper bound on real repetition. No new threshold, codebook or attack is added.
The receiver sees only received signed_projection/rho and its fixed receiver
name. Attack truth and message are joined only in reporting after selection.

This is the same development batch. Eight confirmation cases remain
NOT_EXECUTED and outside its denominator. New centered K1/OFF behavior has not
been verified by this K0 payload study; old negative evidence is not replaced.
Report all missing, failed, ambiguous and unsupported slots as well as scored
rows. Support-path error is not a substitute for payload BER or exact recovery.

## User-run delivery

Open `notebooks/paper_results_v1_receiver_payload_followup_colab.ipynb`, select a
GPU runtime and Run all. It mounts Drive first and loads the ordinary editable
`paper_results_v1_receiver_payload_followup_companion.zip` from the development
branch. Source/model revisions are locators, not identity admission checks.

The notebook uses the existing `Paper-Results-V1-Cache` Wan VAE snapshot and the
old recovery run as read-only inputs. It creates one new Drive directory under
`Paper-Results-V1-Receiver-Payload-Followup`, with a pointer that resumes the same
directory on another Run all. Only actual import failures trigger dependency
repair; pip-check status is diagnostic. No framewise, baseline or quality
environment is prepared and no DiT or VAE decode is called.

Temporary received RGB is reused when present. Otherwise the future run decodes
the 28 existing edited M05 MP4s; it does not re-encode or regenerate media.
The original attack source root is used when the recovery artifact only carries
an obsolete temporary RGB path. Media decode attempts, VAE loading attempts and
physical Wan encodes have separate actual records. Encoding uses the original
`load_frozen_vae` → `reencode_rgb24_readback` → `read_payload_general` chain.

The standalone CLI also works without Git:

```sh
python -m experiments.paper_results_v1.receiver_payload_followup_cli \
  --source-run /path/to/saved/recovery --config /path/to/effective_attack_config.json \
  --output /path/to/new/followup --temp-root /path/to/local/rgb --phase init
python -m experiments.paper_results_v1.receiver_payload_followup_cli --output /path/to/new/followup --phase prepare
# Only the following phase loads/executes Wan and may decode saved MP4s:
python -m experiments.paper_results_v1.receiver_payload_followup_cli --output /path/to/new/followup --phase read
python -m experiments.paper_results_v1.receiver_payload_followup_cli --output /path/to/new/followup --phase report
```

`prepare` is saved-score CPU work. It freezes the selectors and groups exact
input maps before `read`. Truth does not participate in selection or grouping.
Case/attack labels locate observations; grouping uses the same received
observation and full map. The source config supplies K0; a different key is
not substituted while reusing K0 evidence. Original reader defaults remain U.

## Reuse and interrupted execution

Reuse compares the same received observation, full index sequence (including
repetitions and length), key and reader contract. Source coordinates are
annotation-only: copied detail JSON rewrites each time row's estimated source
coordinate from the current operation at stride 4×latent_index. FULL old RAW
annotations were null and must not be copied as candidate estimates. Unchanged
sign/zero arrays, votes, decoded bits and receiver-local timing are reused.
Source files remain unchanged. Missing old sidecars remain MISSING, with their
paths/reasons recorded; they do not silently trigger another model read.

Each physical invocation is saved RUNNING immediately before the encode and
COMPLETE when it returns, before CPU transfer/readout. A shared raw read is
saved before its dependent logical rows. Re-entry or external child failure
can finish that fan-out from durable evidence. If an attempted encode has no
durable read evidence, retain FAILED/INTERRUPTED slots instead of repeating it.
Preprocessing failure is not counted as an encode. Independent unattempted
inputs can continue on resume. A hard kernel death can only leave the last
durable checkpoint; Run all performs takeover when the kernel is available.

Return the whole output directory, including `handoff_summary.json`, execution
log, attempts, stage receipts and `run_state/` (state, summary, 90-row CSV,
evaluation report, selector/read JSON and raw vote NPZ). Report generation can
be called again without model work and retains missing/partial rows. No
scientific PASS is inferred from process success or engineering checks.
