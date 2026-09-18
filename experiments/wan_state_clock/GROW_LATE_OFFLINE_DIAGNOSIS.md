# GROW late-window saved-tensor diagnosis

Run: grow_late_control_20260918T080310904782Z. Offline CPU analysis only; no generation, GPU, strength adjustment, media, or model inference. Fixed 12 terminal tensors were downloaded from the specified existing Drive run. All SHA256 values match the run result, and all terminal aggregate records exactly match recomputation. No arm was excluded.

Reproducible command from repository root:

```sh
PYTHONPATH=. python scripts/analyze_grow_late_saved.py --root /mnt/d/CodexData/.codex/grow-late-offline --output /mnt/d/CodexData/.codex/grow-late-offline/analysis.json
```

The JSON retains all 46 x 4 x 16 coefficients per arm, frequency coordinates, per-frequency and per-time signed votes, response summaries and input hashes. Raw tensors stay outside Git.

| Case | A errors (erasures) | B errors (erasures) | A/B soft errors, diagnostic only | OFF temporal-mean energy fraction |
|---|---:|---:|---:|---:|
| p0 s0 | 0 (0) | 0 (0) | 0 / 0 | .40260 |
| p0 s1 | 1 (0) | 0 (0) | 1 / 0 | .65636 |
| p1 s0 | 10 (6) | 7 (4) | 7 / 8 | .99936 |
| p1 s1 | 9 (7) | 12 (7) | 6 / 6 | .99845 |

Hard recovery remains 3/8; soft coefficient averaging also yields 3/8. Soft errors decrease from 39 to 28 overall but do not solve complete recovery, and this post-hoc observation does not select a replacement receiver. No coefficients are below absolute 1e-6: erasures are vote cancellation, not numerical zero coefficients.

The message-dependent A-minus-B difference has the correct payload sign at every one of the 46 x 64 coordinates in all four cases. Its mean signed gains are .23256, .20244, .30568, .29531. Thus the fixed late control transmits a message-dependent coefficient difference; absolute coefficient-sign decoding fails to separate that difference from content bias. This contrast requires two writer runs and is a mechanism reference, never a deployable receiver.

In p1, the OFF coefficient energy is almost entirely time-constant. Common bias (A+B)/2 exceeds half-response magnitude at 82.07% / 97.25% of coefficient-time slots. Many failed bit groups have exactly two repetitions voting +46 and two voting -46; repetition across time faithfully repeats a biased frequency sign and does not supply independent evidence. Example p1 s0 A bit3: repetition vote sums [-46,46,46,-46], signed coefficient means [-.412,.567,1.340,-5.373]. The single p0 s1 A failed bit14 has [14,-28,-36,40], a different weaker-margin failure. Last-step local losses and induced displacement are retained in JSON; the implementation already reports same-history native shadow and terminal equality to final controlled clean.

## One proposed change, one hypothesis

Propose replacing temporal DC repetition with one fixed adjacent-time difference carrier: for every existing selected spatial frequency, use d[k]=(c[2k]-c[2k+1])/sqrt(2), k=0..22. Write the same 16-bit targets into d with the existing half-squared loss, eta .1, amplitude .5 and steps30..49; read sign(d) and combine 4 x 23 =92 votes/bit, keeping ties as erasures. This is one coupled writer/reader carrier replacement, not receiver-only centering of the old signal (which would remove the old time-constant message too). It preserves pair common mode. No frequency/bit selection from truth and no new strength scan.

Hypothesis: fixed temporal differencing suppresses the dominant persistent content bias sufficiently that the same local guidance creates recoverable difference signs. OFF adjacent-difference RMS falls from .24315 to .16137, .24020 to .10899, 2.40247 to .04197, and 2.93609 to .09379. These are empirical nuisance measurements, not predicted marked performance. Pairing was not searched; no alternative timing/frequency/weight variants were evaluated. Pair common-mode preservation and reduced number of constrained coordinates change control energy; measure it rather than claim equal perceptual strength.

Minimal discriminator: retain the four content/seed cases and OFF/A/B fixed roster, change only the temporal carrier as above, measure terminal exact recovery/BER/erasures, pair-domain OFF bias, same-history actual control response and final-step contribution. Keep 8/8 terminal eligibility, no automatic MP4. Existing OFF may only be reused after exact initial/schedule/environment matching; otherwise retain a new OFF. A terminal improvement supports the carrier hypothesis only; a terminal failure rejects this fixed implementation. Temporal alignment attacks, MP4 survival, quality, false-positive calibration and generalization remain untested. This proposal is not implemented or run by this audit.
