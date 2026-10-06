# Receiver-Estimated-Align-V1 (Stage2)

This isolated, fixed receiver diagnostic consumes the three saved G M05 received RGB observations: FULL181, CROP177 and SHORT89. Source generation, M05 strength/writer/codec, public sync codebook and the canonical payload reader remain unchanged. It is a user-run experiment; the delivered notebook has SOURCE_SHA=None until reviewed source publication.

The receiver freshly encodes each RGB observation with the framewise VAE once, scores K0 and K1 independently over the complete public offset roster, and seals all six blind readouts. An estimate requires complete finite candidate/local evidence and the unchanged unique canonical argmax (tie_atol=1e-12); sync_accepted remains false. No minimum score or gap threshold is introduced. Tied, incomplete, missing or nonfinite evidence leaves the aligned logical slot unresolved. The FULL181 singleton is only geometric evidence.

For each key's estimated offset, p=offset%4. Input mapping is [0]*p + list(range(N-p)): prepend p copies of the first received frame and discard the last p original received frames, keeping N unchanged. Synthetic outputs are indices 0..p-1; p0 is identity. This extends the Stage1 p1 operation deterministically to p2/p3 without claiming their efficacy has already been established. It neither recovers an unavailable source frame nor asserts cache recovery.

After the sync seal and before any Wan processing, a physical plan is frozen by observation identity, phase and key. Each observation has a baseline p0 encode shared by keys. Aligned p0 aliases its same-key baseline read; equal nonzero phases share an encode but have separate key reads. No cache crosses observations. Aliases are references, not independent observations.

| Fixed logical accounting | Count |
|---|---:|
| Received observations | 3 |
| Fresh framewise encodes / sync reads | 3 / 6 |
| Candidate scores / tubelet rows | 198 / 4820 |
| Payload slots (BASELINE/EST_ALIGN × K0/K1 × 3) | 12 |
| R44 slots / R22 slots | 8 / 4 |
| Logical votes / time-bit rows / final bits | 422400 / 14080 / 384 |
| Physical Wan encodes, complete valid input range | 3..7 |
| Physical key reads, complete valid input range | 6..10 |

Wan FP32 posterior-mode shapes are [1,16,46/45/23,40,64]. Reads remain times 1..44, 1..44, and 1..22. The isolated detailed wrapper checks every final decoded bit and aggregate against the unchanged original reader; its extra FFT is evidence extraction, not a second method read. It records all ordered signs, exact-zero masks, coordinate order and time/channel rows. FFT.real>0 is a one vote; exact zero stays a zero vote. Counter ties retain first encounter in time-major bit::8 order. Per-time diagnostics are not an extra voting layer.

The public runtime configuration contains no message, true offsets or history reference. A separate SHA-bound posthoc configuration is first read after the final payload blind seal. Only then are truth-signed margins, exact/phase offset comparisons and BASELINE-to-EST_ALIGN differences formed. Historical G comparisons check saved sync candidates/summary and baseline per-bit aggregate/decoded values; they do not consume history as the new estimate or trigger retries. Full, failed, missing, tied and K1 WRONG_KEY slots remain in fixed denominators. A history availability failure is reported explicitly without changing receiver decisions.

The notebook has five code cells: exact Drive mount, immutable-source guarded setup, pinned environment, fixed run, compact display. It preserves subprocess cleanup, model release and failure receipts from the successful notebook path. Eight relevant packages reuse Stage1 pins with both framewise and Wan imports probed; pip-check warnings and actual versions persist. No GPU model gate, codec, generation, new quality pass, parameter scan or automated real execution is included.

Local CPU/fake/static validation is engineering evidence only. Real results would concern this saved M05 source and fixed roster; they cannot establish FPR, generalization, a unique phase cause or production readiness. Receiver latent indices are nominal coordinates, not proof of matched independent four-frame source support. The new notebook can be built with:
`python -B scripts/build_video_trajectory_receiver_estimated_align_notebook.py`.
