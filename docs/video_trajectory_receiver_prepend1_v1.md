# Receiver-Prepend1-V1 — Stage1 controlled diagnostic

This fixed user-run diagnostic uses only the saved G FULL181 P0/P1 RGB8 identities in the config. It verifies each full source and the unchanged start1 crop. No codec fallback, new source, framewise VAE, writer, generation, synchronization search or quality measurement is present.

Each condition has exactly four physical observations:

| Observation | Input construction | Source-frame map |
|---|---|---|
| S0_IDENTITY | source[0:177] | f0..f176 |
| S1_ORIGINAL | clip = source[1:178] | f1..f177 |
| S1_PREPEND1_DROP1 | concat(clip[:1], clip[:-1]) | f1,f1,f2..f176 |
| S1_TAIL_REPEAT | concat(clip[:-1], clip[-2:-1]) | f1..f176,f176 |

Both synthetic arms drop f177. PREPEND duplicates f1; TAIL duplicates f176 as a fixed content control. S0 is identity under both rules; OLD_RULE_START0 and NEW_RULE_START0 are aliases of one physical observation, never extra encodes. The operation receives only a 177-frame clip and a fixed public operation string. No message truth, source start, original-source frame or writer evidence enters it.

All eight inputs are 177 frames. One frozen Wan VAE performs eight FP32 posterior-mode encodes, each producing [1,16,45,40,64]; sixteen K0/K1 reads use R44 latent indices1..44. The original four payload channels,32 bits,240 coordinates,time-major bit::8 flattening, FFT.real>0 and Counter first-encounter ties remain unchanged. Full denominator:675840 votes,22528 time-bit rows,512 final bits. Missing/failed observations retain all planned slots and explicit failed artifacts. No threshold, edge removal, channel subset, BER-selected strategy or retry selection is implemented.

Before blind sealing, artifacts contain operation/input-output hashes, received-index maps, synthetic duplicate/drop indices, readouts and detailed votes. Source-frame maps, condition, source start, aliases and expected-message margins join only after seal. Eight baseline key reads (P0/P1 S0/S1 × K0/K1) then compare decoded bits and per-bit aggregate votes with the fixed previous origin result SHA f37c73017d22d0b8ac26d9e9bc08364817626e56ce2922c33e74b6f5b4dcab33. This comparison checks only per-bit aggregate votes and decoded bits in the prior result.json; it does not compare per-frequency votes. The prior origin run has detailed sidecars, which this comparison does not load. History disagreement or absence does not trigger selection or reruns.

All sixteen posthoc reads retain full44-time/channel summaries. Fixed PREPEND-minus-ORIGINAL and TAIL-minus-ORIGINAL differences are descriptive, including every final bit and every time/channel group. K1 is explicitly WRONG_KEY; it is not an FPR estimate. Per-time majority diagnostics do not replace canonical decoding from all1320 votes per bit. Receiver indices are labels, not independent four-frame supports or proof of matched source receptive fields.

The model config is pinned to revision0fad780a534b6463e45facd96134c9f345acfa5b. Runtime remains diffusers0.39.0; the config's0.33.0.dev0 metadata is not the runtime version. Official0.39 encoding clears cache, encodes the first input frame, processes subsequent four-frame chunks with cross-chunk cache, then clears cache; chunks are not independent. Shared VAE helpers are unchanged. TAIL only changes the last input frame, with early/late differences reported descriptively and no hard scientific PASS.

The draft notebook has five code cells and the exact initial Drive mount. SOURCE_SHA=None stops before output directory creation. It reuses the successful origin eight dependency pins, necessary fixed repair, process-group cleanup, failure persistence and Wan-only probe. Actual versions and pip-check warnings remain recorded; no GPU model-name gate or unrelated dependency repairs.

After review and source publication, the upper reviewer binds the immutable source:

    python -B scripts/build_video_trajectory_receiver_prepend1_notebook.py --source-sha <published-source-SHA>

Only CPU/fake/static validation is performed here. Real Run all is left to the user after publication. This known-start controlled diagnostic is not blind synchronization, recovery of f0 or historical cache, proof of unique phase causality, generalization or scientific PASS. Stage2 is not implemented or executed in this Stage1 notebook; contingent on Stage1 result review. M05 estimated-offset correction, R22, strength and smoothing changes are outside this Stage1 implementation.

Targeted local validation:

    /home/richar/projects/CEG-WM/alive/CEG-WM/.venv/bin/python -B -m pytest -q tests/test_video_trajectory_receiver_prepend1_v1.py -p no:cacheprovider
