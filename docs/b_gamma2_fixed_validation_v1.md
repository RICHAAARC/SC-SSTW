# Fixed gamma2 paired validation

This local implementation follows the adopted fixed four-source proposal. It constructs PM2 from each saved P0 terminal using the true margin `max(0, 2-q0)/||d||² * sign*d`. It preserves the terminal write position, key, channels4:16, four ROIs, directions and the existing receiver. It does not multiply PM1−P0 by two, regenerate trajectories, use DiT/gradients or add E181 encoding.

## Colab notebook delivery

Open [the gamma2 Colab notebook](https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/dev/local-joint-state-payload-v1/notebooks/b_gamma2_fixed_validation_v1.ipynb) and choose Run all. No manual source ZIP upload is required. The first code cell only mounts Drive. The next cell directly downloads the available `dev/local-joint-state-payload-v1` source and locates `experiments/wan_state_clock/b_gamma2_fixed_validation_v1_run.py` in an isolated directory. Optional `SOURCE_DIRECTORY` is copied into that directory; the user's source directory is not edited. Actual unreadable sources, unsafe paths and failed imports remain errors. No embedded source overlay, hash, version equality, manifest or Git-clean admission gate is used.

The setup reuses the successful three-group dependency probe and missing-package repair, including ftfy and the actual FqnToConfig/optional torchao conflict. It keeps FFmpeg/ffprobe setup and the shared Drive FP32 VAE cache selection, posterior mode and cache resets. It does not instantiate a generation pipeline.

Input defaults to `/content/drive/MyDrive/Video-WM/B-Terminal-Three-Group-V1/fixed-four-new-v1`. New output defaults to `/content/drive/MyDrive/Video-WM/B-Gamma2-Fixed-Validation-V1/fixed-four-source-v1`. Preserve this output after failure or interruption. An execution artifact makes subsequent Run all read-only before download, source copy, setup or model load. Setup logs alone do not spend the experiment. The runner writes attempt and fixed plan first; started work is never retried or silently replaced.

CLI: `python -m experiments.wan_state_clock.b_gamma2_fixed_validation_v1_run --input ORIGINAL --output NEW_OUTPUT`. `--help` and read-only recovery require no model. Complete plans, references, per-bit outcomes and failure details remain in files; the UI shows summaries.

## Fixed budget and common threshold

Only PM2 has new physical observations: four D181,144 E41, and16 each of codec encode, probe and decode. E41 receives5904 frames; D181 returns724 frames; each codec direction handles2536 frames. These are maximum one-shot counts. The first failure stops subsequent work and leaves fixed NOT_ENTERED slots. Internal VAE block counts are separate from top-level calls.

Calibration sources c01/c02 produce only wrong-key state observations, with no payload/DCT. Eval sources e01/e02 share each E41 between both keys and one DCT projection. New logical counts are216 Q records,54 H137 families/7398 scores,72 eval DCT projections and144 key decodes. State construction/native audits are arithmetic and do not add VAE calls.

The existing background models remain fixed. The two existing exactly90 calibrations use shared72 OFF/P0 null families plus18 PM1-wrong or18 PM2-wrong families. Both must complete before the outer threshold freezes `tau*=max(tau1,tau2)`. This is108 unique null families/14796 candidate slots, not180 independent observations. Eval PM2 physical work begins only after common calibration. Correct-key PM, eval sources and payload answers do not enter calibration.

Detection preserves `maxS > tau*`; candidate membership preserves `S >= tau*`. At equality, C can be nonempty while D is false. Full H137 competitors and all four phases are retained; computing137 scores does not mean137 VAE encodes.

## Cached controls and blind interpretation

OFF/P0/PM1 raw families and phase bit files are reused. The original plans supply only anonymous input/query and fixed-window metadata. A small reader stops at the `regions` field of old bit JSON after reading its leading metadata, so it can recover all physical view IDs without deserializing bit values. Old raw families may lack IDs because the previous runner added those fields after saving; IDs are recovered from the original plan. New C, D, paths and aggregates are rebuilt under the common threshold. Old accepted bits or old C are not substituted for new results.

All48 new eval plans are saved and read back before lazy caches can select bit values. Only after all blind decisions are saved/read back does post evaluation select payload truth. Gamma1's original-tau results remain a distinct historical baseline.

Four arms ×2 eval ×3 fixtures ×2 keys give48 tasks. Each aggregate retains1536 slots:384 positives and1152 unique nulls. Each gamma has192 positives and960 nulls, sharing768 OFF/P0 null slots. Main CDC_M and auxiliary CDC_E/R0/R1/UNION/alias/reverse are recorded. This task excludes independent cross-source bundles.

## Post evaluation and evidence ceiling

Legacy `assess` fields remain intact. Separate `actual_supporting_exact` is computed from actual `supporting_refs`, requiring at least one truly exact same-group source cell. An exact selected reference that did not support the emitted value does not qualify.

The adopted development rule requires e02 to gain at least one supported correct CDC_M slot in every fixture relative to PM1 under common tau, e01's historical96 successful slots to remain, no loss of existing common-threshold supported correct slots, zero specified common/added literal errors, false attribution and eval null emissions, and complete required observations. Missing technical evidence is UNRESOLVED, never a reduced denominator or a tau fallback. Success is development support, not192/192 by definition; exact total recovery is reported separately.

Quality adds only36 P0→PM2 records (four sources × nine scope/layers),5796 frame differences and5760 adjacent differences. It reuses the36 historical P0→PM1 quality records, reports clipping and native update energy, and never silently regenerates missing old media. RMS/PSNR/outside/adjacent fields are descriptive; there is no quality PASS or new PSNR threshold.

These sources and gamma1 outcomes were already viewed when gamma2 was selected. Even a successful run is finite development evidence, not independent confirmation, population FPR/CFAR, general deletion robustness, unique localization or source authentication. Native q=1 was a chosen margin rather than a physical ceiling. The current write is terminal, with no subsequent sampling tail to dilute it.

## Local validation boundary

Targeted tests use CPU arithmetic, synthetic saved-file fixtures and a fake physical backend. They cover the gamma2 active-set change, true-margin shortage audit, calibration/count boundaries, actual-support distinction, cache ordering, first-failure/hard-exit behavior, direct source loading from a candidate publication archive, and setup-before-recovery prevention. They do not load model weights or invoke real VAE/FFmpeg work. The notebook and builder must be rebuilt together after source changes.

Implementation details verified in the local revision: protocol parsing selects only scheduling/model/key fields and skips writer.bits before post; all four saved P0/native-Q inputs are read and validated on CPU before any physical call. Recovery restores the full fixed plan even if ledger.json is missing. Failed blind operations retain their 32-slot status; quality failures retain all36 records and frame/adjacent denominators. Positive emitted bits without actual exact supporting evidence have a separate zero-output requirement, while legacy false-attribution fields remain intact. Historical-versus-recomputed tau1 differences are recorded, never used as an equality gate.
