# Temporal-attack two-pilot recovery handoff

Use [`paper_results_v1_temporal_attack_recovery_colab.ipynb`](../../notebooks/paper_results_v1_temporal_attack_recovery_colab.ipynb) for saved run
`20261010T031617441784Z-2536172d`. Download the notebook, upload it to Colab,
select a GPU runtime with sufficient memory, authorize the first-cell Drive
mount, and choose **Run all**. Return the unique directory printed by the last
cell. A source-run pointer in Drive makes a later Run all reopen that same
recovery directory instead of creating another attempt.

The source run remains unchanged. Its audit is at
`diagnostics/a-line-temporal-attack-20261010T031617441784Z-2536172d-audit/`.
That audit records 480 fixed main rows as 30 `EVALUATED`, 448 `FAILED`, and 2
`UNSUPPORTED`; all 60 baseline rows failed; 10 of 14 quality rows evaluated;
and all eight confirmation sources remained unexecuted. It inventories 116
nonempty MP4 files: 112 main temporal edits and four external-baseline native
publications. The four baseline embeds, four baseline native codec roundtrips,
eight FULL clock encodes, and 22 FULL physical Wan encodes are source-run calls
and stay in a separate ledger.

Recovery copies the old fixed-denominator state and failure history into a new
directory. It opens and validates every reused MP4 as RGB8 with its declared
frame count and 320×512 geometry. It also opens the retained FULL clock
JSON/NPZ, read JSON/vote NPZ, ten quality records, and four 1024×320
side-by-side videos. Paths nested in retained JSON are relocated to the chosen
source run. Missing `/content` RGB caches are decoded again from persisted MP4;
this is a decode, not another publication. One unreadable artifact fails its
own fixed row and does not stop independent rows.

Only this work is new:

- 56 baseline temporal-edit codec roundtrips;
- 112 main framewise clock encodes and 448 logical main receiver rows;
- at most 336 new physical Wan encodes, with the actual map/cache count saved;
- 60 baseline extraction calls; and
- four baseline-versus-OFF quality rows.

The 112 main temporal publications, four baseline embeds, and four baseline
native publications are never regenerated. The 32 FULL receiver rows,
including two `UNSUPPORTED` rows, are reused without reinterpretation.
Confirmation remains `NOT_EXECUTED_BY_NOTEBOOK`. The report keeps all 480 main,
60 baseline, and 14 quality pilot rows plus the complete confirmation
denominators.

Each expensive call is recorded before execution and updated immediately on
ordinary return or failure. A completed or interrupted call receipt is not
silently repeated. If a kernel ends after temporary RGB deletion, the index
stage rematerializes only the local decode cache from saved MP4. Evaluation is
recomputed on every continuation so a report produced during an interruption
cannot hide later completed rows. Phase `COMPLETE` means the phase traversed
its fixed items; the handoff summary separately reports item statuses and the
old/new actual-call ledgers.

This repository task did not execute Colab, Drive, ffmpeg, any model, VAE, GPU,
or package installation. GPU peak memory, elapsed time, actual physical Wan
alias count, and recovered scientific outcomes remain evidence for the user-run
directory.
