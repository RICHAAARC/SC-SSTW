# Residual Clock V1 user-run handoff

This package runs two fixed, causally separate measurements over
`confirm_01`, `confirm_02`, `confirm_07`, and `confirm_03` from checkpoint
`20261012T032440550166Z-c3026cfb` and layer readout
`20261014T003005600270Z-bad88775`.

The residual group constructs exactly one RGB8 PRE candidate:

```text
uint8(clip(int16(NATIVE_PRE) + int16(M05_PRE) - int16(RECON_PRE), 0, 255))
```

It retains 12 payload rows (four cases by NATIVE, current M05, and candidate)
and eight descriptive clock rows (candidate and native/no-clock). The clock
selector is the existing blind `CENTERED_U` path. Identity truth is joined only
after the blind selection is saved. A path, positive gap, `SUPPORTED`, or any
other selector status is not a clock-presence result.

The separate `delete_g0` group reads each old received M05/K0 clip in received
order using its first 161 frames at R40. It compares these four NO_SYNC rows
with four saved blind rows using 181 restored frames at R44. Every complete
saved EVALUATED R44/32-bit blind row is retained independently of oracle-map
equality. Map equality is a post-hoc boolean diagnostic. Path error comes from
the saved receiver path metrics when present and is explicitly unknown when no
saved path-error evidence exists; map mismatch is never converted into a path
error. The reported
`NO_SYNC bit errors - saved blind bit errors` measures the whole synchronization
recovery operation, including restored support length. It cannot be attributed
to the clock score alone or to the new residual writer. An unavailable pair
remains in the four-pair denominator and makes the complete four-case delta
unavailable.

## Planned model work

- One uninterrupted Wan load and eight encode clips: four residual candidates
  and four old NO_SYNC received clips.
- One uninterrupted framewise load and four candidate clips, comprising 92
  physical batches at batch size eight.
- Saved native framewise latents are reused. If they are unreadable, at most
  four native fallback clips add another 92 physical batches.
- At most four unique saved MP4s can supply missing received RGB8 caches. An
  uninterrupted run needs at most four readbacks. Interrupted readback attempts
  may be retried, so cumulative actual attempts can exceed four and are retained.
- The workflow performs no generation/DiT call, VJP, Wan decode, oracle encode,
  new attack, or codec encode.

## Outputs and resume behavior

Run
`notebooks/paper_results_v1_residual_clock_v1_colab.ipynb` with **Run all**.
Its first code cell only mounts Drive. A fixed active pointer selects one Drive
task directory under `Paper-Results-V1-Residual-Clock-V1`.

`run_state/residual_clock_state.json` is the authoritative mutable state.
Latents, RGB8 inputs, vote sidecars, score matrices, blind selections, post-hoc
path errors, and quality records are flushed and atomically replaced before a
unit is marked complete. The journal writes a durable snapshot every 60 seconds
and at phase exit. A new process changes stale `RUNNING` attempts to
`INTERRUPTED`, skips complete units, and retries only unfinished units. Ordinary
per-unit corrupt or unreadable cached artifacts become explicit `FAILED` or
`MISSING` rows while later fixed units continue.

The handoff directory includes `summary.json`, `evaluation_report.json`, three
fixed-row CSV files, `attempts.json`, `stage_receipts.json`, `execution.log`,
and the complete `run_state` directory. Reports preserve all 12 residual
payload rows, eight residual clock rows, and eight delete_g0 comparison rows.

## Local validation boundary

CPU/stub checks cover the integer candidate formula, eight-clip Wan schedule,
R44/R40 reads, native-latent reuse, 92-batch accounting, interruption recovery,
per-unit corrupt-latent isolation, fixed four-pair delta semantics, notebook
syntax, isolated companion imports, and ZIP/source equality. Local saved JSON
records also confirm the eight reused payload-control rows and four saved blind
rows. Their current maps happen to equal the offline oracle, but that
observation is not an eligibility condition.

These checks do not run a real model, GPU, Colab, Drive, codec, or saved MP4
readback. The user-run output is required before making claims about payload
survival, clock visibility, synchronization value, or scientific success.
