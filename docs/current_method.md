# Current released method and remaining research

Updated 2026-10-04. **main is the only authoritative release branch.**
Its executable scope is the complete GROW fixed video reference, independently
usable from a clone or extracted source ZIP.

## Established limited mechanism

The retained run 20260929T124706736859Z demonstrates native Wan MULTI25–49
generation-time control, real frozen VAE decode, RGB8/H.264 MP4 save and blind
reading after independently reopening that MP4 and posterior-mode reencoding.
MULTI and LAST recover the fixed 32-bit payload exactly. All 3 videos,
24 readouts and 48 post-read evaluations remain recorded.

The payload is repeated across latent time. The receiver uses full fixed
geometry; it does not recover temporal alignment. MULTI includes the successful
last step, so its superiority/necessity is unproven and is not required for
publishing this limited closed chain. Quality remains diagnostic.
See [construction and real-run evidence](grow_video_reference_v1.md).

## Components in the released tree

- Pure FFT-real payload construction, keyed coordinate assignment, local
  mean-MSE control and truth-free bit extraction.
- Frozen Wan model/VAE loaders, native scheduler stepping and tensor identity.
- Wan normalization/cache handling, RGB8/H.264 save and independent RGB24 read.
- Fixed OFF/MULTI/LAST experiment with separate generation/media workers,
  blind readout sealing, post-read message evaluation and diagnostic quality.
- Git-or-manifest source records, reproducible notebook and standalone checks.

The old terminal state-clock, four-bit RM/44–46 combined execution chain, their
CLI/configs/notebooks and exclusive tests are removed from the current tree.
They are unnecessary dependencies for GROW. Historical standalone mechanisms
are not thereby declared false; their actual evidence stays at its original
scope. [Immutable historical entry index](historical_entries.md).

## Development work outside main

The proposal remains local tube carriers + state-space synchronization written
through real multi-step generation control. Sampling-step time and video time
are distinct. The open same-version chain includes:

1. Local time-state observations surviving the actual VAE/media channel.
2. Blind temporal paths, declared equivalence/ambiguity and support rejection.
3. Time-dependent segment payload whose recovery benefits from alignment.
4. Segment/sequence aggregation with overlap, erasures and conflicts handled.
5. Declared crop/delete/repeat/speed edits and corresponding negative inputs.

OLD8 has terminal/DIRECT path evidence but fails current MP4 path recovery.
Difference, LOWBAND16, CONTRAST and double spatial replicas do not close that
gap. DWELL4 is CPU construction only, with reduced path separation and no new
media result. None of their executables is included in this release.
[Bounded development evidence index](development_evidence_index.md).

Affine-invariant synchronization remains optional when observations justify
it. Arbitrary insertions and interpolated frames require separately declared
models. Repeated payload, local peaks, path correctness and method completion
are distinct claims.

## Release evidence

Source content is tracked by release_manifest.json even without Git.
The release checks target this GROW chain and run its CPU/fixture tests from a
temporary no-.git copy, including save/read/evaluate and failure paths.
External ordinary dependencies and model assets are not bundled.
Engineering release completeness does not imply full research completion.
