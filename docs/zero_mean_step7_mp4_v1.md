# Frozen step7: actual MP4 save and readback

Click **Run all**. Loads the audited total-step7 POINT4 terminal from run20261001T170749052584Z (true rank1, Delta=5.4846719567056e-7). The fixed primary observation is MP4. No optimization, step6 substitution, new generation or parameter scan.

One native FP32 VAE decode produces a single RGB8 raster for two fixed branches:

- RAW444: materialized RGB8→raw YUV444→reopened RGB24, the previous positive reference channel.
- MP4: the existing project encoder, H264/libx264, CRF18,8fps,yuv420p, followed by independent MP4 readback and native VAE encoding. All181 frames remain320x512; reader stays g0/R44, all174 valid candidates and both keys.

Two normalized observations, four path reads, four payload reads, eight message evaluations,696 valid costs. One VAE decode, two VAE encodes, one MP4 save/read; zero Transformer or gradient calls. This should avoid the previous gradient run's87.7minute workload and disk spool; actual runtime depends on model loading and hardware. No GPU-name gate is added. Existing current-Python dependency repair and fresh-child execution are reused.

The start terminal/result/parent raw444 bytes, normalized observation and both raw readers are checked against their saved identities. Historical OFF/PAYLOAD_MULTI/OVERLAP_MULTI controls remain references, not new FPR samples. Fresh RAW444 replay differences are reported, not hidden. A failed branch remains in the fixed denominator while the other branch can finish.

Inspect both branches' Delta, rank, nearest competition in retained costs, public-margin shortfall, payload and actual RGB quality. Quality is relative to the same marked input raster and raw444, not OFF. RAW444-vs-MP4 changes both chroma subsampling and lossy compression, so a difference does not isolate H264 alone. Exact repeated payload bits remain a separate readout from path recovery.

This is one frozen terminal, one fixed codec configuration, fixed phase and one source. It does not establish generation-time trajectory watermarking, unknown-phase/fragment synchronization, calibrated FPR or broad robustness. Source is immutable when bound below. Local validation is CPU/static with simulated VAE; real pretrained VAE execution is left to the user.


Parent result SHA256:96023f82ca5eed73c57ae161a22bc3e45d79321dc127951c8767ddc60ad49827. Frozen terminal SHA256:2fa1a8b4b6c3954c7627e7e986b1b5d7e186f7095e05cc76a1c96d8e1e2118a2. Existing method, runtime and receiver files are unchanged. Publish source first, then bind the immutable user-run notebook.
