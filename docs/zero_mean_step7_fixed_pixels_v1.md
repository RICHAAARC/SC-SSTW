# Frozen original step7 pixels: encoder replay, raw420 and MP4

Click **Run all**. Reuses original step7 POINT4/rgb8.pt and its saved positive raw444 roundtrip.rgb from run20261001T170749052584Z. The pixel files and raster hashes are fixed. No terminal decode or new watermark update. The later MP4 run's newly decoded pixels are not used.

Three observations use one fixed native FP32 VAE encoder:

- SAVED_RAW444: directly re-encode the exact saved positive raw444 RGB. This isolates encoder/readout replay from decoder and color-conversion replay.
- RAW420: original saved RGB8→materialized raw YUV420→reopened RGB24→VAE encode.
- MP4: the same original saved RGB8→existing libx264 MP4,CRF18,8fps,yuv420p→independent RGB24 readback→VAE encode. MP4 remains the fixed primary endpoint.

Three observations, six path and payload reads, twelve message evaluations,1044 valid costs. Three VAE encodes, zero VAE decodes, zero gradient or Transformer calls. Same g0/R44 reader, all174 valid candidate paths, both keys and both messages. No candidate deletion, threshold change, best-point selection or scan. Each branch preserves failures and the other branches can finish independently.

The original anchor had rank1 and Delta=5.4846719567056e-7. The last fresh-decode run changed its pixel raster and yielded raw444 rank2 and MP4 rank46. This run removes decoder replay variation by holding the original pixels byte-exact. Same input bytes do not force identical VAE outputs: normalized max/RMS errors and correct-key path-cost changes are reported, along with actual package versions. No hard requirement to recreate an older CUDA build or use a particular GPU.

If the anchor changes, encoder execution is implicated without decoder ambiguity. RAW420 measures the color420 transport contrast; MP4 versus RAW420 assesses additional standard codec transport. Internal color conversions are not assumed algebraically identical: commands, verbose raw logs and MP4 stream metadata are retained. This is a diagnostic contrast, not unconditional attribution to a single codec operation.

All181 frames remain320x512. Historical negative controls remain references, not fresh independent negatives. Quality is relative to marked original RGB8, saved raw444 and current raw420, not OFF. Exact repeated payload is a separate readout from path recovery. These fixed-source, fixed-phase results do not establish unknown-phase/fragment synchronization, calibrated FPR or generation-time trajectory watermarking.

Existing current-Python setup and fresh-child execution are reused. Local CPU/static validation uses simulated model execution; real pretrained VAE execution is left to the user. Source is immutable when bound below.

Input result SHA256:96023f82ca5eed73c57ae161a22bc3e45d79321dc127951c8767ddc60ad49827. RGB8 file SHA256:4fc1a96ce08f5b0aac1aebf5e43a4a37f133bb6fc37f3895c0ce049399d0175d. Pixel raster SHA256:134d804b3c7740bf314109c956c590ba66297eb366cb28609aaf6d6bf2a25017. Existing method/receiver/runtime files stay unchanged. Publish source before binding the notebook.
