# Native WAM / PixelSeal development baseline

Entry: `notebooks/native_baselines_v1.ipynb`; builder: `scripts/build_native_baselines_v1_notebook.py`. User runs Colab GPU; local checks use artificial adapters/media only. Source release is a separate delivery action; the default remote notebook URL works once these necessary files are published on the selected editable branch. An existing source directory also works, without Git metadata.

The four real input locations are in `experiments/wan_state_clock/configs/native_baselines_v1.json`: MyDrive/Video-WM/B-Rank3R-New-Source-Confirmation-V1/fixed-four-five-v1/m4_inputs/b-rank3r-m01..m04/OFF/precodec.rgb24. Metadata and producer OFF semantics were checked by principal; actual bytes are read/validated by the user-run runtime. No identity BASE, replacement source, automatic generation or model fallback.

WAM uses the public MIT32 checkpoint, ImageNet normalization, official constructors, explicit successful state loading and JND1_3_blue, scaling2.0. It writes one source message per frame in center `[80:240,128:384]`. Nonempty predictions use the official semihard32 decoder; empty masks become wrapper UNKNOWN. PixelSeal is native256, scaling0.2/JND1_1, chunk32/step8, full-frame video writer. One detect call per received view, bilinear/align_corners=False/antialias=True, all-frame mean logits then >0. No second extract_message call.

FULL and CENTER_RGB are actual received pixel views after one H264 CRF18/yuv420p generation at8fps. CENTER_RGB preserves all WAM writing support and25% of the PixelSeal frame; this does not test arbitrary unknown fragments. Native capacities/coverage/repetition differ. Same-source precodec and same-source decoded quality pairs remain separate. No time recovery, WAM temporal vote or calibrated FPR claim.

Full plan:2 loads;724 WAM image embeds/2896 detects;4 PixelSeal video embeds/16 detects;12 each encode/probe/decode;8 precodec plus8 decoded quality pairs. PixelSeal expected4 embedder/96 detector forwards,92 key frames/2896 detected frames.32 reads/5792 frame rows; WAM92672 frame bits, PixelSeal741376, each half positive/control. PixelSeal additionally16 video messages/4096 bits. Failures and not-started slots remain in their original denominators.

Output: `/content/drive/MyDrive/Video-WM_Result/baselines/native-wam-pixelseal-dev-v1/attempt01`. Attempt is reserved before setup. A failure/interruption consumes it; later Run all reads saved outputs without setup, media or model retry. No cleanup of this result directory. `/content/video-wm-native-baselines` holds temporary sources, weights and caches. One notebook instance at a time; the ordinary receipt is not a distributed lock.

`plan.json`, `events.jsonl`, `ledger.json`, `attempt.json` track actual calls. Blind frame files, packed WAM masks, sufficient selected-logit sums/counts, PixelSeal frame logits and video outputs are saved before `post_truth_access.json`. `evaluation.json` and `frame_results.csv` include all planned frame rows; `quality.json`, `models.json`, `result.json` retain quality/settings/cost/failure. `media/` retains twelve planned full videos when returned. Core JSONs also receive ordinary cloud copies. A path or upload is not a scientific PASS.

Standalone CLI in a prepared environment with official source/weights under the chosen vendor directory:

```bash
python -m experiments.wan_state_clock.native_baselines_v1_run --config experiments/wan_state_clock/configs/native_baselines_v1.json --vendor /content/video-wm-native-baselines/vendor
```

It executes the same bounded runner without notebook/scripts and treats an existing attempt as read-only. No model imports occur for `--help`. Missing real dependencies or weights are explicit failures; there is no silent retry. Colab bootstrap prepares imports/source/weights automatically before calling the internal reserved worker. CPU fixture checks: `python -m unittest tests.test_native_baselines_v1 -v`. These are not real GPU/codec results.
