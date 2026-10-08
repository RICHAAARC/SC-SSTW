# SC-SSTW trajectory-sampling video watermark mechanisms

`main` is the release branch. The research position is **基于轨迹采样嵌入思想的视频水印方法**. GROW remains a concrete implementation source rather than the umbrella method name: the [author repository](https://github.com/luopengchen/GROW), retained implementation commit `6aa69a9c5d4a9e75df457fcca8dfc71b64a6b870`, and the [CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Luo_GROW_Watermark_Generation_with_Progressive_Guidance_for_Diffusion_Models_CVPR_2026_paper.pdf) remain attributed. The executable FFT-real construction and its distinction from the paper's DCT setup stay documented in [the fixed GROW reference](docs/grow_video_reference_v1.md). This tree keeps the complete GROW payload chain and a narrow whitelist of measured terminal synchronization/receiver mechanisms. Conditional joint V1 is integrated from its audited source closure; this release integration does not merge the development branch or claim a new main-version real-model execution.

| Entry | Fixed scope |
|---|---|
| `grow_video_reference_run` | Unchanged native 50-step OFF/MULTI/LAST GROW, 3 MP4 / 24 reads / 48 evaluations; its payload reader retains all 46 latent times |
| `video_trajectory_payload_framewise_sync_m05_v1_run` | Explicit FULL181 source RGB; fixed M05 target .5, framewise encode/write/decode, single 8fps CRF18 H.264 roundtrip, fresh FULL sync and payload reads |
| `video_trajectory_receiver_estimated_align_v1_run` | Explicit saved FULL181/CROP177/SHORT89; new blind per-key offset/phase then unchanged R44/R44/R22 payload reading |
| `video_trajectory_internal_single_deletion_v1_run` | Fixed C/D 177-frame construction, 709 H0/H1 paths/key, complete-map caching, blind modes sealed before oracle; message evaluated last |
| `video_trajectory_conditional_joint_v1_run` | Fixed self-generating conditional chain: PAYLOAD_MULTI → M05 → two MP4s → 18 blind sync reads → 40 blind payload slots → 4 post-seal oracle slots; one seen development source |
| `video_trajectory_attribution_v1_run` | Research-branch fixed DEV→C1/C2 attribution protocol; its user-run audit retained 192 slots and observed 16 confirmation ACCEPT / 112 REJECT / 0 UNCERTAIN |
| `video_trajectory_attribution_uncertainty_v1_run` | Research follow-up over six exact saved C1/C2 RGB inputs: 22 fixed queries test sync and exact-zero identity uncertainty under the unchanged frozen rule |

Run from the project root of a clone or extracted ZIP. External weights and standard runtime dependencies are not bundled. First install a compatible Torch/Torchvision pair for your platform, then:

```bash
python -m pip install -r experiments/wan_state_clock/requirements-grow-video-reference.txt
python -m experiments.wan_state_clock.grow_video_reference_run --output /absolute/new-grow-output
```

Install FFmpeg and ffprobe with your normal system package manager if absent. GROW and M05 use them for the fixed MP4 roundtrip. The two receiver-only entries consume already saved RGB bytes without codec fallback. `requirements-dev.txt` adds CPU validation tools; it is not a replacement for runtime dependencies. The new Colab notebooks install and record the fixed eight-package runtime used by their original successful versions.

The three new entries require an explicit configuration and a fresh output directory. For example, prepare an M05 job by copying the public template and its companion, then setting the input identity:

```bash
mkdir -p /absolute/m05-job
cp experiments/wan_state_clock/configs/video_trajectory_payload_framewise_sync_m05_v1*.json /absolute/m05-job/
python - <<'PY'
from pathlib import Path
import hashlib, json
job = Path('/absolute/m05-job')
cfg_path = job / 'video_trajectory_payload_framewise_sync_m05_v1.json'
cfg = json.loads(cfg_path.read_text())
rgb = Path('/absolute/source-full181.rgb8')
assert rgb.stat().st_size == 181 * 320 * 512 * 3
cfg['source'].update(path=str(rgb), sha256=hashlib.sha256(rgb.read_bytes()).hexdigest())
# Set the companion's expected message to the source's known payload before running.
# It is evaluated only after the blind seal; do not choose it from decoded results.
for field in ('preparation_config', 'oracle_config', 'posthoc_config'):
    if field in cfg:
        companion = job / cfg[field]
        cfg[field + '_sha256'] = hashlib.sha256(companion.read_bytes()).hexdigest()
cfg_path.write_text(json.dumps(cfg, indent=2) + '\n')
PY
python -m experiments.wan_state_clock.video_trajectory_payload_framewise_sync_m05_v1_run --config /absolute/m05-job/video_trajectory_payload_framewise_sync_m05_v1.json --output /absolute/new-m05-output
```

Inputs are headerless, contiguous `uint8` RGB in `[frames,320,512,3]` order. Copy all same-prefix JSON companions for the selected entry. After editing any companion, recompute its public `*_config_sha256` using the loop above; relative companion paths resolve against the public JSON directory.

| Receiver entry | Input edits before execution | CLI module suffix |
|---|---|---|
| Offset/phase | Set each `inputs` entry's `path` and `sha256` to saved FULL181, CROP177 and SHORT89 RGB respectively; set posthoc labels/message to the known construction | `video_trajectory_receiver_estimated_align_v1_run` |
| Single deletion | Set `source.path` and `source.sha256` in `_preparation.json` to the saved FULL181 RGB, then refresh its hash in the public JSON; retain the fixed C/D maps and oracle geometry | `video_trajectory_internal_single_deletion_v1_run` |

Use `python -m experiments.wan_state_clock.<module suffix> --config /absolute/public.json --output /absolute/new-output`. Preparation may know the construction; blind selection does not consume its truth. Oracle semantics are parsed after blind payload sealing and message evaluation follows the oracle seal.

Geometry, models, public denominators and method parameters remain fixed. Input configuration SHA and full runtime content identity are recorded. Templates have no implicit historical Drive source. Exact old configuration bytes live under `experiments/wan_state_clock/configs/historical`; their original-version records are historical, not new main measurements. Use the original commit links in the mechanism index for exact historical execution. Failures stay in the fixed slots.

Current user-run notebooks: [GROW](notebooks/grow_video_reference_v1_colab.ipynb), [M05](notebooks/video_trajectory_payload_framewise_sync_m05_v1_colab.ipynb), [offset/phase](notebooks/video_trajectory_receiver_estimated_align_v1_colab.ipynb), [single deletion](notebooks/video_trajectory_internal_single_deletion_v1_colab.ipynb), and [conditional joint V1](notebooks/video_trajectory_conditional_joint_v1_colab.ipynb). The research branch carries the published [trajectory attribution V1](notebooks/video_trajectory_attribution_v1_colab.ipynb) and the source-draft [uncertainty follow-up V1](notebooks/video_trajectory_attribution_uncertainty_v1_colab.ipynb). The latter reuses audited saved RGB and does not generate or re-encode video. The conditional notebook uses its repository-bundled fixed configs and regenerates the fixed source; it does not require an external `CONFIG_PATH` before clone.

Notebooks begin with the independent Drive mount and use a published immutable source. For the three saved-input notebooks (M05, offset/phase and single deletion), set `CONFIG_PATH` to the prepared input JSON before user Run-all. Conditional joint V1 instead uses its fixed bundled config and regenerates its fixed source. Source candidates use `SOURCE_SHA=None` and stop before output creation. Each newly published notebook binds its own reviewed immutable source S; existing notebooks retain their recorded bindings and are not rewritten for this follow-up. Agents do not execute model notebooks.

A ZIP has no Git identity: its result records `source_sha=null` plus the SHA-256 content identity and manifest match. A checkout with its own root `.git` also records its actual commit/status; an unrelated parent repository is not attributed to the copied source. The manifest is an integrity/reproducibility record, not a signature or scientific threshold.

See [current mechanisms and limits](docs/current_method.md), [versioned mechanism evidence](docs/mechanism_evidence_index.md), and [original GROW protocol](docs/grow_video_reference_v1.md). OLD8, DWELL4, T05 and abandoned backprop/combined methods remain outside main.

CPU release validation: install `requirements-dev.txt` in a separate environment, then run `python governance/tools/run_validation_profile.py release`. This validates notebook schema and strict published binding, layer boundaries, the content manifest, and full tests in an independent no-`.git` copy. Draft-source candidates intentionally retain the publication-binding failure until N; it is not a scientific verdict.
