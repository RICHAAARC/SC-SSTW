精简公开 Baseline：[`native_baselines_public_compact12_v1.ipynb`](notebooks/native_baselines_public_compact12_v1.ipynb)，CAL32＋TEST500、四配置×12视图；[范围与预算](docs/public_compact12_v1.md)。完整论文入口已默认此流程，原v4独立保留。

有限独立低FPR确认入口：[Colab Run-all](https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/main/notebooks/paper_fpr_validation_v1.ipynb)，复用原TEST-B八源与原查询包，零新增源/推理预算；[概率口径](docs/paper_fpr_claims_v1.md)。

# SC-SSTW trajectory-sampling video watermark mechanisms

> 完整论文实验入口：[paper_suite_v1.ipynb](notebooks/paper_suite_v1.ipynb)（GPU Run all）。接齐十二源C4、原公开Baseline续接、十二源同域五视图对照、QMatch正式统计与论文表格；[范围与固定预算](docs/paper_suite_v1.md)。已有TEST-A与旧失败保留，不用单个TEST-B入口冒充整套实验交付。

> 2026-11-08：main 已选择性整合 [C4 正式时间与条件载荷入口](docs/b_paper_cohort_v1.md)。保留 DCT2＋CDC32、Joint-A/F42/relative、QMatch 和旧 H/L；默认 TEST-A 四源，已消费结果只读，TEST-B 不自动启动。核心、runtime、实验编排、薄 notebook 分层独立。

保存证据的独立只读入口见[报告协议](docs/paper_results_v1/evaluation_protocol.md)、[完整结果索引](docs/paper_results_v1/results_index.md)和[被动 native 接口状态](docs/paper_results_v1/baseline_status.md)。`python -m experiments.paper_results_v1.cli --help` 仅依赖 Python 标准库；原36/P2196账及失败完整保留，不新增模型执行或公平 baseline 结论。

> 2026-10-24：已精选整合[公共多步分段载荷](docs/segmented_payload_v1.md)：50步中25..49控制，四段各8位，净32位；只提供给定时间映射及完整支撑下的读取，不包含A/B盲同步研究。已有真实结果沿用原实验分母，本次整合未重跑模型。分支用途见[入口索引](docs/branch_guide.md)。
>
> 当前用户规则：源码、配置及环境身份仅供定位，B64/哈希/manifest/Git状态/固定版本不作为运行或交付门禁。当前五个 Notebook 与公共载荷入口均直接使用可用源码。

公共载荷独立入口（无需Notebook或Git元数据）：

```bash
python -m experiments.wan_state_clock.segmented_payload_v1_run --help
```

`generate-latent`执行原生50步并保存终态；`plan`、`read-votes`、`read-latent`、`read-rgb`接收调用方明确给出的时间映射。模型运行由用户启动；纯保存票读取不加载模型。具体格式、命令和证据边界见[公共机制说明](docs/segmented_payload_v1.md)。

> 2026-10-09 当前状态：[正式要求与证据对照](docs/research_status_2026-10-09.md)。Attribution V1 已真实运行并审计：C1/C2 两个确认源为 16 ACCEPT / 112 REJECT / 0 UNCERTAIN / 0 技术缺失。后续不确定补充 22/22 完整，16 ACCEPT / 6 REJECT / 0 UNCERTAIN；两类不确定共用同十个探测查询，均 0/10，**完整但未覆盖**。两项均已在研究分支发布，实现尚未进入 main。查询数不是独立来源数；不宣称总体 FPR 为零。

`main` is the release branch. The research position is **基于轨迹采样嵌入思想的视频水印方法**. GROW remains a concrete implementation source rather than the umbrella method name: the [author repository](https://github.com/luopengchen/GROW), retained implementation commit `6aa69a9c5d4a9e75df457fcca8dfc71b64a6b870`, and the [CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Luo_GROW_Watermark_Generation_with_Progressive_Guidance_for_Diffusion_Models_CVPR_2026_paper.pdf) remain attributed. The executable FFT-real construction and its distinction from the paper's DCT setup stay documented in [the fixed GROW reference](docs/grow_video_reference_v1.md). This tree keeps the complete GROW payload chain and a narrow whitelist of measured terminal synchronization/receiver mechanisms. Conditional joint V1 is integrated from its audited source closure; this release integration does not merge the development branch or claim a new main-version real-model execution.

| Entry | Fixed scope |
|---|---|
| `grow_video_reference_run` | Unchanged native 50-step OFF/MULTI/LAST GROW, 3 MP4 / 24 reads / 48 evaluations; its payload reader retains all 46 latent times |
| `video_trajectory_payload_framewise_sync_m05_v1_run` | Explicit FULL181 source RGB; fixed M05 target .5, framewise encode/write/decode, single 8fps CRF18 H.264 roundtrip, fresh FULL sync and payload reads |
| `video_trajectory_receiver_estimated_align_v1_run` | Explicit saved FULL181/CROP177/SHORT89; new blind per-key offset/phase then unchanged R44/R44/R22 payload reading |
| `video_trajectory_internal_single_deletion_v1_run` | Fixed C/D 177-frame construction, 709 H0/H1 paths/key, complete-map caching, blind modes sealed before oracle; message evaluated last |
| `video_trajectory_conditional_joint_v1_run` | Fixed self-generating conditional chain: PAYLOAD_MULTI → M05 → two MP4s → 18 blind sync reads → 40 blind payload slots → 4 post-seal oracle slots; one seen development source |

Run from the project root of a clone or extracted ZIP. External weights and standard runtime dependencies are not bundled. First install a compatible Torch/Torchvision pair for your platform, then:

```bash
python -m pip install -r experiments/wan_state_clock/requirements-grow-video-reference.txt
python -m experiments.wan_state_clock.grow_video_reference_run --output /absolute/new-grow-output
```

Install FFmpeg and ffprobe with your normal system package manager if absent. GROW and M05 use them for the fixed MP4 roundtrip. The two receiver-only entries consume already saved RGB bytes without codec fallback. `requirements-dev.txt` adds CPU validation tools; it is not a replacement for runtime dependencies. The Colab notebooks probe actual imports and repair missing imports when needed; version differences alone never trigger installation.

The three new entries require an explicit configuration and a fresh output directory. For example, prepare an M05 job by copying the public template and its companion, then setting the input identity:

```bash
mkdir -p /absolute/m05-job
cp experiments/wan_state_clock/configs/video_trajectory_payload_framewise_sync_m05_v1*.json /absolute/m05-job/
python - <<'PY'
from pathlib import Path
import json
job = Path('/absolute/m05-job')
cfg_path = job / 'video_trajectory_payload_framewise_sync_m05_v1.json'
cfg = json.loads(cfg_path.read_text())
rgb = Path('/absolute/source-full181.rgb8')
assert rgb.stat().st_size == 181 * 320 * 512 * 3
cfg['source'].update(path=str(rgb))
# Set the companion's expected message to the source's known payload before running.
# It is evaluated only after the blind seal; do not choose it from decoded results.
cfg_path.write_text(json.dumps(cfg, indent=2) + '\n')
PY
python -m experiments.wan_state_clock.video_trajectory_payload_framewise_sync_m05_v1_run --config /absolute/m05-job/video_trajectory_payload_framewise_sync_m05_v1.json --output /absolute/new-m05-output
```

Inputs are headerless, contiguous `uint8` RGB in `[frames,320,512,3]` order. Copy all same-prefix JSON companions for the selected entry. Relative companion paths resolve against the public JSON directory. Hash fields may be absent or retained as optional notes; no refresh is needed.

| Receiver entry | Input edits before execution | CLI module suffix |
|---|---|---|
| Offset/phase | Set each `inputs` entry's `path` to saved FULL181, CROP177 and SHORT89 RGB respectively; set posthoc labels/message to the known construction | `video_trajectory_receiver_estimated_align_v1_run` |
| Single deletion | Set `source.path` in `_preparation.json` to the saved FULL181 RGB; retain the fixed C/D maps and oracle geometry | `video_trajectory_internal_single_deletion_v1_run` |

Use `python -m experiments.wan_state_clock.<module suffix> --config /absolute/public.json --output /absolute/new-output`. Preparation may know the construction; blind selection does not consume its truth. Oracle semantics are parsed after blind payload sealing and message evaluation follows the oracle seal.

Geometry, models, public denominators and method parameters remain fixed. Source, config and environment location notes are optional. Templates have no implicit historical Drive source. Exact old configuration bytes live under `experiments/wan_state_clock/configs/historical`; their original-version records are historical, not new main measurements. Use the original commit links in the mechanism index for exact historical execution. Failures stay in the fixed slots.

Current user-run notebooks: [GROW](notebooks/grow_video_reference_v1_colab.ipynb), [M05](notebooks/video_trajectory_payload_framewise_sync_m05_v1_colab.ipynb), [offset/phase](notebooks/video_trajectory_receiver_estimated_align_v1_colab.ipynb), [single deletion](notebooks/video_trajectory_internal_single_deletion_v1_colab.ipynb), and [conditional joint V1](notebooks/video_trajectory_conditional_joint_v1_colab.ipynb). The conditional entry regenerates its adopted fixed development source; choose its public config and same-prefix companion files explicitly.

All five notebooks begin with the independent two-line Drive mount. Set `SOURCE_DIR` to available source (an unpacked directory without `.git` is valid) and `CONFIG_PATH` to an explicit readable public JSON before Run all. For GROW and conditional joint, the supplied fixed configs are valid explicit choices. `SOURCE_URL` is optional and used only when the source directory is absent. Existing source is never reset or overwritten. `SOURCE_SHA` is only an optional note. The selected source must actually contain the chosen runner; local unpushed changes are not already available from remote Colab links. Agents do not execute model notebooks.

Source or environment recording failures do not block execution. A directory without its own `.git` never inherits an unrelated parent's Git identity. Missing or changed manifests, hashes and exact versions do not prevent execution or reading results. Actual file, JSON, shape, numeric, method, fixed-denominator and blind-information errors remain explicit failures.

See [current mechanisms and limits](docs/current_method.md), [versioned mechanism evidence](docs/mechanism_evidence_index.md), and [original GROW protocol](docs/grow_video_reference_v1.md). OLD8, DWELL4, T05 and abandoned backprop/combined methods remain outside main.

For this record-only cleanup, use the bounded checks in `tests/test_record_only_execution.py` and `governance/tests/test_checks.py`. The release profile also includes physical CPU FFT/codec fixtures; do not run it when only static/fake checks are authorized. Notebook validation checks schema, the explicit entry and required inputs; it does not require a published SHA or content manifest.
