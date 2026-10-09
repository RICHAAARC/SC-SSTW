# Local Joint State+Payload V1：最小真实机制实验提案

**状态：METHOD_ADOPTED / EXECUTION_NOT_AUTHORIZED。** 本文件的单源、双 arm、载体参数与 known-grid 描述性后评口径已经采纳并作为固定配置保存；采纳方法不等于授权真实执行。真实模型、GPU、VAE、FFmpeg、媒体、Colab 与 Drive 均未在本轮执行。

## 要回答的单一问题

固定的局部 paired RGB-DCT energy 构造在同一次真实 50-step Wan 生成中写入 state 与四个不同 8-bit fragment 后，是否能让两类证据共同从 float RGB 延续到同栅格 RGB8，再延续到 MP4。实验不证明新源泛化、FPR、质量优越、盲同步或任意攻击恢复。

相对旧 RGB-DCT lift / MULTI / LOCAL / Structured 和 same-raster STE 路径，本构造的新变量是：接收端直接从 received RGB 读取局部成对能量比；同一局部支持同时承载 32-chip state 与按 segment 轮换的四个 8-bit fragment；写端在每个启用步对 conditional clean `x0` 做无梯度 RGB 变换，再以同一冻结 Wan VAE 的 posterior-mode difference 回注，并只做一次既定联合 cap。RGB 直接读、`E(X')-E(X)` lift、终态/多步介入本身已有历史先例；本方案不含 STE、tail backward、all-candidate margin、forward-accept search 或调参扫描，也不声称解决旧梯度代理与真实媒体方向不一致的问题。

## 已采纳的一次开发源运行

只用一个已见 development source，生成参数完整固定如下：

- source id `yellow_sailboat_dev_s2026100701`；模型 `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`，revision `0fad780a534b6463e45facd96134c9f345acfa5b`；181 帧、320×512、50 native steps、CFG 5、seed `2026100701`、`max_sequence_length=512`。
- prompt：`locked camera, a small yellow ceramic sailboat gently drifting across a shallow clear glass tank on a stone tabletop, steady soft daylight, no people, no cuts`
- negative prompt：`text, watermark, logo, camera motion, cuts, multiple objects, flicker`
- 两个物理 arm：`OFF` 与 `JOINT`，共享同一 initial latent 身份，各自使用完整 pristine scheduler。OFF 是显式零 joint control；`rho=0` 仍会把系数对重分为等能量，`cap=0` 仍会做 VAE 调用，二者都不能冒充 OFF。
- JOINT 参数：`rho=0.5`、conditional-clean normalized latent 联合 `L2 cap=1`。理想、clip 前的局部 pair 能量分成 75%/25%，得到 `q=±0.5`。cap 只在 mask 后 shrink，不对 raw `<1` 扩幅；它不等于旧 native 下一状态 `R*`。CFG 5 下名义 velocity 差是 `-5d/sigma`，FP32 realized 另记，native total state update 不是控制因果效应或终态预算。
- public key 原字符串 `local-joint-state-payload-v1-first-mechanism`；wrong key 为该字符串加 `-wrong`，只作旁证，不建立 FPR。message 严格为 hex `8001a55a`，四个 byte 各不相同且按 byte 顺序、MSB-first；不按结果挑 message。
- 8 fps，FFmpeg RGB24 输入，`libx264`、CRF 18、`yuv420p`；MP4 必须由已经独立持久化的同一 RGB8 raster 字节产生。运行配置固定 `device=cuda`、transformer dtype `bfloat16`、冻结 VAE `float32`，不设具体 GPU 型号 gate，也不自动换 dtype。source id、所有上述字段、device 与 transformer dtype 都由运行配置显式提供，无代码默认。

总物理规模为 2 个 MP4、362 帧、200 次 transformer forward、100 次 native step。JOINT 在 steps 25..49 做 25 次 clean decode 与 50 次 posterior encode，两个 arm 各做 1 次 terminal decode，因此合计 27 decode、50 encode、2 次 MP4 save/read。当前串行 loader 每次进入 VAE phase 都调用 `load_frozen_vae`，所以建议的 OFF+JOINT 名单成功完成时共有 27 次 VAE load 尝试；缓存文件重复加载开销尚未测量。每个联合步先取得 c/u，再将实际 arm scheduler 与大 transformer 切到 CPU/VAE 阶段，完成 1D+2E、释放临时量与 VAE、恢复同一 transformer/scheduler 后才做 CFG/native step；conditioning 与当前 z/c/u 小张量保留在 device。约 25 个 joint VAE 阶段、25 次 joint 往返（按方向约 50 次驻留切换），初始化和两个 terminal decode 另计。不由性能猜测引入 VAE cache、自动恢复或替代驻留策略。

仓内历史工程依据是 `runtime/wan/generation.prepare_generation(load_vae=False)` 与 `load_frozen_vae`，以及历史 `WanMultiBackend.to_vae_phase/to_transformer_phase` 的同一 transformer 串行驻留方式。`diagnostics/receiver-first-20260923/rgb_dct_multistep_real_result_audit_20260924.md` 记录 run `20260924T121641567985Z/S2=d23fe4e` 在 L4 完成 8/8 MP4、多次 LIFT/RESUME 与 FINAL_MEDIA，恢复 CFG `max_abs_error=0`、14 decode/16 encode、0 backward、峰值 allocated 约 11.65 GB（原审计口径）、无 OOM；同一历史的较早介入结果仍为负（SINGLE46 C14/15、MULTI C19/16，而 SINGLE49 C30/30）。这些只支持旧无梯度分相路径可接线，不保证本轮 25 次完整 181 帧三次 VAE 调用的显存峰值、时延或存留效果。

## 固定原始证据与已采纳后评公式

每个 float RGB、RGB8、MP4 层先封存 received-only 的全部 768 行：phase `g=0..7`、slot `j=-1..22`、4 ROI，保留 signed requested coordinates、实际 received indices、每 chip 的 `E+`、`E-`、`q`、expected/available/zero/failed support，以及 SCORED/PARTIAL/MISSING/FAILED。terminal latent 只保存真实 normalized tensor、形状、统计和可获得的同源 OFF 差异，不使用旧 latent reader 冒称新 RGB carrier readout。现有 decoder 输出命名为 **adapter-clamped FP32 RGB [0,1]**；它不是 raw unclipped VAE output。

runner 对 correct key 和 wrong key 分别保存原始目录；一个 key 的提取失败不覆盖另一个已经保存成功的目录，并另存不含 key/message 的 `raw_observation_manifest.json`。独立 posthoc 在读取含配置的 `result.json` 前，先从该 manifest 重读并校验 OFF/JOINT × float RGB/RGB8/MP4 × correct/wrong key 共 12 份原始收据，将全部目录身份写入 `raw_observation_seal.json`，此时 `truth_loaded=false`。只有 seal 完成后才加载固定 key/message，在 `posthoc_result.json` 中对预声明未攻击公共网格 phase 1、slot 0..21 作描述性后评，保留 22×32=704 个 state q 和 704 个 payload q。它不实现 blind path、message decode 或科学 PASS。

State 对 22 个循环 label 偏移全部报告

`C_r = (1/704) * sum_{j=0..21,i=0..31} S[(j+r) mod 22,i] * q_state[j,i]`, `r=0..21`，

并报告 `C_0` 相对其余最大值的 gap；严格并列原样保留，不要求每窗唯一峰。Payload 按预定义 `(j mod 4,b,i)`，只报告真值后评的每 fragment/bit signed mean：每 bit 固定 24/24/20/20 条 local evidence，同时保留每次重复与原始 q，不输出 decoded message。zero-support 属于构造支持缺失并留在固定分母；任一需要的窗口/bit 缺证据则对应指标为缺失并记录原因，不能 drop 后重归一化或用真值补值。

固定描述性进展条件为：MP4 的 `r0 gap > 0`，同时四个 fragment 的全部 8 个 signed mean 都 `>0`。OFF 必须报告相同指标及 JOINT-OFF 差值；若 OFF 也满足，或者 OFF 必要证据不可用，则机制归因未决，不能宣称写入存留。wrong key 仅是旁证。float→RGB8→MP4 使用同一固定规则逐层比较。该后评是 known-grid oracle diagnostic，不是 blind path；证据 routing 也不是 fragment 恢复。输出区分完整有限负结果、零支持构造缺口和 nonfinite/读取等工程失败；缺失项仍留在固定分母，不以一个合并的 missing 状态抹去原因。

## 失败、停止与用户裁定

runner 应逐步流式保存 50 行、模型驻留 attempt/completion/failure、terminal/float/RGB8/MP4 固定层状态和原始观测。OOM、加载、nonfinite、scheduler cursor、保存或读取失败属于工程无结果；有限 zero-support 或载荷符号失败属于构造证据。不得自动重试、换 dtype、分时间 chunk、换 seed/message/frequency/rho/cap 或扫描。若一次完整冻结候选在该 development source 上产生有限负结果，则停止这一冻结候选；这否定的是该源上的双证据闭合，不是整类路线。

当前实现是可选 runtime provider、串行 residency loader、独立显式配置 CLI 和只读 posthoc CLI。CLI 的普通路径会调用真实 50-step sampler；`--preflight-only` 只验证显式配置、完整源码闭包身份和固定记录目录，便于无 `.git` 发布目录检查。驻留事件保存 elapsed time，以及 CUDA allocated/reserved 和进程自上次 CUDA reset 起的累计 peak；CPU 检查对应值为 null。实现不含 model residency 常驻缓存、独立 GPU 型号 gate、自动 dtype fallback、blind selector、message reconstruction 或实验参数默认。当前 CPU 只用 fake transformer/VAE/codec 边界、小张量和合成 raw 目录验证生命周期、50-step 接线、失败停止、同 RGB8 字节输入、双 key 封存、固定 reducer 与 no-git 运行；未验证真实 25 次往返的资源可行性或媒体存留。

### 可执行入口与精确配置 schema

入口只接受包含下列 **8 个且仅有这些** 顶层 section 的 JSON；所有字段必填，代码不补默认值。下列内容已保存为 `experiments/wan_state_clock/configs/local_joint_state_payload_v1.json`，是已采纳方法的唯一固定运行配置；普通命令的真实执行范围仍须另行明确：

```json
{
  "schema": "local-joint-state-payload-real-v1",
  "model": {
    "id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "revision": "0fad780a534b6463e45facd96134c9f345acfa5b"
  },
  "generation": {
    "height": 320,
    "width": 512,
    "frames": 181,
    "steps": 50,
    "guidance_scale": 5.0,
    "max_sequence_length": 512,
    "prompt": "locked camera, a small yellow ceramic sailboat gently drifting across a shallow clear glass tank on a stone tabletop, steady soft daylight, no people, no cuts",
    "negative_prompt": "text, watermark, logo, camera motion, cuts, multiple objects, flicker",
    "seed": 2026100701
  },
  "carrier": {
    "key": "local-joint-state-payload-v1-first-mechanism",
    "wrong_key": "local-joint-state-payload-v1-first-mechanism-wrong",
    "message_hex": "8001a55a",
    "rho": 0.5,
    "cap": 1.0
  },
  "media": {
    "fps": 8,
    "codec": "libx264",
    "crf": 18,
    "pixel_format": "yuv420p"
  },
  "source": {
    "source_id": "yellow_sailboat_dev_s2026100701",
    "development_only": true
  },
  "arms": ["OFF", "JOINT"],
  "runtime": {
    "device": "cuda",
    "transformer_dtype": "bfloat16"
  }
}
```

只做无模型预检：

```bash
PYTHONPATH=. python -B -m experiments.wan_state_clock.local_joint_state_payload_v1_run \
  --config experiments/wan_state_clock/configs/local_joint_state_payload_v1.json \
  --output /new/output/path --preflight-only
```

普通命令会真实加载模型并开始两条完整轨迹，本轮没有执行：

```bash
PYTHONPATH=. python -B -m experiments.wan_state_clock.local_joint_state_payload_v1_run \
  --config experiments/wan_state_clock/configs/local_joint_state_payload_v1.json \
  --output /new/output/path
```

运行完成后，独立后评从已经保存的 `result.json` 与 12 份原始目录读取，不重新生成、VAE encode 或 codec：

```bash
PYTHONPATH=. python -B -m experiments.wan_state_clock.local_joint_state_payload_posthoc_v1_run \
  --run-result /absolute/run/output/result.json \
  --config experiments/wan_state_clock/configs/local_joint_state_payload_v1.json \
  --output /new/posthoc/output
```

wrong key 已是 8-section 配置中 `carrier.wrong_key` 的显式必填字段，runner 只用它保存第二套 received-only raw observation；message 不进入 raw reader。posthoc 才在 seal 后加载固定 truth 并应用 known-grid reducer。wrong-key 结果不决定主条件或形成 FPR，routing 也不等于恢复。

posthoc 在 seal 后还会从 `result.json` 重建同样的去 key manifest，并要求与已封存 manifest 精确相等，以绑定同一次 run、arm、layer 和 receipt。输出分别保存 `mp4_attribution` 与顶层 `outcome_classification`：run 未完成时顶层为 `INCOMPLETE`，任一 correct-key 层为工程失败时顶层为 `ENGINEERING_FAILURE`；wrong-key 单独失败不推翻完整 correct-key 链。非有限 q 或必要 energy 在 derived JSON 中写为 `null` 并保留 missing reason，不能参与正结论；原 raw 文件及其 SHA 不被改写。

上述 `1 source × {OFF,JOINT}` 名单、`rho=0.5/cap=1`、source/seed/prompts/key/message/codec、描述性后评公式与进展条件已经采纳并显式冻结。真实执行仍未授权；执行前不再改变这些字段，也不增加 arm、扫描或自动重试。

runner 冻结代码、CPU 工程分母与 A2/A3/A4/A5 同版审查结果绑定在[主提案的第三阶段收据](local_joint_state_payload_v1_proposal.md#第三阶段-runner-版本绑定与审查收据)；这些历史收据不表示本阶段后评实现已经过同一审查，也不改变真实执行边界。

主提案同节的 constructor 补充收据把当时受影响代码绑定到 `ac822e6`；“不实现 wrong-key/reducer”是该历史版本的边界。当前新增实现仍只形成 fixed known-grid 描述结果与归因状态，不能自动回答 blind recovery、FPR 或科学 PASS；真实执行仍须单独授权。

固定配置、双 key 原始目录与独立 posthoc 的最终受审版本、17 项工程验证及 A2/A3 同版 ACCEPT 记录见[主提案的最终同版审查收据](local_joint_state_payload_v1_proposal.md#最终同版审查收据)。该收据确认方法实现闭合，不改变本文件的 `EXECUTION_NOT_AUTHORIZED` 边界，也不提供真实媒体结果。

整份 raw observation 缺失时仍保留固定 55 项 metric 目录的补充修复与同版审查，见[整份观察缺失目录补充收据](local_joint_state_payload_v1_proposal.md#整份观察缺失目录补充收据)；载体参数和固定配置未改变。

## 单文件 Colab 交付

本阶段交付 `notebooks/local_joint_state_payload_v1_colab.ipynb`。用户步骤固定为：（1）上传并打开 notebook；（2）选择 CUDA GPU runtime，notebook 不按 GPU 型号设 gate；（3）选择一次 **Run all**，并在提示时授权 Drive mount。首个代码单元严格只有 `drive.mount('/content/drive')` 的两行导入/调用。它在独立的 `MyDrive/Video-WM/Local-Joint-State-Payload-V1/<UTC>` 目录先写固定 OFF/JOINT、每臂 50 step、4 层和 6 份 observation 的未完成槽，再准备环境、展开内嵌的 no-`.git` 源码闭包、运行固定真实 CLI，并在 runner 已保存 `result.json` 与 raw manifest 时继续执行 seal-first posthoc。没有运行开关、参数扫描、自动重试或 GPU 型号 gate。失败会保留已有 result/raw/posthoc、流式日志、执行收据和原始主异常；posthoc 仍保留整份 raw 不可用时的 55 项目录。当前 27 decode、50 encode、27 次 VAE load 和 25 次 joint residency 往返的资源可行性未知。

完成或失败后，直接分享/回传整个新建的 UTC 目录，包括 raw observations 与失败记录；最终 summary 单元在异常后可能不会执行，不能只回传屏幕摘要。目录中存在即回传：`notebook_failure.json`、`execution.log`、`fixed_slots.json`、`setup_receipt.json`、`environment_receipt.json`、`portable_source_receipt.json`、`execution_receipt.json`、`notebook_audit.json`、`run/result.json`、`run/raw_observation_manifest.json`、`posthoc/raw_observation_seal.json`、`posthoc/posthoc_result.json`，以及这些清单所在目录的其余 raw/media/failure 文件。两臂 float RGB 与 RGB8 栅格的已知合计约 0.89 GB；模型权重/cache、临时量、12 份 raw JSON、MP4 与日志的额外存储总需求没有测量，不据此编造最低磁盘值。

同目录的 `local_joint_state_payload_v1_portable_source.zip` 与 notebook 内嵌 ZIP 字节完全相同，仅供本地审阅/下载，不是用户额外上传或运行步骤。ZIP 包含 23 个 B-line 必需文件、完整 hash manifest、`git_commit=null`，展开目录不含 `.git`，因此不会继承外层或 Drive 上其他 checkout 的 Git 身份。notebook 分别核对 23 文件 package identity、runner 自身 20 文件 source closure identity 和固定配置字节/语义，不把未发布源码冒充某个历史 SHA。

依赖口径采用较新 conditional-joint 实跑环境的固定组合：torch 2.11.0、diffusers 0.39.0、transformers 4.57.6、numpy 2.1.3、accelerate 1.15.0、safetensors 0.8.0、huggingface-hub 0.36.2、tokenizers 0.22.2、sentencepiece 0.2.2、ftfy 6.3.1。`diagnostics/trajectory-conditional-joint-real-run-audit-20261008/raw/setup/` 的 primary receipts 绑定实际源码 `ac111d0fed253767651929d115c343fe1636c525`，记录 torch 2.11.0+cu130/diffusers 0.39.0、runner rc 0；当前 B 的 generation/VAE/trajectory 关键调用与该实跑版本保持功能兼容。`pip check` 的返回码只写入环境收据，不作为无关 Colab 包冲突的 hard gate。

串行驻留依据另来自 [旧 multistep 审计](../../../diagnostics/receiver-first-20260923/rgb_dct_multistep_real_result_audit_20260924.md)及其 [Drive result](https://drive.google.com/file/d/16TX_EE8s1Zj5xA873WJzbn_wuoQdFLGA/view) 和 [setup log](https://drive.google.com/file/d/1LMz-TtvSCJEJshyV9Pu-sjZp2wRRR0eW/view)：精确 notebook source 为 `6850ee81454f538916bc228f2ecec521f9ac7511`，运行源码为 `d23fe4eeaca81395c57c98fc403e1d37b4dcd6e4`，run id 为 `20260924T121641567985Z`。该 N2/S2 run 使用 torch 2.11.0+cu128、diffusers 0.40.0、transformers 5.16.1、accelerate 1.14.0、huggingface-hub 1.29.0、numpy 2.1.3，在 L4 完成 8/8 media、14 decode/16 encode、0 backward、CFG restore max-abs 0，peak allocated `11652710912` bytes（约 11.65 GB，原审计十进口径）。这套旧环境没有与较新 conditional-joint 环境混写为同一个验证栈；它只证明较小的同 Wan 无梯度释放/重载路径曾真实成功，不能证明当前 27 decode、50 encode、27 次 VAE load、25 次 joint 往返的显存、时间或媒体存留。

notebook 及 builder 的本地验证只执行 deterministic source packaging、代码单元语法和完全 stub 的编排；没有运行模型、权重、GPU、VAE、codec、媒体、Colab 或 Drive，也没有安装/下载依赖。Run all 是之后的真实外部执行动作，仍不能从这些本地检查推断 state 与四个 payload fragment 穿过 MP4。
