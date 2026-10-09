# Local Joint State+Payload V1：最小真实机制实验提案

**状态：PROPOSED_FOR_USER_DECISION。** 本文件提出一次真实运行的完整参数与后评口径；代码实现和 CPU 工程检查不表示这些数值、名单、判别规则或真实执行已获采纳。真实模型、GPU、VAE、FFmpeg、媒体、Colab 与 Drive 均未在本轮执行。

## 要回答的单一问题

固定的局部 paired RGB-DCT energy 构造在同一次真实 50-step Wan 生成中写入 state 与四个不同 8-bit fragment 后，是否能让两类证据共同从 float RGB 延续到同栅格 RGB8，再延续到 MP4。实验不证明新源泛化、FPR、质量优越、盲同步或任意攻击恢复。

相对旧 RGB-DCT lift / MULTI / LOCAL / Structured 和 same-raster STE 路径，本构造的新变量是：接收端直接从 received RGB 读取局部成对能量比；同一局部支持同时承载 32-chip state 与按 segment 轮换的四个 8-bit fragment；写端在每个启用步对 conditional clean `x0` 做无梯度 RGB 变换，再以同一冻结 Wan VAE 的 posterior-mode difference 回注，并只做一次既定联合 cap。RGB 直接读、`E(X')-E(X)` lift、终态/多步介入本身已有历史先例；本方案不含 STE、tail backward、all-candidate margin、forward-accept search 或调参扫描，也不声称解决旧梯度代理与真实媒体方向不一致的问题。

## 建议冻结的一次开发源运行

建议只用一个已见 development source，生成参数完整固定如下：

- source id `yellow_sailboat_dev_s2026100701`；模型 `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`，revision `0fad780a534b6463e45facd96134c9f345acfa5b`；181 帧、320×512、50 native steps、CFG 5、seed `2026100701`、`max_sequence_length=512`。
- prompt：`locked camera, a small yellow ceramic sailboat gently drifting across a shallow clear glass tank on a stone tabletop, steady soft daylight, no people, no cuts`
- negative prompt：`text, watermark, logo, camera motion, cuts, multiple objects, flicker`
- 两个物理 arm：`OFF` 与 `JOINT`，共享同一 initial latent 身份，各自使用完整 pristine scheduler。OFF 是显式零 joint control；`rho=0` 仍会把系数对重分为等能量，`cap=0` 仍会做 VAE 调用，二者都不能冒充 OFF。
- 建议 JOINT 参数：`rho=0.5`、conditional-clean normalized latent 联合 `L2 cap=1`。理想、clip 前的局部 pair 能量分成 75%/25%，得到 `q=±0.5`。cap 只在 mask 后 shrink，不对 raw `<1` 扩幅；它不等于旧 native 下一状态 `R*`。CFG 5 下名义 velocity 差是 `-5d/sigma`，FP32 realized 另记，native total state update 不是控制因果效应或终态预算。
- public key 原字符串 `local-joint-state-payload-v1-first-mechanism`；wrong key 建议为该字符串加 `-wrong`，只作旁证，不建立 FPR。message 严格为 hex `8001a55a`，四个 byte 各不相同且按 byte 顺序、MSB-first；不按结果挑 message。
- 8 fps，FFmpeg RGB24 输入，`libx264`、CRF 18、`yuv420p`；MP4 必须由已经独立持久化的同一 RGB8 raster 字节产生。建议 `device=cuda`、transformer dtype `bfloat16`、冻结 VAE `float32`，不设具体 GPU 型号 gate，也不自动换 dtype。source id、所有上述字段、device 与 transformer dtype 都由运行配置显式提供，无代码默认。

总物理规模为 2 个 MP4、362 帧、200 次 transformer forward、100 次 native step。JOINT 在 steps 25..49 做 25 次 clean decode 与 50 次 posterior encode，两个 arm 各做 1 次 terminal decode，因此合计 27 decode、50 encode、2 次 MP4 save/read。当前串行 loader 每次进入 VAE phase 都调用 `load_frozen_vae`，所以建议的 OFF+JOINT 名单成功完成时共有 27 次 VAE load 尝试；缓存文件重复加载开销尚未测量。每个联合步先取得 c/u，再将实际 arm scheduler 与大 transformer 切到 CPU/VAE 阶段，完成 1D+2E、释放临时量与 VAE、恢复同一 transformer/scheduler 后才做 CFG/native step；conditioning 与当前 z/c/u 小张量保留在 device。约 25 个 joint VAE 阶段、25 次 joint 往返（按方向约 50 次驻留切换），初始化和两个 terminal decode 另计。不由性能猜测引入 VAE cache、自动恢复或替代驻留策略。

仓内历史工程依据是 `runtime/wan/generation.prepare_generation(load_vae=False)` 与 `load_frozen_vae`，以及历史 `WanMultiBackend.to_vae_phase/to_transformer_phase` 的同一 transformer 串行驻留方式。`diagnostics/receiver-first-20260923/rgb_dct_multistep_real_result_audit_20260924.md` 记录 run `20260924T121641567985Z/S2=d23fe4e` 在 L4 完成 8/8 MP4、多次 LIFT/RESUME 与 FINAL_MEDIA，恢复 CFG `max_abs_error=0`、14 decode/16 encode、0 backward、峰值 allocated 约 11.65 GB（原审计口径）、无 OOM；同一历史的较早介入结果仍为负（SINGLE46 C14/15、MULTI C19/16，而 SINGLE49 C30/30）。这些只支持旧无梯度分相路径可接线，不保证本轮 25 次完整 181 帧三次 VAE 调用的显存峰值、时延或存留效果。

## 固定原始证据与拟采纳后评公式

每个 float RGB、RGB8、MP4 层先封存 received-only 的全部 768 行：phase `g=0..7`、slot `j=-1..22`、4 ROI，保留 signed requested coordinates、实际 received indices、每 chip 的 `E+`、`E-`、`q`、expected/available/zero/failed support，以及 SCORED/PARTIAL/MISSING/FAILED。terminal latent 只保存真实 normalized tensor、形状、统计和可获得的同源 OFF 差异，不使用旧 latent reader 冒称新 RGB carrier readout。现有 decoder 输出命名为 **adapter-clamped FP32 RGB [0,1]**；它不是 raw unclipped VAE output。

以下只是供用户采纳的描述性机制诊断，runner 不实现 reducer、blind path、message decode 或科学 PASS：原始 768 行封存后，另文件才可读取写端真值，并只在预声明未攻击公共网格 phase 1、slot 0..21 上后评。保留 22×32=704 个 state q 和 704 个 payload q。

State 对 22 个循环 label 偏移全部报告

`C_r = (1/704) * sum_{j=0..21,i=0..31} S[(j+r) mod 22,i] * q_state[j,i]`, `r=0..21`，

并报告 `C_0` 相对其余最大值的 gap；严格并列原样保留，不要求每窗唯一峰。Payload 按预定义 `(j mod 4,b,i)`，只报告真值后评的每 fragment/bit signed mean：每 bit 固定 24/24/20/20 条 local evidence，同时保留每次重复与原始 q，不输出 decoded message。zero-support 属于构造支持缺失并留在固定分母；任一需要的窗口/bit 缺证据则对应指标为缺失并记录原因，不能 drop 后重归一化或用真值补值。

建议的描述性进展条件为：MP4 的 `r0 gap > 0`，同时四个 fragment 的全部 8 个 signed mean 都 `>0`。OFF 必须报告相同指标及 JOINT-OFF 差值；若 OFF 也满足，则机制归因未决，不能宣称写入存留。wrong key 仅是旁证。float→RGB8→MP4 使用同一固定规则逐层比较。该后评是 known-grid oracle diagnostic，不是 blind path；证据 routing 也不是 fragment 恢复。

## 失败、停止与用户裁定

runner 应逐步流式保存 50 行、模型驻留 attempt/completion/failure、terminal/float/RGB8/MP4 固定层状态和原始观测。OOM、加载、nonfinite、scheduler cursor、保存或读取失败属于工程无结果；有限 zero-support 或载荷符号失败属于构造证据。不得自动重试、换 dtype、分时间 chunk、换 seed/message/frequency/rho/cap 或扫描。若一次完整冻结候选在该 development source 上产生有限负结果，则停止这一冻结候选；这否定的是该源上的双证据闭合，不是整类路线。

当前实现是可选 runtime provider、串行 residency loader 与独立显式配置 CLI。CLI 的普通路径会调用真实 50-step sampler；`--preflight-only` 只验证显式配置、完整源码闭包身份和固定记录目录，便于无 `.git` 发布目录检查。驻留事件保存 elapsed time，以及 CUDA allocated/reserved 和进程自上次 CUDA reset 起的累计 peak；CPU 检查对应值为 null。实现不含 model residency 常驻缓存、独立 GPU 型号 gate、自动 dtype fallback、blind selector、fragment reducer 或实验参数默认。当前 CPU 只用 fake transformer/VAE/codec 边界与小张量验证生命周期、50-step 接线、失败停止、同 RGB8 字节输入和 no-git 运行；未验证真实 25 次往返的资源可行性。

### 可执行入口与精确配置 schema

入口只接受包含下列 **8 个且仅有这些** 顶层 section 的 JSON；所有字段必填，代码不补默认值。以下数值仍是本提案建议，只有用户采纳并另行授权真实执行后才可保存为运行配置：

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
  --config /absolute/path/run.json --output /new/output/path --preflight-only
```

普通命令会真实加载模型并开始两条完整轨迹，本轮没有执行：

```bash
PYTHONPATH=. python -B -m experiments.wan_state_clock.local_joint_state_payload_v1_run \
  --config /absolute/path/run.json --output /new/output/path
```

wrong key 和本文件给出的 state/payload 描述性 reducer 不属于这 8-section runner 输入，也不被生产 runner 消费；它们属于原始 768 行封存后的独立后评，须随后单独实现、冻结并审查，不能把 routing 当作恢复。

请用户一次裁定：是否采纳上述 `1 source × {OFF,JOINT}` 名单、`rho=0.5/cap=1`、source/seed/prompts/key/message/codec、描述性后评公式与进展条件，并另行授权真实执行。若不采纳，请指出要替换的具体字段；代码不会把本提案推荐值变成默认。
