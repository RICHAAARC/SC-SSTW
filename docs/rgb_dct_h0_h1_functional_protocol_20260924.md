# RGB-DCT T49：独立校准与盲 H0/H1 功能性首轮协议

状态：**用户已采纳方法；源码实现待同版审阅，真实 Colab/GPU 待用户运行。** 原采纳草案字节 SHA-256：`b3b287958e6a0df291d031c4fa935022b8d0b1dc2cd43dca8c01ca189fc8df7e`；采纳记录见外层 `diagnostics/receiver-first-20260923/rgb_dct_h0_h1_functional_adoption_20260924.md`。当前两开发来源的 `DEVELOPMENT_CONTROLLABILITY_OBSERVED` 只授权设计下一步，不能给本协议的阈值、名单或判据提供独立验证。真实 Colab/GPU 仍由用户运行。

## 目的和不变项

唯一问题：固定 T49 VAE-lift writer 生成的实际 FULL MP4，能否在完全独立的 OFF 校准后，被只接收 `MP4 路径、key、冻结接收规范、冻结阈值` 的接收流程按 H0/H1 正确判别。此轮为**两评价来源的功能性检验**，不估计低 FPR、总体检出率或跨内容稳健性。

锁定成功开发运行的模型 `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` revision `0fad780a534b6463e45facd96134c9f345acfa5b`，key `WanProjection-first-validation-key-v1`、key ID `785b91ae6b23bfc9`、RGB-DCT receiver spec SHA-256 `0a8cc13d8f99fa87fa2ad8f6e34b953e091261763877b5e39863f179f317f00f`。沿用 320×512、181 帧、8 fps、50 步、CFG 5、同一 negative prompt、libx264/CRF18/yuv420p、FFmpeg RGB24 读回；沿用正向 `A=0.1*c_t*s_b*K`、`E(P+)−E(X0)`、latent 时间 `1:45`、同 history T49 单步、目标 support RMS `0.042943312697648145`。不改变写入强度、符号、时刻、key、接收公式或预算语义。

安装、模型加载、分进程释放、VAE、媒体接口及 Drive 回执优先复用真实跑通的 S4=`a883659010870e613deaa612419cbe2630ec36a3` / N3=`16077c472d1433914f8036b89d6c652f4d0fdff4` 路径。新源码需重新绑定 immutable SHA；不从历史成功推断新运行已经成功。首代码 cell 仅两行 Drive mount，Run all 一条固定命令，新时间戳输出，保留完整 pip/environment/source 日志。

## 预先固定来源

四个 prompt/seed 组合均与当前两开发来源不同，且四者互不重叠；下面的确切 prompt/seed 与角色均**已采纳**，不得按校准或评价得分替换。所有来源从各自固定 seed fresh 生成，不复用旧 MP4、latent 或同批开发结果。

| 角色 / ID | seed | 固定 prompt | 实际 MP4 槽 |
|---|---:|---|---|
| CAL_OFF `cal_top_s2401` | 2026092401 | `locked camera, a single green spinning top rotating slowly on a plain wooden table, stable indoor light, no people, no cuts` | OFF |
| CAL_OFF `cal_lantern_s2402` | 2026092402 | `locked camera, a white paper lantern swaying gently in front of leafy trees outdoors, stable daylight, no people, no cuts` | OFF |
| EVAL `eval_robot_s2403` | 2026092403 | `locked camera, a small red toy robot walking slowly across a plain gray floor, stable indoor light, no text, no people, no cuts` | OFF, NATIVE_H1 |
| EVAL `eval_sunflower_s2404` | 2026092404 | `locked camera, a single yellow sunflower swaying gently against a green field, stable daylight, no people, no cuts` | OFF, NATIVE_H1 |

## 顺序、判定与失败

运行前持久化全部 **4 来源、6 媒体/评分槽、1086 计划帧**。先完成两条 CAL_OFF 实际 MP4 保存、单次读回及盲连续分数，核对各 181 帧、hash、有限性及无失败。仅两条均有效时原子持久化并读回 `FROZEN` 校准记录：`tau=max(S(cal_top OFF), S(cal_lantern OFF))+1e-6`，同时绑定 source/config/protocol/key、两媒体 hash 和分数；评价来源生成必须发生在冻结之后。`1e-6` 只是固定数值 guard，不是 FPR 置信界。

若任一校准 OFF 无效，两个校准槽保留实际状态，四个评价槽标 `NOT_RUN_UNCALIBRATED`，不补来源、不改用评价 OFF 校准。若校准有效，两条评价来源均无条件尝试 OFF 与唯一 NATIVE_H1。`P+` 仅在浮点 RGB 上构造内部 VAE lift；**不生成 PIXEL± MP4，不保留开发期方向门或依据同源 OFF 分数决定是否写入**。有限零 lift/零 native response 分别记为方法负，非有限、OOM、超时或媒体失败记工程无效，均保留固定槽；不回退 OFF、不换来源、不自动重试。

`score_mp4(path,key)` 只接收实际 MP4 与 key；连续分数、媒体身份先落盘，再用已冻结 `tau` 对每个评价视频作 `score>tau` 的 H1 判定（`score<=tau` 为 H0）。阈值构造器只接收预定两条 CAL_OFF 的已持久化分数，不能读取评价分数；评价判定器只接收单条分数及冻结阈值，不能接收 arm、配对 OFF 分数或 writer state。真值标签只在判定落盘后报告合并。同源 `S(NATIVE_H1)-S(OFF)` 可列诊断，但不参与阈值、判定或样本选择。两条评价 OFF 均拒绝、两条 NATIVE_H1 均检出且六槽全部有效，才报告 `FUNCTIONAL_BLIND_PRESENCE_OBSERVED`；完整但误判则报告负结果；任一无效/未运行则报告不完整并逐槽保留，不把失败计作正确拒绝或检出。

## 最大调用账与证据上限

计划上限：4 generation、400 Transformer、202 正式 native scheduler step + 2 unit probe = 204 native total、6 VAE decode、4 VAE encode、0 backward、6 MP4 save/read/score。attempted/completed 分开逐源及总账记录；校准失败后的未调用低于上限但不缩分母。四来源串行独立 worker，可沿用每源 10800 秒超时与子进程组终止；最近 L4 两来源每源约 700–782 秒仅供规划，不能作为本次运行时保证。

两条校准 OFF 的经验尾部分辨率极低。即使本轮四条评价视频全部判对，也**只能**证明这个固定 key/writer/receiver/阈值规则在两个预指定新评价来源上的盲存在判别功能可行；不得宣称低 FPR、总体 100% 检出、视频质量、payload、归因或多步增益。后续低误报目标须另定独立且更大 OFF 尾部审计样本和统计报告方案，不能从本轮挑阈值或回头改名单。

## 已采纳的具体方法决定

已采纳：**2 校准 + 2 评价的规模与上述四组 prompt/seed；`max(两校准 OFF)+1e-6` 和严格 `score>tau`；评价移除 PIXEL± 通道门并固定尝试两条 NATIVE_H1；六槽完整性、结果分类和 204 native/6 MP4 上限。** 若选择原 T40–T49 路线的 9 校准 + 4 评价规模，需要为当前 RGB-DCT/T49 候选另冻名单和预算；不得把旧 C2 协议自动移用。
