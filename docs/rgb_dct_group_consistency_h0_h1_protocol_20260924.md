# RGB-DCT 时间组一致性接收器：独立 H0/H1 功能首轮协议

状态：**按用户本轮采纳的单一候选规则冻结实施输入；尚无新版源码、notebook 或真实 GPU 结果。** 用户固定 `C≥24` 判 H1、`C<24` 判 H0；`24` 来自已见六媒体上的开发选择，不是旧运行的事前阈值。旧运行保持 `FIXED_FUNCTIONAL_NEGATIVE`，旧媒体仅用于开发回归，不进入本轮分母。此协议的新来源、顺序和预算在任何新媒体评分前固定；本轮由用户运行 Colab。

## 唯一问题与接收规则

固定原 T49 VAE-lift writer、key 和 MP4 编解码链后，**全新来源**的实际 FULL MP4 是否能由单一时间组一致性统计量盲判别存在？这是一轮两评价来源的功能首轮，不估计低 FPR 或总体检出率。

接收器输入只准是 `MP4 路径、固定 key、冻结接收规范`。复用原 `runtime.wan.io.read_mp4` 的 FFmpeg RGB24 全 181 帧读回及 `uint8→float32/255→float64` 路径，原 luminance、32×32 块、160 块/帧和 `d[t,b]=DCT(2,1)-DCT(1,2)`。原 key 派生的空间符号为 `s_b`，30 组、中心化单位 RMS 的逐帧时间码为 `c_t`；分母 `D=sqrt(mean_{t,b}(d[t,b]^2)+(1/255)^2)`。逐块去时间均值 `m_b=mean_t d[t,b]`。前 29 组每组 6 帧，末组 7 帧；对每组固定计算：

`q_g=mean_{t∈g,b}(c_t*s_b*(d[t,b]-m_b))/D`，`C=sum_{g=0}^{29} 1(q_g>0)`。

唯一判别量 `C` 为 `0..30` 的整数，`q_g=0` 计非正。完整且有效的 MP4：`C≥24` 判 H1，`C<24` 判 H0。读回、形状、帧数、非有限值或计算失败单列 `INVALID`，不转成 H0。接收器不可访问 arm、prompt、seed、同源 OFF、writer state 或真值。原连续分数 `S` 和逐组 `q_g` 可存审计，但不得成为备用判据；`sum_g(n_g/181)q_g` 应以绝对误差 `≤1e-10` 还原原 `S`，仅作机械一致性检查；若不一致，该媒体记工程无效，不能回退为 H0。不得扫描组数、平滑、阈值、key 或其他 receiver。新版 `C≥24` 接收规范须有独立 spec ID/SHA，配置绑定本协议的精确字节 SHA；下文旧 spec SHA 只标识原 DCT 特征和连续分数，不能单独标识新版判别规则。

## 固定 writer、媒体和环境

以已交付 H0/H1 源码 `S2=c316262fed402c60a5c60b9120729391280b033a` 和 notebook `N=2cde559c0832d5cb98149bf782a3994ea954052e` 为实现起点，新实施在隔离候选分支保留旧版本。锁定 `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` revision `0fad780a534b6463e45facd96134c9f345acfa5b`、key `WanProjection-first-validation-key-v1`、key ID `785b91ae6b23bfc9`、原 receiver 特征规范 SHA-256 `0a8cc13d8f99fa87fa2ad8f6e34b953e091261763877b5e39863f179f317f00f`。沿用 320×512、181 帧、8 fps、50 步、CFG 5、negative prompt `text, watermark, logo, camera motion, cuts, multiple objects, flicker`、libx264/CRF18/yuv420p 和 FFmpeg RGB24 读回；沿用正向 `A=0.1*c_t*s_b*K`、`E(P+)−E(X0)`、latent 时间 `1:45`、同 history T49 单步、目标 support RMS `0.042943312697648145`。不改变写入强度、符号、时刻、key、媒体链或调用预算语义。

环境安装、模型加载、分进程释放、VAE 和日志优先复用已经真实跑通的 S4=`a883659010870e613deaa612419cbe2630ec36a3` / N3=`16077c472d1433914f8036b89d6c652f4d0fdff4` 路径，以及上一轮 H0/H1 notebook 的 torch CUDA 后缀修复。新 notebook 首代码 cell 仅两行 Drive mount；Run all 固定一次真实运行，绑定发布后的 immutable source SHA，新时间戳输出并保留安装、环境、源码、媒体和执行回执。不得把静态/CPU 核验写成真实 GPU 成功。

## 预先固定的全新来源与六槽

以下 prompt/seed 与旧开发六媒体及原 H0/H1 四来源均不同；`REF_OFF` 是**阈值不变的独立阴性见证**，不参与拟合或重定 `24`。四来源各自 fresh 生成，不复用旧 MP4、latent 或 noise。来源顺序固定如下；任何有效误判或单源失败都不替换来源、不跳过后续来源。

| 顺序/角色与 ID | seed | 固定 prompt | 预留 MP4/判别槽 |
|---|---:|---|---|
| 1 `REF_OFF/ref_metronome_s2411` | 2026092411 | `locked camera, a wooden metronome swinging steadily on a plain desk, stable indoor light, no people, no cuts` | OFF |
| 2 `REF_OFF/ref_pond_s2412` | 2026092412 | `locked camera, gentle ripples spreading across a small pond beside still stones, stable daylight, no people, no cuts` | OFF |
| 3 `EVAL/eval_bicycle_s2413` | 2026092413 | `locked camera, a blue bicycle rolling slowly along an empty paved path, stable daylight, no people, no cuts` | OFF, NATIVE_H1 |
| 4 `EVAL/eval_windmill_s2414` | 2026092414 | `locked camera, a small white windmill turning slowly in a grassy field, stable daylight, no people, no cuts` | OFF, NATIVE_H1 |

在任何模型调用前持久化 **4 来源、6 MP4/判别槽、1086 计划帧**。每槽实际 MP4 保存后核验字节 SHA-256/大小；由**唯一一次**接收器 FFmpeg RGB24 读回核验 181 帧、形状与有限性并计算 C，不额外解码媒体。媒体身份和 C 落盘后，固定规则产生判别；真值标签只在判别持久化后合并报告。同源 H1−OFF 的旧 `S` 差值可作描述，不参与 C 或判别。`REF_OFF` 不构成新的阈值冻结门：即使其 C≥24 或无效，只要独立 worker 仍能执行，其余三来源继续按计划尝试。评价来源 OFF 得分也不能决定是否尝试其 NATIVE_H1。每个失败、未运行和超时槽均保留原位及原因；不重试、不补样本。

## 固定调用预算与结果上限

最大账与已验证 2+2 H0/H1 路径相同：4 generation、400 Transformer、202 正式 native scheduler step + 2 unit probe = 204 native total、6 VAE decode、4 VAE encode、0 backward、6 MP4 save/read/score。逐来源与总账区分 attempted/completed；四个独立 worker 串行、每源 10800 秒超时，失败低于上限不缩分母。实现只作必要 CPU/static/fake 验证，审计侧不代跑模型、GPU、Colab 或新媒体。

仅当六槽有效且 **4 条 OFF 全部 C<24、2 条 NATIVE_H1 全部 C≥24**，报告本固定名单上的 `FUNCTIONAL_BLIND_PRESENCE_OBSERVED`。任一有效误判保留为功能负结果；任一无效/未运行则报告不完整并显式列出所有已观察误判，绝不把无效算 H0。即使六槽全对，也只支持两个新评价来源及两个新阴性见证上的功能观察；30 个时间组不是 30 个独立 H0，四条 OFF 不足以宣称低 FPR、总体检出率、视频质量、payload、归因或多步增益。低 FPR 需之后另定足量独立 OFF 样本，不能在本批更改 `24`。

## 实施与审计交付

A1 在新建实施任务中作唯一代码写入者，建立隔离候选 worktree/分支，交付纯 receiver、实际 MP4 接口、固定六槽 runner、CPU/fake 定向测试与可直接 Run all 的 Colab；保持旧分支、旧结果及 `main` 不变。先发布 source commit 并核验远端 SHA，再将 notebook 绑定该 SHA，提交和发布 notebook commit，核对远端分支与 GitHub default branch。A2/A3 对同一待审源码独立审查方法与实现，A4 综合，A5 在 notebook 交付里程碑审计；审计会话最终核验。旧六 MP4 可作 `C=16/11/13/18/28/30` 的开发回归测试，不能混入本轮新分母。用户运行后以 Drive 同次原始记录审核结果，不为得到通过而修改阈值或名单。
