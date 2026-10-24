# 公共分段载荷 V1

本入口提供固定 **4 段 × 每段 8 bit = 净 32 bit** 的生成期载荷写入与条件读取。时间映射由调用方给定；读取器按该映射归组投票，**不估计时间路径，也不构成盲同步**。A 的同步候选、论文评价和 B 的后续同步研究分别保留在各自研究分支，见[分支用途](branch_guide.md)。

## 固定构造

保留已有 50 步 native scheduler 与 conditional-clean FFT-real 局部载荷梯度路径，仅在零基步 **25..49** 写入。有效 Wan 时间为 **1..44**，每连续 11 层对应一个 8 bit 字节；原四个通道重复同一时间段的字节，构成段内冗余，不增加净位数。

| 段 | 有效 Wan 时间 | A 历史固定消息示例 |
| --- | --- | --- |
| S0 | 1..11 | `A6` |
| S1 | 12..22 | `D3` |
| S2 | 23..33 | `9C` |
| S3 | 34..44 | `5E` |

写入 mask 仍为 **44,160** 个系数，包含保护层 0 与 45；两层分别写 S0/S3，读取净载荷时均排除。有效读取总票为 **42,240**；完整可见时，每净位为 11 层 × 4 通道 × 30 个频率位置 = **1,320 票**。原绝对目标幅度 0.5、reference eta 200 对应实际 eta 5520；相同幅度并不保证相同输出扰动或质量。

读取保持 `fft2(norm="ortho").real`、严格 `>0`、原频率坐标与 `slot::8` 规则。精确零投为 0；按 channel、合格接收时间、频率坐标顺序聚合，平票保留 `Counter` 首遇结果。

## 条件读取边界

调用方提供 received-frame 到 source-frame 的时间映射。该映射可以来自独立估计、已知构造或 correct-time 对照；本入口不判断其定位是否正确。每个投票单元要求其四个名义源帧位置都由映射中的真实收到索引支持；为 VAE 补齐的上下文帧不投票。没有合格单元的段保留 `MISSING` 和零票，不用邻段补值。

消息答案不参与解码或路径选择。若调用方给的是 oracle 映射，结果必须标为 correct-time 条件读取；不能据此报告盲同步性能。名义局部映射也不保证 VAE 感受野严格局部。

## 公共入口

从源码克隆或普通源码 ZIP 的根目录调用：

```bash
python -m experiments.wan_state_clock.segmented_payload_v1_run --help
```

普通调用示例（不冒充P2完整源配置）为 [segmented_payload_v1.example.json](../experiments/wan_state_clock/configs/segmented_payload_v1.example.json)。各命令的参数以其 `--help` 为准：

| 子命令 | 用途 |
| --- | --- |
| `plan` | 从调用方给定的时间映射建立合格单元和缺段计划。 |
| `read-votes` | 对已保存的正负票与零系数标记按同一规则归约。 |
| `read-latent` | 对有限、形状合法的 normalized Wan latent 做 FFT 与分段读取。 |
| `read-rgb` | 对给定 RGB 及时间映射完成模型编码和条件读取。 |
| `generate-latent` | 沿既有 50 步原生路径生成分段载荷 latent。 |

实际格式与示例：

- `--source-map map.json` 是长度等于收到帧数的整数数组，范围0..180、非递减；例如 clean 为0..180。映射来源由调用方另行标记，结果统一声明 `CALLER_SUPPLIED`。
- `read-votes --input votes.npz` 需要 `signed_votes`（int8 ±1）和 `zero_coefficient_mask`（int8 0/1），形状均为 `(4,R,240)`，R见plan的 `wan_support`。
- `read-latent --input mapped.npy --key KEY` 接收已经按plan重排RGB后编码的有限浮点 normalized latent；不能直接给任意裁片latent再仅附一个map。
- `read-rgb --input received.npy --key KEY --config model.json` 接收 `(len(map),320,512,3)` 的uint8实际帧数组；入口先重排补上下文，再以 posterior mode 编码一次。`model.json` 至少提供 `model.id`，无固定revision要求。
- 每个 `--output` 是新目录，保存plan/read、原票与执行记录；已有结果不覆盖，失败直接报告，不自动重试。

```bash
# 用户执行真实生成时：不解码、不编媒体，只保存分段终态和50步记录
python -m experiments.wan_state_clock.segmented_payload_v1_run generate-latent \
  --config experiments/wan_state_clock/configs/segmented_payload_v1.example.json \
  --output /absolute/new-segmented-latent

# 读取已有票，无模型、无消息答案输入
python -m experiments.wan_state_clock.segmented_payload_v1_run read-votes \
  --source-map /absolute/map.json --input /absolute/votes.npz \
  --output /absolute/new-segmented-read
```

生成路径的调用为：1次模型准备，50次conditional与50次unconditional transformer前向、50次native更新、25次局部FFT载荷梯度；不反传DiT、不解码、不运行codec。RGB读取路径为1次VAE加载与1次编码；保存latent/票读取不加载VAE。这里是公共接口成本，不是新增实验计划或已完成GPU结果。默认设备CUDA/模型BF16，控制与CFG保持FP32；设备和可用模型路径可由调用方指定。

`plan`/`read-votes` 不调用模型；`read-latent` 读取现有张量，`read-rgb` 与 `generate-latent` 涉及实际模型调用。文件不可读、shape/dtype/有限性或映射支持不合法时直接报告具体问题。源码、配置和环境信息仅供定位，不设置摘要、manifest、版本或 Git 状态相等门禁。

## 已有真实证据

以下来自用户已完成的 **A 原版本运行**及同次保存证据核验。本轮 main 精选整合没有重新执行模型或媒体，也不把旧结果改记为新 main 实验。公开开发方法与运行包见 [A 分段载荷文档](https://github.com/RICHAAARC/SC-SSTW/blob/dev/paper-results-v1/docs/paper_results_v1/segmented_payload_v1.md)和 [P2 包说明](https://github.com/RICHAAARC/SC-SSTW/blob/dev/paper-results-v1/docs/paper_results_v1/distinct_confirmation_p2.md)；精确结果定位另列于[证据摘要](evidence/segmented_payload_v1.json)。

| 保存运行与条件 | 行数 | 计划位 | 结构存在位 | 已读正确位 | 结构缺失位 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原四源 clean correct-time | 4 | 128 | 128 | 128 | 0 |
| 原四源攻击 correct-time oracle | 8 | 256 | 192 | 192 | 64 |
| P2 四新源 clean correct-time | 4 | 128 | 128 | 128 | 0 |
| P2 攻击 correct-time oracle | 32 | 1024 | 640 | 640 | 384 |

原四源固定总账为 **36 行 / 144 个计划段 / 1152 计划位**，含 256 个结构缺失位。144 是相关读取行中的计划段输出数，不能称为 144 条独立消息或独立视频。P2 为 **4 个新源 × 每源 8 个相关攻击视图**，固定 **196 行 / 784 段 / 6272 位**：3968 结构存在、2304 结构缺失；188 READ、4 UNRESOLVED、4 UNSUPPORTED 全部保留。各攻击接收臂均另有 384 结构缺失位，不计正确或恢复；32 个相关视图、多个接收臂和 alias 不增加独立来源或消息数。

这些记录支持固定单 message/key 的分段载荷可读性。A 研究包中的盲路径、覆盖收益、未读、误归属及退步仍按原表解释；它们不是本入口实现了盲同步的证据。正确读出的载荷也不能抹去路径误差。

本地原始证据与主审记录：

- 原四源：`/home/richar/projects/Video-WM/diagnostics/a-line-segmented-payload-result-20261016/same_run/`；主审 `diagnostics/a-segmented-payload-result-principal-audit-20261016/REPORT.md`。
- P2：`/home/richar/projects/Video-WM/diagnostics/a-line-p2-confirmation-result-20261017/same_run/`；主审 `diagnostics/a-p2-confirmation-result-principal-review-20261017/REPORT.md` 与 `root_acceptance.json`。
- [原四源保存目录](https://drive.google.com/drive/folders/1GMc7M5cmgLvQCIk7c_jJiLbZP_KYftWD)、[P2 保存目录](https://drive.google.com/drive/folders/1pQBcfof8-G_udLD1giUD9zCeefePLhuT)。

## 质量与结论上限

已有质量参照 `NATIVE_PRE` **本身已含分段载荷**。candidate_PRE 相对它的差量为新增时钟残差，candidate_POST 的差量包含时钟与 codec；这不是载荷相对同源无水印 OFF 的总质量。P2 同样缺少该 OFF 对照，旧重复消息的质量结果不能替代本分段 writer 的对照。因此不宣称本载荷质量 PASS、无损、总体 FPR、任意消息/密钥泛化、外部竞争力或科学 PASS。
