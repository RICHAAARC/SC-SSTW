# Video Overlap Spatial Zero Mean C1 V1

本候选只改变公开空间 basis：原选择包含 Hadamard 常量列时，原位替换该列；其余编码、写入损失、更新预算和盲评分保持原定义。动机来自已保存 C1 run20260930T062340205444Z 的终点诊断。这是已见结果后的新编码假设，不是已定位的实现 bug 修复，也不是独立确认。

旧 video_overlap_tube 源码、已发布 notebook、原运行与诊断文件均不改。新 namespace 为 video_overlap_zero_mean，method_version=video-overlap-zero-mean-v1。新 C1 config 绑定新机制 config，后者精确绑定旧机制 config 作为不变部分的定义依据。为避免修改冻结模块或 monkeypatch 全局，有限状态、catalog、exact equivalence 和 SSE 代码在新 pure 模块中保持同式，仅公开版本与 basis 派生不同。runtime 不导入 experiments，receiver 不接 writer diagnostics。

## 唯一方法变化

对每个公开 key 和空间 block i：

1. 按原 SHA256(VTOS1/basis/order NUL key NUL decimal_i NUL decimal_c) 和 numeric c tie break 对 c=0..15 排序，无尾 NUL。保留完整原顺序、原先前8列和每个原 slot q 的 VTOS1/basis/sign 符号。
2. 原 selected8 不含 c=0 时，整个 basis 逐元素不变。
3. 若含 c=0，只在其原 slot 用完整原顺序中第一个 c!=0 且不在原 selected8 的列替换，沿用该 slot 的原 sign。其它7个 slot 的列、位置和 sign 均不变。
4. writer 和 blind reader 使用完全相同规则；wrong key 自己独立派生。每个4×4 block 仍是8个正交单位向量，且各列空间和为零。basis_layout 收据保存完整顺序、原/新列、slot、sign、key 指纹；公开返回对象不能改写 basis cache。

没有时间去均值，没有新的评分、edit 惩罚、阈值或力度扫描。空间零均值不等于各 slot 的时间零均值。

## 保留的模型与实验

S_t 仍是 keyed 六位时钟与两位非恒定校验形成的8维码；4个 block 各取两维，tube stride1、age0..3 在同一物理 latent 张量中正交叠加。物理时刻 u 是增强状态 [s_u,s_(u−1),s_(u−2),s_(u−3)]，早期越界为零，晚端不补写。时间单位为 Wan VAE regular latent 窗口，nominal stride4，不是单 RGB 帧或完整感受野。

写入 active 系数仍为8*(45+44+43+42)=1392，全源 alpha=1/sqrt(1392)、target L2=1；没有按 crop 长度或覆盖数重归一。pilot 在 conditional clean 的新 U 投影上做 active mean MSE，eta174，实际 full-tensor update 只缩不放大到 cap L2=1。payload 继续 ch0..3 原 FFT-real32bit、eta5520。合并真实两个 delta 后以 v_cond -= delta/sigma 更新，再用 FP32 CFG5，完整 native scheduler history 连续执行。

Wan model/revision、copper-kettle prompt、seed2026092501、181×320×512、8fps、50steps 均继承。OFF / PAYLOAD_MULTI / OVERLAP_MULTI 三条独立完整轨迹，两个 marked arm index25..49 fresh gradient，每次实际 delta 重算；step25前 z/history/c/u identity 必须相同。新 basis 会改变 rawNorm、cap 触发及后续模型/载荷反馈，不能继承旧运行的25步全 cap，也不声称新旧 payload 轨迹逐元素相等。只保持算法和预算规则，不称跨臂总预算匹配。

固定3 source MP4、15 matched second-codec FULL_RESAVED181 与 CROP4/5/6/7_129、60实际四相位 whole-frame VAE normalized tensors、120 phase/key path+payload read、30 global searches、60 posthoc。完整 catalog357480、有效 cost92016、结构排除265464。FULL共同 R44，crop R31；extra rows 仅保存诊断，terminal R45（1440投影维，含48边界零模板），FULL R44为1408维。盲 path 仍 mean SSE、同 available 分母、全部 zero/one repeat/skip family、全部结构等价与1e-12 numeric ties；一律 uncalibrated/unaccepted。重复 payload 不能证明对齐必要性，正常有限模型 argmin 不等于真实同步成功。

媒体仍采用真实保存→readback→完整帧 VAE posterior mode + Wan normalization；crop context reset 不等于 latent slice。质量同格式相对 OFF 与 OVERLAP/PAYLOAD 的 RMSE/PSNR 单列、无阈值。5 pilot-positive 与25相关 controls 不是独立 FPR 样本。C2、真实编辑及位置相关 payload 均不在本候选执行范围。

## 75个独立 writer 诊断槽

生成前预填3 arms×index25..49共75槽，每槽原子保存 npz，5个 FP32 [45,4,4,2] 数组，总未压缩数字字节2,160,000（约2.06MiB）。所有值由本步已有实际张量 detach/no_grad 后使用同一新 writer U 投影，保留所有 time/block/age/component 和48边界维，不保存 full state，不增加 forward/VAE 调用，不消费 RNG。

| 数组 | 定义 |
|---|---|
| conditional_clean | 控制前 z_pre − sigma_pre*c_FP32 |
| pilot_delta | 实际已 cap、将被合并的 full pilot delta 的投影；OFF/PAYLOAD 未施加的 pilot component 确切为零 |
| z_pre | native step 前实际 latent |
| z_post | native step 后实际 latent |
| cfg_clean | z_pre − sigma_pre*v_actual，v 为真正交给 native step 的更新后 FP32 CFG velocity |

cfg_clean 含 unconditional CFG，既不等于 conditional_clean+5*delta，也不是 solver history 修正后的下一真实 state。若后评使用 conditional_clean+pilot_delta 或 cfg_clean−5*pilot_delta，它们仅是有 FP32 舍入的代数投影派生，后者意为“移除本步 pilot 增量的更新后 CFG clean 投影”，不是另存的观测。

metadata 保存 method version、arm、index、sigma_pre/post（末步后0）、pilot_enabled、shape/dtype。实际 full pilot tensor 计算 spatial L2/patch能量，再记录 projected L2、active/boundary energy、空间 U-complement residual、outside-ROI energy 和 Parseval residual；总能量减 patch 能量保留有符号 FP64 归约舍入，不把投影看不见误称无泄漏。空间 U-complement、投影内模板正交残差、时间 DC/变化量是三个不同空间，均不自动叫纯宿主。

末步 z_post 只与 terminal 在相同新 U/regular范围的投影闭合，不等于保存完整 native state。末端投影/CPU转换/hash 异常独立记 FAILED，不能将已完成的50步 trajectory 改为生成失败。sidecar/投影失败保对应槽；generation 真失败时剩余槽 NOT_COMPLETED。execution_status、writer_diagnostic_status、science_status 分开；诊断缺失不能报全套 COMPLETE，却保留已成功的生成/媒体/盲读结果。

writer table 不进入 blind_readouts.json，不进 path ranking/truth join。所有 primary raw 在后评前 canonical-first 保存，parent 在 child 结束/失败后重读最新 canonical，保留已完成 sidecar/hash。75槽独立于120 reader slots，不能混算。

## 必要本地验证和证据上限

固定代表 keys 检查确定性、完整原排序与精确 slot 替换、其它列/sign 不变、Gram I/列和0、公开缓存隔离。实际 full tensor scatter-add→提取核1392/alpha/norm，新旧理想坐标模板与同式 SSE 一致。仅少量 noedit/repeat/skip、缺失歧义与 exact-copy insert=repeat 的反例，不跑旧 B 矩阵或92016真实成本、不用旧 real tensor 当新编码成功。

模型零调用的 synthetic Transformer + 实际 native UniPC fixture 核三臂前25同态、25次 fresh控制、完整history；具体在 OVERLAP_MULTI 单独双跑启用/关闭诊断，velocity/terminal/history/calls 逐项一致；每条 fixture 运行前后 torch CPU RNG 状态一致（未声称所有 RNG 系统或三臂各自双跑）。独立 FP64 提取与 FP32 投影使用显式容差。检查实际梯度闭式、norm/support、末端诊断失败不丢 terminal、callback/export故障、75槽保留、盲读重命名、partialphase、canonical两窗口和 parent spawn失败。没有加载真实模型/VAE或调用 codec。

新 notebook 是未发布草稿，SOURCE_SHA=None；首格保持精确两行 Drive mount，current sys.executable、成功的 Torch-pair probe +原 grow dependencies、fresh generation/media children、真实保存视频预览。它只固定执行上述 C1，无额外模式。发布 source 并验证 immutable SHA 后方可由主会话绑定；当前没有可交付 Run-all 链接。

已见媒体中去掉旧 DC slots 后错误路径仍可能优于真路径。新列可能受不同空间频率的媒体衰减；本候选不能保证改善终点/媒体同步或质量。旧诊断只提供单假设动机，真实生存与机制因果仍待用户运行。
