# RANK3 四新源阶段2确认

本入口实现已采纳的四新源固定确认：γ=1，复用原 fixed-four-new-v1 的背景与 τ=0.49221603782894896，不重校准。它与旧成功三组入口独立，生成、FP32 VAE 缓存、媒体与 DCT 路径复用成功实现。

在 Colab 打开 `notebooks/b_rank3_four_source_confirmation_v1.ipynb` 后 Run all。第一代码单元仅两行 Drive 挂载；随后从公开 B 分支 `dev/local-joint-state-payload-v1` 下载可用源码到临时隔离目录。发布后的 Notebook 无需源码 ZIP、B64 或摘要检查。发布前本地 CPU 检查不能代替公开链接可用性核验。可选 `SOURCE_DIRECTORY` 会先复制到隔离目录。

默认输入 `/content/drive/MyDrive/Video-WM/B-Terminal-Three-Group-V1/fixed-four-new-v1`，只读取旧背景和冻结阈值。新输出 `/content/drive/MyDrive/Video-WM/B-Rank3-Four-Source-Confirmation-V1/fixed-four-new-v1`。四源 prompt、seed、a 与 eval 角色在独立 JSON 中固定，结果后不替换来源。用户运行入口：

```bash
python -m experiments.wan_state_clock.b_rank3_four_source_confirmation_v1_run --input OLD_DIRECTORY --output NEW_DIRECTORY
```

一次 attempt 先持久化，再进入输入与模型操作。首个失败或中断消耗已 STARTED 单元并停止后续调用；未进入单元保留 NOT_ENTERED。重复 Run all 在下载、setup、运行时导入前只读恢复 receipt，不能补缺或重试。运行日志为 `runner.log`，完整失败明细保留在 receipt、ledger 和 planned_outputs 中。只读恢复没有源码、模型或 `.git` 依赖。

| 固定工作 | 预算 |
|---|---:|
| native50 trajectories / steps / DiT forwards | 8 / 400 / 800 |
| local gradients / terminal state construction | 100 / 4 |
| D181 / E41 | 12 / 432 |
| codec encode / probe / decode | 48 / 48 / 48 |
| E41 Q / H137 families / H score slots | 864 / 216 / 29592 |
| E41 DCT / E41 双 key bits | 432 / 864 |
| native key00 Q / native DCT / native 双 key bits | 8 / 8 / 16 |
| 质量记录 / frame / adjacent | 76 / 12316 / 12240 |

模型顶层调用和可观测内部 VAE blocks 分开记录。全部四源都走 eval 缓存路径；无 E181、额外轨迹、新校准或跨片 bundles。

五读法为 CDC_M、UNION_M、RANK3、RANK2、FIXED_ORIGIN。每读法固定 72 tasks、2304 slots，其中 12 阳性 task/384 slots、60 null task/1920 slots；PM wrong key 为其中 12 tasks/384 slots。局部 CDC 固定6912槽，M3固定69192 path classes和604872 seam aliases。

RANK3 每窗必须有唯一 H137 argmax，三个选中 origin 必须落在原声明 M3 path domain，随后只用各窗 singleton C 建立引用。RANK2 仅 L/R 端点；FIXED_ORIGIN 无存在性判决。原 CDC_M 与 UNION_M 保持旧阈值 full-C，不被 singleton 构造修改。所有五读法使用同一物理观测。

物理缓存可先生成保存，但不在采集阶段 flatten 或消费 bit 值。全部 72 tasks 的五读法 plans/refs 保存并回读后，LazyCache 才开放；全部360盲 decisions保存回读后，后评才读取载荷真值与私有几何。Writer 的已配置载荷只用于既定写入，与 receiver 输入分开。

方法判据：RANK3 实际精确 supporting_refs 支持且 literal 正确的槽至少288、每源至少48；阳性 DETECT至少9、每源至少2；不丢失任何CDC_M实际支持正确槽；literal错误、误归因、无实际支持输出与null DETECT均为零。即使null DETECT不输出位，也立即记录本候选失败。之后仅可完成既定已保存数据的五读法诊断，不再开展物理工作、候选或调参。技术缺失单列 TECHNICAL_UNRESOLVED，不能作为科学负例。

仅当全部方法判据满足，且相对CDC_M的实际支持正确净增至少32槽，才报告固定范围新增贡献。方法达标与贡献达标是两个结果字段，不以工程运行完成替代。

质量保留原72项并新增四个同源完整视频 readback 的 OFF→PM **直接数组比较**；不能相加 OFF→P0 与 P0→PM 的距离。新增项的 nominal support 是全时域 payload ROI 与源帧1..176 state ROI的并集；ROI外指标不是VAE因果support证明。RMS、PSNR、相邻帧残差和 clipping 只报告数值，没有新增质量PASS门槛。真实相位 local bit配对不等于盲恢复的完整组正确。

证据上限为固定四新源、151帧crop或声明的单次删帧。不支持任意短片、精确seam、来源认证、总体FPR、多key安全或DiT多步状态传播。当前交付的 CPU/fake/static 检查只验证接线与记录语义，不执行模型、VAE或媒体实验。
