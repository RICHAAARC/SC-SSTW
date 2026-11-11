# Compact12 四片定向恢复

入口：`notebooks/native_baseline_compact_recovery_v1.ipynb`，选择 GPU 后 Run all。
首格仅挂载 Drive；运行代码在 `runtime/wan/`，普通源码下载及依赖准备在
`scripts/native_baseline_compact_recovery_v1_entry.py`。默认
`PREVIOUS_RUNTIME_STOPPED=True`，因为原四片已经 CLOSED。

新结果写入原正式实验目录下的 `recovery/repair01`。原分片、原 v4 包、
成功和失败记录均不覆盖。临时数组仍在正式结果目录之外的原 cache 下。

## 固定范围

原缺口为 6 个源的 87 个逻辑视图和 11 个质量阶段，70 个原始视图组。
用户另外授权 ucf-0643 PixelSeal 正例整臂替代，需要额外测原 9 个完整视图和
3 个质量阶段。本次执行范围为 96 个逻辑视图、14 个质量阶段、79 个原始视图组；
原缺口清单仍保留，论文全局分母不增加。组数不是 GPU 调用数；实际费用以新 ledger 为准。
原方法、消息、编辑 plan、阈值、CAL32/TEST500、12 视图和全局分母保持。
不新增 LPIPS，不修改 DCT2＋CDC32 或旧 H/L。

1. 先恢复已保存原始批次和 CPU 后评。仅使用所选 receipt lineage 的最后已保存
   epoch；registry 缺失时，只有全部 effective receipts 能唯一关联一个 epoch 才恢复。
   不按分数选 epoch，不把最新 STARTED 回退成旧完整 epoch。
2. PixelSeal/FrameIndex 的 OFF 原始读量共享，逻辑后评分别调用原 evaluator。
   已有完整 post 直接跳过；路径优先级与原全局表一致，旧完整不能越过新 incomplete。
3. 质量按缺失 stage 取原配对数组/媒体。原 G1/G2 不在时，只能从已保存原 writer/
   canonical 像素按原 H264、CRF18、25fps 恢复质量素材；新媒体与费用独立保存，
   不接入旧 native epoch。单个 stage 失败留原因，真实磁盘写入失败停止。
4. CPU 未解决项才进入 GPU。读得出的 COMMITTED 批次不重算。部分 epoch 的输入丢失、
   committed 数组不可读、epoch 关联不明等保留未决，不整视图隐式补跑。
5. ucf-0642 的 PixelSeal 和 FrameIndex 正例臂各 12 视图全缺，按已授权原源、原查询
   完整重建。新 writer/media/native 不混入旧臂批次，旧 receipts 单列保留。
   原方法/格式失败不会因为输入恢复而自动重试。
6. ucf-0643 PixelSeal 按原方法、消息、源和编辑计划完整重建 12 视图及 3 质量阶段。
   旧结果保留；新臂从原始读取、CPU 后评、质量到全局表只使用新臂结果。
   GPU 前先建立新臂固定空槽；新臂失败/缺失时保持未知，不能回填旧 9 个成功视图。
   `shards/part2/replacements/` 明确记录整臂选择和旧结果位置。ucf-0642 的两个新臂
   使用同样的整臂隔离，任何原观察都不拼入新臂。

## 本地 CPU 交付复用

Notebook 自动读取 `recovery/repair01/local_cpu_recovery.zip`。只补缺失证据文件，
已有 Colab 文件不覆盖，runtime/owner/lock 和派生 `physical_counts.json` 不导入。
任何导入写入前检查修复根、修复四片和原四片的实际运行状态，导入期间持有修复根
runtime lock。证据先完整解压到内存，再写临时文件并原子发布；中断不会留下半最终文件。
CPU `physical_calls.jsonl` 的 STARTED 成本事件按原事件地址追加一次，原预算累计值
仍从旧分片继承；不会把累计计数再次算成新增成本。非法 zip 路径和符号链接拒绝。
所有复制、已有路径跳过、排除项及新增成本写到 `cpu_package_imports.jsonl`。

父会话本次实际本地恢复已得到 3 个视图和 5 个质量阶段；新增模型 0，质量专用
codec encode 5、probe 8、decode 8。后续 Colab 从这些已完成记录继续；本文件不把
工程检查或恢复完成当作科学成功结论。

## 输出与本地命令

- `cpu_recovery.json`：逐组原始读量、CPU 恢复与剩余原因。
- `physical_plan.json`：实际补算前的缺失批次候选地址及完整臂重建范围。
- `physical_recovery.json`：各项返回、失败和未决，不把候选数当实际模型次数。
- `shards/partN/physical_calls.jsonl`：新增物理成本；`physical_counts.json` 包含继承成本。
- `global/tables/`：复用原 `build_tables` 按固定槽全局汇总，不平均分片率或区间。
- `result.json`：工程恢复状态；科学解释仍需独立核验。

仅 CPU、部分本地镜像不生成全局表：

```bash
python -m runtime.wan.native_baseline_compact_recovery_run_v1 \
  --cpu-only --no-tables --previous-runtime-stopped \
  --parent /path/to/mirror --output /path/to/cpu-repair01 \
  --cache /path/to/mirror-cache --old /path/to/mirror-old-v4 \
  --old-cache /path/to/mirror-old-cache
```

该 CPU 入口不加载模型。真实新物理调用只由用户运行的 Colab 完成。
