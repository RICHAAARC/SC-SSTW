# 固定短片与多数重复 CPU 分析

新入口 `notebooks/paper_results_v1_short_reference_v1_colab.ipynb`，由 `build_short_reference_v1_colab.py` 构建。CPU Colab Run all，从原独立四源Drive读取4份真实接收clock Y，固定输出 `Paper-Results-V1-Short-Reference-V1/resumable-short-reference-v1`。不加载模型、解码媒体或读取消息。

四源全部保留，固定起点37，长度9/17；另各在几何中点额外重复10/18份，得到19/35帧多数重复。4源×4视图×RAW/CENTERED/同一DC参考=48行、960臂帧对应，4独立来源。按每个收到视图重新计算CENTERED；REF逐帧patch均值参考及公开增益不调参。保存所有状态和完整路径；多数复制是特定机制压力，不代表一般重复失效。

同Drive恢复先检查已完成行的必要selection JSON及有限float64正确形状的q NPZ；完整SUPPORTED/UNSUPPORTED/UNRESOLVED均跳过，附件缺失或损坏仅补对应廉价CPU行且保留历史。比较已记录的具体方法值和实际view映射，已知方法不同直接报告；旧可选记录缺失仍可读现有结果，不检查源码/版本/摘要。下载失败时使用同Drive可读缓存ZIP。缺失输入不自动编码补造Y。CLI从普通无.git源码运行：

```bash
python -m experiments.paper_results_v1.short_reference_v1_cli --source-run /path/to/resumable-independent-four-v1 --output /path/to/short-reference
```

2026-10-16本地真实保存Y的48次CPU结果：RAW路径0/16，CENTERED8/16（普通短片8精确、多数重复8未解决），REF12/16（4行错误）；REF绝对位置误差和1509，RAW15815。CENTERED未解决8行使固定分母误差和不可计算，不能写作全16行零误差。无新GPU载荷证据；CENTERED继续作为原任务默认，DC参考长片段6/12对CENTERED12/12的负结果保留。此结果支持特定条件差异，不足以选择通用替代或推荐GPU参考实验。
