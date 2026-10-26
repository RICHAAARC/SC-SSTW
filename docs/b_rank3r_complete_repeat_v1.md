# B-RANK3R 完整重播 M2

本入口实现已采用的唯一M2：γ1旧四来源12个完整MP4，经r=8、b=49/102的左右完整重播形成24个151帧片段；只比较原RANK3与独立RANK3R。本入口随B分支源码发布，可直接Run all，无需手动上传源码ZIP；本M2入口尚未执行真实模型、VAE或媒体实验。

[在Colab中打开](https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/dev/local-joint-state-payload-v1/notebooks/b_rank3r_complete_repeat_v1.ipynb)。

输入默认 `/content/drive/MyDrive/Video-WM/B-Rank3-Four-Source-Confirmation-V1/repeat-20261025-v1`；输出独立为 `/content/drive/MyDrive/Video-WM/B-Rank3R-Complete-Repeat-V1/fixed-four-v1`。方法输入须保留旧background/protocol、72任务的planned_outputs/evaluation及12 full MP4。threshold仅尝试读取为历史记录，缺失、损坏或与参考τ不同均保存具体状态，不阻断、不选择替代τ；两排名读法始终不用τ。缺失不会重新生成；不依赖diagnostics文件。

原RANK3保留961类，RANK3R采用4247类/35743 seam aliases。a=0..30是旧设计先验，负d=-r对每b须1<=r<=min(b,151-b)，全部151映射在source0..180、三窗origin在H137。实际只测r8；b只作别名，不能声称精确seam恢复。重复同源引用不增加32槽覆盖或独立票，相等值不增信、冲突擦除。

完整执行固定codec encode24/probe36/decode36（12源解码共享左右fixture）、E41=288、Q=576、H family144/score19728、DCT288/双key位解码576；生成/DiT/gradient/state/D181/E181/native均0。各读法48任务/1536槽：256正、1280null；PM wrong-key256是null子集。旧72仅重放扩域membership及旧actual-support标志，不重算Q/S/bit。全部96 reader plans保存回读后才消费位cache，96 decisions回读后才打开当前payload真值。

先持久化attempt/固定计划，首技术失败或中断停止后续物理调用，已STARTED消费、余项NOT_ENTERED；后续Run all只读恢复，不补缺。普通CLI也拒绝孤立启动收据及任意M2执行痕迹。Notebook本次新收据读回后用专用内部fresh-start交接，同样拒绝任何runtime执行痕迹并独占创建attempt；没有公开override/retry开关。

Notebook首cell仅两行Drive挂载，默认下载B分支可用源码到本次会话临时隔离目录。成功repeat的SETUP、ftfy/torchao实际缺失修复、ffmpeg及FP32 VAE/cache沿用；模型/HF缓存和瞬态tensor留/content，Drive保存JSON和24最终MP4，无源码ZIP或摘要门。

完整观测下RANK3R须正DETECT>=6/8且每源>=1/2、actual-supported-correct>=192/256且每源>=32/64，旧384零损失，旧60+新40 null零DETECT，字面错/错误归属/无实际supporting exact支持输出均0。null DETECT立即记候选失败，即使无bit输出；只可完成已保存的既定逻辑诊断，不追加物理工作或改域。完整观测未达标为METHOD_NOT_SUPPORTED，缺测/调用失败为TECHNICAL_UNRESOLVED。

旧质量采用record-only wrapper：固定expected_rows=76、12316帧/12240相邻对，原可用records全部保留，缺失/无效行明确列出且不进入方法判据或阻断运行；新增24条直接edited RGB→最终codec readback整幅距离与相邻残差，共3624帧/3600相邻对，复用内存数组、零额外codec/VAE，不使用水印support mask或感知PASS。保存状态独立于方法状态：Drive API实读核心JSON和24视频，/content生成可下载结果备份，缺项明确保留，不触发补跑。

CLI：`python -m experiments.wan_state_clock.b_rank3r_complete_repeat_v1_run --input <saved-repeat-root> --output <new-result-root>`。源码无.git可导入，`--help`及已有结果读取不加载模型。CPU/fake/static仅证明接线，不是重复鲁棒性结果；M3/M4不在本入口。
