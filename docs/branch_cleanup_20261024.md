# 分支整理远端收尾记录

日期：2026-10-24。用户明确接受旧历史GitHub／Colab分支链接失效，并授权远端删除与本次整理文档提交／push。此次只做引用和导航整理，不改变研究方法或结果。

## 最终入口

| 入口 | 保留分支 | 本地工作树 |
|---|---|---|
| 正式公共机制与只读论文外层 | `main` | `worktrees/SC-SSTW-main-publish` |
| A论文、结果和同步对照 | `dev/paper-results-v1` | `worktrees/Paper-Results-V1` |
| B后续开题同步研究 | `dev/local-joint-state-payload-v1` | `worktrees/Local-Joint-State-Payload-V1` |
| 旧主checkout及C2A入口 | `c2a-2a-colab-preparation` | `alive/SC-SSTW`，不是main工作树 |

实时origin heads、本地heads和活动worktree分别为 **4／4／4**，GitHub默认分支仍是main。A/B/c2a的HEAD、工作内容及四个目录保持；main仅前进本轮整理文档提交。

## 两阶段事实

| 项目 | 最初 | 第一阶段 | 本次收尾 |
|---|---:|---:|---:|
| 本地分支 | 39 | 4 | 4 |
| 实时origin分支 | 36 | 35 | 4 |
| 登记工作树 | 38 | 4 | 4 |

第一阶段完整保存34个历史文件树和原39条heads所需提交历史，清理历史本地ref与工作树登记，远端只删core-integration。31条历史远端当时为旧入口兼容保留。第二阶段取消此兼容保留条件，通过实际远端删除移除31条，并刷新本地remote-tracking导航；没有创建tag，没有重复搬动或重建归档。

## 本次实际删除的31条远端分支

- `dev/bidirectional-cross-confirm-v1`
- `dev/content-background-existence-v1`
- `dev/flow-fixed-key-v1`
- `dev/flow-tube-multistep`
- `dev/flow-tube-objective-alignment`
- `dev/flow-tube-state`
- `dev/flow-tube-state-guidance`
- `dev/flow-tube-sync-diagnostics`
- `dev/flow-tube-terminal-feedback`
- `dev/flow-tube-uniform-tanh`
- `dev/grow-single-step-jacobian`
- `dev/grow-spatial-carrier`
- `dev/grow-video-frequency`
- `dev/image-trajectory-reference-v1`
- `dev/keyed-presence-t40-t49-v1`
- `dev/media-channel-retention-v1`
- `dev/public-statistic-control`
- `dev/real-video-public-observation`
- `dev/rgb-dct-b2-terminal-receiver-v1`
- `dev/rgb-dct-group-consistency-h0-h1-v1`
- `dev/rgb-dct-h0-h1-functional-v1`
- `dev/rgb-dct-local-gradient-v1`
- `dev/rgb-dct-multistep-comparison-v1`
- `dev/rgb-dct-structured-terminal-feedback-v1`
- `dev/rgb-dct-temporal-balanced-v1`
- `dev/rgb-dct-terminal-gradient-v1`
- `dev/trajectory-attribution-v1`
- `dev/trajectory-payload-framewise-sync-v1`
- `dev/video-inversion`
- `dev/video-trajectory-blind-validation-v1`
- `dev/window-state-mse-v1`

另：`dev/sc-sstw-core-integration`已在第一阶段删除，不属于本次31条；`release/conditional-joint-v1-main-20261008`、`dev/flow-tube-partial-observation`和`dev/videoseal-receiver-diagnostic-v1`原来没有live origin，因此没有冒称本次删除它们。

删除前的实时35条与上轮清单一致，没有新增活动分支，也没有将其它并行任务分支混入删除名单。四个保留ref不在删除集合。用户接受历史分支URL失效，不再承诺旧链接持续可用。

## 当前取码与检查范围

只读检查四个保留树的实际Notebook／builder取码：当前调用只依赖上述保留分支及仍由它们承载的现有固定提交。没有发现对待删31分支的真实运行依赖，因此本次不修改Notebook、builder或方法代码，也没有将旧实验整包合入main。

必要检查为：实时远端heads、默认分支、精确提交文件范围、文档格式，以及四个保留入口的远端Notebook可读性／JSON解析。不运行Notebook、模型、GPU、VAE、FFT、codec或CPU科研回放；入口可读不等于科学通过。没有新增摘要、版本、Git-clean等运行门禁。

本次main精选提交范围仅`docs/branch_guide.md`与本文。其他工作树未提交研究内容不参与此次提交。

## 本地材料和按需恢复

完整材料仍在用户本机：

```text
/home/richar/projects/Video-WM/archive/branch-cleanup-20261024/
  repository-history.bundle     # 原39条本地heads及祖先
  trees/<原工作树名>/           # 34份完整文件树，含未提交/untracked/ignored
  records/<原工作树名>/         # 对应index、patch和定位记录
  README.md                     # 直接查阅及按需恢复说明
```

项目根`diagnostics/branch-cleanup-20261024/REPORT.md`保留第一阶段回交并追加本次收尾；`remote_final_*`记录本轮实时清单、实际远端删除及发布核对。这些本地路径不代表GitHub上传了bundle或研究数据，也不是结果读取／运行准入条件。历史材料未删除；需要历史内容时直接查阅这些普通文件或按README恢复，不依赖已经移除的分支URL。
