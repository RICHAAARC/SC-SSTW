# 分支与结果索引

2026-09-20 用户授权的最小整理。两条活跃方法支线继续推进；历史配置固定不等于否定整个方法。此次只整理入口和已有记录，无模型运行、调参、合并 main、删除历史提交或工作树。

Git 中保存轻量索引与 manifest；原始 JSON/log 位于各工作树 `evidence/raw/<run>/`，本地 Git exclude 排除，不随 clone 下载。manifest 保存原始 Drive ID/URL、来源类别、字节数、SHA256 和获取时间。文件哈希证明本地字节完整性，不证明科学有效性。已有本地文件、用户粘贴摘要与新下载 Drive 原文件分别标注。未下载视频、tensor、模型或源码 ZIP；因此这是结果记录归档，不是完整重跑资源包。

## 10 个方法/历史/发布分支

以下 SHA 是研究源码/已发布入口的固定锚点，不是不断变化的文档 HEAD。英文名对应新规范分支；两个旧中文 tag 是兼容入口，不算额外研究路线。

| 分支 | 状态与用途 | 固定源码 / notebook 入口锚点 | 结果与结论边界 | 下一步 |
|---|---|---|---|---|
| dev/flow-tube-state-guidance | 活跃；LAST 末步管状块状态控制 | dev 源892013f；holdout源0797064；crop源7298170；holdout-crop源3b8ee00；最新入口eb06976 | [四批原始记录](INDEX.md)：完整视频dev五模式8/8；holdout-crop每起点4/4。裁剪窗口对齐不等于精确帧级同步或低FPR | 保留已验证固定配置，继续未证实能力 |
| dev/flow-tube-multistep | 活跃；完整早期投影与末步补偿 | 源b57c13b；入口018ab4c | [本轮记录](../../Flow-Tube-Multistep/evidence/INDEX.md)：LAST/EARLY_LAST各8/8，EARLY_ONLY6/8或7/8；EARLY_LAST控制和画质代价更高。当前只有用户摘要，原Drive包未取得 | 继续方法推进；本轮只是固定配置消融 |
| dev/video-inversion | 固定正结果对照；初始噪声写入、已知prompt反演 | paper源7fa72ef；入口b6caddc | [三批原始记录](../../Video-Inversion/evidence/INDEX.md)：STATE与STATIC/search14均4/4源全视图正确；shift0各2/4与3/4；评估OFF仅2源 | 不追加无目标调参；状态优势、低FPR、质量和一般鲁棒性未证明 |
| dev/grow-video-frequency | 历史保留；固定已有配置与消融 | paired源51b774a；入口b6e470c | [记录](../../GROW-Video-Frequency/evidence/INDEX.md)：paired MULTI/LAST终态均8/8，MP4硬读4/8与7/8，软读3/8与5/8 | 保存部分正证据，不把整GROW判失败 |
| dev/grow-single-step-jacobian | 历史保留；单步30控制负消融 | 源001bc59；入口b95f01c | [原始记录](../../GROW-Single-Step-Jacobian/evidence/INDEX.md)：LOCAL/JAC完整消息各层0/8，输入VJP实际完成 | 结论限当前时刻、载体、预算与自由续程 |
| dev/flow-tube-state | 历史保留；旧受限Flow及clipped配置 | 旧源fca6f1f；clipped源8b55bce；入口d5877c5 | [原始记录](../../Flow-Tube-State/evidence/INDEX.md)：旧Flow归因不稳、质量四项通过；新取得clipped实际完成但NO_SELECTION | 更正此前未核验状态；固定负结果，不自动补跑 |
| dev/public-statistic-control | 历史诊断；亮度DCT二维公共统计约束 | 原入口891988b；规范入口40ac028 | [原始记录](../../Public-Statistic-Control/evidence/INDEX.md)：run08拒绝更新；split02在A100完成12TF/3VAE/5backward，未执行候选写入或接受判定，science_denominator=0 | L4修复仍未验证；不把拆分诊断完成等同方法成功 |
| dev/real-video-public-observation | 历史诊断；固定镜头公共观测/局部warp | warp源f54917a；原入口1698fbf；规范入口8476c0f | [原始与独立审阅记录](../../Stage1-Public-Observation/evidence/INDEX.md)：6视频完成，动态q覆盖11/12/11/15低于24/30 | 仅固定构造工程负结果，不否定一般公共观测 |
| c2a-2a-colab-preparation | 待整理历史开发树 | dff5dd344f198a08a00076a965e3b85ee724f933 | 24独有提交；14 tracked修改+21 untracked文件，未纳入本次提交 | 单独判断保存方案，不阻断活跃方法 |
| main | 稳定正式发布；终态投影state-clock基线 | 源591c104d06b3c149f00544622de9aefb4b159ad2；入口3f0a5fafa7c2aa56fbc69bed17649accf49ae152 | 保持默认分支与发布树；新实验没有自动合入 | 保持稳定 |

源代码完整 SHA 以各 manifest 内实际记录为准；入口短 SHA 均可在同一仓库解析。Public 原文件未完整记录执行 Git SHA，891988b 是入口锚点，不追认历史实验版本。Warp f54917a 来自既有独立审阅的来源记录，不从 pending 原始结果推断。

## 新补取证据纠正

- clipped run `velocity_clipped_margin_20260917T134515071140Z`：四case terminal/media完成exit0，最终NO_SELECTION。rho1.0五模式均正确3/8，rho0.3为2/8；holdout NOT_RUN_NOT_AUTHORIZED。MATCHED仅限saved-zero终态相等，非原模型/embedding身份证明。此前“未核验实跑”已被本次原文件补证替代。
- public `after47-split02`：returncode0，12TF/3VAE/5backward完整；runtime明确A100-SXM4-40GB。它不能替代L4验证，也没有新写入成功证据。
- FlowTubeMultistep当前搜索及Video-WM根目录列举未返回原目录；只归档用户原文摘要，保留该缺口，不包装为原始结果包。

## 两个旧名的兼容映射

| 旧中文兼容 tag（本地与远端保留，原中文分支已移除） | 固定提交 | 新规范名称 |
|---|---|---|
| dev/生成端公共关系载体/二维图像统计-持续生成约束 | 891988bfeaedb82a622d9ae3aac7a3e1531d66d5 | dev/public-statistic-control |
| dev/真实视频公共观测/帧差加权质心-固定镜头单主体 | 1698fbfc1a7081d0538090fe549d8aabe3d1f049 | dev/real-video-public-observation |

改名前相对其他研究分支各有12/14独有提交，现由新英文分支共同保存。旧SHA notebook仍可能fetch/clone旧分支，所以以同名tag保留旧入口。新4+2入口仅固定改名前默认实际获取的源码并验证HEAD，非追认旧实验SHA。warp既有f54917a绑定未改。历史日志、结果及notebook历史版本不改写；本次将旧中文branch原子转换为同名兼容tag，不移除工作树。

## c2a 未提交工作：只读分类

没有修改、提交、删除或stash这些35个文件；没有识别出可直接当临时垃圾删除的文件。它们主要是正式main已接收的核心/适配器副本及main刻意不收录的历史兼容材料。

- 逐字节相同于main的8文件：tests/test_state_clock.py、tests/test_wan_projection.py、main/tube_state/{__init__,projection_margin,state_clock}.py、requirements-dev.txt、runtime/wan/{__init__,generation}.py。
- 仅末尾空行不同的3文件：runtime/wan/{io,quality,vae}.py。
- 文档/契约：.codex/project_contract.md、AGENTS.md、README.md、governance/policies/method_readiness.yaml、docs/current_method.md、governance/docs/rpgov_adaptation.md；这些内容与main不同，保留开发/兼容边界。
- 历史适配迁移：runtime/c2a/{chain,generation,protocol}.py、runtime/c2t1/run.py、runtime/tstwv2/{projection_margin,state_clock,run}.py；main无这些旧路径，不可仅因新核心已存在就删除。
- 检查工具/配置/测试：tests/test_synthetic_interface.py、governance/harness/checks.py、governance/policies/validation.json、governance/tests/{test_checks,test_compatibility,test_equivalence_tool}.py、governance/tools/{run_validation_profile,verify_extraction_equivalence,verify_formal_runner_candidate}.py。
- 历史检查报告：docs/local_refactor_review.md、governance/reports/extraction_equivalence.json。后者是dff5dd3基线的CPU抽取等价记录，不是新增科学结果。

后续若保存，需按上述归属精确选文件；本次不对整个dirty树打包提交。c2a整理不是其余分支工作的前置条件。
2026-09-20 后续命名整理：本地与远端现在仅10个英文分支，两个中文名称仅为固定兼容标签。已在全新临时仓库验证历史 fetch 与 clone --branch 两种取源方式，均得到原SHA。
