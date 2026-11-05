# 正式公开视频 Baseline：WAM32 / PixelSeal256，nominal family FPR 1%

本入口实现已采纳 REV_A 的公开视频给定完整消息验证。默认真实来源为 UCF101 500 CAL 组＋500 TEST 组，11个固定时间条件×两代媒体。WAM中心四分之一、原生32位；PixelSeal全幅、原生256位。原生盲提取、给定消息验证、画质和资源分别保存。Wan生成域同源比较另列，旧v1/v2、主项目与旧结果不改。

本地人工CPU检查仅验证工程；尚未执行真实媒体、模型或Colab，不能由此声称FPR已证或论文科学成功。

## 入口

- Notebook：`notebooks/native_baselines_paper_fpr1_v3.ipynb`，用户选择GPU后Run all。
- Builder：`scripts/build_native_baselines_paper_fpr1_v3_notebook.py`，首cell仅Drive mount。
- CLI：`python experiments/wan_state_clock/native_baselines_paper_fpr1_v3_run.py --vendor /content/video-wm-native-baselines-paper-v3/vendor`。
- 默认config：`experiments/wan_state_clock/configs/native_baselines_paper_fpr1_v3.json`，同目录真实 `.sources.json`。
- 只读后评：同一CLI加`--evaluate-only --output <已保存结果目录>`；不调用模型或codec。
- 固定结果目录：`MyDrive/Video-WM_Result/baselines/native-wam-pixelseal-paper-fpr1-v3/attempt01`。原AVI、OFF/WM媒体、失败、部分输出、native/statistics与日志均保留，不参与缓存清理。

正常Run all自动准备源码/依赖、下载选定AVI、源准备、CAL读取/定标、TEST读取及后评，无人工搬包。源码入口可编辑，未建立hash/B64/manifest/固定引用/版本/Git-clean门禁。直接检查实际文件可读、格式、数值、方法语义与来源分组。

## 实际数据范围

默认名单来自[UCF101公开镜像](https://huggingface.co/datasets/sumith2425/UCF101)。完整2525个(action,group)经过文件头盘点，13320路径中按每组clip序查首个可读取、至少181帧、25fps、正几何的clip；8950次最多4096-byte头请求确认1145组合格。1284组所有已查头不满足条件，另96组含头读取错误；全局108 timeout、9 URL error保留，17个最终合格组也有较早clip错误。它是已确认可读的长片子集，不代表其余组都短或抽样完全无偏。

1145组按action/group排序后PCG64 seed20261105一次排列，前500 CAL、随后500 TEST。固定1000组覆盖73个动作：CAL 69、TEST 71；group无交集。余145组不作失败替补。每条直接使用原`url`及`dataset_path`，保留HandstandPushups目录/HandStandPushups文件名等大小写；`clip`是数字编号，不重建URL。默认表顶层明确header-only/no-decode：成功文件头不等于181帧已实际可解码。

Colab每clip仅取前181个连续实际解码帧，保留自然25fps，不用fps插补、循环或pad补短片。最大整数8:5中心裁剪后bilinear缩放H320/W512；320×240源裁x0:320/y20:220。实际解码不足即消费原单位并保留失败，不换来源。原AVI已含源压缩，G1/G2指本实验追加的H264 CRF18首代/第二代。规范化时保存相邻RGB绝对变化均值作描述，不据此筛内容或宣称光流覆盖。

[原作者论文](https://arxiv.org/abs/1212.0402)说明同group共享背景/演员。已知group隔离减少相关片段泄漏，但不证明全部组绝对独立或公开预训练无重叠；本项目未参与设计的TEST仍是按已知组隔离的held-out测试。自然域结果不冒称Wan同源、相等载荷/空间支持或完整101类泛化。

## 判决与统计

模型输出前以PCG64 seed202611052001为每source/method分配一个完整原生随机query，22视图共用；碰撞保留，每次只查该query，不搜消息库。TEST另抽不同wrong-query（query XOR均匀非零mask），仅作描述，不能宣称单独校准到1%。

WAM保持原生mask sigmoid>0.5选像素、`msg_predict_inference(semihard)`最终均值>0.5的hard bits；view score为所有正常非空收到帧对query的一致位数max。正常全空mask=-1/无消息，不是工程失败。WAM原生视频消息仍N/A，没有投票。Pixel保持全部收到帧payload logits无权平均后>0，score为256位一致数；不去重、不用检测通道0、不再次detect。盲输出先持久保存，query validator后接；decoder只收到实际RGB，不收query、writer truth或编辑映射。

每方法500 CAL OFF的每源22视图max，升序第496个为tau，严格`score>tau`才接受，tie拒绝。对应交换性假设下5/501≈0.998%的边际family目标，不是固定阈值下任意总体的95%保证；任一单条件事件包含于family事件，不是22个各自精确1%。tau=-1/32/256退化原样记录，不降阈值救结果。

CAL缺任一必要观测，该方法`CALIBRATION_INCOMPLETE`；不缩N、补低分或重跑。仍执行原计划TEST native/quality，只不给该方法验证判决；另一方法独立。只有完整保存view才给主score；部分已保存帧继续进入native明细。TEST family已见完整view FP则1，无FP但有缺失则UNKNOWN，22完整且全拒才0。

TEST每条件和family固定500组；记录planned/completed/unknown、FP/TP、coverage与`positive/N`到`(positive+unknown)/N`，工程缺失不当零误报，正例unknown不当成功。完整时95%双侧Clopper–Pearson及单侧95%上界；缺失时取固定分母上、下计数的保守区间包络，完整子集区间另列。0/500单侧上界约0.597%；2/500实测0.4%且上界超过1%，只限制置信声称，不引入PASS门或重调规则。按action聚类2000次bootstrap、seed202611052002作敏感性，零事件退化bootstrap不替代精确上界。

## 固定物理预算和恢复

1000 source-prepare＋2000 source-method为消费单位；100个10源shard只调度。每unit调用前持久STARTED；完成、失败、中断都消费该unit，下次Run all只续NEVER_STARTED。模型按method外层/source内层在同shard复用，每次最多一个GPU模型；首个实际load记该unit预配槽，后续`REUSED_NO_CALL`，不强制每源重新加载。实际重连/耗时记录，不能承诺单会话完成。

| 项目 | 计划上限 |
| --- | ---: |
| WAM B8 detect API/forward | 645000，处理5025000帧 |
| WAM B8 embed API/forward | 11500，处理90500帧 |
| Pixel detect API/forward | 33000/180000，处理5025000帧 |
| Pixel embed API/forward | 500/500，11500 keyframes |
| clip-method tasks / frame rows | 66000 / 10050000 |
| 结果媒体encode/probe各 | 24000 |
| 结果媒体decode | 37000 |
| 原AVI normalize / metadata probe | 2000 / 至多1000 |
| quality precodec/G1/G2 | 1000/1000/11000 |
| model_load | 每source-method最多1，总≤2000 |

prepare每源OFF首代＋11个G2 encode/probe，只decode首代；两个方法各decode持久OFF 12次，TEST WM每方法再12次，因此decode=1000+24000+12000。两个TEST writer各另normalize原AVI一次，连prepare共2000次。未执行和失败槽保留，forward hook与API/实际帧分列，不能用逻辑帧当forward。

源获取预算是1000个任务。每次`urlopen`记`download_calls=1`；urllib可自动跟随重定向，底层HTTP跳数未测量，保存`http_requests=null/NOT_MEASURED`，不将下载函数调用冒称实际HTTP请求数。物理回执按全部预配槽铺底：从未启动为NOT_STARTED，已消费但缺账为UNKNOWN_UNIT_LEDGER，真实RETURNED/FAILED/REUSED_NO_CALL覆盖对应槽；额外保存的未知操作另列。损坏结果JSON按文件报告读取错误并保留原件，继续显示可读材料，不触发模型重跑。

WAM FP32/eval/inference_mode、同source/view连续B8，尾批5/1/3/7/7不pad、不混图像统计；OOM不降低batch、换精度或重试。Pixel保持chunk32/step8。批处理不承诺与B1 GPU逐位一致或实测显存/速度。codec单命令600秒有界停止；notebook停止或worker退出后清理整个进程组，含已退出leader遗留后代。

## 保存材料与资源边界

每source-method用一个append-only `native.bin`保存按批NPZ数值记录，另有单一events日志与固定plan；按批返回后持久保存。中途断写只保留此前可读记录，未落盘的已返回读量为OUTPUT_MISSING，不重算。后评`post/<method>/<source>/frames.npz`配合`evaluation.json`提供固定帧行的状态、correct/wrong/unknown、源帧与重复实例索引；not-received源索引单列。原生bits/sums/means仍在raw native文件中。

WAM保存bits、valid、selected_count N、公开中心64:192区域count I、sums/means、mask_shape，不保存502.5万个空间mask文件。固定无空间攻击下WM IoU=I/(N+16384-I)，OFF IoU=(N==0?1:0)；不支持任意新ROI事后重算。bits来自原parser，float64诊断均值不能替代原生hard bits。Pixel保存完整257通道FP32 frame logits及同次原生frame/video bits。

WAM主要数组payload约2.779GB，Pixel logits约5.166GB，另有NPZ容器/逐批记录、Pixel bits、状态/索引、日志、原AVI和全部OFF/WM MP4；这些不是完整Drive上限，媒体压缩字节数须实测。当前源/当前view大数组用完释放，GPU同一时刻只留一个模型；不缓存1000源RGB。全部正式结果永不作为临时缓存删除。

## 验证

`python -m pytest -q tests/test_native_baselines_paper_fpr1_v3.py`只运行人工CPU/静态检查，fixture URL明确为fixture.invalid，不进入正式默认来源。真实Run all需要用户Colab GPU、Drive、网络及实际模型/媒体；本次交付未执行这些步骤，也没有将人工数据结果填入论文表。
