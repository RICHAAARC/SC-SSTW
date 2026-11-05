# Joint-A 固定候选内硬码字一致性：单新源验证

入口为 notebooks/b_joint_time_relative_validation_v1.ipynb，builder为 scripts/build_b_joint_time_relative_validation_v1_notebook.py。用户选择GPU后Run all，顺序执行新P0生成、同host捕获、原CPU求解、native replay、NEW/BASE全部五视图、原hard/F42与条件比较、盲保存和后评价。首cell仅Drive mount，不需要人工下载上传科学输入。SOURCE_REF或可编辑SOURCE_DIRECTORY用于寻找可用源码，不作版本或完整性准入。

单新内容 joint-relative-r01，seed2026110501；prompt在配置 experiments/wan_state_clock/configs/b_joint_time_relative_validation_v1.json。181×320×512、8fps、原Wan50步/CFG5与DCT2写端不变。BASE有DCT2身份但没有Joint-A时间包；接收端E41/project/identity-bit=0，本轮身份/CDC32与joint成绩未评价。

仅在原F42已有完整可评多候选W内比较。所有候选共同使用原packet未校正的lane0..4×R0/R1×bit0..62共630个已知物理硬位；F为派生解析，不作为第三物理copy。公开模板为原msg=8j+age与BCH63/24，完整整数Hamming唯一严格最小才输出条件位置。并列、缺位、非法、startup不同支持均保留整个W；原空/唯一含startup沿用。没有扩大候选、全域最近、半径/margin、路径非降或truth选择。条件边保留完整笛卡尔积及INVERSION，原F42嵌套独立保存。

固定FULL181、CROP101、DELETE1 180、REPEAT1 182、DELETE3 178；每arm822节点/817边，两arm1644节点/1634边及1,035,720物理硬位槽。全部失败、未决和缺失仍占固定分母。评价报告新增正确、保留歧义、新错唯一、原正确节点/边损失、启动区与BASE输出；无可比较新歧义只报告保留性。条件唯一位置不是水印存在证明或正式科学PASS；反例保存，不换源调参补跑。

外层78调用槽分为prepare5、solve5、replay2、receive66。四个独立进程使用VAE load2、D181 capture1（内部baseline head46）、head replay46；100 DiT、25 DCT2 payload-gradient事件；solver K≤4200/KT≤4200、prediction K1。codec encode/probe/RGB decode各12，nativeY/hard/F42/relative各10。原writer编码1790，接收raw24 decode≤24660，新候选模板encode≤82200，硬位比较≤10357200；这些是不同层计划上界，不是已执行数。stage事件记录外层开始/返回/失败，internal_calls记录模型和K/KT，relative计数记录实际模板与比较量。中断的细粒度decoder计数不能由未保存packet重建，按未知保留。

固定结果：MyDrive/Video-WM/B-Joint-Time-Relative-Consistency-V1/fresh-r01-v1/attempt01。源码/权重/约11.39GB feature及大像素缓存使用/content/video-wm-joint-relative-v1；原q0/head/系数、媒体、raw读量、盲输出、native回执、后评和失败记录永久保存。原CPU solver 7200秒/4000迭代/16GiB live-array估计；接收8GiB RSS、每hard600秒/累计6300秒，codec每次300秒。这些不保证整段GPU峰值或时限；实际资源/文件/模型失败直接停止。

准备前即消费attempt，失败或中断也消费；后续Run all只读，不补模型/媒体或后评。不要同时运行两个实例。Notebook中断会终止runner所属进程组，包括尚存stage/codec。运行时硬重置可能保留不完整末次终态，仍不能补跑。

CLI不依赖notebook或Git，可从可编辑源码目录运行：

```sh
python experiments/wan_state_clock/b_joint_time_relative_validation_v1_run.py --help
```

后评依据实际调用账与保存文件区分NOT_STARTED、FAILED、INTERRUPTED、OUTPUT_MISSING、BLIND_SAVED，部分执行或账不足单独保留。缺失节点/边的科学指标为null，不计已读未决；只完成部分NEW时明确部分执行，不能当已执行零错误。所有计划行仍保留。

本线已完成实现、15项不同人工CPU工程检查及A1–A5分阶段审查。真实新源GPU、模型和媒体尚未执行；工程接受仅支持接口、记账与停止行为，不证明方法效果。远端发布及回读状态由主方另行记录。
