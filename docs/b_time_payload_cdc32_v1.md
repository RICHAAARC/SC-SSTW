# c01 时间到 DCT2＋CDC32 的同链路读取

本实现依据 `diagnostics/time-to-identity-same-chain-method-review-20261031/FINAL_IMPLEMENTATION_SPEC.md`，接入既有 c01 五个二代 H264 编辑媒体及 V3 submission。代码准备和人工工程验证不表示真实身份已读回；工程检查阶段未执行真实数据回放、媒体/模型/GPU/Colab，发布及远端链接由主会话核验。

## 入口与输入

完整入口 `notebooks/b_time_payload_cdc32_v1.ipynb` 与 `scripts/build_b_time_payload_cdc32_v1_notebook.py` 同步，首代码格仅 Drive mount。默认输入 `/content/drive/MyDrive/Video-WM/B-Conditional-Time-Validation-V1/fixed-c01-v1`，仅打开 `receiver/received/{FULL,CROP,DELETE1,REPEAT1,DELETE3}/received.mp4` 与 `submission.json`。新独立输出 `/content/drive/MyDrive/Video-WM/B-Time-Payload-CDC32-V1/fixed-c01-v1`。不会使用 materialization/B/QIM.mp4、P0或旧M3/e01缓存，不重新writer或时间读/提交。

独立命令为 `python -B -m experiments.wan_state_clock.b_time_payload_cdc32_v1_run --config /absolute/config.json`，也可直接运行脚本；无.git源码可用。已有输出通过 `--output` 在配置与科学导入前只读恢复。源码普通下载/安全解压，实际导入修复，无摘要/B64/manifest/精确版本/Git状态门。

## 真实路径与资格

`main/tube_state/b_time_payload_cdc32_v1.py` 只接匿名input_id、收到长度与最终时间字段。每view query起点为0、(N−45)//2、N−45，各有p0..3，共12个E41。E41读取实际收到顺序a..a+40；runtime复用原FP32 Wan encode_view、一次uint8/255、posterior mode、mean/std归一化及逐窗口cache reset，返回CPU FP32 [1,16,11,40,64]。原DCT2 project/decode直接复用R0/R1、row1..10、8票、strict>0与Counter遇见次序。

B只接受完整合法的CONDITIONAL_OUTPUTS_SAVED最终位置。四帧输出恰为[4t−3,4t−2,4t−1,4t]、t1..44才记CELL_GEOMETRY_FULL；未知、内部删重、guard等逐行保留。每query/group取最早合格t，冻结同t全部实际(a,row,region)引用，实际eid去重。完整41映射的未知与非单位delta独立记录，不拒绝其它四帧合格cell，不外推startup。几何资格不证明原181编码上下文等同。

W是B冻结局部计划的完整循环置换：t加11模44、group加1模4，实际eid/row/region/位地址和失败不变，不重选。U无时间输入，m(r)=r、仅p0，错位四帧据实擦除，不floor救答。各臂独立reconcile/read/aggregate，仅同view跨三个query条件并集，不跨arm或攻击union。

CDC32新增显式read_time_local/aggregate_time入口，共享原ref读取和聚合内核。旧plan_local/H137/S/tau/D及旧输出不变；新接口无D/ANY_LOCAL_DETECTED/LOCAL_FULL_C，标记existence_test=NOT_EVALUATED、source_authenticated=false。保留原冲突擦除、整局部引用技术缺失、跨query部分证据与非投票条件并集规则。

## 一次执行、记录和预算

先排他保存attempt和256行账本，再保存全部五view三臂plans.json；之后才加载VAE、E41或打开位缓存。每op planned/started/returned/status分开，返回后发现超限仍记录returned=1及AFTER_CALL_RETURN；失败耗时计入累计，不补跑。不论何处失败、中断或断进程，再次Run all均只读。

attempt同时保存固定槽骨架。硬终止缺少blind_terminal时，CLI、entry和notebook依据已保存操作回执与独立local/final/raw metadata只读恢复480/1440槽及raw60单元；未完成、缺失或非法槽保留UNRESOLVED_RECEIPT，未完成op的已写文件只作为unconfirmed artifact，不臆造成功值或refs。恢复不打开bit cache/truth，不调用planner、聚合或后评。若attempt本身不可读且源码缓存也缺失，notebook直接报告读取失败；有可用源码的独立CLI仍能重建固定未决槽，无需重跑。

|逻辑操作|次数|
|---|---:|
|保存时间读取与计划|5|
|probe / decode / encode codec|5 / 5 / 0|
|VAE load|1|
|E41 / DCT project / keyed decode|60 / 60 / 60|
|local read / aggregate|45 / 15|
|合计|256|

固定raw身份9600位、票槽76800、local1440槽、final480槽（每臂160）；失败或未执行保持全部final槽和raw bank分母。原822时间节点/817关系只作为保存输入。DiT/D181/head/solver/K/KT/training/time_read/BCH/C/L/submit重算均0。

CPU RSS8GiB、codec300秒、model load900秒、E41每次600秒/累计7200秒、CPU接收与保存1800秒。codec使用既有单communicate管道及terminate→wait30→kill；RSS后台观测并在操作边界/codec循环检查。模型/E41的时间上界在返回或失败后观察，不能称调用中硬中断保证；超限停止后续，不改dtype、tiling、分辨率、模型、窗口或GPU门槛救跑。全wall和RAM/VRAM另记。

同一RSS监视从第一次plan读取前持续到完整盲终态保存后，包括异常退出。CPU时间包含末端blind_terminal保存及监视关闭；之后写resources完成回执，再读truth。若保存后才观察到超限，明确记录AFTER_BLIND_TERMINAL_SAVE，后评证据不完整，核心op已有returned计数不抹除。末端资源回执及后评/展示I/O另记，不以事后计时承诺保存中硬中断。

输出包括plans.json、raw_bank.json、bank中的原DCT缓存、local、final、state/counts/resources、blind_terminal.json和evaluation.json。失败后的缺失local/final以metadata终结，不虚增read/aggregate物理调用。只有全部盲槽终态保存后才打开原 `receiver/private/attack_truth.json` 及同源固定writer消息A6D39C5E。

## 后评上限

逐view/arm/slot分别保存字面值正确性、四帧精确cell、group归属、实际支持、错误归属、结构擦除、冲突、技术未读及unknown。selected refs与supporting refs分列；任何支持引用的错误归属都不能被另一正确引用抵消。完整41上下文单列，不能混为四帧资格的一票硬门。

B对W/U分别报告共同输出BER、错→对/对→错、新增与丢失正确支持。覆盖增加不是BER改善，W定义性错组本身不证明收益；字面同好而标签不同不称解码改善。B反例不以净增抵消，完整运行全拒/全擦除不通过，技术或真值不足保留EVIDENCE_INCOMPLETE；完整32位须四组均有实际支持且逐位正确。

当前是已见c01的same-chain development；旧native181/181超精确目标、4000迭代最优性未证、画质未证和所有历史失败保持。没有同媒体QIM前后身份配对，不声称无损或相对旧M3零损失。四组8bit在组内重复，不鉴别同组cell/逐帧身份/任意长位置；Ø、总体FPR、物理来源、跨段/长视频、速度变化/插帧仍开放。
