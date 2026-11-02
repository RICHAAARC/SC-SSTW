# 固定 n01 原生 Y 整链用户运行包

当前采用依据为根目录 diagnostics/startup-new-source-method-review-20261102/FINAL_METHOD_PLAN.md 与 SOL_FINAL_CONFIRMATION.md。源固定 b-rank3-n01；旧 RANK3/RANK3R 已用，只称限定当前机制留出，不称全项目未见/IID。无运行时换源或默认消息。

入口 `notebooks/b_native_y_source_confirmation_v1.ipynb`；builder 为 `scripts/build_b_native_y_source_confirmation_v1_notebook.py`。首cell仅Drive mount，设置已填源及独立输出，模型可配置，普通分支/源码目录可编辑。Notebook bootstrap仅普通下载/安全解压及调用已跑c02原entry/setup/worker路径，不作摘要、版本、Git状态验证。首次整包预约之后，A/B/time/payload的失败和未进入槽均消费；相同OUTPUT再Run all仅元数据读取。

链路：原A8457/B57（FP32特征仍在A/features，原CPU FP64 solver4000/7200s/16GiB，实际输出文件系统容量检查）→两首代BASE/QIM RGB媒体prep14→十view nativeY盲82→仅QIM五view原payload256。合计8866异质逻辑预算，不等于模型调用数；codec50=12encode/19probe/19decode。model_load3、D1811内baseline_head46、head_replay46、E4160，新DiT0。公共码表在唯一codebook项生成一次，保存五组并flatten供F/C共用，无旧c02 NPZ。

五view FULL181/CROP101(38..138)/DELETE1删44/REPEAT1重复44/DELETE3删43..45，两个arm共1644nodes/1634edges/8220lane/1035720replica位槽；payload raw9600/vote76800/local1440/final480，每B/W/U160，BASEpayload0。time8GiB/6300s/view600s/codec300；payload原load900/E41600每次/7200总/CPU1800/RSS8GiB。

时间读端只收到像素、公开码表与原H/C/L/submit语义。固定mapping仅由fixture构造保存至private，不反馈读端；所有时间/载荷盲终态后才统一time_post1/payload_post1/summary1。消息只读原SOURCE/protocol.json的config.writer.bits，实际非法/不可读失败不回退。全部B/W/U plans由原payload实现先冻结，再60次物理E41。

post按完整分母分别保存H/final位置、精确边与真实删重事件；保留未决，非启动覆盖单列，不加新覆盖阈值。BASE任意SUPPORTED_RELATION_SET即有限错误接受（即使exact0）；同时报exact、NEW L、局部BCH命中、C DEFINED与实际j评分。BASE是有身份无时间，不是全无水印。B各view32/32字面正确+实际refs归属+物理支持正确为目标，错归属/无支持/字面错优先保留反例。共同BER、输出增减和正确支持/归属修正分开；无新RGB/Y对照，不能声称Y收益。

profile未知、native精确目标失败与solver最优性未证均原样报告。仅一个当前机制来源案例，不证一般FPR、来源认证、跨消息或长视频。CPU人工检查和交付不等于真实GPU结果；由用户选择GPU后Run all，agent不运行科学cell。直接CLI亦可通过配置文件使用，但消费后不得补跑。

工程验证使用人工CPU/stub，仅覆盖新接线、预算、盲顺序、失败消费、控制判据、源码入口和恢复。实际notebook科学cell未执行；用户Colab运行后再审查保存证据。
