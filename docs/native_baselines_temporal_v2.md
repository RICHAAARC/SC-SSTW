# WAM / PixelSeal 时间域开发批次

本入口在四个已经参与研究的 m 来源上测原生重复载荷存留。它是开发实验，尚未执行真实模型或产生性能结果；不能称独立测试、泛化或完整论文 baseline。WAM 原生32位、中央四分之一写入，PixelSeal 原生256位、全幅写入；两者容量、冗余和支持不同。所有时间条件均以完整空间画幅接收。

用户在 `notebooks/native_baselines_temporal_v2.ipynb` 选择 GPU 后 Run all。首 cell 只挂载 Drive，完整入口自动完成依赖准备、单次写入、两代媒体、实际盲读取、保存与后评价。无需下载中间结果再回传。结果固定保留于：

`/content/drive/MyDrive/Video-WM_Result/baselines/native-wam-pixelseal-temporal-dev-v2/attempt01`

失败和中断也消费此 attempt；再次 Run all 只读，不重装、不补调用或补评价。不要同时运行两个实例。该目录与所有失败包不参与常规清理；源码、权重和临时解码数据放 `/content/video-wm-native-baselines-temporal-v2`。原 v1 notebook、配置、结果目录保留。

输入沿用 `Video-WM/B-Rank3R-New-Source-Confirmation-V1/fixed-four-five-v1/m4_inputs/b-rank3r-m01..m04/OFF/precodec.rgb24` 及旁边的 JSON sidecar，uint8 `[181,320,512,3]`、8fps。读取实际字节/几何/时基，不能以 identity BASE 替代。代码与输入定位记录不设摘要、Git 或精确依赖版本门禁。

## 固定条件与媒体

零基源帧0..180；每源 PCG64(20261105+100×来源序号) 一次排列，D18/D54 为其前18/54的集合。所有方法、arm 和通道共享映射；实际映射在任何模型/codec前写入 `edit_truth.json`。

| 条件 | 编辑 | 收到帧数 |
|---|---|---:|
| IDENTITY | 全部原顺序 |181|
| TRIM_S26_L129..TRIM_S29_L129 | 起点26、27、28、29分别取129帧 |129各|
| DROP18 / DROP54 | 删除嵌套D18/D54，余帧顺序不变 |163 /127|
| BURST18 / BURST54 | 删除81..98 /63..116 |163 /127|
| REPEAT_FRAME18 | D18每帧后立即重复一次 |199|
| REPLAY_BLOCK18 | 0..98接81..98接99..180 |199|

每源一份共享OFF及两种WM各做一次完整G1 H264 CRF18/yuv420p/8fps。G1_EDIT_RGB在这次实际读回RGB上取映射后直接detect；G2_EDIT_H264将同样的编辑RGB再次编码、probe、decode后实际detect。两个通道均保留IDENTITY；G2比较用同代无编辑对照。G2−G1包含压缩与编辑上下文交互，不能称纯时间错位。OFF的各G2媒体只建一次，两个方法共享；任何裁后重嵌、重排FULL logits代替detect、重复帧去重均不在此入口。

WAM/PixelSeal适配器及模型参数原样复用 v1。PixelSeal实际L帧的logits等权均值，尾批全保留；WAM空mask记32位UNKNOWN，无视频投票。两者不输出源时间位置；时间恢复、GAP/REPEAT/REPLAY预测、同一残片身份—时间联合正确性均为N/A。其他未接入方法单列NOT_IN_BATCH/ADAPTATION_UNVERIFIED。

## 预算与输出

| 固定计划 | 数量 |
|---|---:|
| 模型load |2|
| WAM embed / detector forward |724 /26800|
| PixelSeal video embed / detect |4 /176|
| PixelSeal embedder / detector forward |4 /960|
| PixelSeal写入关键帧 /检测帧 |92 /26800|
| encode /probe /decode |各144（12 G1＋132 G2）|
| 读取任务 /帧行 |352 /53600|
| WAM frame-bit槽 |857600|
| PixelSeal frame-bit槽 /video-message槽 |6860800 /176（45056位）|
| precodec /G1 IDENTITY /G2同条件质量对 |8 /8 /88|

WM/OFF各占对应位分母的一半。`plan.json`、`events.jsonl`、`ledger.json`保留每次API调用的开始、返回、失败及未开始状态；forward由适配器实际hook计数。计划数与实际返回数分列。崩溃后的STARTED在只读回执中标INTERRUPTED_CONSUMED，不覆盖原证据。正式输入与保存输出可能较大；原始 WAM masks 逐帧保留，不额外调用模型获取统计。

`blind/` 保存实际帧位、WAM mask与选区logit统计、PixelSeal完整帧logits和原生视频位。`blind_terminated.json`先落盘；随后才由评价层读取消息和编辑映射。编辑器能见到映射，模型detect及其盲postprocess只收到RGB，不见消息、seed、roster或编辑参数。

`frame_results.csv`和`source_condition_channel_results.csv`保留每source/method/arm/condition/channel的全部计划收到位、correct/error/unknown、可读BER、coverage及完整消息状态；`video_results.csv`保存PixelSeal视频级对应结果。`source_retention.csv`另存181源帧的保留、NOT_RECEIVED和重复收到数量，不能把删除帧记为已恢复或混入接收器UNKNOWN。`evaluation.json`还包含同代IDENTITY比较、等源等条件汇总、未定义BER单元数量、全部读错/失败记录和N/A接口。OFF字面匹配不是FPR或认证。没有科学PASS阈值。

## 源码入口

纯协议在 `main/baselines/native_paper_temporal_v2.py`；runtime负责实际编排与独立后评；CLI和薄notebook共用入口。可编辑源码副本无需 `.git`：

```bash
python experiments/wan_state_clock/native_baselines_temporal_v2_run.py \
  --config experiments/wan_state_clock/configs/native_baselines_temporal_v2.json \
  --scratch /content/video-wm-native-baselines-temporal-v2/work \
  --vendor /content/video-wm-native-baselines-temporal-v2/vendor
```

该命令是真实用户执行入口。仓库内人工CPU检查使用 `tests/test_native_baselines_temporal_v2.py`，只验证接线、映射、预算、盲边界和失败处理，不能证明真实WAM/PixelSeal或codec跑通。
