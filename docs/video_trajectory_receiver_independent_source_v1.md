# 预固定新源小复核（Milestone1B）

本入口实现已采纳的单次新源规则，当前为工程候选，真实结果须由用户手动 Run-all 后完整审计。prompt 固定为：

> locked camera, a small blue toy car slowly rolling left to right across a wooden tabletop, steady soft daylight, no people, no cuts

seed 固定 2026100601；唯一 condition 为 M05_FRAMEWISE_SYNC、projection target 0.5。新源在规则冻结后生成，不依结果重抽 seed 或换源。这是一个预固定新源的小复核，不能预称独立泛化成功。

| 固定阶段 | 计划调用与支持 |
|---|---|
| 原 G 生成协议 | PAYLOAD_MULTI；181×320×512、8 fps；一次轨迹、50 native steps、conditional/unconditional 各50、local control/payload gradient 各25、native decode 一次 |
| M05 准备 | 一次 source RGB 读取、一次 framewise VAE encode/write/decode、一次 raster 保存、一次完整 MP4 save/probe/readback |
| 四窗接收输入 | 本次 M05 FULL181 received RGB 读取一次，CPU clone [2:179]、[3:180]、[38:127]、[39:128]；切片后无 codec |
| 同版接收器 | 4 framewise encodes、532帧、70个 batch8 批次；8 sync reads、392 candidates、9456 local rows |
| 固定逻辑分母 | 16 payload reads、506880 votes、16896 time-bit、512 final-bit；R44/R44/R22/R22 |
| 实际物理接收 | Wan load 一次；完整可读时4…12 encodes、8…16 key reads；同观察同phase共享encode，p0 aligned引用同key baseline |

生成子进程直接复用成功 G 的 source_worker 与原轨迹语义，退出后才启动独立 M05 准备子进程。准备进程只组合原 M05 writer、framewise VAE 和媒体原语；不调用会增加 P0/P1/M1/quality 的 G media_worker。准备进程退出后才加载接收模型。先保存新源 RGB 身份，再记录 M05 writer receipt、一次 libx264 CRF18/yuv420p/8 fps 完整 MP4 及回读 RGB 身份；无旧 G RGB 依赖或 codec fallback。

Writer 合法读取 prompt/key/message，四窗准备合法读取 crop start。接收 score、estimate、physical plan 和 phase correction 只用收到 RGB、公开参数及本 key 的新 blind estimate，不消费 writer 证据、消息或 crop truth。沿用 Stage2 全候选 complete/finite/unique 判定、offset%4、prepend 首帧后等长裁尾及详细票；每 key 独立估计。sync seal、physical plan seal、payload seal 依次完成，评价 POSTHOC 配置仅在最后一次 seal 后读取。没有这四窗的历史对照查询。

在任何源准备、codec、接收失败或可捕获中断下，16个逻辑槽及失败理由仍保存；无唯一估计不补真值或 phase0。子进程终止、组清理与 receipt 持久化沿用已成功包装；不重试换源，也不新增臂、质量测量、阈值或扫描。按实际 attempted/completed 另报物理调用；别名不是独立证据。

Notebook 保持5个代码单元与精确首 mount；SOURCE_SHA=None 在创建输出目录前拒绝未发布草稿。复用成功8包 pins，另固定同 G 保存环境的 sentencepiece 0.2.2 与 ftfy 6.3.1；依赖探针覆盖 WanPipeline、双 VAE 类及生成依赖。pip check 非零单独记录，不等于模型导入或 runner 失败，不自动扩张环境修复。A1 仅运行 CPU/fake/static 验证，未执行真实模型、GPU、媒体或 Drive 操作。

1A 已接受的同开发源四窗结果不作为本新源结果。精确 offset、phase、最终 bit 与全部 time/channel 余量分开报告；全 time-bit 正不是硬 PASS，K1 是描述对照，sync_accepted=False。未覆盖动态路径、存在/拒绝、FPR、唯一 phase 因果或广泛泛化。

入口：experiments.wan_state_clock.video_trajectory_receiver_independent_source_v1_run  
Builder：scripts/build_video_trajectory_receiver_independent_source_notebook.py
