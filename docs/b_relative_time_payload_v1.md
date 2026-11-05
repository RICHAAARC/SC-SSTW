# r01 relative 时间到 DCT2＋CDC32

本入口读取已有 r01 NEW/BASE 各五份收到媒体。固定时间输出先经 allowlist 保存，规划完整物理 bank 和所有引用后，才执行实际 E41 和读位。结果是一个已见来源上的有限策略比较，不是科学 PASS，也不隔离纯 phase/context 因果。

用户 Colab 入口：`notebooks/b_relative_time_payload_v1.ipynb`；同步 builder：`scripts/build_b_relative_time_payload_v1_notebook.py`。第一 cell 只有 Drive mount。源码默认从可用开发分支读取，也支持已有可编辑目录；没有 Git、源码摘要或版本相等门禁。完整 CLI：

```bash
python experiments/wan_state_clock/b_relative_time_payload_v1_run.py
```

CLI 的 `--config` 可定位可读输入/模型，`--output` 可定位新输出；固定媒体 roster、几何、key、控制与预算不从 config 改写。默认输入为 `/content/drive/MyDrive/Video-WM/B-Joint-Time-Relative-Consistency-V1/fresh-r01-v1/attempt01`；新输出为 `/content/drive/MyDrive/Video-WM/B-Relative-Time-Payload-CDC32-V1/fixed-r01-v1/attempt01`。

五策略分别为 B_NEW（冻结条件时间）、U_NEW（公开 received clock、四 phase）、W_NEW（固定 n+1/越界 None、首次独立规划）、K_NEW（B 几何和实际 watermark-wrong/key01）、P_BASE（NEW 时间配对到 BASE 自己的真实 eid/cache）。P_BASE 不是独立盲同步；NEW/BASE 同 key00 和原 writer 消息。W不循环改组，不保证比B更差，也不根据位结果换偏移。

主逻辑预算：10 planning、10 probe、10 RGB decode、1 VAE load、120 E41、120 canonical project、180 keyed decode、75 local、25 aggregate，合计 **551**。NEW 每投影分别 decode key00/key01，BASE 仅 key00。raw28800 位、vote使用230400、物理canonical coefficient153600；local2400/final800槽。五view相关，独立内容来源为1。终态保存、一次后评、五组增量质量单列。

VAE沿原 c01 实际收到 RGB24→41帧切窗→FP32 posterior mode→原 normalized latent，编码前后清cache。每query/group在读位前选最早 FULL cell；完整41帧context另列，不以连续context作为新的资格门。DCT2和CDC32原八票、Counter tie、冲突擦除、技术缺失、实际eid去重与条件并集保持；新增明确scope入口。

整 attempt 首先原子预留551条操作和全部raw/local/final计划槽。失败/中断消耗；再次Run-all或CLI只读，包括不重后评、不补未开始调用。已返回投影/位只在同一次执行内部共享。科学worker的外部监督包括所有规划、模型、媒体、保存和后评：12600秒墙钟、8GiB进程树RSS；按0.1秒轮询，TERM最多等待2秒后KILL整个进程组，记录实际延迟。CUDA峰值单列，无型号白名单。源码/依赖安装在worker外另计时；临时内容放`/content/video-wm-relative-payload-v1`，不清理原结果或`Video-WM_Result`。

原 c01 codec adapter 的单次 codec 300 秒实际超时仍沿用并在 backend 回执单列；没有进入旧 time receiving context，旧6300秒时间接收、1800秒CPU和7200秒E41累计限制没有启用。显式指定已有`--output`时，即使当前配置文件不可读，CLI也直接读已存结果，不重启准备或科学worker。

保存入口：

- `attempt.json`、`planned_slots.json`、`raw_planned_slots.json`、`state.json`、`counts.json`：固定分母、开始/返回/失败/未知；占位不伪造成功零值。
- `raw_terminal_slots.json`、`raw_terminal_summary.json`：初次盲闭合从已保存key输出及RETURNED操作回执归并的固定28800逐位实际状态，逐位关联物理地址、key、操作和原始bit状态；未确认文件仅保留引用，缺失/损坏回执不伪造值。原raw计划表仍是计划，再次Run-all只读已有终态，不补建或重算。
- `plans/`、`plans_frozen.json`：仅条件映射及完整冻结引用；`bank/`：实际canonical projection、独立keyed记录及缓存。
- `local/`、`final/`、`blind_terminal.json`：全部固定盲槽。全部闭合后才开原`private/attack_truth.json`和`protocol.writer.bits`。
- `post/`、`evaluation_summary.json`：逐槽/引用字面正确、四帧/组/同组错cell、真实支持、context、失败/擦除，B/U与B/W共享/不同refs分层，B/P同实际位置配对；没有O读位。
- `quality/`、`quality.json`：复用十次decode，五view NEW/BASE增量MSE、PSNR和maxabs，尺度[0,1]、data_range=1；零差异用`psnr_db=null, psnr_is_positive_infinity=true`，不输出非法JSON Infinity。
- `resources.json`、`backend_resources.json`、`extra_counts.json`：监督进程树峰值、实际墙钟/终止延迟、CUDA与分项时长。`worker.log`保留实际失败。

四组8位各在11cell内重复。字面32位正确不认证组内位置；任何错支持不被另一正确ref抵消。K只描述这一固定错钥，不存在身份检测/FPR判据。无clean OFF，只能报告时间包增量身份/画质；原native精确目标未达、solver最优性未证及旧失败保持。CPU假后端/人工数组检查是工程证据，真实GPU由用户运行。

notebook本身在任何本地/远程bootstrap获取前，先用标准库检查已消费输出并展示保存summary与记录路径；已有结果读取不依赖源码或网络。最后展示cell的只读变量分支独立闭合。首次运行才获取bootstrap，首cell仍仅挂载Drive。
