# 有限开发集强度校准：固定最小协议

状态：本地实现与 CPU/Fake 验证；尚未发布，未执行模型/GPU/Colab/Drive。
清单：`configs/velocity_calibration.json`，不根据生成结果更换内容或种子。

| 阶段/内容 | 固定 prompt 原文 | seed |
|---|---|---|
| dev p0 | locked camera, a single small red sailboat gliding slowly across a calm lake, gentle ripples, stable daylight, no people, no cuts | 20260916、20260917 |
| dev p1 | locked camera, a single yellow tram moving slowly along a straight tree-lined street, stable overcast daylight, no people, no cuts | 20260916、20260917 |
| holdout p0 | locked camera, a single white swan swimming slowly across a quiet pond, soft morning daylight, no people, no cuts | 20261001 |
| holdout p1 | locked camera, a single blue cable car moving slowly across a mountain valley, stable afternoon daylight, no people, no cuts | 20261002 |

**1. 开发终态全表。** 4 cases × rho `[0.1,0.3,1.0]` × A/B =24强度行；
另保留每 case ZERO_A/B。ZERO_A 是 OFF，同次 terminal A/B 定共享 R。
每 case/message 在 a=0 求一次 q 并冻结；各预定强度独立恢复相同前缀及完整
UniPC 历史，`epsilon=rho*0.999*epsilon_cap`。不再优化 q、不逐视频选强度、
不 clip/自适应缩小/重试。44/45/46 控制、47..49 完整尾程、5280系数、
normalized 坐标、原 codebook/loss、BF16 CFG/FP32 控制与嵌套 checkpoint 不变。
保存实际有效 FP32 控制、U/D、loss、正确/竞争 margin、增益曲线、资源与全部失败。
相邻强度的增益斜率/反转仅供观察，不引入新的饱和阈值或线性外推。

**2. 预定候选筛选。** 每 rho 固定8行，全部完整、有限且 U/D预算与response检查合法
才具有完整配置资格。按最差正确-竞争 margin 降序、平均 margin 降序、rho升序取前2。
负 margin 可作为低证据候选；`margin>0` 也仅为终态筛选信息，不能替代读出。
规则筛掉记 `NOT_SELECTED`，缺失/失败保留原状态和分母；不隐去不利强度。

**3. 开发 MP4 与统一参数。** 每候选覆盖全部4 case×A/B。复用已保存终态，
OFF/terminal A/B 媒体在候选间共享，不重生成尾程。采用原 CRF18/yuv420p MP4，
四 origin VAE重编码和原 state_clock 全五模式完整盲搜；truth仅在read返回后报告。
候选合格要求每case OFF/terminal完整，terminal及FLOW A/B五模式均唯一正确；
FLOW相对同次OFF的 saved RGB MSE、residual temporal MSE 均≤1.5×对应terminal。
0/0质量比定义为0，正数/0不合格；1.5为既有工程容忍度，并非验证过的感知阈值。
合格候选按全部8行两指标的最差质量比升序、平均比升序、rho升序选一个统一rho。
任何缺失/写入失败均不因部分文件可读而升格完整。无合格则 `NO_SELECTION`。

**4. 独立验证。** 统一rho先于holdout执行写入selection.json；两新内容/新seed仅评估
该rho。每case仍按同一算法独立求一次A/B零点方向，不重新选rho。保留OFF/terminalAB/
FLOWAB及全部4个message行；使用同一预算/五模式/质量判据报告loss、margin、质量比，
不调用选择函数、不自动宣称科学PASS。无开发选择则 `NOT_RUN_NO_SELECTION`，不是执行失败。
排名不是FPR；有限开发集与2case holdout不建立广泛泛化结论。

| 最大调用（无失败且选2候选） | dev4case | holdout2case | 合计 |
|---|---:|---:|---:|
| 普通Transformer | 736 | 272 | 1008 |
| backward / outer replay上限 | 8 / 80 | 4 / 40 | 12 / 120 |
| VAE decode / MP4 save | 28 / 28 | 10 / 10 | 38 / 38 |
| VAE encode | 112 | 40 | 152 |

三类block计数各最多 outer次数×实际block数，单位分开。dev实际媒体为每case
`3+2*C` decode/save与`4*(3+2*C)` encode；C=0则无媒体/holdout。每case独立子进程
释放模型和计算图，终态与媒体分阶段；日志/配置/固定清单/所有result均保留。

入口：`python -m experiments.wan_state_clock.calibration_run --manifest
experiments/wan_state_clock/configs/velocity_calibration.json --output <new-dir> --stage all`。
可独立运行develop、development-media、holdout以复用终态；不自动重试失败case。

Notebook：`notebooks/velocity_calibration_colab.ipynb` 首格为精确独立Drive mount。
当前SOURCE_COMMIT=None，明确待发布，不假绑定旧源码、不嵌PAYLOAD。
获得新阶段发布授权后，先发布含新入口/清单的源码，再用
`python scripts/build_velocity_calibration_notebook.py --source-commit <published-full-SHA>`
生成绑定版，fresh fetch校验后单独发布Notebook交付提交。已有方向Notebook完全不变。
