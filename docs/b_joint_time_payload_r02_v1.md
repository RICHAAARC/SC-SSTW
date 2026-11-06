# r02 单新源联合时间与载荷

采用规格：项目根 `diagnostics/relative-time-payload-next-priority-review-20261106/FINAL_METHOD_PLAN.md`。
本入口复用已运行的时间链与 DCT2/CDC32 链，将新海浪内容 `joint-relative-r02`（seed 2026110602）从生成串接到联合盲终态、两项后评和展示。新内容检验迁移，未改写 writer、Joint-A、F42、同630硬位条件严格最小或 CDC32 语义。

用户入口是 `notebooks/b_joint_time_payload_r02_v1.ipynb`。第一代码 cell 仅挂载 Drive；选择 GPU 后 Run all，期间无需搬文件或切设备。builder 为 `scripts/build_b_joint_time_payload_r02_v1_notebook.py`。CLI 为 `python -m experiments.wan_state_clock.b_joint_time_payload_r02_v1_run`，支持 `--config`、`--output`、`--scratch`。源码定位可编辑，无摘要、固定版本或 Git 状态门。

输出固定默认 `/content/drive/MyDrive/Video-WM/B-Joint-Time-Payload-R02-V1/fresh-r02-v1/attempt01`，包含 `time_chain/`、`payload/`。源码、权重与大缓存置于 `/content/video-wm-joint-time-payload-r02`。不清理或改写旧结果、失败包或 `Video-WM_Result`。

执行顺序：原采样中 DCT2 → 同 host 捕获 → 原 CPU solver → 原生回放 → NEW/BASE 五视图与盲时间 → 所有身份 refs 保存 → E41/实际 keyed decode/CDC32 盲输出 → 联合盲终态 → 时间 post 一次 → 载荷 post 一次。旧 r01 入口默认行为保留；r02 显式传 source、validator 与 defer_post。时间成绩不决定载荷是否执行，语义歧义照常传递。真正上游执行/依赖失败则不启动依赖它的物理读取。

根 attempt 用独占记录在执行前登记两个操作账和全部固定槽：78＋551＝629 外层操作；时间1644节点、1634边、1,035,720硬位；载荷28,800 raw位、230,400 vote uses、153,600物理系数、2400局部/800最终槽，每策略160。内部 VAE load 3、E41 120、project 120、keyed decode 180；codec encode/probe/RGB decode 为12/22/22，Y decode10单列。保存、两次 post 和五组质量另计；旧1522引用不是新预算。

保留 solver 4000迭代/7200秒/16GiB live-array估算；时间接收8GiB RSS、单次hard600秒/累计6300秒；codec单次300秒。载荷子进程含联合闭合和两次post，共12600秒、进程树RSS8GiB，CUDA峰值记录。生成/capture/native replay 的整个阶段限额未认证，其外层进程树RSS与wall单独记录。原 `finite_candidate_saved/native_handoff_eligible` 保留，`NUMERICALLY_UNRESOLVED` 不是新门槛。

每个 time stage 和 payload worker 以独立进程组运行，正常结束和中断均清理所属子进程。普通技术错误先闭合盲槽再做可用后评。外部SIGTERM/SIGINT或KeyboardInterrupt另作消费终止：r02监督层先记录中断意图，再抛出异常；旧worker捕获异常后也不能抹掉该意图。时间阶段中断后不启动payload或post，仅闭合固定盲槽，未开始post记NOT_STARTED；载荷中断不进入post，第一项post中断也不启动第二项。payload watchdog耗尽时同样不越过预算补跑。无法捕获的SIGKILL或强制终止宿主可能使最终闭合未写完，最初根attempt仍保存全分母；复开只读，不补写科学结果或重做post。

根 `joint_blind_terminal.json` 在两阶段盲输出保存之后写入。`time_terminal_slots.json` 对未确认RETURNED的文件保留未知；r02时间post不评分此类文件。payload复用逐位已返回receipt闭合。`posts.json` 与 `payload/extra_counts.json` 分别记录联合后评和旧worker中被延后的post，二者不能重复计数。失败post也消费，不重做。

B冻结NEW时间，U收到时钟四phase，W固定+1且越界/未知→None，K固定watermark-wrong/key01，P_BASE借NEW映射但读取BASE自己的eid与缓存。P_BASE有相同身份载荷，并非clean OFF。分别报告共同位BER、支持增失、时间错误、错误归属、擦除、错钥及质量。单源和相关五视图不证明总体泛化、FPR或科学PASS；旧W/K局限、原native目标未达与solver最优性未证保留。

本地只运行人工 CPU fixture、AST/CLI及只读检查；未运行真实模型、保存实验媒体、GPU、codec、Colab或Drive。工程接受与新科学结果分开。
