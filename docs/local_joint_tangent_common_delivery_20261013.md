# B 纯切向共同方向最小验证

默认沿用既有步长 0.10，减为单次最小验证；原 BASE 半径上限保持 1。CPU 准备与定向检查已完成，GPU 由用户运行，GitHub 发布由协调者完成。

本轮检查同一保存 CONT_FINAL 上的纯切向共同方向，是否实际提高原六个负位的连续余量，并记录其余 26 位、状态和画质的变化。原 BASE 为既有 JOINT BASE，累计半径上限保持 1。不得由位数不变直接推断六位没有改善；也不得把一阶预测视为位覆盖、状态或画质保护。

- 输入：Six-Bit-Final/resumable-six-bit-final-v1/run/inputs/fixed_inputs.pt 中的 base/final，以及 diagnostic_directions.pt 中的 tangent_common。方向不重算、不归一化、不投影。
- 候选真实运算：`candidate = (START.double() + step_size * saved_direction).to(torch.float32)`。方向只在 START 点切向；直线步会改变 BASE 半径。实际候选超球明确报错，不自动缩步或投影。
- 固定三视图：BASE、START_FINAL、TANGENT。名义 1 次加载、3 次普通解码（138 分块），0 新 VJP；无 encode/DiT/native/scheduler/codec。
- 每视图保存 88 窗口、1408 chips、32 余量＋22 状态相关＋G；共 264/4224/165，所有缺失与失败保留。
- 主比较 TANGENT−START_FINAL：六位连续余量变化、全 32 位得失、全部状态、相对 BASE 和 START 的全帧 float 质量。每一参考各保存 181 帧与 180 时间差、实际 RMSE/max_abs 峰值及邻域、对应裁剪与固定帧。BASE-relative 画质只表示从既有 JOINT BASE 的累计变化，不是相对 OFF 的总水印失真。
- 可选读取六份同点梯度，CPU 计算 `g_i · (candidate.double()-START.double())` 以及计划增量预测；缺失/无效预测如实记录，不阻断候选、不重做 VJP。其他 26 位、状态、画质导数保持未测。

直接运行入口：`notebooks/local_joint_tangent_common_v1_colab.ipynb`；普通伴随源码 ZIP 与 builder 同步，源码 ref/路径可编辑，不设 B64、哈希、manifest、干净 Git 或精确版本门禁。

稳定输出：`MyDrive/Video-WM/Local-Joint-State-Payload-V1-Tangent-Common/resumable-tangent-common-v1/`。固定输入、方向、候选终端与预测在模型加载前提交；逐视图 float RGB、raw q、分数和质量完成即保存。每 60 秒保存最后已提交状态及 progress，保留上个可读回退。它不保存进行中的 decoder，也不保证最多损失 60 秒工作；中断的解码整体重做，并累计所有实际调用。完整 RGB 可用于 CPU 报告和图像修复，完成后 Run all 零模型调用。任一视图读出或质量失败不阻断其他可完成视图。

验证：定向 CPU/stub 测试覆盖原球几何、真实 FP32 增量、固定三视图、可选预测缺失、单视图失败隔离、报告/画质恢复、解码中断及 RGB 提交前中断、timer 快照、损坏 JSON 回退和物理调用累计；notebook 按 cell 顺序 stub 运行、重复 Run all、子进程退出清理、普通 ZIP 解包和静态语法。真实 Colab/GPU 由用户运行，本地测试不代表方法或科学 PASS。
