# A固定单行补读

用户已采用卡40提案并授权准备运行包；本包仅补 `distinct_confirm_20261017_01/SHORT17_HOLD_MEDIA/DISTINCT_RAW_U` 的16个未知存在位。默认CENTERED不变；原P2的196行/6272位和后验64行/2048位不改，新增独立补读记录及overlay。固定4段32位=16存在+16结构缺失，原路径误差1、区间正确的结果保留。

[直接Run all的Colab入口](https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/dev/paper-results-v1/notebooks/paper_results_v1_single_read_v1_colab.ipynb)在主会话审查并将本轮文件push到现有dev后可用。[本地notebook](../../notebooks/paper_results_v1_single_read_v1_colab.ipynb)、[直接下载伴随包](../../notebooks/paper_results_v1_single_read_v1_companion.zip)、[builder](../../experiments/paper_results_v1/build_single_read_v1_colab.py)。本轮实施者未commit/push。

## 用户执行

选择GPU运行时并直接Run all，首代码单元仅原drive.mount，不force_remount。无需手动ZIP上传。使用同一Drive的原P2目录及已有Wan VAE缓存；按已保存计划从35帧同次received.npy或POST.rgb8中映射17帧。优先存在的received.npy，否则选择已有POST；一旦打开选定容器，错误就保留失败，不换容器重试。已查normalized缓存不匹配，旧025在output9取received8，本计划取received9，两帧保存RGB精确类不同；本包不搜索/拼接/近似复用其它latent。

最多1次FP32 VAE加载、1次encode、1次原read_segments（批FFT2/新票归约各最多1次）。已有模型从缓存根目录的available、根目录或已有子目录定位；目录名和版本不是准入条件，加载前选定一个，不在失败后换模型。仅实例化AutoencoderKLWan，复用P2的posterior.mode及mean/std归一化，不调用WanPipeline。新checkpoint下载、framewise/DiT/梯度、score/U/rowmax、旧票重聚合、codec、媒体decode/probe、新源/新视图、扫描、重试均0。

固定输出：

```text
/content/drive/MyDrive/Video-WM/Paper-Results-V1-Single-Read-20261018/fixed-single-read
```

`attempt.json`在任何输入/模型工作之前独占创建并flush/fsync；state在input/load/encode/read各阶段之前保存attempt=1。先保存chosen_input，再打开选定容器；新normalized、votes和read_result先保存，盲进程结束后独立评价模块才接入消息。read端只接收public plan、key及同次RGB/normalized；不会读取旧P2 state、消息、真路径、oracle或其它臂答案。

本次一旦开始，不自动续做失败/中断后的链，即使normalized已保存而read尚未开始，也不会在Run all中续读。保存的完成、失败、部分、损坏记录都不能获得新预算。已存在科学产物或attempt时，Run all跳过源码下载、依赖准备和所有物理调用，只显示保存JSON；不要删除固定目录/尝试记录来重跑。STARTED但无完成记录表示运行中或中断/未完成，保留原状态，不猜测已成功。

回交整个固定输出目录。主结果为summary.json、row_result.json及overlay.json；盲证据为blind_plan.json、normalized.npy、votes.npz、read_result.json，attempt/state/log保留失败与尝试。完整存在位BER仅在16位全部有效保存时计算；缺失16位不算正确，新16位自己的错误数/16与可选RAW合并错误数/640分列，不预设零错。只关闭一个后验16位缺口，不成为独立确认或修复路径误差。

## 工程验证与执行缺口

本次11个定向CPU/stub检查通过；notebook结构/代码AST、无.git伴随包读取保存结果已验证。stub只用合成小数组/令牌，无真实像素、旧latent/votes、模型、FFT或Drive操作。真实模型加载、GPU编码/读取、Drive持久写入及跨Colab中断行为尚未执行，需用户在主会话发布后按上述Run all验证并回交固定目录。本包没有GPU型号、摘要、源码引用或精确依赖版本门禁。
