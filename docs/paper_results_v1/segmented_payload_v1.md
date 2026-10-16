# 固定四源真实分段载荷用途验证

入口为 `notebooks/paper_results_v1_segmented_payload_v1_colab.ipynb`，由同名 builder 构建普通源码 ZIP。用户在 GPU Colab 直接 Run all；固定输出 `Paper-Results-V1-Segmented-Payload-V1/resumable-segmented-payload-v1`。本交付的实际模型、媒体和真实读取尚未由 agent 执行。

## 实际方法变化

生成中真实替换原恒定载荷 target，保留原50步 native scheduler、conditional-clean FFT-real 局部梯度、后25步写入及CFG5。固定32bit `A6D39C5E` 分成 `[A6,D3,9C,5E]`，有效Wan时间1..44每11层一段；四个原通道在该时间层重复同一段8bit。0/45保护层分别写S0/S3，不读作净位。

原mask44160、绝对幅度0.5、reference_eta200→eta5520不变；每净位11层×4通道×30频率=1320票，等于原44层×1通道×30。有效总票42240。新四条trajectory沿用既有四prompt/seed，未复用旧terminal，未增加旧载荷或OFF生成。梯度和损失逐步保存；相同幅度不保证相同输出扰动。

接收仍用原RAW/CENTERED U与visible-span；UNALIGNED从收到位置0按顺序对应。局部Wan时间r映射到 `anchor//4+r`，只有其四个名义源位置全部在估计路径中有实际收到索引支持才投票。填补仅作VAE上下文。完全无单元的段保留MISSING，不能用邻段补值。原FFT ortho real、strict >0、Counter先遇平票顺序保持。该名义映射不保证VAE感受野严格局部；真实串扰/边界混合由本次结果揭示。

## 固定读取与解释

| 类型 | 固定行 | 计划段 | 计划位 | 真值结构缺失位 |
|---|---:|---:|---:|---:|
| 两视图×三盲臂×四源 | 24 | 96 | 768 | 192 |
| 两视图correct-time oracle×四源 | 8 | 32 | 256 | 64 |
| clean correct-time oracle×四源 | 4 | 16 | 128 | 0 |
| 合计 | 36 | 144 | 1152 | 256 |

CROP_MID取 `[37:126]`，正确映射完整单元为S0/S1/S2/S3=2/11/9/0；DELETE_S1删45..88，正确映射为11/0/11/11。攻击在同一candidate_POST RGB上取帧，不第二次编码。盲计划在clean答案之前全部提交，包括缺失或失败计划。随后按源读取clean；四字节完整精确才执行该源其余8行，否则保留 `NOT_RUN_SOURCE_UNREADABLE`。该前提不是科学PASS，不能改写盲计划，也不能缩减固定分母。

评价单列真实结构缺段、可见但未读位、已读错误位、错误归属、段内容与顺序。错误路径可能对完全已删段误报内容，这不算恢复丢失信息。BER只在真实存在且读到的位上计算，另报完整计划分母与未读数。oracle是独立可读性诊断，不给盲臂填失败或选路。

summary按CLEAN、攻击ORACLE、CENTERED_U、RAW_U、UNALIGNED分别汇总固定行/段/位、读取/未读、结构缺失、已读错误和错误归属；未读和NOT_RUN不当零误码，完整可见位分母BER仅在该组全部可见位均已读取时提供。

## 一次物理尝试和恢复

每unit在调用前持久记录STARTED；失败、中断、完成未知和产物损坏均消耗该unit。下一次Run all仅复用可读完成产物或继续从未开始的unit，不自动补做已开始的GPU/codec/selector/reader。实际shape、dtype、有限性、路径与支持合法性仍检查；源码、版本、摘要及Git状态不作门禁。

| 物理单元 | 上限 |
|---|---:|
| 新生成 | 4条，200native step / 400DiT / 100局部FFT |
| native Wan decode | 4 |
| framewise encode / decode | 各8clip / 1448帧 / 184batch8 |
| CPU clock write | 4 |
| candidate codec encode/probe/readback | 各4 |
| U selector | 16 |
| Wan encode / 分段reader | 各最多36 |
| generation prepare/load | 4，每源1 |
| native Wan / reader Wan / framewise load | 各2，正常1+恢复1 |

恢复load只为尚未开始的原预算单元，不重做失败单元；每阶段每次调用至多尝试一次load。用完则保留预算耗尽和下游缺失。相同源、相同接收视图、完整相同received-index map才共享encode；同R不同map不共享。阶段分进程释放模型，依赖按可达剩余单元准备。下载失败可读同Drive缓存ZIP。

24盲槽必须全部有持久计划或不可重试终态后才读任何clean。暂缺Y而上游仍可达则等待；已耗尝试/预算且无可读Y时，先冻结真值无关MISSING计划，不调用selector。旧状态若已越过消息读取边界，未存盲槽直接冻结MISSING，不能事后补选择。Notebook中断终止同一进程组的CLI和worker。已保存投票与JSON按同一纯CPU规约复核，不一致保留MISSING，绝不触发新的物理读取。

质量参照仅为本轮新分段NATIVE_PRE。candidate_PRE差反映时钟残差，candidate_POST与N_PRE差包含clock及codec，不叫纯clock效果。输出RMSE/MAE/PSNR和逐帧MSE；未另生成OFF。

## 工程验证范围

定向CPU fixture覆盖真实target四段写入与原mask/幅度、固定映射/完全缺段零票、native循环目标传递、一次attempt/中断/附件丢失不重做、load上界、exact-map共享、先存盲计划、clean失败保留8行及无模型恢复。交付Notebook代码解析、普通ZIP一致和无.git CLI初始化均检查。fixture不是实际Wan或MP4证据；GPU可读性、真实跨Colab恢复、用途收益及科学结论待用户运行。没有阈值扫描、候选替换或自动GPU执行。
