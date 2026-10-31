# 条件时间输出接口 V1

本地工程实现，当前输入未配置、未执行真实验证。采用任务见外层 diagnostics/conditional-time-output-implementation-20261031/ADOPTED_IMPLEMENTATION_TASK.md；方法权威见 conditional-time-interface-method-review-20261031/FINAL_REVIEW_RECOMMENDATION.md。

新增的是显式提交和后评接口。DCT2＋CDC32、原 hard 和原双副本融合 C/L 不变。旧 H 与旧精确关系另存原对象；只从旧未决位置的 SUPPORTED／DEFINED／epsilon=1e-12 唯一合法原 N 候选取 S。先去与 H 逆序的 S_H，再在冻结 T 上一次去全部逆序参与者。H 可空；固定 H 与剩余候选后做原区间整数可扩展性复核，传播不新增节点。收到相邻端点明确才新增精确边；不跨未决、不补起点/步长、不重算 L、无 A/B/AND/次峰/带权 G。

LOCAL_CONFLICT 保持局部未决，相关端点和边不得新增；全部旧边（包括无精确端点的边）直接核对delta区间、原N、kind和公开相邻节点/path的冲突状态。旧 IMPLEMENTATION_UNRESOLVED／无可行路径／不支持资格、格式非法或可见旧输出矛盾阻止该 view 新增。输入或结构问题保留 old_hard/original_L 和错误，不修补历史输出。S_H 在合法精确 N/H 下通常已经被原区间隐含排除，仍按采用定义显式实现并单测其隔离性。

有效旧节点在检查L/关系的新增资格之前独立放入final_nodes；因此L字段或旧边语义错误时，该view不新增，但有效旧H和原旧关系仍在最终输出中。真正不可读/非法节点保留原始old_hard及old_node_errors；对应可定位列表槽标为INVALID_OLD_NODE，不推造接收坐标或输出索引，也不删除其它可读有效旧节点。节点容器本身不可读时不猜节点列表。后评仍按错误状态记录证据不完整，不因旧H保留而把失败view改判成功。

## 本地入口

独立 CLI，不依赖 notebook、scripts、Git 或源码摘要：

```bash
python -B -m experiments.wan_state_clock.b_conditional_time_output_v1_run --config /absolute/input-config.json
# 或从任意 cwd 直接调用完整脚本路径；--output 可覆盖配置输出位置。
```

默认配置 experiments/wan_state_clock/configs/b_conditional_time_output_v1.json 中 entry、output、输入路径均为 null。直接调用返回 UNCONFIGURED、零调用，不创建 attempt。没有确认未用同 writer 正例；禁止填 e02 换剪法代替新源。主方只读 ASSET_INVENTORY.md 的结果不被冒充真实输入就绪。未来一次真实执行需另获当前范围授权。

仅选一个 entry：semantic、raw、media。source_note 是可选定位文字，不是身份门。output 为唯一固定输出目录；第二次及后续同目录调用在读取新配置/科学输入之前返回 READ_ONLY_EXISTING_ATTEMPT。失败、中断、未调用行全保留，不转其它入口补领；非空无 attempt 的目录返回 OUTPUT_CONFLICT。实际输入目录和输出不得重合/包含，保护原材料。缺真实路径不预约；实际语义/格式读取发生在预约后的固定调用内。

|entry|配置输入|核心项数|
|---|---|---:|
|semantic|views 五个键各指向完整 JSON，或 {hard:旧blind.json路径,L:旧candidates.json路径}|5|
|raw|views 五个键各指向 received raw.npz|26|
|media|media 指向181帧512×320/8fps原正例MP4|43，codec17|

views 固定键 FULL、CROP、DELETE1、REPEAT1、DELETE3，长度181/101/180/182/178。原 MP4 编辑固定 FULL、[38,139)、删44、44后再放44、删43/44/45，第二次 H264 使用现有 CRF18/yuv420p 路径。raw26=读取5+码表1+hard/C/L/submit各5；media43=源probe/decode各1+码表1+每view encode/probe/decode/time_read/hard/C/L/submit各1。semantic5 每view包含所需读取、提交及保存，文件IO次数不是核心项数；码表/C/L重算0。

semantic 完整对象格式为 {"hard": 原hard对象, "L": 原candidates对象}；也可指定两个原JSON路径。hard 必须有 path_status、qualification、完整 nodes（received_frame/status/interval/output_index）和全部 relations；每条旧关系，无论是否已有精确端点，都必须含 received_left、delta、kind、status、output_endpoints、output_missing_indices、output_repeated_index。无精确输出时相应output字段保留null，delta/kind/status仍保留原图实际语义，不能省略。L 必须有 qualification、epsilon、完整 nodes（received_frame/model_status/L_status/best_count/model_index/candidate_index）。只含索引表不能替代这些语义；不从省略字段推断图、资格和旧边。

raw.npz 使用原 q/d0/d1/soft/mask 字段，各形状 [收到帧,5,2,63]，mask布尔。浮点字段在原hard融合之前统一转为FP64并核对转换后有限性，避免较窄dtype改变融合算术。运行时复用原码表、hard、C、L，保持FP64/block1024/epsilon1e-12及原公共域。无模型/writer/GPU执行。依赖 numpy、psutil；media另需可用ffmpeg/ffprobe。没有精确版本或摘要门，真实导入/IO/格式错误直接记录。

truth 为可选后评 JSON 路径，格式 {"views":{"FULL":[源索引,...],...}}；raw/semantic 缺truth保持证据不完整。media 无外部 truth 时，盲输出全部终结保存后再存固定编辑映射用于后评。未知 truth 单元保留 UNKNOWN；truth 不传入 method/runtime。接收器看不到原源长、编辑映射或成绩。

## 记录和判据

固定822节点、817收到相邻边、517860原始位槽、4110lane槽，一个来源五个相关编辑。attempt/state/counts/resources/blind_terminal/result.json 保留调用与失败；received/<view>/ 保存原观测/硬图/C/L（适用时）、semantic.json 和 submission.json，evaluation.json 保留全位置/边、候选损失、错、未知和未新增。固定窗口结束后才读truth。晚发生资源错误也保留已保存可读提交用于后评，但不能冒充完整成功。后评文件实际保存失败时，result摘要明确为EVIDENCE_INCOMPLETE，不从内存评价报告有限支持；同OUTPUT仍只读、不补评。

后评分别计旧节点、旧边、新节点、新边。新边主正确性要求绝对端点+delta/kind+适用缺失/重复索引全对；delta-only单列。错旧H参与的新边仍计新边错。一个原N非单点新增正确节点必须自身触及新增正确边，不能将不同节点的证据拼接成满足条件。

COUNTEREXAMPLE：至少一个新增错节点/边，正确净增不能抵消。EVIDENCE_INCOMPLETE：执行/保存/后评/unknown不完整。NO_ADDED_GAIN_OR_TEST_POWER：完整但无同节点非退化新增边见证，全拒不算PASS。FINITE_ADDED_SUPPORT：仅本批新增部分、零错零未知、旧输出未变、结构无错误且有同节点见证。旧H自身truth错误单列限制总体；continuous/repeat/gap只由对应新增正确事件支持。顺序一致不认证绝对定位，单源不证明总体可靠性、Ø、来源身份或完整同步，n0..3启动区未定义保持。

资源上限RSS8GiB、码表300秒、每view接收/提交/保存600秒、接收累计3300秒，codec每次300秒另账；全墙时另记。已有可靠codec修复不改动。异常消耗固定attempt，没有自动重试或跨入口预算。

## Notebook 与工程检查

notebooks/b_conditional_time_output_v1.ipynb 与 scripts/build_b_conditional_time_output_v1_notebook.py 同步；首代码单元仅两行Drive mount。算法、配置及循环在源码。该独立CPU组件仍需实际CONFIG_PATH；SOURCE_DIRECTORY可留空，入口会普通下载源码，按实际导入错误准备numpy/psutil，仅media准备ffmpeg。已有attempt时先只读，不下载/安装/重跑。用户要求含新正例物化的完整Run-all入口另见 notebooks/b_conditional_time_validation_v1.ipynb 与 docs/b_conditional_time_validation_v1.md；它默认明确c01 P0和独立输出，物化预算与43项验证分开。当前仅完成本地发布准备，未启动Colab、写Drive或由A1发布远端。

tests/test_b_conditional_time_output_v1.py 用人工JSON/小数组验证方法边界、三预算实际编排、失败/中断消耗、盲先于truth、后评反例和薄notebook。媒体/码表/BCH读使用人工adapter；真实runtime只以人工语义输入核对提交和资源边界。没有读取真实e01/e02/new-source数据、媒体解码或模型。这些只构成工程证据。
