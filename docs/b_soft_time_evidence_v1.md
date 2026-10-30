# L/G 离线软信息诊断

采用2026-10-30三方最终规格，仅对旧盲候选作节点模型排名。旧 DCT2＋CDC32、writer、正式对应/资格/关系保持；新增正式确定与关系/事件输出均为0。真实软回放由用户未来在CPU Colab运行，当前工程交付只使用人工fixture。

实际notebook为 `notebooks/b_soft_time_evidence_v1.ipynb`；builder为 `scripts/build_b_soft_time_evidence_v1_notebook.py`。普通CPU只需NumPy/psutil，无模型/codec依赖安装。资源监视和公共BCH码表生成复用既有time-recovery模块的相应函数；不调用旧BCH译码、旧receiver或媒体函数。

独立CLI无需notebook/scripts/report/.git，可使用默认config或显式路径：
```
python -B -m experiments.wan_state_clock.b_soft_time_evidence_v1_run \
  --source-attempt /path/to/old-time-recovery --output /path/to/new-lg-diagnostic
```
默认config为 `experiments/wan_state_clock/configs/b_soft_time_evidence_v1.json`。旧目录为Drive中B-Framewise-QIM-Time-Recovery-V1/fixed-e01-ten-views-v1，新目录为B-Soft-Time-Evidence-V1/fixed-e01-lg-v1。旧目录只读；preflight/log/cache同级。科学尝试先保存全计划和计数；raw/old blind各十份且各一次、码表一次、C/L/G各十次。失败中断消耗不补领，独立view可继续，重开在下载/setup/science前只读。预检只查文件存在性，不打开raw或blind内容。

每view仅取raw中的soft/mask，旧blind一次读入并逐字节复制为新目录views/vXX/old_blind.json。评分器从旧图提取完整N、状态及资格，不接收arm/view、181原终点、私有mapping或45/16名单。逐副本hinge后先平均同bit有效副本，再平均固定有效lane/bit。每帧每j只算所属lane一次，n束只查五个j；完整域92672、block1024，候选不裁剪。N含n<4整行C=0且STARTUP_MODEL_UNDEFINED，冲突/异常/无观测保持，不被G单值晋升。

L与G共用C；G用含当前c=C/R的F/B，Phi=F+B-c，自由首尾/非降，前后缀min，不另算关系或拼一条路径。保存delta=Phi-J_star原值与dG_C=R*delta；不剪负微残差。L用C单位epsilon1e-12，G_native用J单位，G_aligned用换算C单位。两种G集合差异单列尺度效应，不替换原始集合。

## 文件与数值解释

每个view保存C.npz（全部C、bounds、offsets、有效支持数），L.npz（全部dL和首选bool mask），G.npz（全部Phi/delta/dG_C/native/aligned masks）和blind_ranking.json（每行状态、集合大小/单点模型首选、J_star、归因）。第r行完整候选是bounds[r]的闭整数区间；其数值和集合掩码由offsets[r]:offsets[r+1]索引，故全部并列及竞争索引均可恢复，无top-K。C/L.json保存定义状态与局部最小值。无路径或旧图异常保留未决，不能救回；NPZ的inf仅用于这种无可用图的未执行min-marginal，不充当有效排名。

原始L→G_native、L→G_aligned的纠正/损害/并列变化始终完整计数，不被数值归因列吞掉。附加归因仅解释已存FP64 C之上的图算术：令M=max(C)，eps_machine为float64机器epsilon（2^-52），k=8R+32，gamma=k*eps_machine/(1-k*eps_machine)，使用保守C单位算术包络gamma*R*M。非负路径的至多R次除法/加法及F+B-c-J、R倍换算均在该保守操作数覆盖内；min不引入加法舍入。局部差使用gamma_4*M。这个包络不重求图、不改epsilon、不拒绝运行，不是数据模型、概率/置信或精度准入阈值。

summary直接比较已算出的完整L与G掩码并保存逐行同集合标记；两边都MODEL_TIE、基数相同但成员不同仍记TIE_CHANGE。G单独失败时，后评只读取新输出C的bounds/offsets/状态及L已保存best mask/状态，保留已RETURNED的L，不补算或重读旧输入。未执行、调用失败、已返回但保存记录不可读分别记NOT_EXECUTED、FAILED_UNRESOLVED、SAVED_RESULT_UNRESOLVED；任一分支不可用时转移为UNRESOLVED_COMPARISON，不当UNCHANGED。旧正式node/relation嵌套格式损坏仅使该view未知，BASE旧标签缺失/损坏也单列未知，其他view后评继续。

只有L集合对其局部舍入包络稳定，G_aligned出现唯一模型首选且与所有竞争者的分差超过2倍图包络加既定epsilon，并有超出算术包络的C单位min-marginal变化时，标CONSTRAINT_RANKING_DIFFERENCE_AT_STORED_C。其余L/G集合差异保留NUMERIC_ATTRIBUTION_UNRESOLVED；仍原样展示全部分差和转移。该判断仅排除共享已存C后的这一保守浮点算术解释，不证明C模型正确、输入噪声可靠或新增正式恢复。输入C本身的模型/观测误差未认证；小差异、边界并列或不满足条件者不能记图收益。微负残差另计，不裁剪。

十view所有盲结果终结后才打开source/private/attack_truth.json；post分类45非启动/16启动未决及旧确定、BASE。旧761/715按实际保存结果机械计数，BASE旧资格/j0与模型排名并列。十视图1644帧/1035720位槽/8220lane槽/1634旧关系均保留失败/未定义/未执行。

资源预算不变：码表300秒，单view输入/C/L/G/盲写600秒，累计6300秒，实际RSS8GiB。真实CPU耗时和排名表现未验证，旧popcount时长不是本算法保证。所有报告只是同批保存数据的条件模型诊断，不新增可靠对应、关系、身份/画质保持或一般误报率证明；原native精确落点失败、原数值停止与最优性未证保持。
