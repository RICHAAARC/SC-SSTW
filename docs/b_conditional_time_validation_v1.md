# c01物化与条件时间输出完整验证入口

用户2026-10-31要求“包含，补齐新正例物化和固定验证后给链接”。本地入口已实现，真实执行由用户Run all；A1不代执行模型/媒体/GPU/Colab，不发布远端。最终发布及链接核验由主协调处理。

## 默认来源与复用边界

默认SOURCE为 `/content/drive/MyDrive/Video-WM/B-Terminal-Three-Group-V1/fixed-four-new-v1`；实际输入 `private/b2-m3-c01/P0_terminal.pt`，source_id=`b2-m3-c01`，模型位置沿既有 `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`。按原配置c01、c02顺序取首个c01，依据是旧完整P0文件条目已确认、当前时间包/融合L/条件接口开发与结果范围未发现c01/c02使用记录；不看新源分数选来源。

c01/c02确有历史身份校准/相关诊断用途，不是全研究未见内容。此处只作为当前时间包融合L/条件输出规则的未用来源，不增加身份/统计独立或总体泛化结论。详见外层 diagnostics/conditional-time-output-colab-delivery-20261031/SOURCE_SELECTION.md。未读取P0张量或真实接收数据，文件实际可加载性、FP32形状和有限性仍在原capture的唯一调用内检查；缺失/不合法原样失败，不换e02/c02或重新生成。

## Run-all实际接线

`b_conditional_time_validation_v1_run` → 整包attempt → 现有 `b_minimal_l_materialization_v1_experiment.run_stage(A)` → 同次调用内可用完整A缓存 → 原run_stage(B) → 有限完整native和已有QIM文件 → 已审查V3 `b_conditional_time_output_v1_experiment.run(entry=media)` 固定43项。

原物化API新增显式protocol/receiver_codec_calls参数；旧默认e02/34/40仍服务旧入口，完整新入口显式传c01/17/23。没有全局猴补丁。A/B的模型、求解、目标构造、容差、缓存/阶段消耗及raw读取实现都复用；原DCT2＋CDC32、融合L、旧H/旧精确关系不变。

新整包输出为 `/content/drive/MyDrive/Video-WM/B-Conditional-Time-Validation-V1/fixed-c01-v1`。在任何A/B实际执行前以排他创建顶层attempt。任何后续Run all都只读，即使上次只完成A、B尚未消费，亦不借原独立A/B入口的恢复能力再取得一次B执行。只有当前同次顺序调用能推进尚未消费的下一阶段。A失败/缓存不足不补录或求解；B直接约束失败无新媒体就不读接收；中断停止整包后续。未执行接收的43行保留于receiver_not_executed.json，822/817固定分母仍列明，不伪造接收attempt。

原B可能返回NUMERICALLY_UNRESOLVED但保存有限完整native与QIM文件；此时按既定上限进入43项，其源probe/decode在预算内确定媒体实际可读性，不额外试读。native/solver失败和optimality=UNPROVEN保留在materialization/A、B及顶层stage_results；receiver_evaluation_status单列，scientific_pass始终未作整体成功裁决，接收结果不修补或盖过物化失败。

接收入口预检失败或在预约前中断，同样保留43项未执行行、零次started/returned及具体退出原因；不生成内层attempt。接收已预约后中断则保留内层真实计数和状态，顶层明确标记PACKAGE_INTERRUPTED，不以未执行记录覆盖已开始的调用。两类退出后重新Run all均只读。

## 独立预算与分母

|类别|A物化|B物化|接收|
|---|---:|---:|---:|
|model_load|1|1|0|
|D181|1|0|0|
|BASE head / QIM head|46 / 0|0 / 46|0|
|solver / targets|1 / 1|0 / 0|0|
|K / KT|4200 / 4200|1 / 0|0|
|BASE / Aa约束投影|0 / 0|1 / 1|0|
|物化四层time_reads|4|4|另计time_read5|
|codec encode/probe/decode|1/1/1|1/1/1|5/6/6|

物化共codec6，接收codec17，整链codec23。接收43=源probe/decode2+码表1+五view各encode/probe/decode/time_read/hard/C/L/submit八项；不是旧两臂85/75或codec34。不新增DiT、E41或训练，不重新运行身份writer；本轮新时间包沿既有P0复用当前时间writer物化。物化两臂四层保留912240位槽；接收只QIM五相关view，固定822节点、817相邻边、517860位槽、4110lane，一个来源。

求解沿原4000次、每20检查、7200秒、16GiB；目标1e-3、预算1e-7，最优性UNPROVEN。原feature缓存按实际通道空间检查（至少32GiB），不以GPU型号作门。接收RSS8GiB/码表300秒/每view600秒/累计3300秒，codec每次300秒另账、terminate等待30秒后kill；全墙时另记。四文件既有codec修复必须随发布带上。

## 源码与恢复

完整notebook首格仅两行Drive mount；默认SOURCE_DIRECTORY=None，通过可编辑SOURCE_REF普通下载入口和源码包、安全解压，使用原已跑通安装/导入修复，再调用独立CLI。没有hash/B64/manifest/Git-clean/精确版本检查或strict/override。模型/算法/配置/长循环在源码。实际下载/导入/格式/空间失败直接报告；准备失败未启动物化，科学调用开始后由整包和各阶段记录消耗。

已有顶层attempt时，notebook在下载或安装之前用stdlib展示attempt/state/counts/result；缓存和源码都丢失也不补跑。独立CLI支持无.git可编辑源码，已消费时无需numpy/torch/psutil/runtime/scripts。普通新运行可用：

```
python -B -m experiments.wan_state_clock.b_conditional_time_validation_v1_run --config /absolute/config.json
```

配置见 experiments/wan_state_clock/configs/b_conditional_time_validation_v1.json；notebook与builder为同名文件，完整入口为 b_conditional_time_validation_v1，而 b_conditional_time_output_v1 notebook是独立CPU组件。真实未用时间包物化/43项接收尚未执行；人工测试和静态发布检查不能称科学有限新增支持。
