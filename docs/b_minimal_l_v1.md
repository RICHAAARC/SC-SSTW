# 单 e02 最小 L 候选确认与时间媒体准备

当前交付是用户采用任务卡后的工程实现。单 e02 未参与本次逐帧 QIM/C/L 调整，但不是全项目从未使用的来源。先由用户运行 GPU notebook 生成时间媒体，再运行独立本地 CPU 入口。工程检查不证明真实效果。

方法沿用原 writer 的 L5/S1/BCH/ROI/两副本/强度/U/4000 次 solver/阈值。DCT2+CDC32 与旧正式结果不变。F18/F19/F32/F33/F34 的全拒、共同映射、次峰、错归属和擦除失败，以及 F38/F39 冗余/保持失败仍保留。本次实际变化是新 e02 的独立输入与固定 L 候选接收，没有改写 writer 或新增可靠放行。

## 用户 GPU Run all

实际 notebook：`notebooks/b_minimal_l_materialization_v1.ipynb`；同步 builder：`scripts/build_b_minimal_l_materialization_v1_notebook.py`。首代码单元仅 Drive mount。五个代码单元仅负责设置、加载普通源码、启动与显示。

默认输入 `/content/drive/MyDrive/Video-WM/B-Terminal-Three-Group-V1/fixed-four-new-v1`，唯一 terminal `private/b2-m3-e02/P0_terminal.pt`。元数据存在不代表张量合法，实际加载检查有限 FP32 和 `[1,16,46,40,64]`。默认输出 `/content/drive/MyDrive/Video-WM/B-Minimal-L-V1/fixed-e02-materialization-v1`，A/B 各自有消费预约和账本。模型位置可用默认 Wan-AI/Wan2.1-T2V-1.3B-Diffusers 或普通可用路径；不以版本或摘要相等作门槛。

```bash
python -B -m experiments.wan_state_clock.b_minimal_l_materialization_v1_run --source INPUT_ROOT --output NEW_MATERIALIZATION_ROOT
```

| 调用 | A | B | 总上限 |
| --- | ---: | ---: | ---: |
| model_load | 1 | 1 | 2 |
| D181 | 1 | 0 | 1 |
| baseline_head / head_replay | 46 / 0 | 0 / 46 | 各46 |
| solver / targets | 1 / 1 | 0 / 0 | 各1 |
| K / KT | 4200 / 4200 | 1 / 0 | 4201 / 4200 |
| BASE_constraint_P / Aa_constraint_P | 0 / 0 | 1 / 1 | 各1 |
| codec encode/probe/decode | 各1 | 各1 | 各2，共6 |
| 四层 time_reads | 4 | 4 | 8，共912240位槽 |
| E41 / DiT / training | 0 | 0 | 0 |

A 的第一次 BASE/preclip raw 已保存后将同一 q 传入 solver，按 replica-major 转换 `[181,10,63]`，目标只构造一次。B 读取该目标，通过码类网格和 `-16 < q0-target <= 16` 直接核对 nearest/低值平局语义，不重建目标，不引入新阈值；两个直接 P 与物理层读分开记。B 单次 K 的 Aa 缓存用于原生 head replay，不再次 K。

CPU FP64 solver 沿原混合 GPU 物化流程：4000 迭代、每20检查、7200秒、原16GiB live-array bound。A 正常结束、所有固定步骤返回且有完整缓存和有限系数时，数值未决也可进入首次 B。实际中断、缺缓存、层读/媒体或释放失败不重做 A，不启动依赖 B。B 线性失败如实保留已消费 model/K/P，后续 head/media/读量未执行；有限完整 native 即使目标1e-3或预算1e-7不达也完成固定层读，失败保留。独立可执行层读失败继续其余既定层，不补读。

A/B codec 实际使用有界子进程：每次300秒，terminate后最多30秒再kill，不重试。记录每次 codec 和阶段完整墙时；不声称 GPU 峰值已实测。A/B 再开只读；A完整而B未预约可首次B，预检在输出同级目录。源码/缓存/安装/worker均跳过已消费且无新阶段的包。

## 本地 CPU 接收

媒体分别在 `A/BASE.mp4` 和 `B/QIM.mp4`。本地使用可读副本的路径；不需要 torch、模型、notebook、scripts 或 Git。

```bash
python -B -m experiments.wan_state_clock.b_minimal_l_v1_run \
  --base-media /local/e02/A/BASE.mp4 \
  --qim-media /local/e02/B/QIM.mp4 \
  --output /local/e02-minimal-l-confirmation
```

可选 `--config` 使用普通 JSON。bundled `experiments/wan_state_clock/configs/b_minimal_l_v1.json` 指向上述 Drive 媒体及独立 `B-Minimal-L-V1/fixed-e02-ten-views-v1` 输出；显式提供三路径无需配置文件。媒体未齐报告 CPU 确认未执行，不消费该尝试。消费后即使来源或配置缺失也只读原记录。

固定 QIM/BASE 各 FULL181、CROP[38,139)101、DELETE44(180)、REPEAT44(182)、DELETE[43,46)(178)。第二代 H264/8fps/CRF18/yuv420p 不变。codec encode10/probe12/decode12 共34次；time_read/hard/C/L 各10、码表1、G0、模型/写端/求解0。全链 codec 为 GPU6+CPU34=40，GPU层读8与CPU读10分栏。

分母10视图、1644帧、1035720位槽、8220lane槽、1634关系，QIM822帧/817关系。完整公共域0..92671一次生成五个码表桶，flat仅重排相同码字，不再次生成。旧hard最多152352768比较、C最多152352768个frame×j计算，合计304705536为两个不同内核的项目数。C_replica_bit_terms=126*C_frame_j 是两副本×63位的稠密逻辑等价项上界，总上界19196448768；C_replica_bit_terms_kind明确标记其为逻辑展开上界，不是实际物理标量运算次数或计时测量。真实实现先平均副本hinge，再每j查63位代价。C元数据缺失/损坏只将C计数记未知及C_error，保留独立可读的hard和L后评。

码表与重排300秒，每view的time_read、raw保存/重读、hard/fused/blind保存、C/L与候选保存全部纳入600秒，累计接收6300秒，实际进程及子进程 RSS 8GiB。codec在接收计时之外，每调用300秒与terminate30秒，另报codec耗时和完整尝试wall time（不含前置setup/preflight）。资源close失败保留在result/evaluation，不能洗掉已完成记录。

接收端只看收到RGB/soft/mask与公共协议。原hard完整formal/关系保存在 `received/vXX/blind.json`；完整C、dL、首选布尔集合、bounds/offsets及状态单独保存。仅原资格SUPPORTED_RELATION_SET、行DEFINED、唯一L首选才有candidate_index。startup、冲突、无观测、并列不晋升。候选不组成joint path/Δ/关系，formal新增和关系新增恒0；旧formal保持报告单独存储。

全部盲步骤终结并落盘后才读新私有mapping后评。逐视图报告hard正确/错/未决/未知、truth排除N、旧未决的L正确/错/并列/未定义/无资格、旧确定丢失/改动与候选分歧。BASE结构资格、soft可评分分母和候选假输出分列；全不可评分为NO_SOFT_NEGATIVE_EVIDENCE。未知不填0，不以净增掩盖错归属。

e01 的45/16/761/715仅属既有开发报告；本包没有e01输入路径或重读raw/BCH/C/L/G。新增formal与关系仍0，不新增身份、可靠放行、一般FPR、推广或科学PASS结论。
