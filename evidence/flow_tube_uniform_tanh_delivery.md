# Uniform-Tanh-44/46 交付卡

状态：上级最终发布审计已接受，候选分支已 push，固定源码和 notebook 的远端内容已核验；未运行 GPU、真实模型、Colab 或 Drive 写入。

## 版本与输入

- 基线：`29c634c251c879e37cbe674a23b89d0052553377`。
- 分支：`dev/flow-tube-uniform-tanh`；最终可执行/notebook 审查版本 N3：`081266cab1eb2839a26d7db6e24ea41b3315348e`。
- 初始源码提交 S：`5dc70aa3f6918bc4f29acdf01e1b570be0ce7b46`；缺资产分母修复 S2：`5934a46e965d584a9fee4c75d5e029cfca1e6e47`；无仓库导入的 notebook 定位修复 S3：`3dd3d0ec7c69dd67fcd6e7acf94ae0f3434ba084`。
- notebook：`notebooks/flow_tube_uniform_tanh_colab.ipynb`，固定 pin=S3，首代码单元严格为独立两行 Drive mount；Run all 只有固定实验，无模式菜单或硬件白名单。
- 固定模型 revision：`0fad780a534b6463e45facd96134c9f345acfa5b`。历史 response-selection source 的 revision 为 null，因此不声称历史权重身份相同。
- 固定输入 run：`flow_tube_response_selection_20260921T013844126172Z`。2026-09-21 只读 Drive 元数据检查确认 run 目录与两个 case 目录均列出 generation/config/book/prompt/negative/OFF_nodes/OFF_snapshots；未在该检查中下载大张量或重验 hash。用户运行时由 runner 对旧 manifest 逐文件 hash，并核对 saved44 input/history fingerprint；缺失或不匹配保留失败。

## 已发布入口

- notebook 预期输入路径：`/content/drive/MyDrive/Video-WM/FlowTubeResponseSelection/flow_tube_response_selection_20260921T013844126172Z`。纯标准库 locator 每次均在整个 `MyDrive` 下递归搜索同名 run 目录；唯一结果交给 runner，零结果回退上述预期路径，多个不同结果明确报歧义。缺文件由 runner 记录固定失败分母。
- 默认输出：`/content/drive/MyDrive/Video-WM/FlowTubeUniformTanh/flow_tube_uniform_tanh_<UTC>`。
- 已发布 Colab 链接：[固定 N3 notebook](https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/081266cab1eb2839a26d7db6e24ea41b3315348e/notebooks/flow_tube_uniform_tanh_colab.ipynb)。固定 notebook pin 为 S3；由用户 Run all，agent 未执行 Colab。
- 2026-09-21 发布核验：仅推送 `dev/flow-tube-uniform-tanh`；GitHub raw 读取的 N3 notebook 及 S3 runner/config/runtime/builder/tests 六份文件均与本地对应 Git 对象逐字节相等。远端 notebook 的两行 Drive mount、S3 pin 与无输出状态一致。notebook SHA256：`8fa3d535d0f27b4c7bb874ce36c905f1094be122cc991481cb1683620d554061`。远端 `main` 在发布前后均为 `3f0a5fafa7c2aa56fbc69bed17649accf49ae152`。

## 固定方法与预算

- 两个原 dev 内容/seed × `OFF/SINGLE46_A/B/MULTI44_46_A/B` = 10 逻辑视频；三层媒体 × 四相位 = 120 receiver encode 固定分母。
- `R*=0.042943312697648145` 是四个完整历史 dev LOCAL native-response 预算的最小值。SINGLE46 使用 `R*`；MULTI44/46 各使用 `R*/2`。这是历史 oracle provenance；新运行不读取未来 LAST 预算。
- 全轮固定：84 Transformer、52 formal native、12 unit probe、4 second-control same-history zero shadow、12 CPU clean-leaf backward、10 decode、10 MP4 save、120 encode。
- MULTI46 在受控44与自由45之后的真实 latent/完整 scheduler history 上重新计算 gradient、shadow 与 probe。47..49 自由尾程，无末步补写。

## 本地验证

- 命令：`python -m pytest -q tests/test_flow_tube_uniform_tanh.py tests/test_flow_tube_clean_proxy.py tests/test_objective_alignment.py tests/test_flow_tube_multistep.py`
- 结果：14 passed。覆盖真实 `python -m` CLI、精确调用计数、controlled history、shadow/probe 不污染、无未来预算依赖、预算累计/平方和、第二控制失败保留第一步诊断、10/120 失败分母、三层实际栅格和 absolute correct/wrong/gap。
- 本地解释器绝对路径：`/home/richar/projects/CEG-WM/alive/CEG-WM/.venv/bin/python`；torch 2.5.1+cpu、diffusers 0.39.0，属于 CPU/fake 工程证据。notebook 固定安装 torch 2.11.0+cu128、diffusers 0.40.0；尚无本轮真实 GPU/model 证据。
- S2 定向验证：`python -m pytest -q tests/test_flow_tube_uniform_tanh.py -k locator`，1 passed、5 deselected。首选 run 目录即使缺文件仍被交给固定 runner；完全缺目录也传入预期路径；多个不同 run 目录明确报歧义。
- S3 定向验证：`python -m pytest -q tests/test_flow_tube_uniform_tanh.py -k notebook_locator`，1 passed、5 deselected。测试从仓库外 cwd 执行 builder 写入 notebook 的同一段纯标准库 locator source；无需把 clone 加入 kernel `sys.path`，不导入 torch/diffusers runner，并保留 S2 的缺失/歧义语义。
- 上级真实顶层缺输入 CLI 检查：`/home/richar/projects/Video-WM/diagnostics/project-status-20260921/uniform_tanh_cli_missing_input`，在 N=`9d6a3e2e8af4bf6c9942f37caf9c75517d36c8d5`、上述 venv、`CUDA_VISIBLE_DEVICES=-1` 下，四个真实 child 均落盘失败并以 exit 1 收束；10/30/120 与4配对分母完整，model/scheduler/VAE attempted 均为0。该检查由上级完成，本分支未重复运行。

证据上限：仅证明固定实验可执行结构、预算与记录语义；不证明真实运行完成、等能量、等终态、存在性阈值、FPR、感知质量或推广性。

## 同版本审查

- A2 方法审查：对 N=`9d6a3e2e8af4bf6c9942f37caf9c75517d36c8d5` PASS，无方法阻断；固定 R*、受控 live history 重算、预算/非主张与无新 future-budget 依赖均符合任务卡。S2/S3 未改变 runtime/config 方法核心，该 PASS 适用于最终 N3 的方法范围。
- A3 实现/证据审查：初审 P1 是原 notebook `valid_source` 在 OUTPUT/runner 前要求两个 case×7文件全齐，缺资产时丢失10/30/120分母；S2 已关闭。N2 从 kernel 外部 cwd 导入 clone 内 runner，A3 实际复现 `ModuleNotFoundError: experiments` 并新增入口 P1；S3 改为纯标准库内联定位。A3 对 N3 最终 PASS：两项 P1 均关闭，无剩余问题；复核了 clean HEAD/diff、缺失不筛选、外部 cwd 同源与实际 notebook locator、pin=S3 和代码 AST。
- A4 综合：PASS，A2/A3 无未决分歧，N3 可交 A5。依据为14项原套回归、1项最终 locator 定向验证、上级真实顶层缺输入0模型调用证据，以及方法核心 byte 不变；结论保持 CPU/fake 工程上限。
- A5 里程碑审计：PASS，无阻断，N3 可回交上级发布审计。独立核对 source pin、历史 oracle `R*` 及 SINGLE/MULTI 分配、真实受控 history 上的第二控制、47..49 自由尾程、10/30/120/4 固定分母、84/52/12/4/12预算、缺输入0模型调用和 import-free notebook 均一致；未把 CPU/fake 提升为 GPU、真实模型或科学结果。
