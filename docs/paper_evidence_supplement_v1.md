# 论文证据补充 V1

本轮只修改证据汇总、概率说明和有限成像评分入口。原DCT2/CDC32、时间路径、QMatch、源、攻击和阈值均沿用；不重启F40–F44。工程交付不代表新质量结果或科学PASS。

## 保存证据

`python -m runtime.wan.paper_evidence_supplement_v1 assemble --main-assembly PATH --wan-root PATH --frozen-root PATH --compact-root PATH --output NEW_PATH`

输入是既有JSON/CSV；不加载模型、不解码、不调用接收器、不重做后评、不抽新查询。输出必须与输入目录分离。若保存到正式补充包，使用 `paper_evidence_supplement_v1/attempt01/saved_evidence`。质量执行独占兄弟目录 `attempt01/quality`，两者不会争用一次执行目录。

固定十二Wan源×五视图；480 baseline posts、720固定声明、360冻结view/key、36共享查询族。四态是MATCH、MISMATCH、UNRESOLVED、INPUT_UNRESOLVED；未决非已证无水印，P0不等于OFF。JSON布尔值严格解析；null、缺键和非法类型保留未知，不做真假强制转换。

主线节点、精确端点对、事件类别、时间支撑载荷、baseline原生逐帧BER/完整视频词分别列示。FrameIndex为248身份+8时间适配参照；其他baseline无时间输出列N/A。生成期写和首代媒体后处理写、消息容量不同，因此不作等扰动/等容量排名。CROP为时间切片，DELETE3为固定三帧删除。

## 条件概率

接受词先由冻结证据E形成，不接受查询参数。每view/key最多一个32位完整词；逻辑未决为空集，输入技术缺失/非法为未知槽。每源/臂十槽，已知去重词d、技术未知u给出[d,min(10,d+u)]/2^32，前提是Q给定E独立均匀。P0/NEW若条件Q≠M则先去掉M，分母2^32−1；OFF分母2^32。

设计上界360/2^32、实际保存接受集uniform32 union、剔除M的固定E条件wrong-query union三者分列。随机Q可能恰好是真消息，固定q0/q1不套均匀查询模型。任何条件0都不表示经验总体FPR为0。36族仅概率模型，新增抽样0。未发现随机查询回执只描述实际搜索输入范围，不断言历史从未执行。

Compact12另列公共域500源，每方法OFF 0/500的单侧95%界仅为单方法参考，不是四方法同时置信覆盖。

## 用户Run all质量入口

`notebooks/paper_evidence_supplement_v1.ipynb` 与 `scripts/build_paper_evidence_supplement_v1_notebook.py` 同步。首代码格仅Drive挂载。随后自动加载普通可用源码并准备依赖，再调用独立CLI：

`python -m runtime.wan.paper_evidence_supplement_v1 quality`

也可通过 `--result-root`、`--output`、`--cache` 指定实际路径。默认结果根 `/content/drive/MyDrive/Video-WM_Result`；新质量结果为 `paper_evidence_supplement_v1/attempt01/quality`。源码和权重缓存在 `/content`，不放正式结果目录。可编辑无`.git`源码同样可用，无摘要、manifest或精确版本门禁。

固定84逻辑流，每源main OFF=input0、P0=input5、NEW=input10的received.mp4，以及wam/pixelseal/videoseal/frameindex的FULL.mp4，均为原始最终二代媒体；不取first.mp4或t12修补候选。固定16索引0,12,…,180；缺帧保留，不clamp/补抽/填充。只复用相同实际媒体指针，84逻辑行不变。

1模型加载、最多84媒体读取、1344计划评分槽。MUSIQ-SPAQ eval/inference FP32，RGB uint8转float，长边>512按比例bilinear缩小且antialias=False，再除255；无裁剪/放大。逐帧输出原始MUSIQ分，不冒充VBench官方全帧或总分。实际VideoReader/get_batch/model forward与逻辑槽计数分开。

完整clip需16帧全齐；partial_mean另列。NEW−P0、P0−OFF、NEW−OFF仅同源完整固定索引可配对。baseline还需质量记录中的明确绝对OFF媒体路径与当前main OFF一致；泛称“same second-generation view”和当前runner默认路径不足以证明历史引用。现有旧记录缺明确路径时保留48个baseline配对未知，仍计算绝对WM分，不扩展84流。

先预约再物理调用。失败/中断消耗此包；再次Run all在下载、安装、模型构造前只读，绝不自动重载或补评分。小操作回执保存调用状态和已得帧分，完整表保留1344槽。若进程硬终止而无摘要，重开只读已有回执，缺项仍未知。

输出quality_frames、quality_clips、quality_pairs（CSV/JSON）、quality_summary.json、REPORT.md及操作记录。没有PASS/领先阈值；不测时间连贯、闪烁、短暂伪影、审美、提示词或全面视频质量。旧PSNR只说明参考相对像素偏离；不同阶段保存wall不支持速度比。

## 工程验证与当前交付

定向fixture覆盖布尔三类未知、冻结接受集去重/真词排除、前处理、缺帧不替代、固定分母、模型/解码/数值失败、中断和只读重开、明确OFF引用、builder一致性及notebook→entry→CLI→fake backend的完整接线。CPU/fake不代表真实权重兼容、GPU或视频评分成功。真实Run all和notebook渲染输出尚未执行。

本次尚未提交或推送。当前notebook源码引用是待发布位置，不是已核实可用的公开Run-all链接；主会话处理本轮发布权限后才能交付远端入口。不得复用历史t12发布许可。

官方实现依据：[imaging_quality.py](https://github.com/Vchitect/VBench/blob/master/vbench/imaging_quality.py)、[utils.py](https://github.com/Vchitect/VBench/blob/master/vbench/utils.py)。
