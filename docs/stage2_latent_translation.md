# Wan 共同 latent 平移：阶段二二维位置原语

本次交付为真实运行模块、固定配置、CPU 定向测试和薄 Colab；GPU 尚未执行。未来一次 GPU 运行只检验共同空间重定位的二维响应，不承诺主体动态轨迹、六点模板写入或 AISB。当前帧差 observer、v2、模板、128 预算均不改；本原语无需运行 v2/扫描。

唯一机制：Wan2.1-1.3B 的 8 步推理完成零基 step 5 后，将当前 B,C,T,H,W latent 的每个时间/通道切片共同平移 ±0.25 latent pixel；X/Y 两方向共四干预，加 OFF1/OFF2 两次无干预。正x向右、正y向下，双线性插值、border复制、align_corners=True；不wrap、不在最终像素上平移，不注入q。OFF精确clone不经过插值。每臂复制完整 UniPC 状态，历史缓存不平移、不清空、不重置，然后正常执行step6/7及VAE。历史仍在原坐标，故不能假设剩余生成具有平移协变性。

0.25 latent格点在VAE空间倍数8下仅对应理想约2输出像素，不是最终像素位移真值或q目标。它是固定小扰动起点：足以区别于零操作，同时避免整格大搬移；边界复制、插值平滑、VAE和剩余去噪仍可能改变画质与位移。失败后不扫层/强度/种子。

## 固定范围和判据

精确配置见 configs/stage2_latent_translation.json：Wan官方revision 0fad780a534b6463e45facd96134c9f345acfa5b，diffusers0.35.2，BF16 + CPU offload，单prompt/seed1275，320×512、49帧、8fps、8步、CFG5。12 prefix transformer calls + 6×4 continuation calls =36；6次VAE，6个MP4以及6份相同RGB的无损诊断文件。无梯度/额外VAE。仅未来一次GPU运行，3600秒包括模型加载、全部分支与CPU读回；外层独立进程组超时强制终止，保存实际exit/完整六身份，无自动重跑。建议未来获批后使用L4 24GB或更高的BF16 GPU，记录实际显存峰值；没有把历史L4成功当本轮显存保证。

原production decode_video以5Hz、maxframes300/maxpixels2073600/timeout60取灰帧，其同一不可变frames直接传原observe_frames（delta12/support .0001–.5）。原源49帧PTS必须与i/8一致；输出q轴为实际规则i/5，不声称source index=固定倍数。完整灰字节流、源PTS、原q/valid/reason/support/mass保留。preencode无损RGB和MP4均走这条硬阈值读出，前者只外部解释VAE/压缩差异；主结论来自保存MP4。

六臂仅使用同一个共同valid原索引交集构造两个响应列，不compact重新配对。缺失/初帧全部逐行保留；要求共同valid占全部非初帧≥80%，且至少10对。该覆盖门防止少数幸存时点冒充全段响应，仍不代表其余20%成功。

c_a=mean(q_a+−q_a−)/2，C=[c_X,c_Y]，J=C/.25；报告完整2×2矩阵及SVD，允许旋转、反射、轴混合，不要求逐轴对角/正号。f=max(RMS_vector(q_OFF1−q_OFF2),32×2^-23)。预设 smin(C)>5f、κ(C)≤10。numeric floor是人为工程裕量，不是float32/BF16精度保证；5倍/κ10是本小试的保守分流选择，无科学校准。

另对每列单位方向u_a=c_a/||c_a||，分别相对OFF1和OFF2要求 mean(q_a+−q_OFFk)·u_a>f、mean(q_a−−q_OFFk)·u_a<−f；零列不成立。这防止正负两臂共同向同侧漂移却中心差满秩，不承诺每帧都反向。每时刻odd/even（分别对每OFF）、均值、时间漂移全报，不用漂移另调阈值。

质量门：每干预臂相对每OFF，在原坐标全49帧RGB[0,1]绝对RMSE≤.05，preencode与MP4均报/均须满足。约12.75/255的RMS预算允许小位移本身代价，但并不保证感知质量、语义/主体身份保持。无配准、美化ROI或质量损失后增门。门值只用于拟议工程决定，不迁移旧G1 .02或v2 epsilon。

充分共同覆盖、双侧响应/SVD及质量门都满足才 ENGINEERING_PRIMITIVE_READY；完整可评但预算不满足为 ENGINEERING_BUDGET_NOT_MET；模型/时轴/缺臂/关键覆盖不足为 OPERATIONAL_BLOCKED。所有原始失败保留，不能把缺失当零响应。结果仅支持进一步具体控制方案，不自动扩展时序写入或生成运行。

## 传播分层证据

保存prefix、每臂实际injected与final latent；报告注入实际绝对/相对RMS。正常续程后外部比较 final_a 与 T_delta(final_OFFk) 的RMSE及位移方向投影增益，不将该预测返回生成。随后分别保存真实VAE preencode RGB和MP4；两层原observer的完整q及六臂矩阵解释剩余去噪、VAE/边界、压缩读出变化。控制位移不冒充实际图像主体位置参照；本轮未建立主体几何中心真值。

## 旧路径复用与失败区别

只复用 archive/SC-SSTW-Feasibility/src/sstw/flow_guidance_embedder.py:739–770 的完整UniPC复制语义、951–997正常续6/7，以及 flow_guidance_saved_mp4.py:228–275 的Wan加载/offload/VAE/保存流程。新包不import旧archive。旧S1仅内部attention探针，没有正常续程/VAE/MP4，不能作完整路径成功证据。

已只读真实 G:/我的云端硬盘/SC-SSTW-Feasibility/sstw-v2-g1-resume-glass-garden/sstw-v2-g1-12307e2866c50f8d.zip 中 output/result.json：NVIDIA L4、torch2.11.0+cu128、diffusers0.35.2；4新+4复用MP4，28 transformer calls/6VAE；旧chroma-cosine梯度机制虽矩阵满秩且κ通过，轨迹residual .9761/.6886、RGB质量 .317/.320与.0891/.0889均未达旧.02，终态NOT_FEASIBLE。新方法不复活该梯度、不用其色度投影/pairedclean获取或阈值。共同几何平移是不同机制，同时其上限更窄：共同位置原语并非时序轨迹写入。

版本依据：[Diffusers v0.35.2 WanPipeline](https://github.com/huggingface/diffusers/blob/v0.35.2/src/diffusers/pipelines/wan/pipeline_wan.py)，默认temporal4/spatial8。本地当前无模型/媒体新运行；源码来自已读版本语义，GPUAPI/显存仍需未来一次实际检验。

## 文件与执行

main/sc_sstw/latent_translation.py 为方法；runtime/stage2/wan_translation.py 为真实模型运行；experiments/stage2/run_latent_translation.py 为单次进程组限时入口；notebooks/stage2_latent_translation.ipynb 为薄Colab；源ZIP自含以上代码及原observer依赖，无Windows/旧archive绝对依赖、无需push。首cell仅两行独立Drive mount，无force。

CPU检查与notebook静态检查结果见交付根日志；GPU单元未执行，不能以静态编译替代真实模型验证。Notebook在另行批准一次3600秒GPU预算后，将RUN_GPU_AFTER_APPROVAL设True从上到下运行；原目录/日志已存在则停止，不覆盖旧产物。

当前审阅：A2已静态核方法/运行器并APPROVE，root已核Colab与进程组硬截止；均非GPU实证。CPU全部9项已实际通过：使用已有torch2.5.1+cpu解释器、CUDA_VISIBLE_DEVICES为空，测试真实CPU grid_sample、BF16返回、OFF精确clone、独立完整调度历史，以及混合二维/共线/双侧偏置/缺失分母。未执行GPU/模型。Notebook六个代码cell已分别Python AST解析，源ZIP已在临时独立目录真实import成功、不加载模型。

依赖核对：实际旧G1 notebook仅固定diffusers0.35.2，其他依赖未固定；旧原始包未留其余精确版本。新notebook沿该已使用装法，仅精确pin diffusers，补齐sentencepiece，普通install现有/兼容transformers、accelerate、ftfy、safetensors、huggingface_hub、numpy、Pillow，不用更早S1的numpy1.26/transformers4.49降级组合。未来runtime记录全部实际版本，API兼容与显存仍属于未执行缺口。

最终CPU证据：cpu_tests.log 的实际命令退出0、9/9通过；先前cpu_tests_predependency.log的5项skip保留为历史。官方CPU wheel下载曾在120.8/196.3MB网络ReadTimeout退出1，未重取；随后仅借用已有CEG-WM .venv的torch2.5.1+cpu解释器执行Video-WM测试，未引入CEG方法/配置或修改其环境。平移测试实际调用CPU torch.grid_sample；调度续程测试使用明确的UniPC接口stub和mock velocity，只检验fork隔离/固定索引接线，未调用真实Diffusers调度数值或Wan模型。CPU验证不替代Wan/UniPC真实模型运行。


## GitHub Colab delivery

The notebook now clones https://github.com/RICHAAARC/SC-SSTW.git at the explicit development branch and prints the resolved commit. No source ZIP upload is needed. The GPU execution approval flag remains explicit; publishing does not run the model.
