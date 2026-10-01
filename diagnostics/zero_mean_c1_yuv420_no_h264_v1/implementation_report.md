# 无H264 raw420控制：本地候选交付

本轮已实现独立runtime、runner、builder、未绑定Notebook与必要tests。此前设计config.local_stage为冻结时的历史“未实现”记录，保持原字节；当前状态见本报告和local_validation，不把历史false字段当最终交付状态。旧writer/receiver、公共VAE/io和旧证据均未修改。

固定链为已保存uint8 RGB24→真实raw yuv420p文件→重新打开raw→真实RGB24文件→完整181帧FP32 VAE mode编码；0Transformer、0VAEdecode、0H264/MP4decode。保3组两段转换及实际文件、3normalized、6新path/payload reads、1044有效成本/23490目录/22446排除、6path后评/12message评估。原两层6norm/12raw按SHA引用，0额外infer。primary/reference/package及固定对手可用性单列，失败全部留槽。

第一段自动转换；第二段仅承接已观测SOURCE的left色度标签。矩阵/范围/传递/原色unknown保持auto。实际verbose命令日志、版本、退出码、字节和hash保存；left元数据、相同pix_fmt或相同版本不证明第一段与旧x264前缓冲的采样格点/滤波完全一致。这是无H264 raw420往返控制，不是保证纯H264单因素隔离；无第二控制矩阵或新scale/BT709/601/range/sws_flags开关。

新raw提交后才读取父result携带的固定原FULL.phase0对手并做truth/参考后评。R44投影q、mu均1408维；mu=composite_signs(key)[:44]*alpha，alpha=1/sqrt1392，1360非零+48零边界，mu能量1360/1392。gain=dot(q,mu)/||mu||²；orthogonal residual为全1408维||q−gain*mu||；同号仅1360非零mu，q=0不算同号。全部是无阈值后评，不改变选路。true rank/unique/start/invented_edit和固定对手差同时保留。

必要验证首轮9passed/0skip/10.06s；实际stdout保存在pytest_initial.txt。只在Notebook末格追加Retained result路径后，既有notebook定点1passed/8deselected/0.75s，stdout为pytest_display_delta.txt；未再跑全套。测试使用真实形状的合成张量、物化的mock raw字节、原infer/full family及fake VAE/FFmpeg进程；未执行真实颜色转换、编解码、VAE、GPU/模型，未重算任何真实旧成本。二段失败保首段、短文件拒绝、canonical两窗口、独立引用失败、worker组TERM/wait/KILL后重新读盘均通过。生成功能没有加入当前候选。

参数取证只实际调用ffprobe show_streams/show_format、ffmpeg/ffprobe version及help；没有show_frames/read_intervals。三份小SOURCE复制后SHA核对，Q8仅Drive stat和继承期望SHA，不重读大float/raster。设计16文件SHA/bytes全部未变。设计tree_recipe的“sorted”文字不准确：记录的e67eaf47ab39be045add519dcd55108df6057923280abb1ed81998966b9d9608是files列序；真正完整path排序是3f8561ab9a461e828c0b7c830b9e03137f84be197689cf5115ade6ca79dbb6f4。保留原设计字节；本次candidate manifest明确按完整path排序后哈希。

Notebook为notebooks/zero_mean_c1_yuv420_no_h264_v1_colab.ipynb，SOURCE_SHA=None，尚未发布/绑定，不提供假Colab链接。current Python、工作Torch配对probe、fresh child沿已成功流程；父worker使用独立进程组，监护失败时连同FFmpeg子孙清理后才加载最新canonical结果。此时没有commit/push/merge。工程mock通过不表示真实链可运行、科学成功、低FPR、质量通过或同步成立；真实执行需后续发布绑定后由用户完成。

主审独立后评收据：/home/richar/projects/Video-WM/diagnostics/zero-mean-yuv420-main-review-20261001/reference_posthoc_check.json，SHA6947cb23b33166d012007d329edc9cb3478466298c8e87712484708e2c81f54c，核对的runner SHA287767088d4e12fbf5640bda56576985d65ecacdbf50d5ca4f84bfb0c90743a6与当前完全一致。12份同run已审raw的rank/unique/top/canonical/invented_edit/truecost/固定对手差12/12一致；只调用后评，不做infer/转换/VAE。正臂Q8 gain0.8271081983、orth2.5498871981、sign894/1360；SOURCE gain−0.2011725711、orth6.8873331015、sign695/1360。这是既有参考字段的独立一致性证据，不是新raw420结果；A1未重算。
