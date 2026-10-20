# Card77 — B内历史复现与现Wan观察算子对照

三部分分开：

1. `legacy_saved_reproduction_v1_run`只读旧SyncTube_results.zip中实际21000条保存事件、冻结旧阈值，重放分数/融合/旧TPR和writer-trace .09敏感性；不跑旧GPU、不重新标定。旧framewise时间对齐方法身份和原结果保留；旧TPR不是逐cell归位/缺失段恢复率。
2. `main/tube_state/legacy_reference_v1/`隔离复用旧partition、position-key signature、offset/scale候选打分/排序、fusion；in-memory projection_margin保持原循环/范数/舍入，排除旧文件hash/cache准入。公开reference接口没有writer trace、真消息或真攻击参数输入，固定关闭support；reference_id/shape/keys必须预先声明，keyed参考符号不是消息答案。公开候选grid与历史按clip_length/speed_ratio选分支分开。仅CPU可移植性/合成参考，不声称旧规则已等价迁移Wan。
3. `b_legacy_migration_v1_run`保持现C73载体、0.5强度、rho严格<1/16及B default/A/净32；仅比较clean A/D在RGB8/MP4×full181/local9下的响应。两种载体/信息条件的旧与新结果不能当纯VAE因果对照。

## 用户固定12E包

输入为已完成C73同次保存文件：`B-C73-Identity-V1/fixed-seen-host-v1/media/{P0,S_base,S_ID}/{rgb8.pt,readback.pt}`。producer对两种文件均用torch.save保存uint8[181,320,512,3]，readback来自保存MP4实际解码；C73 ledger各read/quant返回，路径来自实际runtime，不是旧NPY native缓存。用户包按实际可读性/格式处理，不以摘要/版本比较替代。

完整历史E对181帧一次clear-cache、posterior.mode、float32 normalized latent[1,16,46,40,64]，与原C73同适配器/sample模式；取t16/28。局部E对a61/109各取5context+4target，只读末latent，cache前后清理。VAE load1，新full RGB E3/full MP4 E3/local RGB E6，共12E，名义294 encoder块；D/quant/codec/FFT/新writer/candidate/梯度0。全历史一次E提供两个目标，不能按两个E计。

原C73 MP4 local A/D直接复用：P0索引00/03，S_base06/09，S_ID12/15，各key00/key01，共12份JSON；只复制保存y/s/rho/status，不再encode、projection或classify。旧reset来自先前执行run，新full/局部RGB是未来run，记录双方模型/runtime来源，不要求版本/摘要相等，也不将runtime漂移误归为纯history效应。

48匹配向量、2208身份槽=8正/2200负；其中36新向量1656槽（6正/1650负）与12复用552槽（2正/550负）分别报告。每个layer/history分支都是12向量552槽、SID正确钥两个合法身份。所有失败/未定义保留，各arm独立，不把全拒读当成功。码成员不是完整盲路径/精确接缝证书。全部数值/复用先存，numeric marker后才读取角色、先前result/quality/历史writer统计作后评；P/S只作归因，不是盲接收输入。

72个固定账本单元；每调用前持久attempt/STARTED，失败/中断耗槽。后续Run all整个attempt只读，不自动恢复/补跑/扫描。无资源硬上限，无hash/manifest/Git-clean/固定版本门禁；实际读/格式/模型调用错误保留。旧50/55正确cell、400/440正确归位原始位及错归属/虚假覆盖记录单独保留在docs/evidence，当前不重新归位或读载荷。新operator payload BER/全视频coverage/path error未测；同码成员retention增失不冒充BER或完整同步收益。新E不重写媒体，质量引用同C73六表，非新质量实验。

## 可执行入口

    python -m experiments.wan_state_clock.b_legacy_migration_v1_run --config experiments/wan_state_clock/configs/b_legacy_migration_v1.json
    python -m experiments.wan_state_clock.legacy_saved_reproduction_v1_run --archive /available/SyncTube_results.zip --output /local/history-audit
    python -m experiments.wan_state_clock.legacy_reference_v1_run --input-latent /available/framewise_latent.npy --reference-id pre_registered_id --reference-shape 32,4,32,32 --output /local/reference.json

notebooks/b_legacy_migration_v1.ipynb与builder同步；直接Run all固定任务，无模式菜单。首代码单元只有Drive mount，下一单元从随notebook交付的普通可编辑源码文本落成本地文件；没有B64、预期摘要、手动上传ZIP依赖或指向未发布新代码的URL。另交普通portable源码ZIP用于完整CLI/离线编辑。真实12E由用户执行；agent仅旧记录算术与必要CPU/静态，不启动GPU/Colab或写Drive，不commit/push/main。
