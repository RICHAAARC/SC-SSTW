# 零均值 C1：匹配重放的五层传递诊断设计

输入固定run20260930T113335113338Z/source05beeaa423fa4f2cf95d69cff23e4c5eda4c5841，生产HEAD53d03ede。writer、零均值reader、state code、loss、预算、旧结果均不改。本轮本地只开发/CPU fake/静态验证，真实VAE与codec仅留给后续发布绑定后的用户notebook，SOURCE_SHA=None。

## 盘点与必要对照

本地saved有132gz、75npz、6pt、2json、2log；6pt为3terminal和3历史FULL.phase0。Drive原运行另有18mp4和全部60phase tensor（合计63pt）。历史clamped float、量化RGB8、一次source-MP4重编码latent从未保存。继承上一轮94消费输入/15源文件验证，不重跑完整成本或张量核验；本设计只读JSON/目录。

最终主链必须匹配重放：不能拿新decode的float/Q8与历史source/FULL直接当唯一五层，从而把跨执行数值漂移误叫codec因果。每臂由同一已保存terminal只decode一次，形成以下新链：

| 主层 | 输入与保存物 |
|---|---|
| TERMINAL_PREFIX44 | 原terminal完整张量/hash引用；新R44评分 |
| FLOAT_CLAMPED | FP32 decode(z*std+mean)，decoded/2+.5后clamp；保存float，重开后VAE mode encode/normalize |
| RGB8_NO_CODEC | 同一float用原quantize_rgb8_no_codec，保存uint8；重开后float32/255再encode |
| MATCHED_SOURCE_MP4 | 同Q8/255交原io.encode_rgb（其再次量化须逐像素等原Q8），保存新SOURCE；实际readback再encode |
| MATCHED_FULL_RESAVED | 从上述同一次SOURCE readback用原codec二次保存新FULL；实际readback再encode |

所有编码输入均为181×320×512完整帧；normalized均[1,16,46,40,64]。主支持仅regular1..44，共R44，1408投影维；regular45仍保留在完整张量中，不入主评分。固定g0不是phase恢复，不能先裁177帧改变VAE context。terminal原R45/89候选只作历史语境，不混入新R44/174比较。

原FULL.phase0的3tensor/6raw仅历史参考，不作为新主链层。新/旧SOURCE与FULL的6个媒体hash对照单列，可不同；原始floatbuffer不存在，不能推断重解码与原buffer逐字节相同。环境、VAE revision、FP32/归一化与代码身份完整记录。

## 分母、保存和执行

3臂OFF/PAYLOAD_MULTI/OVERLAP_MULTI×5层×2key固定30path raw、30payload reads、30path posthoc和60message评估。每条件3915目录/174有效，全30条件117450目录、5220有效成本、112230结构排除，原infer/full family与全部ties不变。true clock/OKOK/NOPE只在raw保存后加入；重复payload不证明同步必要性。

15个主normalized记录=3terminal引用+12新编码张量。计划3VAE decode、12encode、6新MP4保存、6新MP4读取、0Transformer、无crop矩阵、无四phase重编。payload读出调用原实现，不改FFT或票数。历史3normalized/6raw/6mediahash对照另外计数，不冒充主链成功。

每臂保存float RGB约355,860,480原始字节及uint8约88,965,120字节，3臂合计约1.243GiB，另12个新normalized约86.25MiB（不含容器开销）。这是为避免后续重复decode；原媒体/hash只引用。preclamp只记录超界比例、幅度和平均clamp改变量，逐帧/小chunk归约，不另存原始大buffer。输出逐臂顺序执行并释放内存，所有新文件写独立目录。

VAE薄adapter只复制原decode的mean/std、FP32、cache清理、decode/2+.5/clamp语义并增preclamp统计；共用vae/io和method不改。未clamp前域为映射到RGB的decoded/2+.5；该层仍是decode+映射+clamp+encode的复合边界，不能叫纯VAE编码损失。

## 失败与比较

初始化完整槽后先做可用terminal CPU评分/历史引用，再fresh VAE worker重放。VAE load/decode失败不抹去terminal与历史参考；FLOAT encode失败不阻塞由同float产生的Q8或媒体分支。源层失败只使依赖层缺失，独立臂继续，所有raw/candidate/评估分母保留。结果canonical-first，child终止后重开最新记录；primary、history reference、整体包完整性分开，history缺失不伪称主链failed。

局部投影与每time/block/age保留；固定层内全排名按原infer做，truth只事后比较正确路径与已定赢家，层间同R44统计。主链匹配减少重放混淆，但不自动支持因果外推、一般编辑、FPR、phase恢复或质量PASS。旧完整四相位/crop结果只引用为语境。

## 文件范围和验证

仅新增：2层运行适配（runtime/wan/zero_mean_c1_layered_transport.py、experiments/wan_state_clock/zero_mean_c1_layered_transport_run.py），本config/doc，builder、未绑定notebook、tests及诊断收据。无新的main方法模块，无旧文件修改。复用原成功current-Python/Torch配对probe/依赖/fresh-child/notebook首两行Drive mount与独立输出目录。

必要CPU只用完整布局mock VAE/IO，核preclamp和原语义、同Q8 raster、原infer完整R44 family、全分母及依赖失败不抹成功、blind raw后truth、source/hash匹配及notebook静态。没有真实VAE/codec调用，不重跑原方法矩阵。设计冻结后按主审已授权方向实现，发布和真正用户执行另行完成。
