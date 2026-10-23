# VideoLLM 视频压缩支持与 Qwen3.8 接入

更新：2026-10-12。固定参考为 `VideoLLM-Comp` 的 `compression-algorithm`
提交 `7e3ae876bc6306a4179f389e5f960225cd9e3a79`。

原有三基座的 27 个正式 video binding、54 个 phase 单元保持原状态。新增
`vqtoken.video.qwen3_5`，对应 Qwen3.8 的 projector-level 单视频生成路径；目前只有
`generate` 有真实 checkpoint 单样本 LMMS-Eval smoke 证据，`forward` 仍是 declared，
未将其他视频算法宣称为 Qwen3.8 已支持。该 smoke 只验证生成与 token/layout 接线，
不代表 benchmark 质量或论文复现结果。

## 支持矩阵

每格的“推理／训练”分别对应独立的 `generate`／`forward` SupportUnit；不从 image
变体、其他基座或另一 phase 继承支持。

| 算法 variant | LLaVA-OneVision | Qwen2.5-VL | Qwen3-VL | Qwen3.8-27B |
|---|---|---|---|---|
| `metok.video` | 推理／训练通过 | 推理／训练通过 | 推理／训练通过 | 未迁移 |
| `prunevid.video` | 推理／训练通过 | 推理／训练通过 | 推理／训练通过 | 未迁移 |
| `visionzip.video` | 推理／训练通过 | 推理／训练通过 | 推理／训练通过 | 未迁移 |
| `fastvid.video` | 推理／训练通过 | 推理／训练通过 | 推理／训练通过 | 未迁移 |
| `vqtoken.video` | 推理／训练通过 | 推理／训练通过 | 推理／训练通过 | 单样本推理 smoke 通过／训练未验证 |
| `dytok.video` | 推理／训练通过 | 推理／训练通过 | 推理／训练通过 | 未迁移 |
| `longvu.video` | 推理／训练通过 | 推理／训练通过 | 推理／训练通过 | 未迁移 |
| `llamavid.video` | 推理／训练通过 | 推理／训练通过 | 推理／训练通过 | 未迁移 |
| `dycoke.video` | 推理／训练通过 | 推理／训练通过 | 推理／训练通过 | 未迁移 |

精确证据位于 [evidence](evidence/)；文件名为
`<algorithm>_<backbone>_<generate|forward>.json`。Qwen3.8 smoke 使用本地单视频 JSON task，
固定 VQToken `K=32`，记录 `6270→32` 个视频 token；`smoke_nonempty=100` 仅表示生成非空，
不是准确率。每份 manifest 记录代码版本、
checkpoint、配置、固定样本、命令、产物、压缩数量、布局检查、GPU admission 和峰值。
原始运行产物保留在 manifest 指向的本地或服务器 Dev 运行目录，未作为模型权重提交到 Git。

## 唯一推荐配置

配置真值是 [`config.py`](../../../mllm_comp/compressors/video/config.py)
中的 `CANONICAL_VIDEO_CONFIGS`。空 `config: {}` 会解析成下列完整配置；不同值、
未知字段及非 canonical ablation 不继承这里的支持声明。

| variant | 冻结配置与定义性阶段 |
|---|---|
| METok | `segmentation_method=similarity,num_segments=4,keyframe_method=similarity,keyframe_ratio=0.4,top_k_events=3,distinguish_important_events=true,s1_important=2,s2_important=4,s1_normal=4,s2_normal=8,pool_mode=bilinear,non_keyframe_method=adaptive_pool,keyframe_pool_enable=false,inside_compression=true,pruning_layers=[3,10,19],ratio=0.76`；事件/关键帧池化与三次 decoder pruning。 |
| PruneVid | `tau=0.8,cluster_ratio=0.5,temporal_ratio=0.25,k=7,min_tokens_for_cluster=14,enable_vtp=true,vtp_selected_layer=10,vtp_alpha=0.4`；时间聚类、静态/动态空间压缩与 VTP。 |
| VisionZip | `dominant_num=126,contextual_num=21,attn_layer=-2,hidden_state_layer=-2`；逐时间切片的 dominant selection 与 contextual merge。 |
| FastVID | `retention_ratio=0.10,dyseg_c=8,dyseg_tau=0.9,stprune_d=0.4,dtm_p=4,dtm_beta=0.6`；DySeg、STPrune、DTM。 |
| VQToken | `num_clusters=32,adaptive=false,use_cross_attention=false`；全视频固定 K 聚类。 |
| DyToK | `mode=dynamic,base_compressor=visionzip,base_config={dominant_num:126,contextual_num:21,attn_layer:-2,hidden_state_layer:-2},llm_config={target_layers:[16,17,18,19,20,21,22,23]},upper_limit=196`；独立检测模型注意力与固定总量的动态 VisionZip 分配。 |
| LongVU | `window_size=16,threshold=0.83,max_frames=400,min_frames=1,dino_interp_size=576,vision_hidden_size=1024,query_num=144,connector_depth=3`；冻结 DINOv2 时间选择与可训练 SVA 融合。 |
| LLaMA-VID | `num_query=32,bert_type=qformer,compress_type=mean`；本地 BERT QFormer 文本条件 context token 与 mean-content token，每时间单元输出两个 token。 |
| DyCoKe | `merging_ratio=0.7,decoder_prune_layer=3,kv_drop_ratio=0.8,attention_similarity_threshold=0.9`；TTM 与生成阶段的视觉 KV view pruning。 |

DyCoKe 的生成阶段已验证：层 2 的完整 KV attention 决定视觉子集，层 3 及后续层使用
该稀疏 KV view；相似度达到阈值时复用视觉选择。物理 `DynamicCache` 继续保留完整、
只追加的 K/V，所有非视觉 token（包括新生成 token 和 LLaVA newline）保持可见。
这是对 attention 所用 KV view 的压缩，**不声明物理 KV 存储显存下降**。

## 基座与本地资源

验证 checkpoint 为 LLaVA-OneVision Qwen2 0.5B、Qwen2.5-VL 3B Instruct、
Qwen3-VL 2B Instruct。原验证使用 `mllm-comp-platform-dev` 中的
PyTorch 2.8.0+cu128、Transformers 4.57.3；复核时使用当前 lane 对应的环境。

| 本地辅助资源 | 用途与所有权 |
|---|---|
| SigLIP SO400M patch14 384 | LLaVA 原生 vision tower；METok 的文本/跨基座图像信号及 DyToK detector。冻结辅助资源不进入 trainable compressor checkpoint。 |
| DINOv2 Giant | LongVU 的冻结外部 encoder；provider metadata 键 `longvu_dino_path`。 |
| BERT Base Uncased | LLaMA-VID QFormer 初始化；metadata 键 `llamavid_bert_path`。左截断 256，加入 `[DEC]`，不把 assistant target 用作条件。 |
| LLaVA-OneVision 0.5B detection model | DyToK 的冻结独立 detector，读取 decoder 层 16–23 的视觉注意力。 |

DINOv2 Giant 与 BERT 的原验证版本为：
`AI-ModelScope/dinov2-giant@0c788bb0964e17e4d3a68871e50057c997d3978c`、
`AI-ModelScope/bert-base-uncased@e36409b0e76ac5ebd555fdd3938f523ba67c7c23`。
验证使用本地路径和 `local_files_only=true`。
LongVU/LLaMA-VID 不依赖或声称具有方法专用训练权重。

这里是功能迁移，不是逐层架构复刻：LLaMA-VID provider 复用标准 BERT embeddings
和 encoder FFN，并添加隔层 query-only 视觉 cross-attention；没有复刻参考实现中
独立的 query FFN 和去除 token-type embeddings 的结构。文本条件 context 与
mean-content 两个定义性输出阶段均已实现并验证。

LLaVA 保留原生 32 帧拼接与结构 newline。Qwen2.5 保留原始 `video_grid_thw`、
时间间隔及稀疏 MRoPE；Qwen3 同步 MRoPE、三个 DeepStack 分支与
`visual_pos_masks`。不把不规则压缩结果伪装成新的密集 grid。

固定验证中 Qwen 视频均为 1 FPS：通常 `max_pixels=128*(2*patch_size)^2`；
LongVU 为支持 12×12 SVA query raster，使用
`min_pixels=256*(2*patch_size)^2,max_pixels=320*(2*patch_size)^2`。
每个压缩运行与其 baseline 使用相同预处理。该输入尺寸差异不是算法 ablation，
也不构成跨算法效率比较。

## 最小训练与保存/重载

固定推理 doc IDs 是 `3849/442/28/205`。训练 fixture 只由测试 doc `442`
（video `3064389881`、答案 `D`）派生，所有单元都使用相同问题、选项和 target，
batch=1。这是代码正确性检查，不能视为训练集或 benchmark 结果。

历史证据中，DyCoKe、LongVU、LLaMA-VID 单元采用三步，其余单元采用五步。
每步要求实际压缩、有限 loss、backward、非零梯度和参数更新；
最后一步保存，重新加载原始基座和轻量 checkpoint 后再做压缩 forward。
轻量 checkpoint 不包含 optimizer、scheduler 或 RNG 状态，不能作为完整 Trainer
resume 的证据；Swift 恢复由独立的真实 Trainer 验证承担。

训练自由的算法仅更新基座 final norm；LongVU/LLaMA-VID 还训练完整 provider-owned
压缩模块，分别验证 89/268 个参数张量的非零梯度和更新。外部 DINO、detector、
辅助 SigLIP 保持冻结。LongVU/LLaMA-VID 保存已有 final-norm 轻量路径，以及通过
`compression_state_dict`／`load_compression_state_dict` 严格保存重载的
`compression_model.safetensors`；没有把完整基座或冻结外部 encoder 打包进去。

LongVU、LLaMA-VID 的六个训练单元可通过
[`validate_provider_video_training_checkpoint.py`](../../../scripts/diagnostics/validate_provider_video_training_checkpoint.py)
复查。选择 `--algorithm longvu|llamavid`、`--backbone llava_onevision|qwen25_vl|qwen3_vl`，
显式传入本地 checkpoint、auxiliary checkpoint、NExT-QA Parquet/video 目录、代码版本、
设备与独立输出目录。入口固定三步，并执行显存 admission；不得重用已有证据输出目录。

每次 GPU 启动先读取现有进程与剩余显存，显式分配设备并保留 16 GiB 安全余量。
可在有其他进程的卡上运行；两个独立任务只在不同 GPU 和不同日志/输出目录运行。

## 验证边界与历史审查

本迁移不复现论文质量或效率，不运行完整 NExT-QA，不声明收敛、benchmark 提升或
无损压缩。只支持这里的 canonical 配置、batch=1、单样本单视频，不覆盖音视频联合压缩、
多视频、长期训练及其他 ablation。现有 image/text 和 ImageLLM 支持单元保持独立。

2026-09-07 的迁移审查以 `7e40131` 为候选，`c0d2dbc` 修复 DyCoKe 对非视频批量
请求的隔离问题，受影响部分复查通过。该版本的服务器 unit/contract 结果为
`1842 passed, 1 skipped`，本地为 `1465 passed, 223 skipped`；跳过不计为通过。
收口时 54 份 manifest 对应的原始 report 均已核对，GPU 运行版本以各 manifest 为准。
这些是历史验证结论，不是当前提交的全量回归结果；后续改动与验证见[项目进度](../../progress.md)。


## 证据与诊断代码的维护边界

`mllm_comp/compressors/video/evidence.py` 逐项列出 56 个支持单元的证据类别、
源码提交、记录日期和产物校验引用。详细历史数据仅维护在本目录 `evidence/` 的
对应 JSON 中；registry 不再复制历史 token 数、梯度、训练损失或检查次数。
注册发现不读取这些产物文件，安装包中的 Python 模块即可恢复精确支持状态。
删除某个单元的引用会使该单元恢复 declared；缺少必要证据时不能声明 active 或训练已验证。
这些证据状态不作为研究运行门禁；执行仍需满足实际模型、phase、配置和布局约束。
测试保留精确身份、证据完整性与 JSON 校验，算法数值和基座布局测试继续独立执行。

文档中的 `validate_*video*_checkpoint` 命令和参数继续有效。公共实现分工为：

- `scripts/diagnostics/video_training_checkpoint_common.py`：标准训练参数、入口成功/失败报告、
  teacher-forcing 状态、模型加载与 fresh reload 共用加载约束、前向/反向/optimizer 更新，
  以及原有的参数证据和轻量 checkpoint 保存/重载机制。
- `scripts/diagnostics/qwen_video_checkpoint_common.py`：Qwen 视频 placeholder、M-RoPE、
  completion-only label 检查和 Qwen3 加载；其他算法不再借用 VQToken 训练入口的这些工具。
- `scripts/diagnostics/fastvid_video_checkpoint_common.py`：真实 FastVID 输入 raster、信号形状、
  有限值和 canonical 配置检查。

各命令仍保留算法阶段、基座布局、信号提供器和重载后行为检查；多层、DeepStack、
独立 detector 或 provider-owned trainable 模块的差异不通过通用开关抹平。
运行记录现在表示当前请求，累计调用计数继续用于验证触发次数；完整轨迹见 Manifest 引用。

诊断按错误责任维护，不再把 kernel 的分配、聚类、池化公式重写成报告检查，也不要求
四条自然视频再次穷尽确定性单元测试已覆盖的算法分支。算法数值与梯度由 kernel 和
原生小模型的实际张量测试负责；真实 checkpoint 诊断继续观测目标命中、processor
grid/timing、MRoPE/newline/DeepStack、正确 phase 的缓存、有限输出和训练参数更新。
各样本检查后不再复制一份 `defining_path_coverage`／`algorithm_coverage` 布尔汇总。
完成时标准输出仅给出状态、`report.json` 路径和简短统计；完整实际证据保留在报告文件，
不在日志中重复打印数组。训练中的逐步 loss 和更新摘要继续输出。
使用稳定布局的训练命令只在首步记录完整 `native_layout`，后续步骤仍实际检查布局、
loss、梯度和更新；fresh reload 仍核对恢复权重及实际压缩 forward。

日常运行独立数学、公共生命周期和少量诊断成功/失败门禁；执行机制或基座接入变化时
运行对应原生小模型及真实 checkpoint 分支。发布支持或排查具体单元时，使用这里的
精确 CLI、资源和配置复现该单元。历史 54 份 manifest 保留原始结果，当前诊断精简
不表示所有单元都在新版本重新运行，也不改变其历史来源。
