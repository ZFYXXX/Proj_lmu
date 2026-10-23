# LLaVA-OneVision 原生多模态与压缩适配层

本目录选择仓库内固定的 source-tree 路线，而不是运行时猜测不同
Transformers/LLaVA 分叉的内部结构：

- vendored commit：`ee14dbdde72ebb106ad22a2fd31ea9488f0ab97a`；

- 入口：`third_party/backbones/LLaVA-NeXT/llava/model/language_model/llava_qwen.py`
  的 `LlavaQwenForCausalLM`；
- 多模态拼接：`llava/model/llava_arch.py` 的
  `LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal`；
- 视觉塔：`llava/model/multimodal_encoder/siglip_encoder.py` 的
  `SigLipVisionTower` 和 `SigLipEncoderLayer`。

`image_compression.py` 定义了版本化、可机器检查的 source bridge。Provider
必须显式发出四个物理回调，适配层不会 monkeypatch 整个 AnyRes 函数：

1. `vision_block_post_attention_residual_pre_mlp`：Encoder ToMe；
2. `vision_encoder_output_pre_anyres_pack`：ToMe dense restore；
3. `post_anyres_pack_pre_llm_splice`：广义 Projector VisionZip；
4. `llm_decoder_layer_output_image_tokens`：LLM PyramidDrop。

关键前置条件也属于协议：每个 AnyRes crop 的 encoder 状态必须隔离；
VisionZip payload 必须带 packed-layout、`image_newline` 索引、最终视觉 attention
与 keys、原始 `image_sizes` 和 `modalities`；PyramidDrop 必须由 decoder loop
调用。压缩执行根据 `kept_indices` 同步修复 attention mask、position ids 和图像/newline
索引；生成路径还需处理各层 KV cache，其支持证据按方法和 phase 分别维护。

这里的 fake-model 测试只形成 CPU contract evidence，bridge 实例的
`runtime_metadata()` 仍以 `real_checkpoint_verified=false` 表示“这次 CPU 实例本身
没有产生真实 checkpoint 证据”。它不是全局 capability 状态。项目已经在独立真实
checkpoint smoke 中验证 LLaVA-OneVision 的 ToMe、VisionZip 和 PyramidDrop，并由
多模态 registry 将三个方法的精确 image/forward SupportUnit 标记为 active，验证范围为
`batch_size=1`、`use_cache=False`。
PyramidDrop 的 image/generate 单元也已为 active；ToMe、VisionZip 的 generate 单元
仍为 declared，不能从 forward 证据外推。上述方法是本目录的参考切片，其他方法的
generate/KV cache、一步训练和完整训练范围见
[`docs/support/imagellm/status.zh.md`](../../../docs/support/imagellm/status.zh.md)。

新增[14 组对比方法](../../../docs/support/imagellm/comparison_methods.zh.md)也绑定本基座，
保留 AnyRes crop/unpad/newline 语义；其 CPU 原生小模型证据与历史图像矩阵独立。
[MWOP（Mask 形式）](../../../docs/guides/fixed_mask.zh.md)提供另一条头、路径与 FFN
遮罩 Operation 路径。这些新增精确单元保持 `declared`，组合限制分别由方法说明声明。

## Baseline facade

`create_llava_onevision_backbone()` 提供同一路线的最小 baseline：共享一个延迟
加载器，通过 `llava.model.builder.load_pretrained_model` 取得 tokenizer、model、
image processor 和 context length，再用 `tokenizer_image_token`、`process_images`
以及固定的本地 video decode 路线构造 source-native 输入。Facade 支持单样本
`text/image/single-video -> text`，但不支持音频、单样本多视频或 image/video
混合。视频验证配置固定均匀采样 32 帧、stride-2 bilinear spatial pooling 和一个
`<image>` placeholder；原始 `image_sizes`、native `modalities`、帧索引、时间戳、
AnyRes 参数与 newline 语义都会进入 `ForwardState.modality_metadata`。

## Video projector-to-LLM splice

`video_compression.py` 定义独立的
`video_projected_features_pre_llm_splice` provider boundary。算法只接收投影并
pool 后的 patch token；`grid`、`frame` 或 `one_token` 模式产生的
`image_newline` 是不可压缩结构 token。Legalizer 按
`VideoCompressionResult.representative_indices` 将压缩 patch 与原 newline
重新交织，并同步压缩 attention mask 和 ignored labels、重建一维 position IDs。
Identity 与 exact-budget 的 framework-neutral packing/ordering 测试可在无 PyTorch
的控制环境运行；真实 tensor、checkpoint、训练和各算法证据仍由精确 video
SupportUnit 门槛管理，不能从 facade capability 或 ImageLLM evidence 推导激活。

九个 video 变体在本基座已有独立真实 checkpoint 推理与最小训练/重载记录，
具体范围见[VideoLLM 状态](../../../docs/support/video_compression/status.zh.md)。
