# OV 对比方法接入

本页说明 `OV对比方法.zip` 新增实现的框架接入范围。
实现位于 `mllm_comp/compressors/comparison/`。新增 **14 组、17 个 algorithm ID、51 个三基座 binding**。
CDPruner/DPPruner 合为一组；AdaLLaVA-L/H 和 `adallava` 别名合为一组。

三基座分别为 `llava_onevision`、`qwen25_vl`、`qwen3_vl`。新增 SupportUnit 均为 `declared`：
原生小模型 CPU 前向、梯度、缓存、生成和持久化测试不等于真实 checkpoint 的质量、速度或论文结果复现。
不修改既有方法的验证状态，不把压缩包的历史运行记录移作本框架证据。

## 实现和执行阶段

| 组 | algorithm ID | 实际接入内容 | stage.execution_phase |
| --- | --- | --- | --- |
| CDPruner / DPPruner | `cdpruner`, `dppruner` | 指令相关 DPP / 纯视觉 DPP | `forward` / `generate` |
| EADP | `eadp` | 指令熵筛选、空间平滑、重要性配额和 facility-location 选择 | 同上 |
| HiPrune | `hiprune` | 浅层锚点及邻域、深层注意力补足预算 | 同上 |
| VLM-Pruner | `vlm_pruner` | 原生 K 特征、空间距离选择和带权聚合 | `forward` / `generate` |
| TwigVLM | `twigvlm` | 共享基座初始化 Twig 分支、TTP、FinalWipe | `forward` / `generate` |
| CoreMatching | `core_matching` | prefill 核心神经元统计及 Token 选择，decode 索引 FFN | `forward` / `generate` |
| DOP | `dop` | VisPruner + 深度截断的 DOP-V，以及 depth-only；逐层紧凑 KV | `prepare` |
| ShortV | `shortv` | 选定层保留完整 K/V，跳过视觉 Q/O 与 FFN 更新 | `prepare` |
| Skip-Vision | `skip_vision` | 主导/上下文分组聚合、双可学习摘要、FFN 跳过和 prefill 后 KV 收缩 | `prepare` |
| RedundancyLens | `redundancy_lens` | Hollow Attention，probe 激活通道裁剪 | `prepare` |
| YOPO | `yopo` | 校准 head mask、深层视觉读取截断、可选局部注意力与 FFN mask | `prepare` |
| AdaLLaVA | `adallava_l`, `adallava_h`, `adallava` | latency token、L/H 调度、训练时直通采样、推理计划复用 | `prepare` |
| Short-LVLM | `short_lvlm` | 校准层删除及 SCP 子空间输入投影 | `prepare` |
| Structural-Pruning-LVLM | `structural_pruning` | 校准 block-influence 层策略，prefill/decode 同步跳过 | `prepare` |

`prepare` 表示在模型加载后安装方法；安装后作用于 forward 和生成，不表示只在准备阶段计算一次。
Operation 方法保留原生 decoder loop；Token 方法复用已有的布局、标签和缓存修复，或在原生 decoder 输入边界执行同步选择。

所有新增 binding 限定 **batch size 1、单张静态图像、eager/SDPA 原生稠密模型**，不包含视频、多图、packing、padding-free、MoE 或量化模型。
一次启用一个新增 comparison stage，与既有图像压缩 stage 使用互斥运行槽。

## 基座适配

- **OneVision**：保留真实 AnyRes crop/unpad 布局。CD/DP/EADP/HiPrune/DOP-V 在 crop 上计算选择，再映射到 packed 图像；一般保留 `image_newline`。预算是每 crop 的 patch 数，最终 Token 数会受到 unpad 和换行符影响。VLM-Pruner 与 TwigVLM 的整个视觉区间选择可去除换行符。
- **Qwen2.5-VL**：基于真实 `image_grid_thw` 的 post-merger 矩形网格，保留三轴 MRoPE；HiPrune 使用每层真实 `cu_seqlens` 的窗口注意力，按原生 window permutation 恢复空间顺序。
- **Qwen3-VL**：保留原生 Q/K norm 和 MRoPE。Token 选择、带权聚合和 Skip-Vision 的同一映射同步作用于 DeepStack；Twig 分支沿用相应层的 DeepStack 注入。
- CDPruner/EADP 只提取当前 ChatML user 指令，排除 system、角色标记、图像标记和受监督答案。需要标准 Qwen ChatML 模板。
- Qwen 没有 OV 的多 crop 预算：`tokens_per_crop` 在 Qwen binding 中表示当前单图的 **post-merger Token 总预算**。不能直接把 OV 总 Token 比例解释为 Qwen 的相同比例。
- DOP-V 在 OV 使用指定 SigLIP 层的特征和注意力；Qwen 使用指定原生视觉层的 patch 特征分组及聚合注意力，并在 merger 输出选择 Token。它与论文的 Qwen 输入缩放策略不同。

## 配置

完整双用途训练/评测示例位于 [`configs/comparison/`](../../../configs/comparison/)：
[`llava_onevision_dop_v.yaml`](../../../configs/comparison/llava_onevision_dop_v.yaml)、
[`qwen25_vl_dop_v.yaml`](../../../configs/comparison/qwen25_vl_dop_v.yaml)、
[`qwen3_vl_dop_v.yaml`](../../../configs/comparison/qwen3_vl_dop_v.yaml)。
设置对应的模型路径环境变量、`MLLM_COMP_TRAIN_DATASET` 和 `MLLM_COMP_RUN_DIR` 后，通过通常的 `mllm-comp train/eval --config ...` 使用。
这些是接口冒烟策略，不是经质量校准的推荐预算；尚未用真实 checkpoint 跑完 Swift/lmms-eval。

例如在仓库根目录先检查 Qwen2.5-VL 的双用途配置（不加载模型、不创建运行产物）：

```bash
export QWEN25_VL_MODEL_PATH=/absolute/path/to/Qwen2.5-VL-3B-Instruct
export MLLM_COMP_TRAIN_DATASET=/absolute/path/to/image_train.jsonl
export MLLM_COMP_RUN_DIR=/absolute/path/to/runs/qwen25_dop_eval
python -m mllm_comp.cli.main eval --plan_only --config configs/comparison/qwen25_vl_dop_v.yaml
export MLLM_COMP_RUN_DIR=/absolute/path/to/runs/qwen25_dop_train
python -m mllm_comp.cli.main train --plan_only --config configs/comparison/qwen25_vl_dop_v.yaml
```

真实执行时去掉 `--plan_only`；训练和评测使用不同的新输出目录。OV、Qwen3-VL 分别改用
`LLAVA_ONEVISION_MODEL_PATH`、`QWEN3_VL_MODEL_PATH` 和相应配置。评测只要求该任务
消费的模型与输出变量，训练还需图像训练数据；计划通过不验证权重、数据或校准质量。

精确身份为 `algorithm=<ID>`、`variant=<ID>.image`、`binding=<ID>.image.<backbone>`。
前三组使用 `placement=projector_level`、`method=token_compression`、`object_family=tensor_slot`。
VLM-Pruner、TwigVLM、CoreMatching 使用相同 method/object_family 和 `placement=llm_level`。
其余使用 `placement=llm_level`、`method=operation_compression`、`object_family=module_unit`。

以下是可放入 `pipeline.stages` 的 DOP 配置。深度 1 仅用于接口冒烟，正式实验应加载针对当前权重校准的深度：

```yaml
- name: dop
  enabled: true
  plugin: dop
  algorithm: dop
  variant: dop.image
  binding: dop.image.qwen25_vl
  placement: llm_level
  method: operation_compression
  object_family: module_unit
  execution_phase: prepare
  modalities: [image]
  target_modalities: [image]
  config:
    token_reducer: none
    mha_depth: 1
    mlp_depth: 1
```

Token 方法示例：

```yaml
- name: cdpruner
  plugin: cdpruner
  algorithm: cdpruner
  variant: cdpruner.image
  binding: cdpruner.image.qwen3_vl
  placement: projector_level
  method: token_compression
  object_family: tensor_slot
  execution_phase: forward
  modalities: [image]
  target_modalities: [image]
  config: {tokens_per_crop: 128}
```

训练 Token stage 使用 `forward`；三基座的 Token 方法评估使用 `generate`。
Operation stage 在训练和评估时都保留 `prepare`。

| 方法 | 必要/常用参数 | 约束 |
| --- | --- | --- |
| CD/DP | `tokens_per_crop` | OV 使用 `anyres_assembly: onevision_unpad`、`position_id_mode: reindex` |
| EADP | `tokens_per_crop`, `alpha`, `beta`, `spatial_kernel_size` | Qwen 按真实长宽平滑；支持单行/单列网格 |
| HiPrune | `tokens_per_crop`, `alpha`, `object_layer`, `vision_layer` | object 层为一基索引；vision 层为零基，可用 `-1`；Qwen 默认浅层按视觉编码器深度映射 |
| VLM-Pruner | `prune_after_layer`, `retain_ratio` | 必须保留后续 decoder 层 |
| TwigVLM | `twig_start_layer`, `twig_layer_count`, `average_retention_ratio`, `final_wipe_layer_fraction` | 分支不得超出深度；FinalWipe 位于 TTP 后、末层前 |
| CoreMatching | `prune_after_layer`, `token_core_ratio`, `sentence_core_ratio` | 可选 `minimum_visual_retain_ratio` / `maximum_visual_retain_ratio` 是预算适配；decode 当前要求普通 Linear FFN |
| DOP | `mha_depth`, `mlp_depth`, `token_reducer`, `tokens_per_crop` | `vispruner` 为 DOP-V 默认路径；`none` 仅裁剪运算深度，深度不得超过当前模型 |
| ShortV | `replaced_layers` | 针对当前 decoder 校准，零基索引 |
| Skip-Vision | `attention_visual_retain_ratio`, `ffn_visual_retain_ratio` | 两个摘要模块可训练；小图保留必要组可能达不到名义预算 |
| RedundancyLens | `attention_reduction_layers`, `ffn_reduction_layers`, `attention_range`, `operation_backend` | 支持原 mask 语义及 `indexed_structural`；不等同于稀疏 attention kernel |
| YOPO | `head_mask_source`, `head_mask_path`, `head_prune_start_layer`, `layer_drop_start_layer` | `file` 使用当前 checkpoint 校准 mask；`none` 是明确关闭 head pruning 的消融 |
| AdaLLaVA-L | `num_prefix_layers`, `active_layers`, `target_flops_retain_ratio` | `adallava` 为 L 别名；active 包含固定 prefix |
| AdaLLaVA-H | `num_prefix_layers`, `active_units`, `target_flops_retain_ratio` | GQA 组对齐；原生头数/深度必须匹配；非整除 FFN 通道按连续近等长组分配 |
| Short-LVLM | `pruned_layers`, `candidate_layers`, `projection_pairs`, `calibration_asset_path`, `subspace_rank` | 还需一致的校准 FLOPs 与几何字段，见下文 |
| Structural | `pruned_layers`, `original_layer_count`, `target_flops_retain_ratio` | retain ratio 必须等于保留整层比例，至少保留一层 |

## 权重、校准和复现范围

- **TwigVLM** 固定为 `initialization_only`，没有训练过的 twig checkpoint；不能报告为训练后的论文模型，也没有 self-speculative decoding。
- **Structural-Pruning** 覆盖输入包中的 ShortGPT depth 分支，不包含全部宽度剪枝分支或恢复训练结果。
- **DOP、CoreMatching、Short-LVLM** 沿用输入包注明的论文公式/适配实现，不新增“官方等价复现”声明。
- **YOPO** 保留 `qwen2vl_pre_softmax` 和 `llava_post_softmax` 两种不同语义。框架默认要求 `head_mask_source: file` 及当前 checkpoint 的 `head_mask_path`。包内 `official_qwen2vl_mask14` 只在明确选择时加载，并检查 28×28 几何；相同几何也不等于校准适用于当前权重。可选邻域 mask 需要与实际图像网格匹配的 `neighbor_grid_width`；OV AnyRes 多视图不应当作单一矩形，默认关闭该消融项。
- **Short-LVLM** 需要外部 SCP `.pt`，压缩包没有提供所引用的二进制文件。资产包含 `format: short_lvlm_scp_asset_v1`、`projection_pairs`、`metadata.hidden_size` 和 `bases[str(retained_layer)]`；basis 形状为 `[hidden_size, subspace_rank]`。配置中的 `flops_reference_decoder_layers`、`calibrated_flops_ratio`、`calibrated_absolute_target_error`、`target_flops_ratio` 必须与策略一致。当前用可卸载的输入投影执行 `W(I+VVᵀ)`，不原地改权重；前向等价不构成相同优化轨迹或加速声明。
- AdaLLaVA / Skip-Vision 新增参数随 `model.save_pretrained()` 写出 `comparison_adallava.safetensors` / `comparison_skip_vision.safetensors`，从该目录再次安装时恢复。原生完整 checkpoint 的保存恢复有测试；PEFT/Swift adapter-only checkpoint 的新增参数保存尚无实跑证据，不能直接假定覆盖。
- AdaLLaVA 训练当前要求 `gradient_checkpointing: false`。其他声明有独立检查点重放测试的方法见测试文件；不要把一个方法的验证外推到全部方法。
- FLOPs 公式和输入包中的固定 OV 校准几何仅作源实现参考；Token 保留比例、模拟 FLOPs、物理参数减少和实测延迟是不同指标。

## 验证入口

2026-10-03 在 Dev 环境以 `CUDA_VISIBLE_DEVICES=""` 运行相关回归，**378 项通过**。
范围包括全部新增测试、三基座 provider bridge、MWOP、Swift 生命周期、压缩 session/definition/registry
及 `tests/integration/test_modality_split_cpu.py`；没有占用 GPU 或启动真实权重 benchmark。

新增测试为 `tests/unit/test_comparison_*.py`：DPP 独立参考、原生三基座训练/解码、VLM 带权梯度重放、
矩形网格、Qwen DeepStack、SCP 资产、调度器和摘要模块保存恢复，以及 51 个精确 binding 的公开配置/训练传输检查。
框架训练适配器分别记录 Token 变化和实际运算压缩，Operation 方法不伪造 Token 减少证据。

```powershell
python -m mllm_comp.cli.main discover --kind compressor --name cdpruner
python -m pytest -q tests/unit/test_comparison_registry.py
```

完整原生测试需要项目的 Python 3.10.20、PyTorch、Transformers 4.57.3，以及 OV vendored LLaVA。
真实 checkpoint 的 lmms-eval/ms-swift 全链路、目标数据集指标、吞吐、显存和新增参数的 adapter-only 重载仍需另外验证。
