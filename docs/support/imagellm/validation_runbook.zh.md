# ImageLLM 多算法验证手册

> 适用范围：历史 24 个已激活算法 × 3 个基座的一步压缩感知训练；历史矩阵不含 D2Pruner、Bifocal、14 组新增对比方法、MWOP（Mask 形式）或分模态模块替换

新增 Bifocal 的专用配置和证据边界见[当前状态](status.zh.md#bifocal-接入)。
本手册的 72 个训练案例不作为新增方法或 Operation 的验收结果。

新增对比方法的独立入口见文末第 7 节；不要使用历史 72 案例结果替代新增方法验证。

## 1. 固定输入

```bash
export DATASET=/path/to/EITNLP-Img-Post-dataset/train_10k.jsonl
export QWEN25=/path/to/Qwen2.5-VL-3B-Instruct
export QWEN3=/path/to/Qwen3-VL-2B-Instruct
export LLAVA=/path/to/llava-onevision-qwen2-0.5b-ov
```

运行前确认当前 editable install 指向本次验证的 workspace，并检查目标 GPU 剩余显存；不得终止其他任务。

## 2. 按改动选择检查

日常只运行直接覆盖改动的文件；下面列出共享入口，完整矩阵复核时才需要全部执行。
评分来源、应用位置或重算变化时，另选对应 `test_image_*_positions_cpu.py`，
不以配置检查替代原生小模型验证。选择原则见[测试说明](../../../tests/README.md)。

```bash
python -m pytest -q \
  tests/unit/test_imagellm_registry.py \
  tests/unit/test_imagellm_config_renderer.py \
  tests/unit/test_image_decoder_lifecycle.py \
  tests/unit/test_sgl_image_kernel.py \
  tests/unit/test_ms_swift_integration.py \
  tests/unit/test_qwen25_vl_image_compression_provider_bridge.py \
  tests/unit/test_qwen3_vl_image_compression_provider_bridge.py \
  tests/unit/test_llava_onevision_ee14dbd_provider_bridge.py
```

这些测试覆盖精确支持声明与执行合同、配置生成、共享 LLM 层执行生命周期、SGL teacher forcing、三 provider 和 ms-swift 路由。D2Pruner 当前仍为 declared；新增原生小模型测试不会自动生成真实 checkpoint 的 `TRAIN_STEP` 证据。

可用下列诊断脚本查看当前 phase 的全部精确单元；它只报告 registry，不加载模型：

```bash
python -m scripts.diagnostics.report_imagellm_support --format json
```

## 3. 完整复核：72 个一步训练案例

需要复核整个历史矩阵时使用以下命令；局部改动不要求重跑全部案例。

两张 GPU 可并行运行前两个矩阵，完成后再运行 LLaVA：

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train.verify_imagellm_train_step \
  --backbone qwen25_vl --model_path "$QWEN25" --dataset "$DATASET" \
  --output outputs/imagellm/train_step/qwen25_vl_train_step.json

CUDA_VISIBLE_DEVICES=1 python -m scripts.train.verify_imagellm_train_step \
  --backbone qwen3_vl --model_path "$QWEN3" --dataset "$DATASET" \
  --output outputs/imagellm/train_step/qwen3_vl_train_step.json

CUDA_VISIBLE_DEVICES=0 python -m scripts.train.verify_imagellm_train_step \
  --backbone llava_onevision --model_path "$LLAVA" --dataset "$DATASET" \
  --output outputs/imagellm/train_step/llava_onevision_train_step.json
```

每份结果必须为 `passed: 24`、`failed: 0`。失败案例会继续执行其余方法，便于区分共享错误和单方法错误。

## 4. 配置矩阵

`scripts.train.render_imagellm_config` 从 registry 的 variant/binding 生成配置，不维护第二份算法默认值。
当前覆盖 24 × 3 份 eval 和 23 × 3 份 Swift train 配置；训练配置使用 `forward`，
测试同时检查实际 Swift 启动计划。SGL 只有 generate Binding，生成器拒绝其 train 请求，
不把原生 teacher-forcing 证据当作 Swift 接入证据。
LLaVA-OneVision 训练配置会额外写入仓库内的 `third_party/backbones/LLaVA-NeXT`
作为 `train.args.local_repo_path`，无需复制或另行下载源码树。该配置关闭
`vit_gradient_checkpointing` 但保留 language-model `gradient_checkpointing`，因为 pinned
ToMe 视觉 callback 不能在 Swift 的视觉编码器重放中持有 request-local 状态。

示例：

```bash
python -m scripts.train.render_imagellm_config \
  --algorithm sparsevlm --backbone llava_onevision --mode eval \
  --model_path "$LLAVA" --dataset MME --eval_task mme \
  --output_dir outputs/llava_sparsevlm_eval \
  --output /tmp/llava_sparsevlm_eval.yaml
```

该生成器仍拒绝 D2Pruner，因为它没有纳入已验证配方。用户提供 prior 的研究运行
与此生成器的范围分开，具体要求见[状态页](status.zh.md#d2pruner-边界)。

## 5. 公共训练入口

```bash
export QWEN25_VL_MODEL_PATH="$QWEN25"
export MLLM_COMP_TRAIN_DATASET="$DATASET"
export MLLM_COMP_TRAIN_RUN_DIR=outputs/imagellm/public_train/qwen25_tome
mllm-comp train \
  --config configs/train/imagellm/qwen25_vl_tome_train_step.yaml
```

成功标准：Swift 完成 `global_step=1`，压缩 evidence 同时包含 install、token-reducing forward 和 backward，随后写出 checkpoint 与 Manifest。启动校验检查实际 Binding、phase、模型结构和训练布局约束；缺少历史 `train_step_verified` 不阻止合法研究运行。压缩未触发不能作为成功证据，也不能回退为 baseline。

## 6. 历史矩阵的复核范围

- registry：25 algorithms、75 bindings、78 active inference units、72 train-step verified units；
- 72/72 实机案例通过；
- 141 份生成配置通过 definition validator，其中 69 份训练配置可构造 Swift 启动计划；
- 本次改动涉及的 provider、共享生命周期及 ms-swift 检查无失败；
- D2Pruner 的 6 个单元仍为 declared；
- `train_verified` 仍为 false，不宣称完整训练闭环。

## 7. 新增能力独立验证

新增 comparison 不在上述 renderer 或历史 72 案例范围内。在固定 CPU 模型环境按实际改动
选择 `tests/unit/test_comparison_{registry,projector,decoder,operations,policies,qwen}.py`；
文件名中的花括号只是清单表示，执行时填写具体文件。注册与公开配置检查可先运行：

```bash
python -m pytest -q tests/unit/test_comparison_registry.py
```

模型依赖缺失导致跳过时不能记为通过。原生三基座 CPU 的前向、梯度、缓存、生成、
部分 checkpoint 重放和新增参数完整模型保存恢复已有相关回归；精确范围见
[对比方法说明](comparison_methods.zh.md)。

真实模型验证使用该页三份 DOP-V 配置或对应方法配置，先检查计划，再执行单样本评测和
一步训练，记录实际 Token/Operation 执行、loss/梯度、缓存与保存重载。每次运行使用独立
输出目录，并按方法核对校准资产。尚未完成这些新增真实模型验证；不得将历史矩阵计数
或 378 项 CPU 结果记作此次 GPU/完整训练/质量证据。

Bifocal、[MWOP Mask](../../guides/fixed_mask.zh.md)和
[分模态替换](../../../mllm_comp/modality_split/README.md)各按自己的配置、组合与恢复合同验收。
