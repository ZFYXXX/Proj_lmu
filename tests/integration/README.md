# 集成测试

CPU 小模型测试使用真实模型组件，检查布局、位置、cache、梯度和计划重放；真实
checkpoint smoke 使用目标权重验证完整模型路径。两者都不替代质量评测或训练收敛实验。

## CPU 小模型

使用项目固定的 PyTorch/Transformers 环境，按本次修改选择相关文件。例如音频执行族：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m pytest -q \
  tests/integration/test_qwen25_omni_audio_compression_cpu.py \
  tests/integration/test_qwen25_omni_llm_token_cpu.py
```

图像位置测试为 `test_image_*_positions_cpu.py`，视频位置测试为
`test_video_*_positions_cpu.py`，规则测试为 `test_*_rules_cpu.py`。部分文件包含显式要求
两个设备的用例，在 CPU 环境下跳过；这类跳过不代表双设备路径已验证。

原生小模型承担正常压缩执行、prefill/decode 和 GC 梯度对照；unit 测试保留独立数学、
索引/权重、布局和故障注入。不要为同一正常执行再搭一套完整假模型。参数重载、optimizer
下一步恢复和完整 Trainer 恢复是不同责任，不能用前一种通过来声称后一种已验证。

新增三基座 comparison 的原生小模型测试目前位于 `tests/unit/test_comparison_*.py`，
使用相同的模型依赖；测试覆盖及新增参数保存恢复限制见
[对比方法接入](../../docs/support/imagellm/comparison_methods.zh.md)。这些 CPU 证据不激活真实模型支持单元。

## 真实 checkpoint smoke

运行前按工作区规则确认当前 lane、环境和远端目录，检查 `nvidia-smi`，并把 `GPU_ID`
设为本次分配的物理 GPU 编号。单卡进程内该设备映射为 `cuda:0`。模型使用
[项目固定最小权重](../../docs/progress.md#代表验证权重)，产物写入当前 lane。

先设置实际模型路径并创建输出目录：

```bash
export QWEN25_OMNI_MODEL_PATH=/path/to/Qwen2.5-Omni-3B
export QWEN3_VL_MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct
export CUDA_VISIBLE_DEVICES="${GPU_ID:?Set GPU_ID to the assigned physical GPU}"
mkdir -p outputs/smoke
```

Qwen2.5-Omni 全模态输入、Thinker/Talker 输出和 trace：

```bash
MLLM_COMP_RUN_QWEN25_OMNI_GPU_SMOKE=1 \
QWEN25_OMNI_SMOKE_REPORT="$PWD/outputs/smoke/qwen25_omni.json" \
python -m pytest -q tests/integration/test_qwen25_omni_gpu_smoke.py -s
```

Qwen3-VL 原生 text/image/video、DeepStack 与 text trace：

```bash
MLLM_COMP_RUN_QWEN3_VL_GPU_SMOKE=1 \
QWEN3_VL_SMOKE_REPORT="$PWD/outputs/smoke/qwen3_vl.json" \
python -m pytest -q tests/integration/test_qwen3_vl_gpu_smoke.py -s
```

这个 pytest 入口直接调用同一个诊断流程，保留三模态 native validator 与报告断言。
也可以直接运行 CLI；两条命令无需为同一次验证重复执行：

```bash
python scripts/diagnostics/validate_qwen3_vl_gpu_smoke.py \
  --model_path "$QWEN3_VL_MODEL_PATH" \
  --report_path outputs/smoke/qwen3_vl.json
```

Qwen2.5-Omni Segmentwise Top-K：比较 baseline、100% identity 和 25% 保留预算，并记录
Token 与 timing/compression report：

```bash
MLLM_COMP_RUN_QWEN25_OMNI_AUDIO_COMPRESSION_GPU_SMOKE=1 \
QWEN25_OMNI_AUDIO_COMPRESSION_SMOKE_REPORT="$PWD/outputs/smoke/audio_compression.json" \
python -m pytest -q tests/integration/test_qwen25_omni_audio_compression_gpu_smoke.py -s
```

Audio-Pyramid 的 prefill/decode/backward 与 SPADE 的结构改写、导出和重载：

```bash
MLLM_COMP_RUN_QWEN25_OMNI_LLM_LEVEL_GPU_SMOKE=1 \
QWEN25_OMNI_LLM_LEVEL_SMOKE_REPORT_DIR="$PWD/outputs/smoke/llm_level" \
python -m pytest -q tests/integration/test_qwen25_omni_llm_level_compression_gpu_smoke.py -s
```

音频 Projector smoke 还保留真实 checkpoint 的 CPU/eager 入口，仅用于验证插桩，
不提供 GPU 性能证据：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
MLLM_COMP_CPU_SMOKE_THREADS=8 \
MLLM_COMP_RUN_QWEN25_OMNI_AUDIO_COMPRESSION_CPU_SMOKE=1 \
QWEN25_OMNI_AUDIO_COMPRESSION_CPU_SMOKE_REPORT="$PWD/outputs/smoke/audio_compression_cpu.json" \
python -m pytest -q tests/integration/test_qwen25_omni_audio_compression_gpu_smoke.py::test_qwen25_omni_audio_compression_identity_and_r25_cpu -s
```

当前验证结果与支持边界维护在 [support 状态页](../../docs/README.md#具体能力与验证)，
历史运行报告保留在对应 run artifact 中。
