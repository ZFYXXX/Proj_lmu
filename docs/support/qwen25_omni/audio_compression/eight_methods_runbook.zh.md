# Qwen2.5-Omni 八方法按需验证

> 类型：Runbook；适用范围：当前 Dev 环境中的算法、真实模型与训练诊断。

按本次改动选择验证职责，不必先跑完整 unit suite。每个命令独立执行，失败立即退出，
日志和真实报告保存在 `VALIDATION_ROOT`；不再维护前置关卡、PASS 标记或恢复 session。

| 需要发现的错误 | 命令 | 何时运行 |
| --- | --- | --- |
| 算法数值、预算、顺序、原生 CPU 接入 | `oracle METHOD` | 对应算法或执行位置改动 |
| 真实 checkpoint 输入、压缩与生成 | `gpu TARGET` | 对应基座/执行机制/依赖改动，精确支持重现 |
| 实际 loss、反向和音频值路径梯度 | `backward METHOD` | 训练路径改动 |
| 上述三种责任一起验证 | `check TARGET` / `check all` | 对相应支持范围作完整开发复核 |
| Swift 原生启动或单步训练 | `plan METHOD` / `swift METHOD` | 配置或 Swift 接入改动 |
| 质量与耗时、显存、Token 统计 | `quality` / `profile` | 实际研究比较 |

`METHOD`：`segmentwise | atome | ltbm | sa_map | encoder_topk | progressive |
audio_pyramid | spade`。`TARGET`：`segmentwise | atome | extensions | llm`。

## 准备

在已解析的 Dev workspace 和对应环境中执行。先确认模型目录、可用显存和 Python
包导入路径；GPU 由当前运行环境选择，脚本只要求看到一张卡。

```bash
export QWEN25_OMNI_MODEL_PATH=/path/to/Qwen2.5-Omni-3B
export CUDA_VISIBLE_DEVICES=1
export VALIDATION_ROOT="$PWD/outputs/omni8_$(date +%Y%m%d_%H%M%S)"
export VALIDATOR="$PWD/scripts/diagnostics/run_qwen25_omni_eight_method_validation.sh"
export PYTHONUNBUFFERED=1
nvidia-smi
python -c "import mllm_comp; print(mllm_comp.__file__)"
```

仅运行 `oracle` 不需要模型或 GPU。外部训练/评测依赖有问题时，单独运行
`python scripts/diagnostics/validate_framework_integrations.py`；该诊断不加载模型。

## 选择检查

```bash
bash "$VALIDATOR" oracle ltbm
bash "$VALIDATOR" check segmentwise
# 需要复核全部八方法时：
bash "$VALIDATOR" check all
```

`check` 顺序为 Oracle → 真实 checkpoint → backward；`check all` 依次执行
Segmentwise、A-ToME、LLM 两方法、四个 Encoder/Projector 扩展。

`gpu llm` 同时验证 Audio-Pyramid 的 prefill/decode/cache/backward，以及 SPADE
的结构改写、backward、export/reload；两者没有重复的 `backward` 命令。
`gpu extensions` 共享一次真实模型加载验证四方法，训练仍按方法选择实际值路径。
训练 forward、prefill 和 decode 的证据不能互相替代。

失败后可以直接重跑对应命令，无须补跑前置阶段。若该 GPU/backward 命令已经产生
报告，请先换一个 `VALIDATION_ROOT`，保留旧证据，也避免跳过的测试读取旧报告后误报成功：

```bash
VALIDATION_PYTEST_FLAGS="-x -vv -s" bash "$VALIDATOR" backward ltbm
```

Oracle 日志位于 `02_oracle/<method>/`；真实推理位于 `03_gpu/`；backward 位于
`04_backward/`。查看 pytest 结果与实际报告；缺依赖/GPU导致的 skip 不算通过。
三种责任都通过只说明当前开发路径可用，不证明论文复现、收敛或性能提升。
各方法的具体范围见[八方法契约](eight_methods_contract.zh.md)。

## Swift、质量与 profile

```bash
export MLLM_COMP_TRAIN_DATASET=/path/to/train_audio.jsonl
bash "$VALIDATOR" plan all
bash "$VALIDATOR" swift atome

export VALIDATION_MAX_NEW_TOKENS=16
bash "$VALIDATOR" quality baseline
bash "$VALIDATOR" quality atome
bash "$VALIDATOR" profile baseline
bash "$VALIDATOR" profile atome
```

训练数据遵循 ms-swift 原生格式；单步 smoke 不代表完整 optimizer/scheduler/RNG
恢复。完整训练与恢复见[训练指南](../../../guides/1_run_training.zh.md)。

质量与 profile 可以独立运行，不要求先完成 backward。两者共用 `max_new_tokens`
和 `use_cache=true`，检查实际 Manifest 的非空有限指标与实际插件配置。profile
另外产生 `efficiency_summary.json` 并验证 timing/memory/token traces。`all` 会先
跑 baseline 再跑八种方法。Segmentwise 可通过 `SEGMENTWISE_KEEP_RATIO`（默认
`0.25`）与 `SEGMENTWISE_RUN_TAG`（默认 `r25`）选定研究配置。

修改代码、模型、数据或生成设置后使用新的 `VALIDATION_ROOT`。比较时核对实际
产物的协议与样本，不能仅凭日志成功或旧目录中的报告声明提升。脚本不再冻结或
恢复环境变量；重新连接后使用本次实验自己的环境设置。
