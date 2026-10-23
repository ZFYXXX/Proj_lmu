# 复用执行机制的研究方法

这些文件是可导入的完整研究示例。算法、必要参数与注册在同一个文件中；实验仍走
`pipeline.extensions`、`run_experiment` 和原有训练/评测后端。

第一次使用可从 FirstK 开始：它只保留输入音频的前 K 个 Token。修改
`FirstKStrategy.compress` 中的选择逻辑，修改 `ALGORITHM` 为自己的名字，再在
`first_k_audio_train.yaml` 中填写对应名字、模型路径、数据和压缩比例即可运行。
注册调用保留原预算和配置规则，框架据此生成内部描述；无需编写改名插件类。

| 示例 | 修改算法的位置 | 复用机制 |
| --- | --- | --- |
| [first_k_audio.py](first_k_audio.py) | `FirstKStrategy.compress` 选择时间顺序的前 K 项 | Omni 音频 Projector 的预算、输入和布局处理 |
| [ltbm_audio.py](ltbm_audio.py) | 保留 LTBM 机制、声明自己的方法身份和配置 | 音频分组、合并和重放 |
| [image_selection.py](image_selection.py) | 每次请求的 `select` 返回保留的图像位置 | 三个视觉基座的 LLM Token 选择、值处理及布局修复 |
| [stride_video.py](stride_video.py) | kernel 选择及旁路 `replay` | 视频 Projector 与 Qwen3 DeepStack 接入 |
| [stride_recent_video.py](stride_recent_video.py) | stride Token 选择，以及 decode 时保留最近的视觉 KV 位置 | 与 DyCoKe 共用 Projector、DeepStack 和稀疏 KV 视图机制 |

研究示例不继承原方法的支持证据。每个执行器仍调用它原有的方法协议；替换 kernel
必须同时提供兼容的 strategy 或 executor factory。完整说明见[开发指南](../../docs/guides/2_dev_compression_method.zh.md)。

图像示例返回 `ImageTokenSelection`，只决定当前序列中的内容位置。共享执行代码
保留文本和 LLaVA newline 等受保护位置，生成压缩值与布局，并处理训练重算。
选择发生在直接前向或生成的 prefill；增量 decode 不再次选择图像 Token。

视频两个示例直接提供方法对象，复用现有 executor factory。`stride_recent_video` 的
`new_decode_policy` 为每个 cache 创建自己的决策对象；示例不计算注意力相似度，也不
保留上一请求的历史。它保留全部非视觉 KV，视觉 KV 只改变本次读取视图，物理缓存完整保留。

例如，在已有 Qwen3-VL 视频评测配方中替换 pipeline：

```yaml
pipeline:
  extensions: [examples.compression_extensions.stride_recent_video:register]
  stages:
    - name: video
      algorithm: researcher_stride_recent_video
      placement: projector_level
      execution_phase: generate
      config:
        stride: 4
        offset: 1
        kv_keep_tokens: 2
        decoder_prune_layer: 3
```

这里的 Token 选择发生在视频 Projector 接入时，KV 视图选择发生在增量 decode；
`decoder_prune_layer` 是首次使用稀疏 KV 视图的层，不是首次删除输入 Token 的层。

## 用同一份配置完成训练与恢复

在仓库根目录、已安装的统一环境中运行。模型和数据由使用者提供，数据格式沿用
ms-swift，不增加框架私有数据格式。`PYTHONPATH` 让 console 命令及其 worker 都能导入
仓库示例；自己的研究包已安装到环境时无需这项示例路径设置：

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export QWEN25_OMNI_MODEL_PATH=/absolute/path/to/Qwen2.5-Omni-3B
export MLLM_COMP_TRAIN_DATASET=/absolute/path/to/audio_train.jsonl
mllm-comp train --config examples/compression_extensions/first_k_audio_train.yaml --plan_only
mllm-comp train --config examples/compression_extensions/first_k_audio_train.yaml
```

计划中查看 `output.effective_config` 与 `output.plan`；正式运行后从 `result.json`
取得实际 checkpoint 和运行目录。示例只设置两步 full 训练，训练规模、冻结范围和
设备选择通过原生 Swift 参数调整。独立新算法的数学在 `FirstKStrategy.compress` 中修改。

在上述环境中启动 Python，使用相同入口；`PYTHONPATH` 也传给原生 Swift worker：

```python
from mllm_comp.config import load_config
from mllm_comp.experiments import restore_config, run_experiment

config = load_config("examples/compression_extensions/first_k_audio_train.yaml")
plan = run_experiment(config, task="train", plan_only=True)
print(plan.to_dict())
trained = run_experiment(config, task="train")
if trained.status != "succeeded":
    raise RuntimeError(trained.error)

continued = restore_config(trained.paths.run_dir, task="train")
continued.train.max_steps = 4  # 总步数，不是额外再训练四步
resumed = run_experiment(continued, task="train")

inference = restore_config(trained.paths.run_dir, task="eval")
inference.eval.tasks = ["mmau_test_mini"]
inference.eval.sample_limit = 1
result = run_experiment(inference, task="eval")
print(result.to_dict())
```

恢复会保留方法注册，重建所需执行阶段，读取完整权重，并为后续实验选择新目录。
音频训练配置中的直接 `forward` 与评测的 `generate` 由恢复入口按目标任务选择。
`train.max_steps` 必须大于已有 checkpoint 的总步数；PEFT 或仅诊断参数产物的边界见
[恢复指南](../../docs/guides/1_run_compression_experiment.zh.md#从运行恢复)。

对照已有算法时，把配方的 `algorithm` 改为 `segmentwise_topk_audio`，继续使用同一
模型、数据及预算；实验结果保存在不同目录。比较质量需要同一评测协议，见
[比较运行](../../docs/guides/1_run_compression_experiment.zh.md#比较已完成的运行)。

仓库外使用时，把所需示例复制到自己的可导入包，并将 `pipeline.extensions` 改为
`my_research.methods:register`。原生后端 worker 也必须能导入这个包。示例不要求修改
`mllm_comp` 或导入 `tests`。
