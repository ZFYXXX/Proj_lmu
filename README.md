# MLLM-Comp-Platform

> **2026-10-23：当前工作区默认只启用图像 Projector-only 方法。**
> Decoder/Encoder 方法已从默认发现和运行入口隔离，旧配置、补丁和备份可恢复归档。
> EADP、PruneSID、MMTok 的来源、分类、运行配置与验证边界见
> [Projector 工作区说明](docs/guides/projector_only_20261023.zh.md)。
> 下方旧版全模态说明属于历史平台能力；需要审计时使用 `MLLM_COMP_METHOD_PROFILE=full`。

> Accelerating the development of multimodal compression with `MLLM-Comp-Platform`.

多模态压缩研究与实验框架，支持现有图像、视频、音频方法及基座接入。训练使用 ms-swift，评测使用 lmms-eval，CLI/YAML 与 Python 共用实验入口。

按你要完成的任务开始：

| 任务 | 入口 |
| --- | --- |
| 使用已有方法，训练、评测或恢复模型 | [运行压缩实验](docs/guides/1_run_compression_experiment.zh.md) |
| 在 LLaVA-OneVision / Qwen2.5-VL 上使用 MWOP 的头、路径和 FFN 遮罩 | [MWOP（Mask 形式）](docs/guides/fixed_mask.zh.md) |
| 在 Qwen2.5-VL / Qwen3-VL 中按图像、文本分支替换 Attention / FFN | [分模态执行与保存恢复](mllm_comp/modality_split/README.md) |
| 使用 Qwen2.5-VL Bifocal 图像压缩 | [实现范围与配置](docs/support/imagellm/status.zh.md#bifocal-接入) |
| 使用新增 14 组 OV 对比方法及 Qwen 适配 | [三基座实现、配置与验证边界](docs/support/imagellm/comparison_methods.zh.md) |
| 修改评分规则或复用机制增加新算法 | [开发压缩方法](docs/guides/2_dev_compression_method.zh.md)、[可运行的扩展示例](examples/compression_extensions/README.md) |
| 定位失败、修改执行机制或生命周期 | [排查运行时问题](docs/guides/3_troubleshoot_runtime.zh.md) |

先完成对应任务；需要了解内部能力模型时，再查[架构](docs/platform/architecture.zh.md)和[接口规范](docs/platform/interface_spec.zh.md)。

## 目录

- [当前状态](#当前状态)
- [环境安装](#环境安装)
- [核心流程](#核心流程)
- [最小运行](#最小运行)
- [常用运行入口](#常用运行入口)
- [项目目录](#项目目录)
- [能力边界](#能力边界)
- [文档](#文档)


## 当前状态

平台已经接通原生模型、压缩执行、Analysis、lmms-eval 评测和 ms-swift 训练，
CLI 与 Python 共用 `run_experiment()`。方法能安装、某个配置能运行、训练或质量已经
验证，是三种不同的结论。

2026-10-02 主线合入音频研究脚本与比例实验入口、Qwen 分模态模块替换与 RefCOCO
定位评分、Qwen2.5-VL Bifocal 和双基座 MWOP（Mask 形式）。新能力的配置、CPU 测试与真实权重
证据分别记录；Bifocal、分模态替换和 MWOP Mask 的精确 SupportUnit 仍为 `declared`。
合入范围与待完成验证见[近期主线合入](docs/progress.md#2026-10-02-主线合入)。

2026-10-03 主线继续合入 14 组 OV 对比方法，并适配 LLaVA-OneVision、Qwen2.5-VL、
Qwen3-VL，共新增 17 个 algorithm ID、51 个 binding。相关 CPU 回归 378 项通过；
新增精确 SupportUnit 均为 `declared`，真实权重训练/评测及质量、速度仍待验证。
当前四个基座的注册表共 68 个 Algorithm、72 个 Variant、176 个 Binding，包含候选与别名，
不表示 68 篇方法均已验证。实现、配置与限制见[对比方法接入](docs/support/imagellm/comparison_methods.zh.md)。

从实际声明查询当前基座、执行位置和验证证据：

```bash
mllm-comp discover --kind backbone
mllm-comp discover --backbone qwen25_omni --modality audio --phase forward
mllm-comp discover --backbone qwen25_vl --modality image --phase generate
```

`executor_available` 表示存在执行器；`support` 中的精确证据描述已完成的检查。
选择参数或组合后，仍需先用 `--plan_only` 检查配置，再执行真实实验。
支持解释与模型验证记录见[压缩方法支持范围](docs/support/compression_methods.zh.md)；
[项目进度](docs/progress.md)记录里程碑，不作为第二份能力清单。

## 环境安装

下面是当前完整 GPU 环境使用的安装顺序。这套已验证环境使用 Python 3.10.20、
PyTorch 2.8.0 的 CUDA 12.8 build；其他 CUDA 版本不能直接复用这里的 PyTorch 和
`flash-attn` wheel。

### 1. Clone 项目并创建环境

项目包含 Git submodule，首次 clone 时必须一并拉取：

```bash
git clone --recurse-submodules git@github.com:HarrisonWu42/MLLM-Comp-Platform.git
cd MLLM-Comp-Platform

conda create -n mllm-comp-platform python=3.10.20 -y
conda activate mllm-comp-platform
```

如果已经 clone 但缺少 submodule，在仓库根目录执行：

```bash
git submodule update --init --recursive
```

### 2. 安装 PyTorch

```bash
python -m pip install -c constraints/unified-cu128.txt \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
conda install -c conda-forge ffmpeg=7 -y
```

### 3. 安装平台和外部框架

以下命令在仓库根目录执行。源码版本由仓库的 submodule commit 固定；当前基线为
lmms-eval 0.7.2 和 ms-swift 4.4.0，不通过远端分支名选择安装版本。

```bash
python -m pip install -r requirements/unified-cu128.txt \
  -e . \
  -e "./third_party/frameworks/lmms-eval[all]" \
  -e "./third_party/frameworks/ms-swift"
```

[统一环境约束](constraints/unified-cu128.txt)固化已验证环境中模型、训练与媒体处理的
关键兼容链，包括 Transformers、TRL、Datasets、NumPy 和 Protobuf。所有依赖在同一次
解析中应用约束，因此不再依靠先升级再手工回钉的安装顺序。
[安装清单](requirements/unified-cu128.txt)列出这套环境使用的直接依赖；它不是完整的
传递依赖 lock。基础 `pip install -e .` 只安装轻量配置/核心依赖，不等于完整 GPU 环境。

`qwen-omni-utils` 使用 PyPI 包；`flash-attn` 使用与 Python 3.10、PyTorch 2.8 和 CUDA
12.8 匹配的 2.8.3 本地 wheel。替换下面的路径后安装：

```bash
python -m pip install -c constraints/unified-cu128.txt /path/to/flash_attn-2.8.3-*.whl
```

需要 DeepSpeed 时再安装，继续应用同一约束：

```bash
python -m pip install -c constraints/unified-cu128.txt "deepspeed==0.18.9"
```

### 4. 可选：安装 LLaVA-OneVision

只有运行 LLaVA-OneVision 时才需要下面这些依赖。命令仍从仓库根目录执行：

```bash
python -m pip install -c constraints/unified-cu128.txt "einops-exts==0.0.4"
python -m pip install -c constraints/unified-cu128.txt "shortuuid==1.0.13"

LLAVA_ROOT=./third_party/backbones/LLaVA-NeXT
python -m pip install --no-deps -e "$LLAVA_ROOT"
```

LLaVA-NeXT 的上游安装声明使用另一套旧环境，因此只安装源码；其运行依赖沿用上述统一环境，
不允许它重新解析并替换 PyTorch/Transformers。

### 5. 验证安装

```bash
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
mllm-comp --help
python scripts/diagnostics/validate_framework_integrations.py
```

最后一条命令检查 lmms-eval、ms-swift、平台插件和启动参数，不会加载模型或数据集。

## 核心流程

```text
YAML / Python config
  -> run_experiment: 配置、计划与本次运行目录
  -> eval/train integration 或 analysis 样本循环
  -> 加载模型、准备压缩并执行模型 forward/generate
  -> result.json + 原生产物 + 成功 Manifest
```

每个 backbone 由 processor、model、state builder、validator、legalizer 和 runner 组合而成。模型固定能力由 catalog/backbone 实现声明；模型与实验相关的模态参数写在本次运行的 YAML 中；外部框架只负责其原生执行边界。

## 最小运行

安装完成后，第一个真实 checkpoint 建议使用单样本、单层的 Analysis：

```bash
MODEL_PATH=/absolute/path/to/Qwen2.5-VL-3B-Instruct \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/analysis/run_analysis_method.sh representation_dynamics
```

成功后应生成 analysis artifact、`AnalysisRecord` 和 `manifest.json`。脚本保留
`outputs/analysis/methods/<backbone>/<method>/<timestamp>/` 的显式输出协议。

也可以从 Python 运行同一实验核心：

```python
from mllm_comp.config import load_config
from mllm_comp.experiments import run_experiment

config = load_config("examples/analysis/methods/qwen25_vl_representation_dynamics.yaml")
config.backbone.model_path = "/absolute/path/to/Qwen2.5-VL-3B-Instruct"
result = run_experiment(config, task="analyze", device="cuda:0")
print(result.to_dict())
```

`task` 也支持 `eval` 和 `train`，复用对应配方；`plan_only=True` 或 CLI 的
`--plan_only` 只返回计划、不写产物。每次调用使用配置副本与独立运行资源。
普通配置省略 `output` 时默认写入 `runs/<experiment.name>/<run_id>/`；显式
`run_dir` / `--output_dir` 是精确目录，省略 Manifest 路径时采用该目录下的
`manifest.json`。非空运行目录或已有 Manifest 会被拒绝。

正式执行前保存 `effective_config.json` 和状态为 `running` 的 `result.json`；
后者在结束时记录成功或失败，成功实验另写 Manifest。CLI 可能同时输出后端日志，
自动化读取 `result.json` 获取结构化状态与错误。Python、目录覆盖和失败处理的完整
用法见[运行压缩实验](docs/guides/1_run_compression_experiment.zh.md)。

## 常用运行入口

Qwen2.5-Omni 图像/音频训练：

下列训练脚本保留显式 `OUTPUT_DIR` 约定，每次实验需使用新目录。恢复训练的 checkpoint
作为输入指定，新的结果仍写入独立目录。

```bash
MODEL_PATH=/path/to/Qwen2.5-Omni-3B \
DATASET_PATH=/path/to/train_image.jsonl \
OUTPUT_DIR=/path/to/image_sft_output \
bash scripts/train/qwen2_5_omni/train_img_sft.sh

MODEL_PATH=/path/to/Qwen2.5-Omni-3B \
DATASET_PATH=/path/to/train_audio.jsonl \
OUTPUT_DIR=/path/to/audio_sft_output \
bash scripts/train/qwen2_5_omni/train_aud_sft.sh
```

Qwen2.5-Omni 音频压缩训练：

```bash
MODEL_PATH=/path/to/Qwen2.5-Omni-3B \
DATASET_PATH=/path/to/train_audio.jsonl \
OUTPUT_DIR=/path/to/segmentwise_output \
bash scripts/train/qwen2_5_omni/train_aud_sft_segmentwise_topk.sh

MODEL_PATH=/path/to/Qwen2.5-Omni-3B \
DATASET_PATH=/path/to/train_audio.jsonl \
OUTPUT_DIR=/path/to/atome_output \
bash scripts/train/qwen2_5_omni/train_aud_sft_atome.sh

# default 不启用效率或逐层调试采集；需要时显式追加：
# --profile  仅采集效率数据
# --debug    仅采集逐阶段/逐层 Token 统计
```

Qwen2.5-Omni 双卡 MME/MMAU 评测：

```bash
MODEL_PATH=/path/to/checkpoint \
bash scripts/eval/lmms_eval/qwen25_omni_2gpu.sh

MODEL_PATH=/path/to/checkpoint \
bash scripts/eval/lmms_eval/qwen25_omni_2gpu_aud.sh
```

## 项目目录

```text
mllm_comp/    平台核心、backbone、pipeline、analysis 与 integration
configs/      项目级完整可运行配置
scripts/      train/eval 入口、分析与研究实验脚本、可复用诊断工具
examples/     最小 analysis/eval/train 与 compression definition 示例
tests/        unit、contract、integration 与 regression
docs/         当前状态、稳定设计、接口和人工验证手册
third_party/  固定版本的 lmms-eval、ms-swift 与可选 LLaVA-NeXT
```

## 能力边界

- `eval` 使用 lmms-eval，`train` 使用 ms-swift；Analysis 使用同一模型与压缩运行入口。
- lmms-eval 多卡评测是每卡一份模型的数据并行；具体请求协议限制见[外部框架接入](docs/platform/framework_integration.zh.md)。
- Qwen2.5-Omni 的训练目标是 Thinker；Talker 训练与压缩尚未提供。
- 研究方法可通过显式注册复用现有机制。缺少历史支持证据本身不阻止合法研究运行；接口、phase、模型结构、布局和组合约束仍须满足。
- 单步训练、cached generate、保存恢复、完整收敛和质量/效率结论分别需要对应证据；单个方法通过不能推导任意组合通过。

## 文档

从 [文档索引](docs/README.md) 开始阅读。核心入口：

- [Guide 索引](docs/guides/README.md)
- [项目里程碑与 TODO](docs/progress.md)
- [文档总索引](docs/README.md)
- [平台架构](docs/platform/architecture.zh.md)
- [顶层命令设计](docs/platform/top_level_commands.zh.md)
- [核心概念](docs/platform/core_concepts.zh.md)
- [接口规范](docs/platform/interface_spec.zh.md)
- [压缩方法支持范围](docs/support/compression_methods.zh.md)
- [运行压缩实验](docs/guides/1_run_compression_experiment.zh.md)
- [开发压缩方法](docs/guides/2_dev_compression_method.zh.md)
- [Qwen2.5-Omni 文档](docs/support/qwen25_omni/README.md)
- [ImageLLM 多算法状态](docs/support/imagellm/status.zh.md)
- [lmms-eval 与 ms-swift 接入设计](docs/platform/framework_integration.zh.md)
- [量化与现有压缩的联合接入设计](docs/platform/quantization_integration.zh.md)
- [外部框架 Human 验收](docs/support/integration/validation_runbook.zh.md)

项目文档使用中文；代码标识、配置键和必要的技术术语保留英文原名。
