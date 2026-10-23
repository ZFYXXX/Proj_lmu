# 运行评测

> 类型：Guide
>
> 文档层级：1 / 平台使用者
>
> 目标结果：通过 lmms-eval 完成一次可追溯的 baseline 或 compressed 评测
>
> 适用范围：当前 eval integration 与 backbone；压缩研究配置按实际执行合同运行，验证状态由精确 SupportUnit 单独说明
>
> 当前状态：Active

`mllm-comp eval` 负责解析平台配置、激活可选压缩 runtime、构造 lmms-eval 原生启动
参数、收集指标并写入 Manifest。dataset、split、prompt、generation 和 metric 的
任务语义由固定版本的 lmms-eval task YAML 拥有；Qwen RefCOCO REC 路径另应用下述显式
坐标评分协议，比较历史结果时须记录其版本。

## 1. 最小评测

从仓库根目录运行 Qwen2.5-Omni MME baseline：

```bash
QWEN25_OMNI_MODEL_PATH=/absolute/path/to/Qwen2.5-Omni-3B \
bash examples/eval/eval_qwen25_omni_3b_mme.sh --sample_limit 1
```

Qwen3-VL 对应入口为：

```bash
QWEN3_VL_MODEL_PATH=/absolute/path/to/Qwen3-VL \
bash examples/eval/eval_qwen3_vl_mme.sh --sample_limit 1
```

先查看启动计划时添加 `--plan_only`。计划通过后仍需执行真实命令，才能证明模型、
数据、平台 bridge 和 lmms-eval 请求链路可用。

## 2. 完整流程

### 选择配置

- 教学 baseline：`examples/eval/`；
- Qwen2.5-Omni MME/MMAU：`configs/eval/lmms_eval/qwen25_omni_*.yaml`；
- ImageLLM forward/generate probe：同目录的 `*_imagellm_*` 配置；
- Qwen2.5-VL Bifocal：[`qwen25_vl_14bench_bifocal.yaml`](../../configs/eval/lmms_eval/qwen25_vl_14bench_bifocal.yaml)，[精确支持仍为 declared](../support/imagellm/status.zh.md#bifocal-接入)；
- 新增三基座 DOP-V：[`configs/comparison/`](../../configs/comparison/)，其余新增方法的 stage 见[接入说明](../support/imagellm/comparison_methods.zh.md)。Token 评测使用 `generate`，Operation 保持 `prepare`；精确单元为 `declared`；
- 14 benchmark 入口：`scripts/eval/lmms_eval/*_14bench_1gpu.sh`。

先用 `--sample_limit 1` 完成最小真实运行，再取消限制执行正式评测。`sample_limit`
改变样本数，只用于链路检查，不能把所得指标当作完整 benchmark 结果。

### Baseline 与 compressed

Baseline 使用 `compression_mode: none` 并记录 `target: baseline`；compressed 运行必须
使用有效的 compression definition，并记录 `target: compressed`。两次运行应保持：

- checkpoint、revision 与 dtype 相同；
- task、split、prompt/chat 路由相同；
- generation kwargs、seed、batch 和样本范围相同；
- lmms-eval 版本和数据环境相同；
- 只改变 compression definition 与对应输出路径。

当前 Qwen2.5-Omni 音频方法的可运行配置位于 `configs/eval/lmms_eval/`，算法边界见
[Qwen2.5-Omni 状态](../support/qwen25_omni/status.zh.md)。ImageLLM 的精确 active
格子见[ImageLLM 状态](../support/imagellm/status.zh.md)。

### 单机多卡

多卡使用 lmms-eval 数据并行，每个 worker 各加载一份模型，不是 tensor parallel：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
bash examples/eval/eval_qwen25_omni_3b_mme.sh \
  --num_processes 2 --main_process_port 29601
```

默认 launcher 为 `accelerate`，可显式选择 `torchrun`。多卡时不要固定
`--device cuda:0`，每个 worker 必须按 local rank 选择设备。

### RefCOCO 定位评分

平台的 Qwen2.5-VL / Qwen3-VL 插件为 RefCOCO、RefCOCO+、RefCOCOg 的 bbox REC
任务安装 `qwen-grounding-v2` 评分。它解析显式 bbox，并按基座确定坐标系：

- Qwen3-VL：将 0–1000 坐标归一化。
- Qwen2.5-VL：按实际处理后图像的宽高归一化；推理桥将预期尺寸与 `image_grid_thw`
  核对，不按原数据集图像尺寸或输出数值大小猜测坐标系。

结果记录 IoU、ACC@0.1/0.3/0.5/0.7/0.9 和 Center_ACC。无法解析或几何无效的框不作为
有效预测；非 REC 任务沿用原任务评分。实现不修改共享 lmms-eval 子模块源码。
Baseline、compressed 与历史结果比较时，应统一此协议和 processor 的像素边界；
坐标评分修复引起的分数变化不代表模型质量提升。
源码与 CPU 测试见[评分实现](../../mllm_comp/integrations/lmms_eval/grounding.py)和
[协议测试](../../tests/unit/test_grounding.py)，不据此宣称完整 benchmark 已重跑。

## 3. 输出与验收

一次通过的评测至少应产生：

- lmms-eval 原生结果和必要的 sample log；
- 平台 `manifest.json`；
- baseline 或 compressed `MetricRecord`；
- checkpoint、task、effective plugin spec 与依赖版本记录；
- compressed 运行中的 stage identity、runtime metadata 和适用 invariant。

Manifest 存在但没有有限数值指标，不算正式评测完成。进程退出成功但 compression
没有实际触发，也不能作为压缩结果。

## 4. 能力边界

- `--plan_only` 不证明模型加载或指标正确；
- 单样本 smoke 不证明 benchmark 质量；
- 多卡只扩大数据吞吐，不解决单卡模型显存不足；
- HPC 上 lmms-eval task YAML 的数据路径可能是有效环境配置，不得被平台同步覆盖；
- 失败重跑使用新的输出目录，避免把部分结果和新运行混合。

需要新增 task 时阅读[开发评测任务](2_dev_evaluation_task.zh.md)。
