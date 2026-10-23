# 本次本地验证

## 2026-09-19：SA-MAP 50% 独立入口与 75% 移除

- 修改范围仅限本目录；没有修改平台算法实现、注册表或目录外文件。
- `run_sa_map.sh` 现在依次运行 keep10、keep25、keep50；移除 SA-MAP keep75 的五份 YAML。
- 新增 `run_sa_map_keep50.sh`，仅运行五个 benchmark 的 keep50，复用原有配置，输出仍为 `sa_map/keep50/<task>/`。
- 当前共 7 个 SH、90 份 YAML，其中 SA-MAP 15 份。保留的 SA-MAP 配置仍使用 threshold=0.80。
- 7 个 SH 与 `common.bash` 全部通过 `bash -n`。
- 实际执行平台 CLI `--plan_only`：SA-MAP 默认入口 15 个计划、50% 独立入口 5 个计划，全部通过；已核对任务顺序、档位和输出目录。
- 其他五种方法各选 mmsufinal，累计 15 个计划通过；确认共用调度调整后仍保留各自原档位和输出路径。合计 35 个计划通过。
- 使用占位模型路径，不加载模型、数据或 evaluator。上述检查不代表 7B LoRA 真实评估通过；剩余 SA-MAP 档位仍受分组数量条件限制。

以下为 2026-09-15 的历史验证记录；其中 95 份 YAML、6 个 SH 和 SA-MAP 四档描述是当时的配置，当前范围以上述 2026-09-19 记录为准。

## 2026-09-15：候选执行链路修复与状态晋升

- 修复 UnSeg preflight：按 mask 将 padded `[audio, mel, time]` 整理为 `[mel, valid_time]`；同时覆盖 native mask、仅长度与已拼接输入。
- 新增独立数值与真实随机初始化 Transformers encoder 测试共 45 项，全部通过；连同相关回归共 **97 passed，3 skipped**。三个跳过项均为未启用的真实 checkpoint smoke。
- 验证环境：临时 Python 3.11、PyTorch 2.8.0、Transformers 4.57.3、NumPy 1.26.4；没有使用预训练权重，未验证 CUDA/FA2 或项目锁定的 Python 3.10.20 环境。
- UniAvg、UniSamp、UnSeg 的 forward/generate SupportUnit 已从 `declared` 更新为 `cpu_contract_verified`；catalog 仍为 `target`，开发执行已启用，正式 active 所需的 `real_checkpoint` 证据仍缺失。
- 新增 `test_qwen25_omni_projector_candidates_real_checkpoint`：在单 GPU 加载一次权重，执行 baseline 及七组设置并输出 JSON。此入口尚未实跑，不自动修改注册状态。

以下保留创建 sweep 时的检查记录，其中 declared 状态已被上述 CPU 验证状态替代。

日期：2026-09-15。范围：新增 projector 六方法 sweep 的配置、启动计划与已有算法 CPU 行为；未运行服务器上的 7B checkpoint。

## 配置与入口

- 95 份 YAML 均通过平台 `load_config`、`validate_runnable_config` 和对应 plugin 的 `validate_config`。
- 逐份解析 algorithm、variant、Qwen2.5-Omni binding；确认 projector placement、target site、compressed 模式和 benchmark 对应关系。
- 6 个 SH 与 `common.bash` 均通过 `bash -n`。
- 实际执行六个 SH 的 `--plan_only`：Segmentwise 20、UniAvg 15、UniSamp 15、UnSeg 5、LTBM 20、SA-MAP 20，共 95 个计划。校验了档位/任务顺序及各自独立的输出路径。
- 计划检查使用占位模型路径，不加载模型、judge 或数据，不能据此确认服务器推理可用。

注册表实际查询结果（forward / generate）：

| 方法 | 状态 |
| --- | --- |
| Segmentwise Top-K | active / active |
| UniAvg | declared / declared |
| UniSamp | declared / declared |
| UnSeg | declared / declared |
| LTBM | active / active |
| SA-MAP | active / active |

候选方法能够通过完整 V2 身份生成开发运行计划；本次没有修改这些状态。

## 既有测试与数值检查

运行既有测试：

```bash
python -m pytest -q \
  tests/unit/test_segmentwise_topk.py \
  tests/unit/test_projector_extension_methods.py
```

结果：**33 passed，1 warning**。本地 d2l 环境的 PyTorch/NumPy 桥接出现 `_ARRAY_API not found` 警告；本次测试完成并通过，没有验证 NumPy 张量互转或服务器依赖组合。

额外运行未写入项目测试集的 CPU 数值核验：

- N=1、7、25、100；Segmentwise、LTBM、SA-MAP 分别检查 10%、25%、75%、50%，共 48 个选择/合并案例。
- 同样四种长度，UniAvg、UniSamp 分别检查整数 K=10、4、2，共 24 个池化/采样案例；对照独立窗口均值及 `hidden[::K]`。
- 上述 **72 个案例全部通过**，核验输出长度、有限数值，以及适用的索引顺序或输出值。SA-MAP 档位核验使用可满足预算的正交输入，不代表真实音频已校准 threshold。
- UnSeg 的已知输入产生边界 `[2,4]` 和三个正确段均值；恒定输入输出一个自然段；错误的 preflight keep_count 被拒绝。

实际复现的限制：

1. UniAvg 和 UniSamp 的 spec 均拒绝 `keep_ratio=0.75`，错误要求 `1/K` 且 K 为整数。
2. UnSeg 的 spec 均拒绝 0.10、0.25、0.75、0.50 四个固定 keep_ratio。
3. SA-MAP 对完全相同的 10 个 token、threshold=0.80、目标保留 7 个时，因自然 grouping 仅剩一个组而报错。默认 threshold 不是全数据集四档预算的保证。

## 尚未验证

- 远端代码同步、任务 JSONL / Parquet / 音频缓存路径。
- Qwen2.5-Omni-7B LoRA checkpoint 的实际生成、四 GPU 运行及 MMAU-Pro evaluator。
- UniAvg、UniSamp、UnSeg 的完整真实模型验证与正式支持晋级。
- SA-MAP 在五个目标 benchmark、四档保留率上的 threshold 校准。
- 各方法的任务质量、实际平均保留率、延迟、吞吐量和显存收益。

需要在服务器的小样本运行中检查 Manifest 与 `debug_diagnostic*.jsonl` 的实际压缩报告，再进行完整评估。
