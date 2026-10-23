# MMSU-Final Silent Redundancy 分析

本目录实现四阶段实验，目标不是把“低变化”直接等同于冗余，而是区分：

- 低变化、低 attention，删除后无明显损伤的真实冗余；
- 低变化、高 attention，删除后损伤更大的任务相关信息。

Qwen2.5-Omni audio encoder 没有 CLS token。本分析在 encoder 层 3、6、9 使用所有 query/head 的 mean received self-attention；Qwen token 是覆盖全部 Mel 频率的一维时间 token，因此图中使用垂直时间带，而不是 AST 的二维 patch。

## 四个阶段

### 阶段 1：无压缩观察

完整 MMSU-Final 每题只生成一次。observer 从模型实际 `input_features` 计算每个 encoder token 的：

- 五帧 Mel 感受野平均值 `intensity`；
- 五帧时频标准差 `variation`；
- 感受野内部相邻帧平均绝对差 `temporal_change`；
- encoder 层 3、6、9 的 mean received attention。

相关、分位数和二维 rank histogram 均在原生约 2 秒 attention chunk 内计算，避免长短 chunk 之间直接比较。全库只保存充分统计量；预先固定的示例额外保存 Mel、逐 token 分数和虚拟保留 mask。

### 阶段 2：观察性可视化

`report/observational/` 输出：

- 每层、每种声学特征的 rank heatmap；
- 声学五分位中的 attention Top-K 保留率；
- correlation CSV 和 bootstrap 区间；
- 固定示例的五面板时间图，明确画出 chunk 边界和垂直时间 token。

这一阶段只能建立关联，不能证明低变化区域包含关键任务信息。

### 阶段 3：等预算真实 pruning

对最终目标保留率 75%、50%、25%，在层 3、6、9 渐进执行五种 selector：

| selector | 排序依据 |
| --- | --- |
| `attention` | mean received self-attention |
| `intensity` | 对齐 log-Mel 平均值 |
| `variation` | 对齐 log-Mel 标准差 |
| `temporal_change` | 对齐时间差分 |
| `random` | 每题固定种子的随机分数 |

每层 keep ratio 固定为 0.915、0.80、0.64，对应最终约 75%、50%、25%；chunk 取整会让实际比例略有差异。声学分数在第一次删除后按被保留 token 的原始时间 provenance 继续传播。真正的 token 删除、顺序恢复、`cu_seqlens` 更新、pooling 和 projector 均由平台原生 `encoder_attention_topk_audio` 执行。

### 阶段 4：匹配因果删除

在第 9 层单次删除每个 chunk 约 10% token，比较：

- 低变化区域中的高-attention token；
- 低变化区域中的低-attention token；
- 高变化区域中与第一组 attention 值贪心匹配的 token；
- 低变化区域随机 token。

四组使用完全相同的 per-chunk 删除数量。最终报告给出 overall、MMSU 官方 2×3 六组、correct→wrong / wrong→correct、配对 bootstrap 95% 区间和实际 token 比例。

只有当低变化/高-attention象限确实存在，而且删除该组比匹配对照造成更大损伤时，才支持“低变化区域仍保留关键信息”。只有低变化/低-attention组删除近似无损时，才将其称为 redundancy。

## 运行

要求服务器环境包含 Transformers 4.57.3、项目原有 lmms-eval 环境、完整基础或 merged LoRA checkpoint。只含 adapter 的目录不能直接使用。

四卡完整服务器命令见 [TERMINAL_COMMANDS_4GPU.md](TERMINAL_COMMANDS_4GPU.md)，其中包含与你当前 HPC 路径一致的环境、8 题 smoke test、完整 5,000 题运行和分阶段恢复命令。

四卡简化入口：

```bash
cd /home/LuBowen/code/MLLM-Comp-Platform
export CHECKPOINT_PATH=/absolute/path/to/qwen25_omni_checkpoint
export RUN_DIR="$PWD/runs/analysis/silent_redundancy_full_$(date +%Y%m%d_%H%M%S)"
bash scripts/analysis/silent_redundancy_mmsu/run_mmsufinal_4gpu.sh
```

双卡完整实验：

```bash
cd /home/LuBowen/code/MLLM-Comp-Platform
export CHECKPOINT_PATH=/absolute/path/to/qwen25_omni_checkpoint
export RUN_DIR="$PWD/runs/analysis/silent_redundancy_full_$(date +%Y%m%d_%H%M%S)"
bash scripts/analysis/silent_redundancy_mmsu/run_mmsufinal_2gpu.sh
```

先做每卡至少一题的小规模端到端检查：

```bash
# 四卡必须 limit >= 4；完整套件每题共 20 次生成。
bash scripts/analysis/silent_redundancy_mmsu/run_mmsufinal_4gpu.sh --limit 8
```

分阶段运行或恢复：

```bash
bash scripts/analysis/silent_redundancy_mmsu/run_mmsufinal_4gpu.sh --phase observe
bash scripts/analysis/silent_redundancy_mmsu/run_mmsufinal_4gpu.sh --phase compare
bash scripts/analysis/silent_redundancy_mmsu/run_mmsufinal_4gpu.sh --phase causal
bash scripts/analysis/silent_redundancy_mmsu/run_mmsufinal_4gpu.sh --phase report
```

后续阶段必须复用同一个 `RUN_DIR`。已存在且 `coverage.json` 标记完整的条件会被跳过；不完整目录不会被覆盖，应先保留日志并把该条件目录改名后重跑。完整套件为每题 1 次 observation + 15 次等预算比较 + 4 次因果删除，即 5,000 题共 100,000 次生成。

如果任务 YAML 的数据路径不可用，可设置 `MMSU_JSONL` 指向 MMSU-Final JSONL；音频路径解析和原官方 scorer 仍由 lmms-eval 负责。

## 主要输出

```text
RUN_DIR/
├── observe/                         # 阶段 1 原始记录
├── conditions/
│   ├── attention_keep50/            # 阶段 3 条件之一
│   ├── temporal_change_keep50/
│   └── causal_low_change_.../       # 阶段 4 条件
├── report/observational/            # 阶段 2 图表
├── report/final/                    # 阶段 3/4 汇总
└── suite_plan.json
```

关键表格为：

- `report/observational/correlations.csv`；
- `report/final/condition_accuracy.csv`；
- `report/final/six_group_accuracy.csv`；
- `report/final/paired_outcomes.csv`；
- `report/final/summary.json`。

## 本地静态/CPU 检查

```bash
python -m pytest -q scripts/analysis/silent_redundancy_mmsu/test_silent_redundancy.py
bash -n scripts/analysis/silent_redundancy_mmsu/run_mmsufinal_4gpu.sh
bash -n scripts/analysis/silent_redundancy_mmsu/run_mmsufinal_2gpu.sh
```

CPU 测试只验证时间对齐、chunk 内稳定 Top-K、观察统计和因果匹配，不替代服务器上的真实 3B/7B、FlashAttention 2 和多 GPU smoke test。
