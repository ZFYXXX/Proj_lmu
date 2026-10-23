# 压缩方法支持范围

## 查询实现与证据

论文位置、执行机制和验证证据分别回答不同的问题：

| 问题 | 查看哪里 | 能得出的结论 |
| --- | --- | --- |
| 论文在哪里减少 Token 或计算？ | 下方论文位置分类与论文说明 | 算法的原始语义；不表示当前基座已经实现 |
| 当前配置是否有执行路径？ | `discover` 的 `executor_available`、`targets`、`required_handles`，以及真实任务的 `--plan_only` | 存在执行器且静态配置满足契约；模型运行仍需实际验证 |
| 已经验证到什么程度？ | 精确 Binding/模态/phase 的 `support` | 已通过的检查、缺失的证据及已有测试来源 |

```bash
mllm-comp discover --backbone qwen25_omni --modality audio --phase forward
mllm-comp discover --backbone qwen25_vl --modality image --phase generate
mllm-comp discover --backbone qwen3_vl --modality video --phase generate
mllm-comp discover --backbone qwen25_omni --method operation_compression --phase prepare
```

结果由注册的 Algorithm/Variant/Binding 和 `SupportUnit` 投影生成；本文不再手工复制
Active 数量、phase 清单或完整支持矩阵。可按 `--name`、`--variant` 和 `--binding`
缩小范围；研究模块使用与运行配置相同的 `--extension package.module:register`。
Python 入口为 `mllm_comp.experiment_tools.discover_capabilities()`。

`support` 的 `status`、`passing_evidence`、`missing_evidence` 和 `evidence` 说明历史验证范围；
`inference_active`、`train_step_verified` 与 `train_verified` 各自表示相应证据，不能互相替代。
没有 SupportUnit 的研究 Binding 仍可被发现，`support` 为空；缺少证据本身不阻止合法研究运行。
未实现的机制、未知配置、模型结构不满足或真实执行冲突仍会拒绝。

一次实验中改变位置、层号、自定义规则或组合方式，会改变历史证据的适用范围。
可执行不代表新位置已获得标准配方的质量、训练或效率结论；多个单方法证据也不能证明组合。

## 基座验证记录

- [Qwen2.5-Omni 状态](qwen25_omni/status.zh.md)：原生输入输出、音频压缩和训练/恢复的验证条件。
- [ImageLLM 状态](imagellm/status.zh.md)：图像方法的基座适配、限制和证据出处。
- [OV 新增对比方法](imagellm/comparison_methods.zh.md)：14 组、17 个 ID 的三基座接入与独立验证边界。
- [VideoLLM 状态](video_compression/status.zh.md)：独立 `.video` variant、时空对齐和 decode 证据。
- [MWOP（Mask 形式）](../guides/fixed_mask.zh.md)：双基座图像/文本 Operation 的 CPU 证据与 GPU 验证边界。
- [Qwen 分模态模块替换](../../mllm_comp/modality_split/README.md)：图像分支 Attention/FFN 与独立缓存、保存恢复合同。
- [外部框架验证手册](integration/validation_runbook.zh.md)：lmms-eval 与 ms-swift 的验证操作。

状态页保留具体 checkpoint、数据、运行条件与已知限制；精确支持状态以 `discover` 返回的代码声明为准。

## ImageLLM 论文位置分类

下表保留既有论文方法的分类。2026-10-02 新合入的 Bifocal 当前仅接入 Qwen2.5-VL 的
Projector-Level 单图 forward/generate；实现位置不作为其完整论文覆盖声明。两个精确
SupportUnit 仍为 `declared`，专用配置与测试范围见 [Bifocal 接入](imagellm/status.zh.md#bifocal-接入)。

本表严格区分两种位置：

- **论文压缩位置**：论文真正减少 Token 数量或跳过计算的位置；表中的 `●` 只表示这一项；
- **实现 target**：框架安装 hook 的物理位置。结构恢复、位置修复、注意力校准和跨 Backbone
  适配仍会出现在 binding 中，但不因此成为新的论文压缩位置。

因此，`AlgorithmDescriptor.supported_placements` 表示当前实现能安装在哪些位置，不能直接
用作论文分类。`—` 表示论文没有在该位置定义核心压缩；它不表示框架一定没有辅助 hook。

<table>
  <thead>
    <tr>
      <th rowspan="2">方法</th>
      <th rowspan="2">发表</th>
      <th rowspan="2">Input-Level</th>
      <th colspan="2">Encoder-Level</th>
      <th rowspan="2">Projector-Level</th>
      <th colspan="2">LLM-Level</th>
      <th rowspan="2">论文分类与适配备注</th>
    </tr>
    <tr>
      <th>Token</th>
      <th>Operation</th>
      <th>Token</th>
      <th>Operation</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><a href="https://arxiv.org/abs/2412.03248">AIM</a></td><td>ICCV'25</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">●</td><td align="center">●</td><td align="center">—</td><td>部分实现：缺少 LLM pruning</td></tr>
    <tr><td><a href="https://arxiv.org/abs/2505.22038">BTP</a></td><td>NeurIPS'25</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">●</td><td align="center">—</td><td>—</td></tr>
    <tr><td><a href="https://arxiv.org/abs/2502.11494">DART</a></td><td>EMNLP'25</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">●</td><td align="center">—</td><td>—</td></tr>
    <tr><td><a href="https://arxiv.org/abs/2512.19443">D2Pruner</a></td><td>AAAI'26</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">●</td><td align="center">—</td><td>Declared：需外部 calibrated prior，未有真实模型矩阵证据</td></tr>
    <tr><td><a href="https://arxiv.org/abs/2503.02175">DivPrune</a></td><td>CVPR'25</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">●</td><td align="center">—</td><td align="center">—</td><td>—</td></tr>
    <tr><td><a href="https://arxiv.org/abs/2403.06764">FastV</a></td><td>ECCV'24</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">●</td><td align="center">—</td><td>—</td></tr>
    <tr><td><a href="https://arxiv.org/abs/2412.13180">FEATHER</a></td><td>ICCV'25</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">●</td><td align="center">—</td><td>—</td></tr>
    <tr><td><a href="https://github.com/PKU-ICST-MIPL/FinePruner_TIP2026">FinePruner</a></td><td>IEEE TIP'26</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">●</td><td align="center">—</td><td>—</td></tr>
    <tr><td><a href="https://arxiv.org/abs/2506.21873">GAP</a></td><td>arXiv'25</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">●*</td><td align="center">—</td><td align="center">—</td><td>平台组合*</td></tr>
    <tr><td><a href="https://arxiv.org/abs/2604.16462">HalfV</a></td><td>ACL'26</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">●</td><td align="center">●</td><td>部分实现：缺少 LLaVA operation path</td></tr>
    <tr><td><a href="https://arxiv.org/abs/2602.03060">IVC-Prune</a></td><td>ICLR'26</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">●</td><td align="center">—</td><td>—</td></tr>
    <tr><td><a href="https://arxiv.org/abs/2508.18264">MMTok</a></td><td>ICLR'26</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">●</td><td align="center">—</td><td align="center">—</td><td>—</td></tr>
    <tr><td><a href="https://arxiv.org/abs/2602.02951">Nüwa</a></td><td>arXiv'26</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">●</td><td align="center">●</td><td align="center">—</td><td>—</td></tr>
    <tr><td><a href="https://arxiv.org/abs/2504.08966">PACT</a></td><td>CVPR'25</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">●</td><td align="center">—</td><td>—</td></tr>
    <tr><td><a href="https://arxiv.org/abs/2403.15388">PruMerge</a></td><td>ICCV'25</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">●</td><td align="center">—</td><td align="center">—</td><td>—</td></tr>
    <tr><td><a href="https://arxiv.org/abs/2410.17247">PyramidDrop</a></td><td>CVPR'25</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">●</td><td align="center">—</td><td>—</td></tr>
    <tr><td><a href="https://arxiv.org/abs/2606.01711">RESTORE</a></td><td>ICML'26</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">●</td><td align="center">—</td><td align="center">—</td><td>对齐；LLM calibration 为辅助步骤</td></tr>
    <tr><td><a href="https://arxiv.org/abs/2503.11549">SAINT</a></td><td>arXiv'25</td><td align="center">—</td><td align="center">●</td><td align="center">—</td><td align="center">—</td><td align="center">●</td><td align="center">—</td><td>部分实现：缺少 LLM-only path</td></tr>
    <tr><td><a href="https://arxiv.org/abs/2412.03324">SGL</a></td><td>CVPR'25</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">●</td><td align="center">—</td><td>—</td></tr>
    <tr><td><a href="https://arxiv.org/abs/2410.04417">SparseVLM</a></td><td>ICML'25</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">●</td><td align="center">—</td><td>—</td></tr>
    <tr><td><a href="https://arxiv.org/abs/2210.09461">ToMe</a></td><td>ICLR'23</td><td align="center">—</td><td align="center">●</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td>对齐；另有框架输出 compact</td></tr>
    <tr><td><a href="https://arxiv.org/abs/2507.20630">TransPrune</a></td><td>CVPR'26</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">●</td><td align="center">—</td><td>—</td></tr>
    <tr><td><a href="https://arxiv.org/abs/2412.04467">VisionZip</a></td><td>CVPR'25</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">●</td><td align="center">—</td><td align="center">—</td><td>—</td></tr>
    <tr><td><a href="https://arxiv.org/abs/2412.01818">VisPruner</a></td><td>ICCV'25</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">●</td><td align="center">—</td><td align="center">—</td><td>—</td></tr>
    <tr><td><a href="https://arxiv.org/abs/2509.24837">ZOO-Prune</a></td><td>CVPR'26</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">●</td><td align="center">—</td><td align="center">—</td><td>—</td></tr>
  </tbody>
</table>

以下论文位置与框架适配需要分别理解：

- **ToMe**：论文只在 ViT / Encoder 内逐层合并 Token。当前实现先执行 Encoder merge，
  为满足各 Backbone 的结构约束临时恢复 dense layout，随后再做一次输出 compact；后者是框架
  适配，因此不再把 ToMe 标成 Projector-Level。
- **AIM**：论文包含 LLM 前的 Token merging 和 LLM 内的 progressive pruning。当前实现只有
  前半段，所以表中保留论文的两个位置，同时明确标为“部分实现”。
- **SAINT**：论文支持 ViT-only、LLM-only 和 Hybrid。当前 `qwen25_vl` 实现 Encoder 路径，
  `llava_onevision` 与 `qwen3_vl` 使用 pre-LLM 适配路径，尚未实现 LLM-only pruning。
- **HalfV**：论文对不同架构采用不同的第二阶段：Qwen 路径继续减少 LLM 内视觉 Token；
  LLaVA / Vicuna 路径停止视觉 Token 的 Attention / FFN 更新。当前三个 Backbone 都复用了
  Qwen Token 路径，尚缺 LLaVA 的 Operation Compression 路径。
- **RESTORE**：Token reduction 与 distinctive-anchor merging 发生在 LLM 前。LLM 内的
  attention calibration 用于补偿 token mass 和 RoPE 距离失真，不减少 Token，也不跳过或
  删除计算模块，因此不属于 LLM-Level Token / Operation Compression。
- **GAP**：论文把 GAP 定义为已有 pruning 方法的 position-ID 修复插件，并没有提出唯一的
  Token 选择指标。`●*` 表示论文实验中的 pruning 发生在 LLM 前；当前 plugin 将一个平台
  baseline selector 与 GAP 的原始位置保留策略组合在一起，不能把 selector 也归因于论文。

D2Pruner 已实现外部 calibrated prior 的加载与校验，尚未提供校准流程；研究运行条件和
正式支持证据分别见 [D2Pruner 边界](imagellm/status.zh.md#d2pruner-边界)。
每个 Backbone 的精确 target site、variant、phase 和验证证据由 `discover` 返回，验证记录见 [ImageLLM 状态页](imagellm/status.zh.md)。

## 新增图像对比方法

[OV 对比方法接入](imagellm/comparison_methods.zh.md)包含 14 组、17 个 ID、51 个三基座 binding：
7 个 Token ID 使用 Projector/LLM 的 `forward`（训练）或 `generate`（评测）；
10 个 Operation ID 使用 `prepare` 安装，实际压缩仍在后续模型调用中发生。
当前范围为 batch 1、单张静态图像、原生稠密模型的 eager/SDPA；精确单元均为 `declared`。
它们不自动继承历史图像矩阵、位置探索或 `operation.rules` 的覆盖，配置与校准要求以接入页为准。

## Operation 与尚未实现的机制

Qwen2.5-Omni 的结构层剪枝和 Qwen3-VL 的 Attention/FFN replacement 都在 `prepare`
阶段改写模型，随后才安装依赖实际层结构的动态压缩。Qwen3-VL replacement 已通过
`compressors/operation_registry.py` 注册 Algorithm、Variant 和 `prepare` Binding，
与 Token 方法共用定义解析和 Session 安装路径。其 capability 声明为 `active`，新注册的
精确 text/image `prepare` SupportUnit 仍为 `declared`，不继承 Token 方法的支持证据。
上述 `.shared_llm` replacement 在 session 结束时恢复原模块，重新加载原始 checkpoint 后需要重新应用配置。

两插件另有 `.image_llm` variant，在 Qwen2.5-VL / Qwen3-VL 上以 `modalities: [image]`
选择图像分支替换，由公共 pipeline 联合编译为物理分模态执行；不能直接调用单个 image
binding 的 executor 来绕过联合编译。原生文本分支与图像分支各自执行，在选中的 Attention
层排除跨模态边，未选择的层仍按原模型执行。精确 `prepare` SupportUnit 仍为 `declared`。
配置、CPU 测试范围与完整模型/分支状态保存恢复见[分模态说明](../../mllm_comp/modality_split/README.md)。
共享模块重建与分模态 sidecar 恢复是不同路径，不互相代替。

MWOP 的 Mask 形式以 `fixed_mask.image_text` 接入 `prepare` Operation，绑定 LLaVA-OneVision
与 Qwen2.5-VL；执行稠密头/路径/FFN 遮罩，精确单元为 `declared`。`fixed_mask` 是该实现
的代码标识。有限组合及验证范围见[MWOP Mask](../guides/fixed_mask.zh.md)，不与分模态替换任意叠加。

```bash
mllm-comp discover --name qwen3_vl_attention_replacement
mllm-comp discover --backbone qwen25_vl --variant qwen3_vl_attention_replacement.image_llm --phase prepare
mllm-comp discover --name fixed_mask --phase prepare
```

上述 Qwen3-VL 方法出现在结果的 `items` 中，可按 backbone、modality 和 `prepare`
phase 查询实际执行器及精确支持证据。`catalog_only` 仅列出尚未进入压缩 Algorithm
注册表的 catalog 条目，不补造 variant、phase 或 SupportUnit；使用精确模态或 phase
过滤器时不会返回这类条目。

Operation 枚举、量化配置或 KV 相关术语的存在不等于实现。是否可执行必须有对应机制与
基座契约；非 identity replacement 也不自动带来质量或效率收益。

## 使用入口

- 配置、运行和比较实验：[运行压缩实验](../guides/1_run_compression_experiment.zh.md)
- 公共 stage 字段：[配置参考](../reference/config.zh.md#compression-stage)
- 新方法与规则接入：[开发压缩方法](../guides/2_dev_compression_method.zh.md)
