# 图像 Projector-only 工作区（2026-10-23）

## 当前范围与分类

仅修改和归档 `/home/ZhouFengyu/fastv-repro/` 内、属于当前用户的文件。
不修改其他用户目录、不移动模型或数据、不更改训练 checkpoint 和历史输出。

分类有三个维度，不能混用：论文完整方法的位置、当前实现实际执行的位置、该
binding 是否通过真实权重验证。`projector_level` 表示 LLM 输入之前的压缩边界，
不表示训练或替换 Qwen 原生 merger。仍使用原生动态分辨率。

| 分组 | 方法/组件 | 当前处理 |
| --- | --- | --- |
| 当前研究方法 | feature_anchor_mean、grid_mean、feature_anchor_adapter（已训练 MLP） | 保留；训练状态单独标注 |
| 免训练选择 | EADP、PruneSID、MMTok、DivPrune、CDPruner、HiPrune、OTPrune、VisPruner、ZOO-Prune | 保留已声明的 Projector binding；不是所有基座均已验证 |
| 选择/聚合 | VisionZip、PruMerge（含 plus 配置） | 保留；Qwen 移植不等同 LLaVA 原位置/原论文结果 |
| 部分方法/平台控制 | AIM 输入合并、Bifocal 当前实现、RESTORE 的可分离输入组件、GAP 平台选择器、DPPruner | 保留实际纯 Projector binding，但不当作独立完整论文复现；DPPruner 是消融 |
| Decoder | FastV、PyramidDrop、SparseVLM、DART、PACT、FEATHER、IVC-Prune、BTP、HalfV、TransPrune、SGL、D2Pruner、FinePruner 等 | 默认 registry 中禁用 |
| Encoder/混合 | ToMe、SAINT，以及含 Decoder 目标的 Nüwa/其他 binding | 默认 registry 中禁用 |

当前默认：19 个 algorithm、21 个 variant、52 个 binding；其中 Qwen2.5-VL 有
17 个 binding。数量是工程声明，不是已验证论文数。以 `discover` 为准。

默认 `MLLM_COMP_METHOD_PROFILE=projector`。过滤检查实际 binding 的全部 target，
排除 Encoder 论文路线、非图像及非 Token 压缩。纯 Decoder/Encoder 原代码仍可能被
共享 provider、声明或旧测试导入，不能粗暴删掉，否则整个框架无法导入。
它们已退出当前运行范围，而不是宣称完成物理代码剥离。

```bash
python -m mllm_comp.cli.main discover --backbone qwen25_vl --modality image --phase generate
MLLM_COMP_METHOD_PROFILE=full python -m mllm_comp.cli.main discover
```

`full` 仅恢复历史 registry，不会自动恢复已归档的配置。不要把混合方法的完整
论文结果标成 Projector-only；输入组件的结果必须带 `input-only/partial` 标签。

## 三个重点对照

| 方法 | 官方来源 | 当前接入 |
| --- | --- | --- |
| EADP | `/home/ZhouFengyu/fastv-repro/research_sources/EADP`，commit `e1a08801461c181871b9ff1cb0803b8e9966f083` | 复用并核对现有选择器，Qwen 默认 alpha=.5、beta=2；支持真实矩形网格 |
| PruneSID | `/home/ZhouFengyu/fastv-repro/research_sources/PruneSID`，commit `4ef17d96b390027f31243bc7e1d27bdee77792a5` | 新增 PSCA/PCA＋组内 NMS＋配额选择，Qwen2.5 原生 forward/generate |
| MMTok | `/home/ZhouFengyu/fastv-repro/MMTok`，现有官方快照 `8e2686836ed5bbe05d064f1d1c20e50c73c70b94` | 复用内核、补齐 Qwen2.5 generate 和 decode telemetry；对齐 Qwen tau_t=.01、pre-merger 2×2 均值及 `Question:` 用户文本 |

保留旧 `/home/ZhouFengyu/fastv-repro/PruneSID` 快照，不覆盖本地兼容修改，也不
把旧独立 `trainfree-repro-20261020` 的循环当成新的统一框架评测入口。

正式比较默认 `K_i=min(128,N_i)`；N_i 为本图实际 post-merger 视觉 Token 数。
不会重复 Token、固定图像分辨率或引入跨层平均预算。PruneSID 可通过现有
`retention_ratio` 接口显式测试比例，但这是另一个实验，不会自动改变固定 K。

必须披露的适配：

- EADP 论文 Qwen 表固定 1008×1008；这里保留原生矩形动态分辨率。
- PruneSID 官方 Qwen2 使用 `max(round(N*need_token_num/576),16)`，且小预算
  可能按语义组重排序；这里采用明确固定/比例预算、原始顺序和原始 M-RoPE。
  PCA 使用局部固定 seed、不污染模型随机状态；小输入会限制 PCA 秩，并防止重复索引。
- MMTok 原版以比例 .2 为默认；固定 K128 属预算适配。仅支持本轮单图、batch=1。
- 不据 CPU 小模型或选择器 parity 宣称复现了论文质量/速度。

原始文本位置、视觉位置、mask、labels 和 cache 在压缩后同步；首个 decode 的
M-RoPE 必须是 compact prefill 最大坐标加一，错误时显式失败。选择不读取答案。

## 统一运行入口与验证

生成器：`scripts/eval/lmms_eval/prepare_projector_trainfree_configs.py`。
生成的 9 份 YAML 位于 `configs/eval/lmms_eval/projector_trainfree_20261023/`，包含
三个方法各自 identity、smoke、full。配置与来源 commit 记录在同目录 protocol.json。
全部通过平台 `eval` 入口，沿用本地 ChartQA/AI2D/OCRBench/DocVQA 任务定义，
没有新增独立数据循环、没有启动训练。

```bash
python -m mllm_comp.cli.main eval --config configs/eval/lmms_eval/projector_trainfree_20261023/smoke_mmtok.yaml --plan_only
```

正式 GPU 顺序：同配置 dense/identity → 每任务 8 条 smoke → 全量。此次只完成
CPU 和执行计划检查，尚未启动新 7B GPU smoke/全量。GPU 有其他用户进程，未抢占。
配置仍引用既有共享 7B checkpoint 路径，仅用于未来只读加载，不更改该目录。

CPU 验证覆盖：

- 三个选择器对本地官方源码的数值/选择集合 parity（PruneSID K32/64/128）。
- PruneSID 确定性、随机状态隔离、精确数量、无重复和退化/非有限值输入。
- 原生 Transformers 小 Qwen forward/generate、KV cache 与首个 decode 坐标。
- identity logits/generation 一致、uninstall 恢复、19 方法默认范围与 full 恢复。

完整遗留接口测试需要显式 `MLLM_COMP_METHOD_PROFILE=full`；历史不带该变量的
Decoder/Encoder 配置现在失败是预期行为，不应为通过旧测试重新默认启用它们。

## 已执行的可恢复清理

归档位置：`/home/ZhouFengyu/fastv-repro/archive/projector_cleanup_20261023/`。

- 70 份非当前 Projector 范围配置（包括 Decoder/Encoder 和其他非当前图像入口）。
- 18 份旧补丁/传输压缩包。
- 17 份 `qwen38-*-prechange-*` 改动前备份。

共 105 项，均保留原内容；没有永久删除。修改前的框架文件还单独保存为
`framework_before_changes.tar.gz`。`moves.json` 逐项记录原位置、归档位置和状态。
数据集、环境、所有模型、训练产物、历史结果与源码仓库均保留。

恢复先 dry-run，确认无覆盖再执行：

```bash
python scripts/diagnostics/organize_projector_workspace.py --restore /home/ZhouFengyu/fastv-repro/archive/projector_cleanup_20261023/moves.json
python scripts/diagnostics/organize_projector_workspace.py --restore /home/ZhouFengyu/fastv-repro/archive/projector_cleanup_20261023/moves.json --apply
```

脚本拒绝越界路径、符号链接、其他用户所有的文件以及覆盖已有恢复目标。
归档不是释放磁盘空间；本次目标是减少工作目录混乱，并保留可恢复性。
