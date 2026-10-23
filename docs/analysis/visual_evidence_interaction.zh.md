# 视觉证据协同验证

研究对象是固定视觉预算下的区域替换交互。即使观察到正交互，也只能直接支持“这些区域在当前背景下共同影响答案”；语义绑定受损、中介 Token 有效、部署加速均需要后续证据。

## 冻结样本与执行

运行目录：`outputs/analysis/visual_evidence_interaction/20261020_v1/`。

- ChartQA 测试集有 2500 题、1509 张 RGB 内容不同的图像；TextVQA 验证集有 5000 题、3166 张不同图像。
- 每任务按图像内容哈希选 500 张，一图一道题。ChartQA 先选，TextVQA 排除已选图像。每任务前 100 张作开发集，其余 400 张作确认集；选择过程不读取答案。
- 每任务开发集中的 8 张用于 smoke。Smoke 数据属于开发集，不能再当作额外独立图像。
- 每任务确认集前 100 张作预算复核；有第二道问题的确认图像按哈希取前 50 张作问题复核。不同问题仍按原始图像聚类。
- 主预算为 128，随机种子为 11、23、47。每种子 37 个替换条件，另有 7 个普通压缩参照；每图主实验共 118 条条件。
- 问题相关区域依据独立的、无答案输入的 dense prompt Layer-0 attention 选择。TextVQA 遵循原任务的首字母大写格式。
- 2×2 Token 块必须完全位于其空间象限；不完整边缘和跨象限块仍可作为背景 Token。范数匹配使用全部合法块的均值/标准差统计，距离为两个维度标准化差的欧氏距离。
- Feature-anchor 原始代表 Token 使用现有 mean kernel 的同一分组及空间代表位置，保留该位置的原始向量；mean 组使用簇均值及簇重心 M-RoPE。这一对照包含表示和位置处理的差异，不能单独归因于均值数值。
- PruMerge+ 使用现有 kernel 的 plus、非自适应、target_count=128、cluster_size=32 设置，记录邻域与全局残差成员。
- 使用 Qwen 原生 decoder 的 KV cache 加速诊断生成；smoke 与 cache-free 路径和 native generate 核对。每个条件显式检查 placeholder、mask、labels、答案预测位置和文本 M-RoPE。

首次准备与启动：

```bash
/home/ZhouFengyu/fastv-repro/mllm-comp-310/bin/python scripts/analysis/prepare_evidence_interaction.py --output outputs/analysis/visual_evidence_interaction/20261020_v1
/home/ZhouFengyu/fastv-repro/mllm-comp-310/bin/python scripts/analysis/run_evidence_interaction.py --root outputs/analysis/visual_evidence_interaction/20261020_v1 --launch
```

样本清单已有时不要再次准备或覆盖。启动器使用文件锁防止重复执行，按 smoke → development → confirmation → question_review → budget_review 顺序运行，每 GPU 一个进程。只管理本实验创建的进程。

每条成功条件独立追加并同步到磁盘；恢复时核对配置、代码和样本哈希。尾部不完整 JSONL 记录会被截去并重算。失败立即使当前分片退出，启动器停止另一个实验分片，保留全部失败记录。相同实现可使用同一启动命令续跑；若代码发生变化，应使用新运行目录，不能混用确认数据。

开发集完成后自动写入 `frozen.json`，记录代码、样本清单和预注册的哈希。确认与复核阶段要求冻结代码及样本一致。开发集耗时给出确认集两卡运行时间的估计；该估计不保证其他 GPU 负载不变。

## 结果与统计

- `cohort.json`、`preregistration.json`：样本和统计定义。
- `samples/<stage>/<sample>/conditions.jsonl`：每条原始预测、得分、保留位置、校验和开销。
- 同目录的 `plan_<budget>_<seed>.json`：候选、对照、共同背景及全部条件。
- `summary.json`、`interactions.json`、`regions.png`：样本完成状态、区域交互和区域图。
- `<stage>_report.json/.md`：完整阶段报告。确认集未完成 800 张前不执行正式确认检验。
- `run_manifest.json`、`logs/`：后台进程、阶段状态、软件/GPU 信息和失败原因。

主终点是完整生成的任务原生得分。对每图的区域对和种子先求平均，再按图像做 10000 次配对 bootstrap。四个预注册单侧检验采用零假设中心化 bootstrap 的 p 值及 Holm 校正：每个任务的问题区域交互是否大于零、是否大于随机区域交互。区间仍包含有意义收益时只报告证据不足，不声称否定所有交互压缩。

ChartQA 同时给出 relaxed score 和严格去首尾空白、忽略大小写的答案匹配。TextVQA 使用原任务对完整人工答案列表的软评分。按 baseline 得分为零、部分得分、满分及 mean 压缩是否保持得分报告子组，子组属于解释性分析。

问题复核在汇总时逐文件验证候选、对照、背景及所有保留集合与原问题完全一致。预算 256 在源 Token 数低于 288 时明确记为不适用；与 K128 的比较只用相同适用图像。

诊断需执行 dense Layer-0 prompt 前向，因此耗时只用于计算预算和执行审计，不能充当可部署压缩器的净加速结果。没有训练新模块，也没有将答案或 Oracle 结果作为部署输入。
