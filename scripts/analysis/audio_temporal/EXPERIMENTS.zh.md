# 音频时间性质实验：实现细则

## 0. 所有实验共享的表征与时间坐标

每条 MMSU-final 请求保持原音频、问题、候选答案、生成配置和官方 scorer 不变。模型
处于 `eval()`，每条请求只允许一个 packed audio。公共 hook 捕获：

| 名称 | 精确位置 | 回答的问题 |
| --- | --- | --- |
| `C` | 两层卷积与位置编码之后、第一层 audio encoder 之前 | 前端产生了多少局部平滑性 |
| `E_l` | 运行前固定的中间 encoder layer 输出 | 平滑/重复怎样随深度变化 |
| `E_L` | 最后一层 encoder 输出 | 深层语义表征是否仍保留局部时间结构 |
| `P_in` | 原生 `avg_pooler` 和 `ln_post` 之后、`proj` 之前 | 可执行替代与 merge 的统一空间 |
| `P_out` | audio projector 输出 | 进入语言模型前的最终音频 token 性质 |

log-Mel 每帧时间步为 `1000 × hop_length / sampling_rate` ms；`C/E` token 使用卷积
stride 后的实际顺序映射。`P_in/P_out` 记录原生 pooler 所覆盖的 encoder 索引组，并用
组中心作为 token 时间。分析使用物理毫秒 lag，而不是在不同 stage 间复用“第几个 token”
这一不等价尺度。

默认固定五个 encoder 深度（含首层和末层）。可用 `--layers` 明确覆盖，但不得根据
MMSU-final 答案结果再选层。预先固定的 example 文档才保存 float16 隐状态，以免全量输出
不可控；所有音频都保存聚合统计。

## 13. 原生 LTBM 输出 token 的时间来源追踪（`provenance`）

本实验不执行原先实验 6 的五种方法网格。独立启用平台已实现的 LTBM projector-level
stage，默认 `keep_ratio=0.5`、`window_size=8`，只对 MMSU-final 跑一次压缩生成。
压缩过程及 audio placeholder 合法化由现有 runtime 执行；分析目录只在 LTBM hook 前
捕获原始 `P_in`，随后按 runtime 报告的每轮 `merge_pairs` 回放来源，绝不重新选择 pair。
压缩前可能有原生 preflight audio-tower 调用；只以最后一次实际生成调用和压缩报告对齐。

一个当前 token 对应一个 `TokenTrace`：

```text
source_indices     原始 P_in token 索引集合
intervals_ms       归并后的一个或多个时间区间
mass               该 token 代表的原始 token 质量
center_ms          以 mass 加权的时间中心
hard_chunks        来源硬分块集合，可含多个块
nominal_span_ms    最早起点到最晚终点
effective_coverage_ms  各真实区间长度之和
max_gap_ms         离散来源之间最大未覆盖空洞
fragments          连续区间数量
crosses_input_boundary 是否跨输入变化边界代理
```

本实验的向量合并严格沿用原生 LTBM 的每轮等权平均与原生输出顺序；`mass` 只用于追踪
来源数量及计算来源时间中心，不参与 LTBM 向量计算。LTBM 的 local window 使用当前轮
偶/奇分区序号，不保证来源时间连续，也不保证不跨 encoder 硬分块。因此保存跨块数量、
多个离散 `intervals_ms`、碎片数和最大空洞。逐轮配对、每个输出 token 的
`source_indices` 均保存在 `samples.rank*.jsonl`，并校验源 token 无丢失、无重复。

最终图：spectrogram + 原生/合并后 token 覆盖；merge 后跨度、最大空洞、fragment 数分布。判据：
名义跨度小不能代替 `effective_coverage/max_gap/fragments`，也不能被解释为深层感受野。
当前输出是 LTBM 压缩任务分数，不包含同次运行的无压缩配对基线；不能仅凭该分数证明
LTBM 没有任务损伤。

## 1. 多尺度时间相似性（`similarity`）

参考依据：speech representation 的时间局部性分析、Sylber 的 segment consistency 思路，
以及 A-ToMe 用相似性驱动音频 token 合并的基本假设。

### 问题

每个 stage 是否具有时间局部性？相似度从几十毫秒到数秒如何衰减？局部重复在 encoder
深度和 projector 前后怎样变化？

### 设计

1. 默认 lag 为 40/80/160/320/640/1280/2560 ms；每个源 token 在同一硬分块中寻找
   最接近目标物理 lag 的未来 token，容差为该 stage 中位时间步的 0.51 倍。
2. 计算 raw cosine；同时给出在单条音频、单个 stage 内沿 token 轴去均值后的 cosine。
   去均值列只是稳健性描述，不恢复已删除的原实验 2，也不做跨音频/说话人配对。
3. 每条音频先独立聚合，再对唯一 `audio_path` 等权汇总，避免同一音频对应多道题而被
   重复加权。
4. 所有 lag 必须留在原生 hard chunk 内；无合法 pair 的格子记为缺失，而不是 0。

### 输出和判据

- stage × lag raw/de-mean heatmap；
- 每个 stage 的 similarity-decay curve；
- stage token 数对照。

短 lag 高、随 lag 稳定衰减支持时间局部性；仅 raw 高而 de-mean 消失说明公共方向影响大。
无论哪一种，都不能单独推出 token 可删除，必须继续看实验 5 或另行设计配对压缩评测。

## 3. 时间区域特性（`regions`）

参考依据：Audio Transformer token pruning 对低变化区域的观察，以及通用音频结构分析中
对静态段、瞬态和边界分开统计的做法。

### 问题

时间冗余是否主要集中在静音、稳定持续声，还是 transient/boundary 也同样可替代？

### 区域定义

MMSU-final 没有逐帧标签，因此只使用输入 log-Mel，完全不读取隐藏表征或答案：

- `silent`：平滑活动度不高于本音频 20% 分位，且不是极高 flux；
- `stable`：非低活动且 spectral flux 不高于 35% 分位；
- `transient`：非低活动且 flux 高于 80% 分位；
- `boundary`：flux 高于 95% 分位，或原生 hard chunk 边缘；
- `other`：其余帧。

这些名称是结构代理，不是语义事件真值。阈值逐音频计算，适合定位相对变化，但不宜比较
不同音频的绝对声压。

### 观察与干预

先按区域报告 80 ms lag cosine。再对每个目标区域用最佳即时邻居替代固定比例的 `P_in`
token，序列长度不变；替代预算以整条序列为基准，若该区域不足则报告实际预算。区域由
输入决定，邻居 donor 不跨 hard chunk。

最终图：region × stage redundancy heatmap；各区域的配对任务分数变化。只有某区域同时
表现为高相似和低替代损伤，才支持“冗余集中于该区域”。

## 4. 连续时间块可预测性（`predictability`）

参考依据：自监督语音表征中的 masked temporal prediction，以及 Sylber 的段内一致性分析。

### 问题

一个连续时间段能否由其外部左右邻居重建？可预测性随缺失时长如何下降？

### 设计

1. 在每个 hard chunk 内构造 40/80/160/320 ms 连续块；实际 token 宽度由 `P_in` 的
   中位时间步四舍五入，并同时记录有效 token 宽度。
2. 预先固定缺失总预算，随机选择互不重叠候选块；首尾必须各留一个外部上下文 token。
3. 三个无训练 predictor：`copy_left`、左右端点 `linear` 插值、左右端点 `local_mean`。
   相同 block size 的三个 predictor 使用完全相同的缺失块，且与对应任务干预一致。
4. 把预测向量写回原位置，保持 token 数、时间位置、问题和标签不变。
5. 同时报缺失位置的 reconstruction relative-L2/cosine 和完整 MMSU-final 配对分数。

最终图：prediction error–block duration 曲线；condition accuracy/delta 表。短块误差低但长块
快速恶化，说明冗余只存在于局部尺度；误差低且任务不降才支持任务相关的可预测冗余。

## 5. 高相似 token 等长替代（`substitution`）

参考依据：token pruning/merging 工作中用保持长度的 counterfactual replacement 验证
importance 或可替代性的思路。

### 问题

高 cosine 只是几何现象，还是代表 token 在任务中确实可互换？

### 四组条件

- `local_high`：320 ms 窗口内最高 cosine 的非重叠 pair；
- `far_high`：同一 hard chunk 内至少 2,000 ms 的 pair，并把其 cosine 尽量匹配
  `local_high` 的目标分布；
- `random_local`：同一局部候选集的固定种子随机 pair；
- `local_low`：局部最低 cosine pair。

每个 pair 用较早 token 替代较晚 token，序列长度和位置完全不变；每条音频的四个条件
共同采用远程匹配实际可完成的公共替代预算，因而实际修改 token 数也相同。输出 pair
cosine、物理距离、相似度匹配误差、表征变化和逐题配对准确率变化。

核心判据：如果 `local_high` 明显优于随机/低相似，且损伤接近 0，高相似才具有因果
可替代性；如果 `far_high` 在相似度匹配后仍更差，说明相同向量仍有不可忽略的时间身份。

## 7. Local merge 与 global merge（`locality`）

参考依据：Segmentwise Pruning 对时间覆盖的强调，以及 Locality Matters 对受限局部
匹配与全局匹配的直接比较思想。

### 问题

在 pair cosine 和最终 token 数都匹配后，时间局部性本身是否仍影响任务？

### 严格控制

1. 汇总所有不跨 hard chunk 的局部候选，选最高 cosine 的非重叠 pair。
2. 汇总同 hard chunk、距离至少 2,000 ms 的远程候选；按 cosine 最近邻逐 pair 匹配，
   且 donor/receiver 不复用。
3. 两组共同预算取远程可完成匹配的数量，因此每条音频 local/global 的输出 token 数一致。
4. merge 都使用同一个 mass-weighted 算子，唯一主要差异是 pair 时间距离。
5. 保存 similarity-match MAE、实际距离、名义跨度、空洞和 fragment 数。

最终图：local/global 的完整 MMSU-final 曲线；pair distance 与 similarity-match 检查；
span/gap/fragment–degradation 对照。若匹配误差足够小、token 数相同而 global 仍显著更差，
才支持 temporal locality 是独立因素。若远程候选不足，公共预算会降低并明确记录。

## 最重要的最小分析组合

若算力只允许优先完成少数实验，建议顺序为：

1. `similarity`：同时捕获基线表征，建立现象并定位 layer 与毫秒尺度；
2. `provenance`：单独运行原生 LTBM，追踪减长后每个输出 token 的实际时间支持；
3. `substitution`：用等长干预检验可替代性；
4. `locality`：在严格匹配相似度后回答“局部性是否本身重要”。

`regions` 和 `predictability` 用于解释冗余“出现在哪里、能预测多长”，非常适合机制章节；
但若主论文篇幅或预算有限，它们应排在上述主证据链之后，而不是替代因果实验。
