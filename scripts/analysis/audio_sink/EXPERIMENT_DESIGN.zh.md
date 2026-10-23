# MMSU / 7B：全部 encoder 层和 head 的无压缩观测

更新：2026-09-22。本文件描述第一阶段的 **无压缩 observe 模式**：使用准确的完整 7B checkpoint（基础模型或已合并 LoRA）和 mmsufinal，所有 token 均保留。实际权重由 CHECKPOINT_PATH 决定，不根据结果目录中的 lora 字样推断。相似度和虚拟 TopK 索引仅用于记录、统计和绘图，不反馈到网络。

当前入口为 [run_mmsufinal_observe.sh](run_mmsufinal_observe.sh)：全 B 每题只生成一次，观察全部 encoder 层和各层全部 heads；7B 为 L0–L31 × H0–H19，共 640 组合。只有 doc 0/1666/3333 保存每层第 0 个 chunk 的详细表示与全部 head 矩阵。此前五层结果仍只覆盖 L0/7/15/23/31，新增层需要重新采集。

此前的 20–23 干预实验保留在独立 suite 模式，详见 [完整 B attention 干预实验](ATTENTION_EXPERIMENTS.zh.md)，当前默认不执行。suite 可修改 head 输出、V/key 路径及特征，并单独进行真实 token 删除；下面“尚未干预”的描述指当前 observe 模式。

已提供独立运行和绘图代码，位于本目录。原 observe 版本曾用 CPU 合成输入与 Transformers 4.57.3 小型 audio encoder 验证；已有五层真实运行结果不代表后续全层改动或 suite 已验证。本次按要求只做静态检查，没有运行模型、pytest 或绘图，也没有提交全层 MMSU 作业。

## 1. 原论文实际支持什么

核对了原文 HTML 与本地 PDF 的 Figure 2 及相关方法段落：

- [Segmentwise Pruning in Audio-Language Models](https://arxiv.org/html/2511.14293v1)，§3.1、Figure 2、§3.3：在 Audio Flamingo 3 的最后一层 Whisper encoder 上展示一条 10 秒 Clotho-v2 音频的 incoming attention。少数高分位置在时间上相邻，提示只按全局 attention 排名可能缺少时间覆盖。图采用 max-normalization，不能从这张图读出实际总质量占比。原文没有把该现象命名为 attention sink；它也不是 Qwen2.5-Omni-7B LoRA + MMSU 的实证。§5.1 的 Bottom-K captioning 结果同样不支持“低 attention 一定更重要”。
- [Accelerating Transducers through Adjacent Token Merging](https://arxiv.org/html/2306.16009v1)，§2.2、Figure 2(a,b)、§3.2：合并依据是相邻 key 向量的 cosine similarity；Figure 2 的虚线是未合并模型中相邻相似度随深度的变化。可以借鉴这个观测设计，但论文对象是 Transformer transducer / LibriSpeech，不能直接复制其阈值或效果到当前 ALM。

本地原文：
- `/Users/andylubowen/Desktop/EIT-code/paper/compression_review/sources/segmentwise_2511.14293v1.pdf`，PDF p.2–3。
- `/Users/andylubowen/Desktop/EIT-code/paper/compression_review/sources/atome_2306.16009.pdf`，PDF p.2–3。

本次工作将两条动机结合：**相邻表示是否相似，以及高 attention 是否聚集在相似的时间邻域。** 它们分别回答局部冗余候选和时间覆盖偏斜，不能相互替代。

## 2. 研究问题与结论边界

| 问题 | 本次直接测量 | 允许的结论 |
|---|---|---|
| 是否具有时间局部冗余？ | K/V/post-MHSA hidden 的相邻相似度与距离曲线 | 存在局部表示相似性、可考虑局部 merge 的候选 |
| attention 是否集中？ | 每个 head 的 incoming r、质量、熵、query 覆盖 | 存在集中读取或 sink 样候选现象 |
| 高分位置是否聚集在少数时间区域？ | Global TopK 标记的覆盖、最长空缺、相邻比例，对比同预算随机 | attention 排名具有时间集中倾向 |
| 是否固定偏好某种位置？ | 跨音频的 chunk 内峰值位置分布、起点质量 | 位置偏好证据，仍需排除声音分布混杂 |
| 聚集位置是否也相似？ | 对齐 r 与相邻 cosine、原空间 Gram 矩阵 | 集中读取与表示相似性的共现 |
| 是否可无损 merge、对任务无用、是归一化锚点？ | 本次没有实施必要的功能干预 | 不作这些结论 |

对“sink”建议使用操作性表述：高于可见性均匀参考的集中读取，在多个 query 和多条内容不同的音频上反复出现。若只是一条音频中某个事件附近有峰，写“时间集中”更准确。没有跨模型通用的 sink 阈值。

## 3. 固定运行条件

- 使用准确的 7B 完整 checkpoint，不混用基础模型与不同 LoRA 版本；只有 adapter 的目录不能直接作为完整权重加载。
- `mmsufinal` 原任务、原 prompt、原 processor、原生成参数、原字母 parser 与 scorer。
- `pipeline.stages: []`、`eval.compression_mode: none`、batch size 1。
- 当前原配置为 BF16 / FlashAttention 2；observer 保留原 kernel 的输出。
- 本地 MMSU JSONL 为 5,000 条，perception 2,580、reasoning 2,420。5,000 个不同路径不保证 5,000 个独立来源场景。
- 不使用 response cache 跳过 forward；保留模型正常生成所用的 KV cache。
- `MMSUFINAL_DATASET_DIR` 只控制 Parquet/音频根目录。JSONL 路径仍由已有任务 YAML 决定；在服务器沿用已验证的数据路径。
- observer 针对 Transformers 4.57.3 的 audio tower，运行时记录实际版本、来源文件、层数、heads、dtype、attention backend、sample rate、hop length 与 n_window。

## 4. Adjacent Token Similarity 的口径

对 encoder 第 l 层，捕获真实 K 投影，以及 MHSA 残差相加后、FFN 前的 z：

```text
s_K(l,j) = cosine(flatten_heads(K[l,j]), flatten_heads(K[l,j+1]))
s_V(l,j) = cosine(flatten_heads(V[l,j]), flatten_heads(V[l,j+1]))
s_H(l,j) = cosine(z[l,j], z[l,j+1])
```

z 位于 `layer.final_layer_norm` 的输入，正好对应本项目 A-ToMe 插入点的表示；另记录每层 FFN 后相邻相似度。K 的 heads 拼接口径与本项目当前 A-ToMe 相似度实现一致，不等同于先对各 head 的 cosine 求平均。

主统计计算全部重叠相邻对 `(0,1),(1,2),...`；另报本项目不重叠候选对 `(0,1),(2,3),...` 的均值。任何跨实际 attention chunk 的相邻对都标成缺失，不假装它们在同一个可见窗口中。

同时计算同一 chunk 内 lag=1/2/4/8/16/32 的 cosine。主图对比 adjacent 与 lag=16：

- adjacent 明显高于较远位置，支持局部相似性。
- 近处和远处都高，需要考虑共同方向、位置编码或表示趋同。
- K 高但 V/hidden 不高，不足以支持内容也相同。
- 深层相似度趋势是待检验结果，不预设一定单调上升。

所有余弦在 FP32 计算。零范数向量和不存在的距离记为缺失，统计保留有效音频数。主曲线先计算每条音频有效 pair 的均值，再对音频等权；不让长音频拥有更多样本权重。

当前脚本没有加入范数差异、中心化/各向异性分解或真实 merge 效果检验，所以“更适合 merge”仅作为下一步候选判断。

## 5. Attention 的方向和 chunk

当前 Qwen2.5-Omni 的 audio encoder 以实际 `cu_seqlens` 为界，chunk 内双向读取，位置编码在 chunk 内重新开始。按每层每 head 分开处理。

完整双向 chunk 的有效长度为 n，A[i,j] 为 query i 读取 key j：

```text
p[j] = sum_i A[i,j] / n
r[j] = n * p[j] = sum_i A[i,j]
sum_j r[j] = n
```

包含 self-attention。当前脚本验证已去 padding 的 packed chunks 和实际 block mask；遇到额外窗口/causal mask 会停止，避免误用 n 倍归一化。

保存逐 head 的平均 peak r、top10% incoming mass、incoming entropy、query entropy、峰值被多少 query 以高于均匀参考的权重读取、峰值起点权重、chunk 局部位置直方图。短尾块单独保存最后一块的 peak r 和全部 chunk 长度。

另外分别统计第一块第一个 token 的 r 和后续块起点的平均 r，`summary.boundary_comparison` 给出 all-head mean 与 CI；只有一块的音频在后者记为缺失。这使绝对音频开头与重复 chunk 边界可以分别检查。

均匀 attention 的 r=1。纯对角线 attention 的 query entropy 很低，但 incoming 仍均匀，因此不称为列 sink。完全并列的峰值平分位置直方图权重，避免 argmax 总取第一个位置制造虚假的开头 sink。位置直方图、起点权重和 top10% 质量均保留实际 chunk 长度对应的均匀参考；主图画位置超额量，避免短尾块或 top10% 取整伪造偏好。

相邻相似度与 attention 集中是独立轴：允许“高相似低集中”“低相似高集中”等结果，不建立人为综合分数。

## 6. 时间坐标和频谱解释

从真实 processor 读取 sample rate / hop length，从 audio tower 读取卷积 chunk 长度：

```text
每个 chunk 的 Mel 长度 ≤ 2*n_window
卷积后有效 encoder 长度 = ceil(Mel_length / 2)
时间中心 = (chunk_mel_start + 2*index_in_chunk) * hop_length / sample_rate
```

若为 16 kHz / hop=160，encoder 间隔约 20 ms、pooling 后约 40 ms。该时间标签是来源位置；深层表示可以包含整个可见 chunk 的上下文。

示例图保存真正送入 audio tower 的 log-Mel，不重新解码另一份音频。图上频率维标为 Mel bin；token 并不是一个频率 bin。当前代码没有高频能量因果分析，也不能从 log-Mel 总和计算线性高频能量比。此前“是否偏向高频片段”的问题保留为后续声学关联分析，不作为本轮结论。

## 7. 无压缩的虚拟 TopK 标记

保留全部 token 后，仅在记录中模拟索引选择，不调用压缩 controller，也不改变模型 hidden states 或序列长度。

为对应当前 Segmentwise 方法，在最后一层 encoder 上：
1. 先把各 head incoming r 求平均。
2. 按原 avg_pool 的坐标把相邻两个 score 求均值；未配对尾 token 不虚构输出坐标。
3. 用同一 k 比较 global TopK、S=10 的 Segmentwise TopK 与 100 次同预算随机。
4. Segmentwise 直接复用项目的索引函数和 `balanced_exact` 预算规则；这是当前项目适配，原论文使用 floor(K/S)。

报告时间段覆盖率、最大单段候选占比、最长未覆盖区间占比、候选相邻比例；保存 cutoff 并列数量与剩余槽位。原项目按时间索引稳定处理同分，若 cutoff 大量并列，不能把该确定性打破并列的结果当作真实集中证据。

Segmentwise 覆盖更均匀是约束的直接结果，**不能用这一点证明 sink**。研究证据来自原始 incoming 分布，以及 global TopK 相对同预算随机的时间聚集程度。

## 8. 可视化产物

输出 PNG 与 PDF；只对真实采集文件绘图，测试输入仅用于程序/版式验证。

| 图 | 内容 | 用途 |
|---|---|---|
| overview | 相邻与较远相似度随层变化，layer×head 集中程度，chunk 局部峰值位置，虚拟候选覆盖 | 全 B 统计，不依赖单案例 |
| sink_diagnostics | layer×head 的 top10% 超额质量、incoming entropy、峰值跨 query 读取、起点超额权重 | 区分集中强度、读取广度和位置偏好 |
| timeline | 实际 log-Mel、incoming r、相邻 K/V/H cosine、虚拟 global/segmentwise 标记、真实 chunk 边界 | 原论文 Figure 2 的时间对齐扩展 |
| attention_distance_l*_l* | 每页最多四层，每层全部 head 的毫秒距离质量与均匀参考差 | 全 B 读取范围；全部 32 层共八页 |
| representation | 每个观测层一个真实 chunk 的 K/H 原空间 cosine 矩阵、PCA 时间轨迹、cosine-r 共现图 | 观察局部表示结构，K/H 矩阵没有逐 head 分解 |
| all_heads | 每个案例、每层的 20 张 query×key 图，共享 0–1 色标 | 对比全部 head 的列集中和局部读取，不只展示极值 head |
| attention | 指定 chunk 中最大列质量 head 的 query×key 图 | 展示竖列集中，明确标注这是极值 head |
| six_group_analysis | 官方两大类/六子类的八列表格、全层曲线、六组各自的 32×20 mass、peak r、块首峰值热图 | 任务分组描述；七个指定 head 的图保留为补充 |

PCA 各图单独拟合，不直接比较不同层坐标轴；只辅助展示，不能替代原空间相似度和集中统计，更不作为语义丢失证明。默认不使用 t-SNE。

一键入口固定 doc_index=0,1666,3333；底层 launch 默认仍为 0,1,2，均不宣称代表全库。极值 head 的选择仅用于解释图，确认性结论仍依靠全库统计。全层/全 head 指汇总覆盖，不表示保存了全部 5,000 条音频的 N×N 矩阵。

六组报告使用 `task_taxonomy.json` 中从官方分类图转录的 47 项映射，结合本地 Parquet 的 id 和 task_name 进行关联。先检查类别一致性、完整覆盖和音频路径唯一性，再重采样汇总；不把 Linguistics 直接当成 Semantic，也不按 music/speech 划分。原始 accuracy、attention mass、incoming r、距离和位置指标分别报告，不能将任务分组差异解释为 encoder 根据文本问题调整 attention。

## 9. 统计与一致性

- attention 指标先在音频内等权汇总 chunk，再音频等权；相似度先在每条音频内汇总有效 pair。
- 95% bootstrap 默认按 audio_path 聚类。若有共同来源场景，应在采集记录中补齐对应 group 字段并传给 report。单个 group 不生成虚假的零宽度 CI。
- concentration 图展示每个 head 的全音频均值；summary 同时给出预先固定的 all-head mean 及其按音频/来源聚类 CI，不通过在 B 上挑最强 head 来制造显著性。
- baseline 与 observer 保存同样的原始 id、audio_path、prediction、原任务 accuracy。按原始 id 配对，不使用包含 worker request_index 的临时 sample_id。
- 先用同一 checkpoint、参数与样本清单做小样本 baseline/observe。比较逐题生成，不用总体准确率恰好相同替代一致性。
- shadow attention 的 A@V 与原 out_proj 输入比较，报告相对 L2 和最大绝对差。FA2/BF16 不保证位级相同；不能自动用某个固定阈值把差异解释为 sink。
- rank 文件合并后检查应有/实际样本数、重复和缺失；缺失不能偷偷从完整 B 分母剔除。
- 如果根据 B 的图选择 head 再在 B 上作检验，结果仍是探索性的；独立确认需要预先校准选择规则。

## 10. 实现边界与验证状态

`launch.py` 使用已有 `LMMSEvalAdapter(command_runner=...)`，仅将子进程入口从 `lmms_eval` 换为本目录 worker。worker 在本进程内临时绑定观察 bridge；每个 rank 安装 observer，之后继续原评估 CLI。最终移除 hooks、关闭文件、恢复本进程的绑定。

逐层读取实际 q_proj/k_proj/v_proj 输出；在 out_proj 输入处重建 shadow 权重并立即汇总，不改原输出、不重复投影、不保存全音频 N×N。只为固定案例保存单 chunk attention 矩阵与表示。

无需改动 `mllm_comp/`、`third_party/`、Transformers 或已有启动脚本。

已覆盖的 CPU 检查：均匀不同长度 chunks、纯对角读取与单列读取的区别、跨 chunk 相邻对、虚拟索引等预算且不改变输入、真实 HF audio encoder 输出不变、时间映射、worker 启动替换、原始 id 配对、统计与绘图。真实 checkpoint / FA2 / 多 GPU / 全 MMSU 必须在评估机器做小样本验证后再完整运行。运行命令见本目录 README。
