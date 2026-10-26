# Qwen2.5-VL 实验结果总汇

核对日期：2026-10-26（UTC+8）。本文件只汇总已保存的结果，不重新训练或评测，不改变正在运行的 Qwen3.5 实验。

## 0. 阅读规则

- 3B 与 7B 分开；全量与抽样分开；不同生成流程/评测轮次分开。表中的空白不是零分，而是该轮没有完整结果。
- ChartQA 使用 relaxed accuracy；TextVQA 使用十个人工答案的共识软分；OCRBench 用原生总分 /1000；DocVQA 用 ANLS。
- AI2D主线表统一使用修正后的“提取答案开头选项字母”评分。旧严格全文匹配保留在历史报告中，两种口径不混用；这次仅调整汇总展示，没有重新推理或覆盖原始评分。其他历史章节仍按其表头明确标注原口径。
- 固定 K128 指每图最终 `min(128,N)` 个 post-merger 视觉 Token，不是输入分辨率固定，也不是跨 Decoder 层平均128。
- 外部方法为本地受控 Qwen 适配/组件对照，不等同于认证复现论文全部设置。Bifocal Stage1 与 Stage1+2 分开。
- 大多数训练分支只有一个训练 seed，数据集已反复用于探索；不能宣称三 seed 稳定性或独立确认。重复运行/复用预测不增加独立样本量。
- 下文保留历史失败方向，但不把 smoke 分数、未完成结果、Oracle 数值当作全量任务成绩。

## 1. 7B 主线与外部对照：精简后的统一框架全量质量结果

本组：ChartQA 2500题/1509独立图像，OCRBench 1000题/930图像，AI2D 3088题/814图像。BF16、eager、原生动态分辨率、greedy；主线 repetition penalty 1.05。除 dense 外均 K128。

| 版本 | ChartQA % | OCRBench /1000 | AI2D 修正评分 % |
|---|---:|---:|---:|
| Dense baseline | 77.28 | 774 | 80.67 |
| EADP，修正后本地适配 | 53.92 | 560 | 75.55 |
| PruneSID，修正后本地适配 | 46.08 | 483 | 73.51 |
| MMTok，修正后本地适配 | 58.16 | 540 | 77.98 |
| Bifocal Stage1 K128，本地组件对照 | 58.56 | 592 | 78.85 |
| 旧聚合 MLP | 68.76 | 628 | 78.95 |
| 训练视觉评分器＋可微 Top-K | 69.16 | 642 | 77.82 |
| 图像全局上下文选择器 | 68.40 | 669 | 78.63 |
| 全局问题上下文选择器 | 70.00 | 645 | 78.59 |
| 已训练相关性＋保护 | 69.00 | 652 | 79.86 |

保留依据：旧MLP是聚合路线参照；视觉评分器是可微Top-K参照；图像上下文是非噪声分支OCRBench最高；问题上下文是ChartQA最高；相关性＋保护是当前代表性主线。所有版本沿用同一修正评分，不逐方法挑指标。

外部方法保留作对照，不因本地成绩较低而从比较中删除。这里均为本地Qwen适配，不是原论文表格数字。其他方法的较早全量轮次见第3节；Bifocal Stage1+2因预算不同见第2.1节。

从主线展示移除：原始均值、未训练评分器、STE、裸逐词残差、OT、原始余弦保护、均值回收、加权残差回收。它们属于明显退化、已有版本的中间步骤或没有可靠新增收益；移出展示不等于删除负结果，原始配置、预测、checkpoint和完整报告仍保留，可用于消融与失败分析。

噪声门单列为历史OCR实验，不纳入本次精简主线：ChartQA65.76%、OCRBench689、AI2D修正78.85%。它有OCR收益但跨任务不稳定，且按用户要求不迁移到3.5。

### 1.1 这些结果支持什么

- 旧 MLP 对原始均值有明显任务收益：ChartQA +10.12pp，OCRBench +37分。
- 训练视觉选择器相对未训练评分器：ChartQA +26.80pp、OCRBench +244分。该比较包含学习过程，不能独立归因于 Top-K 算子。
- 训练视觉选择器相对旧 MLP：ChartQA +0.40pp、OCRBench +14分，但 AI2D修正分低1.13pp；并不存在一致胜出的单一方案。
- 本地 soft/STE 对照：ChartQA 69.16/38.68，OCRBench 642/457。STE 训练前向保留全部位置并把部分 embedding 置零，不是物理删 Token；不能据此否定所有 STE。
- 全局问题上下文相对图像上下文：ChartQA +1.60pp，图像 bootstrap95%区间 [+0.32,+2.85]pp；OCRBench -24分，AI2D近似持平。问题条件化的收益不是跨任务一致的。
- 逐词相关性本身没有超过之前的全局问题版本。保护规则相对逐词统一排序改善 OCRBench/AI2D，但 ChartQA略降。
- 噪声门是 OCRBench 当前历史最佳压缩分支689，但在 ChartQA明显下降；按用户要求未纳入3.5，不抹去历史数据。
- ChartQA主线最佳70.00仍比本组 dense低7.28pp；OCRBench最佳689仍低85分。不能宣称已经恢复 dense质量。
- AI2D修正评分下，旧MLP78.95低于dense80.67；相关性＋保护79.86，距dense0.81pp。旧严格全文匹配曾造成旧MLP高于dense的印象，不再作为主线展示指标。
- 各任务最优来自不同 checkpoint/规则，不能拼成一个“最佳模型”。

### 1.2 最新回收实验的配对不确定性

不重新训练、不改变128个选中索引，保护 Token 内容不变，只改非保护上下文表示。共2方法×6588题=13176请求，6/6全量完成。

| 回收方式 | 任务 | 相对不回收变化 | 图像配对 bootstrap95%区间 |
|---|---|---:|---|
| 均值回收 | ChartQA | -1.48pp | [-2.82,-0.12]pp |
| 均值回收 | OCRBench | -13分 | [-31.81,+5.14]分 |
| 加权残差回收 | ChartQA | +0.04pp | [-0.49,+0.56]pp |
| 加权残差回收 | OCRBench | +3分 | [-4.99,+11.08]分 |
| 加权残差回收 | AI2D 严格 | +0.10pp | [-0.25,+0.44]pp |

结论：均值回收没有收益；加权残差仅基本持平，所有三个增益区间包含零。只针对预算内写回，不覆盖“128个原始 Token＋额外回收 Token”。后者在本次汇总时尚未实施。

## 2. 7B 外部对照：使用修正后的结果

同样三个全量任务，固定K128。旧版54.60/562等成绩保留在历史文件，不再用作当前修正实现的结果。

| 本地受控适配/组件 | ChartQA % | OCRBench /1000 | AI2D 修正评分 % |
|---|---:|---:|---:|
| EADP，修正后 | 53.92 | 560 | 75.55 |
| PruneSID，修正后 | 46.08 | 483 | 73.51 |
| MMTok，修正后 | 58.16 | 540 | 77.98 |
| Bifocal Stage1 K128 | 58.56 | 592 | 78.85 |

修正前 → 修正后（ChartQA / OCRBench / AI2D严格）：

- EADP：54.60 / 562 / 69.79 → 53.92 / 560 / 69.43。
- PruneSID：51.60 / 518 / 69.75 → 46.08 / 483 / 67.58。
- MMTok：54.00 / 532 / 71.73 → 58.16 / 540 / 71.53。

修正包括原始 dtype归一化、文本池化、PruneSID官方语义组输出顺序及其位置/cache同步等。分数变低不等同于修复错误；内核对照和端到端协议是不同验收。但这些成绩仍不是论文原表的同设置复现，也不能用来证明这些论文普遍无效。

### 2.1 Bifocal Stage1+2：单独预算

| 任务 | 得分 | 阶段1平均N | 阶段2平均N | 跨层平均N |
|---|---:|---:|---:|---:|
| ChartQA | 57.20% | 249.71 | 32.96 | 149.08 |
| OCRBench | 328/1000 | 273.23 | 36.27 | 163.21 |
| AI2D修正评分 | 76.49% | 213.52 | 28.37 | 127.56 |

原历史严格评分为70.27%，不混入修正口径。不是固定K128 projector-only，不能与Stage1行作等预算因果消融，也不能直接与论文截图比高低。

## 3. 7B 早期四任务统一平台轮次：保留但不混入当前主表

这些是2026-10-21/22的完整原生聚合结果。该组 dense ChartQA77.32、OCRBench776，与后续主表77.28/774不同；使用本组自己的对照。DocVQA验证集5349题/1285图像。

| 当时本地实现 | ChartQA % | OCRBench /1000 | AI2D修正评分 % | DocVQA ANLS % |
|---|---:|---:|---:|---:|
| Dense | 77.32 | 776 | 80.67 | 90.89 |
| 旧聚合MLP | 68.76 | 626 | 78.95 | 50.48 |
| CDPruner | 54.32 | 516 | 73.06 | 24.57 |
| DPPruner | 56.00 | 560 | 77.85 | 34.77 |
| HiPrune，本地projector组件 | 60.64 | 594 | 78.14 | 39.46 |
| EADP，当时默认配置/旧实现 | 55.60 | 571 | 77.88 | 38.98 |
| DivPrune | 56.68 | 576 | 77.95 | 38.96 |
| VisionZip，本地Qwen组件 | 59.64 | 574 | 78.40 | 39.06 |
| AIM input-merge组件 | 53.52 | 549 | 77.59 | 32.74 |

边界：HiPrune/AIM等这里的projector组件不代表论文完整多阶段方法；VisionZip原论文压缩位置与Qwen合并器适配不同。DocVQA列没有后来修正的 EADP/PruneSID/MMTok 同轮重跑，不跨轮补齐。`gap_attention_k128` 未找到完整原生聚合且平台记录失败，不给它编分数。此前25%比例轮次已标记 superseded，不混入固定K128结果。

## 4. 7B 更早的诊断生成轮次与训练子集

### 4.1 早期全量（诊断生成，非后来统一主线）

| 任务 | 题数 | Dense | 旧聚合MLP K128 |
|---|---:|---:|---:|
| ChartQA | 2500 | 82.28% | 69.48% |
| OCRBench | 1000 | 763 | 631 |
| DocVQA验证集 | 5349 | 90.96% ANLS | 51.81% ANLS |
| AI2D | 3088 | 74.45% | 77.95% |

这是此前对话中82.28%、763、51.81%等数字的来源，不是当前统一主表。诊断 teacher-NLL＋缓存生成与平台原生生成不能混成同一对照。

### 4.2 7B 聚合MLP的400＋400张子集结果

| 任务 | Dense | 未训练锚点均值 | 训练后聚合MLP | MLP相对均值 |
|---|---:|---:|---:|---:|
| TextVQA，400图像 | 78.00% | 65.25% | 74.05% | +8.80pp，95%CI[+4.83,+12.90] |
| ChartQA，400图像 | 83.00% | 67.25% | 75.50% | +8.25pp，95%CI[+4.75,+12.00] |

不是TextVQA 5000题全量，也不是新鲜独立发现确认；这些图像曾用于此前3B实验。不以子集75.50替换全量68.76/69.48。

## 5. 3B：最初的全量免训练结果

3B checkpoint为本机Qwen2.5-VL-3B-Instruct，revision `1b5a7674d40a0ac9acef541a46fed12d710bd36b`。

| 方法 | ScienceQA-IMG 2017题 % | ChartQA 2500题 % | TextVQA 5000题软分 % |
|---|---:|---:|---:|
| Dense | 81.21 | 83.60 | 78.226 |
| Feature-anchor mean K64 | 80.47 | 47.64 | 55.866 |
| Grid mean K64 | 80.47 | 38.24 | 20.538 |
| PruMerge+ K64，本地适配 | 80.47 | 55.80 | 43.570 |
| Bifocal，历史变长适配 | 80.91 | 71.36 | 68.658 |

Bifocal不是固定K64：ScienceQA平均81（范围64–148），ChartQA平均149。ScienceQA三组总分相同不代表逐题预测相同。

TextVQA保存预测已用每题十参考答案重新计分，5000题分数不匹配数为0。字段曾命名 `exact_match`，实际是共识软分，不是字面严格匹配。这项复核不替代历史像素、EOS与cache的完整审计。

### 5.1 ChartQA预算扫描（3B、2500题）

| 方法 | K64 | K128 | K256 |
|---|---:|---:|---:|
| Feature-anchor mean | 47.64% | 62.08% | 75.12% |
| PruMerge+ | 55.80% | 70.84% | 无完整原生聚合结果 |

Dense83.60%。增加预算改善明显，但不证明任何固定K都适合全部任务。

### 5.2 早期3B训练分支（ChartQA K64全量）

| 分支原名 | ChartQA2500题 |
|---|---:|
| mean_only | 50.56% |
| meta_residual | 50.76% |

收益只有0.20pp；不把它们冒充后来7B聚合MLP。原聚合结果见本组native result文件。

### 5.3 3B自然图像跨任务子集（不是全量）

| 任务 | 题数 | Dense | Feature-anchor K64 |
|---|---:|---:|---:|
| GQA | 256 | 58.98% | 57.03% |
| POPE Random | 256 | 89.06% | 90.23% |
| 当时TextVQA抽样配置 | 256 | 76.99% | 54.18% |

不把GQA/POPE256题写成全量，也不据此宣称压缩提高了泛化能力。

## 6. 3B AI2D全量与早期可学习聚合探索

AI2D3088题，历史原报告的“Accuracy”与其“Strict letter”定义不同于第1节统一平台严格分；只在本组比较。

| 方法 | 历史Accuracy % |
|---|---:|
| Dense | 79.05 |
| Feature-anchor mean K128 | 75.39 |
| Grid mean K128 | 73.54 |
| PruMerge+ K128 | 76.30 |
| pool_adapter | 76.59 |
| summary_random | 75.78 |
| summary_residual | 74.32 |
| spectral_residual | 74.00 |
| feature_anchor_adapter | 76.85 |

另有3B的800图像（TextVQA/ChartQA各400）学习聚合子集：

| 方法 | TextVQA软分 % | ChartQA relaxed % |
|---|---:|---:|
| Dense | 78.15 | 84.25 |
| pool_adapter | 50.92 | 64.00 |
| summary_random | 51.65 | 59.50 |
| summary_residual | 50.60 | 57.25 |
| spectral_residual | 51.90 | 56.50 |

不存在稳定残差/谱聚合优势；不将此组与7B训练收益混用。

## 7. 3B几何/相似度消融：合计500张

ChartQA/TextVQA各250图像，K128，共用以前已看过的图像；不是各500，也不是全量。

| 距离/锚点规则 | TextVQA软分 % | ChartQA % |
|---|---:|---:|
| Dense | 77.40 | 84.40 |
| 余弦最远点 | 62.72 | 65.60 |
| 单位化欧氏最远点 | 62.72 | 65.60 |
| 原始未归一化欧氏 | 30.44 | 61.20 |
| 原始欧氏选点＋余弦分配 | 39.32 | 66.40 |
| 单位特征L1 | 61.80 | 60.80 |
| 相关系数距离 | 62.08 | 64.80 |
| 对角Mahalanobis | 63.64 | 61.20 |
| Query-response KV | 55.96 | 44.40 |
| Query-response JS | 35.96 | 46.00 |
| k-means++，三seed均分 | 56.61 | 62.53 |
| k-means十次迭代，三seed均分 | 57.43 | 63.33 |

k-means逐seed（11/23/47）：

- TextVQA k-means++：54.24/60.36/55.24；十次迭代：56.40/58.80/57.08。
- ChartQA k-means++：66.40/60.40/60.80；十次迭代：66.80/60.40/62.80。

单位化欧氏与余弦等价的实验没有收益；Mahalanobis仅TextVQA微升、ChartQA下降，未建立跨任务稳定优势。相关系数距离也未超过余弦。上述数据不能支持继续追求单纯替换距离公式。

## 8. 现象发现：哪些信号成立、哪些不成立

### 8.1 3B区域协同：1000张独立图像

每任务500图像，其中开发100、确认400。确认800图像按图像聚合，种子/区域对不作独立样本。

| 确认任务 | 问题相关区域交互 | 随机区域交互 | 差值，95%CI |
|---|---:|---:|---|
| TextVQA，400图像 | -0.00053 | -0.00076 | +0.00024 [-0.00719,+0.00758] |
| ChartQA，400图像 | +0.00403 | +0.00389 | +0.00014 [-0.00806,+0.00819] |

决策为证据不足，不是“协同绝不存在”。问题复核100图像、预算复核200图像也未建立稳定优势；这两项重复使用主队列图像，不增加独立样本数。原生四条件交互的正负仅是背景/对照条件下区域交互，不是已证实语义绑定。

### 8.2 3B重构残差＋真实Token混合：800确认图像

| 方法 | TextVQA % | ChartQA % |
|---|---:|---:|
| Dense | 78.15 | 84.25 |
| 特征均值＋重心位置K128 | 62.38 | 66.75 |
| 原始代表Token K128 | 58.75 | 62.75 |
| PruMerge+ K128 | 55.75 | 72.50 |
| hybrid/attention | 42.75 | 56.50 |
| hybrid/diversity | 53.57 | 56.00 |
| hybrid/high_residual | 26.15 | 55.50 |
| hybrid/low_residual | 37.30 | 52.00 |
| hybrid/pca_residual | 26.82 | 56.00 |
| hybrid/random，三seed均值 | 37.43 | 57.25 |
| hybrid/spatial_random，三seed均值 | 35.80 | 59.25 |
| summary96 | 26.85 | 52.00 |

结论：手工高重构残差不是可靠任务信息指标，未建立实用优势。不能由此否定所有可学习重构/回收器。

### 8.3 7B问题使用审计：全量三个任务

仅替换选择器问题，LLM始终看真实问题。数字为相对原问题选择的平均Token重合率。

| 任务 | 错问题重合率 | 通用问题重合率 | 错问题得分 | 通用问题得分 | 原问题得分 |
|---|---:|---:|---:|---:|---:|
| ChartQA | 99.70% | 99.11% | 69.92% | 70.08% | 70.00% |
| OCRBench | 99.60% | 99.30% | 645 | 643 | 645 |
| AI2D严格 | 99.87% | 99.59% | 71.66% | 71.73% | 71.86% |

这是当时全局问题选择器的具体诊断：排序/集合对问题非常不敏感，不等于所有query-aware方法都无效。

进一步750图像机制实验（每任务250、9条件）：满足交换适用条件的图像仅ChartQA4、OCRBench30、AI2D2；平均交换0.32/0.46/0.16个Token。报告中的零分变化不能解读为“关键视觉证据没有因果作用”，因为多数干预不适用/过弱，候选区域也没有人工语义核验。

### 8.4 更早800条因果记录的测量问题

2026-10-19旧实验含200条TextVQA字面 `answer` 错误目标；POPE200条仅34个独立图像，其他任务也不能用记录数替代图像数。相关NLL/交互结论须降级或排除。后续E0修复16样本仅作实现校准，不当成科学确认。此问题与第5节TextVQA5000题保存预测复核不是同一实验，不应互相混淆。

## 9. 最初的Stage1内部诊断（3B）

Layer0已完成四任务各300样本、两种分组，共1200样本；数值gate通过，跳过0。Layer2/4/8/16各1200样本也有完成聚合且gate通过；是教师状态重压缩，不是从该层真实压缩后完整任务准确率。

Layer0 Feature-anchor各分支attention-output相对误差（%）：

| 任务 | A | B | C | D | O-Z | O-M |
|---|---:|---:|---:|---:|---:|---:|
| ChartQA | 10.529 | 4.084 | 3.034 | 2.363 | 2.160 | 约0.00002 |
| TextVQA | 10.842 | 4.938 | 3.431 | 3.300 | 2.196 | 约0.00002 |
| GQA | 7.923 | 2.455 | 1.605 | 1.068 | 0.928 | 约0.00002 |
| POPE Random | 7.478 | 2.250 | 1.510 | 0.953 | 0.866 | 约0.00002 |

两分组最大绝对分解闭合残差约2.38e-7；这不是单独对相对阈值的新计算。Oracle近乎恢复输出是数值验收，不是可部署质量收益。

另有Layer0一次压缩的真实轨迹：四任务各64样本，共256，gate通过。教师→压缩文本KL：

| 任务 | Feature-anchor KL | Grid KL |
|---|---:|---:|
| ChartQA | 0.2971 | 0.4384 |
| TextVQA | 0.3519 | 0.5700 |
| GQA | 0.1748 | 0.4104 |
| POPE Random | 0.1628 | 0.3525 |

低秩Value Oracle：8样本，Feature-anchor ChartQA O-Z2.357%→rank4 0.606%；TextVQA1.884%→0.752%。依赖精确O-Z，不是独立可部署低秩压缩器。

质量修正实际接入的100题pilot：ChartQA K64 40.0→mass41.0；TextVQA50.8→48.8。Layer0误差改善没有自动变成答案收益；该TextVQA是软分，旧报告列名 `exact match` 不准确。

## 10. Token、计算量与速度

7B主线 K128 的实际视觉Token数量：

| 任务 | Dense平均视觉N | 压缩后K | 总视觉Token减少比例 |
|---|---:|---:|---:|
| ChartQA | 596.09 | 128 | 78.53% |
| OCRBench | 652.82 | 128 | 80.39% |
| AI2D | 510.19 | 128 | 74.91% |

这是总视觉Token计数比，不是整模型FLOPs下降，也不是保证端到端同比提速；Vision仍处理原始图像。未完成严格受控的生产路径速度/FLOPs对照。

历史生成计时（ms/题，非公平速度排名）：

| 方法 | ChartQA | OCRBench | AI2D |
|---|---:|---:|---:|
| Dense | 348.871 | 897.389 | 无同口径记录 |
| 旧MLP | 377.730 | 1068.037 | 本次未统一计算 |
| 视觉选择器 | 338.294 | 988.790 | 本次未统一计算 |
| 已训练相关性＋保护 | 452.006 | 977.443 | 299.857 |
| EADP修正后 | 345.625 | 1114.724 | 281.171 |
| PruneSID修正后 | 347.344 | 1040.991 | 331.611 |
| MMTok修正后 | 420.225 | 1052.956 | 348.955 |
| Bifocal Stage1 | 407.607 | 891.632 | 277.695 |
| Bifocal Stage1+2 | 406.504 | 1231.905 | 365.542 |

共享GPU负载、生成长度、诊断同步/审计开销不同，禁止由此表宣称固定加速倍数。理论减少LLM序列与已证明端到端提速是不同结论。

## 11. 总结

1. 有效信号：聚合MLP相对未训练均值、已训练选择器相对未训练评分器有质量收益；特征分组通常比规则网格适合难视觉任务；增加K明显改善OCR/图表质量。
2. 尚未解决：相对dense的大幅差距、问题语义真正影响选择、跨任务稳定性、强回收机制、受控端到端加速。
3. 否定/证据不足方向：单纯替换距离未稳定改善；手工高残差不等于任务贡献；OT当前分支下降；均值回收下降；加权残差回收尚无可靠收益；区域协同未获确认。
4. 主线只有固定预算，动态预算尚未验证。“128＋额外回收Token”没有现成成绩。
5. Qwen3.5另开新实验，不把其训练状态或后续结果计入本文件。

## 12. 结果来源索引

仓库根目录为 `/home/ZhouFengyu/fastv-repro/MLLM-Comp-Platform/MLLM-Comp-Platform-main`。以下均为仓库内原始报告/聚合相对位置，可按实验版本查逐样本预测和配置。

| 内容 | 原始文件/目录 |
|---|---|
| 7B初始Top-K、anchor与MLP | `outputs/eval/qwen25_vl/learned_visual_topk_step320_20261023_v1/report.md` |
| soft/STE | `outputs/train/qwen25_vl/topk_mask_ablation_20261023_v1/report_full.md` |
| 图像/问题上下文 | `outputs/train/qwen25_vl/context_visual_topk_20261024_v2/report_full_selective.md` |
| AI2D两套分数 | `outputs/eval/qwen25_vl/context_topk_ai2d_external_20261024_v1/report.md` |
| 逐词相关性 | `outputs/train/qwen25_vl/word_visual_topk_20261025_v2/report.md` |
| 噪声与OT | `outputs/train/qwen25_vl/noise_ot_20261025_v1/report.md` |
| 保护策略 | `outputs/analysis/protected_visual_topk/protected_k128_20261026_v1/report.md` |
| 回收6/6 | `outputs/analysis/visual_token_recovery/recovery_k128_20261026_v1/summary.json` |
| 修正外部对照9/9 | `outputs/eval/qwen25_vl/projector_official_fidelity_k128_20261024_v1/summary.json` |
| Bifocal6/6 | `outputs/eval/qwen25_vl/bifocal_stage1_stage12_20261024_v2/summary.json` |
| 早期四任务平台组 | `outputs/eval/qwen25_vl/projector_dynamic_k128_platform_20261021_v1/full/*/**/*_results.json` |
| 扩展方法平台组 | `outputs/eval/qwen25_vl/projector_extension_k128_platform_20261022_v3/full/*/**/*_results.json` |
| 早期7B ChartQA/AI2D | `outputs/eval/qwen25_vl/anchor7b_chartqa_ai2d_mlp_only_20261021_v3/*/full_report.md` |
| 早期7B OCR/DocVQA | `outputs/eval/qwen25_vl/anchor7b_ocrbench_docvqa_full_20261021_v1/*/full_report.md` |
| 7B聚合MLP确认子集 | `outputs/train/qwen25_vl/anchor_mlp_7b_k128_ddp_20261021_v1/final_report.md` |
| ScienceQA3B全量 | `outputs/eval/qwen25_vl/scienceqa_img_full_k64_20261017_v2/report.md` |
| ChartQA3B全量 | `outputs/eval/qwen25_vl/chartqa_full_k64_20261017_v1/report.md` |
| TextVQA3B全量重评分 | `outputs/analysis/textvqa_evaluation_audit/20261021_v1/report.md` |
| ChartQA预算扫描 | `outputs/eval/qwen25_vl/chartqa_k_sweep_20261017_v1/*/**/*_results.json` |
| 早期3B K64训练 | `outputs/eval/qwen25_vl/chartqa_stage2_k64_20261018/*/**/*_results.json` |
| GQA/POPE子集 | `outputs/eval/qwen25_vl/untrained_crossdataset_20261018/*/**/*_results.json` |
| AI2D3B全量 | `outputs/eval/qwen25_vl/ai2d_full_k128_20261021_v1/full_report.md` |
| 3B学习聚合探索 | `outputs/train/qwen25_vl/learned_hybrid_k128_20261021_v1/final_report.md` |
| 500图像距离 | `outputs/analysis/token_metric_ablation/3b_k128_n500_20261021_v1/all_report.md` |
| 500图像锚点几何 | `outputs/analysis/anchor_distance_ablation/3b_k128_n500_20261021_v2/all_report.md` |
| 1000图像协同 | `outputs/analysis/visual_evidence_interaction/20261020_v1/final_report.md` |
| 重构残差800图像 | `outputs/analysis/visual_reconstruction_utility/20261020_v1/final_report.md` |
| 问题使用全量审计 | `outputs/analysis/selector_question_usage/query_condition_20261025_v1/report.md` |
| 750图像机制诊断 | `outputs/analysis/selector_mechanisms/mechanism_20261025_v1/report.md` |
| 早期测量无效项 | `outputs/analysis/visual_token_phenomena/discovery_measurement_audit_20261019/audit.md` |
| Stage1 Layer0 | `outputs/analysis/visual_merge_attribution/qwen25_vl_layer0_k64_300_each_20261017_eps2/analysis/visual_merge_attribution/visual_merge_attribution.aggregate.json` |
| 一次压缩真实轨迹 | `outputs/analysis/visual_merge_attribution/layer0_trajectory_64_each_20261017/analysis/visual_merge_attribution/visual_merge_attribution.aggregate.json` |
| 低秩Value Oracle | `outputs/analysis/visual_merge_attribution/value_residual_k64_pilot/analysis_report.md` |
| mass实际接入pilot | `outputs/eval/qwen25_vl/attribution_guided_mass_pilot/pilot_report.md` |
| 历史效率审计 | `/home/ZhouFengyu/exa-results/visual-token-efficiency-audit-2026-10-26.md` |

归档、superseded和失败attempt仍保留原地；本文件不重复列每一轮工程smoke得分，不把它们计入科学实验样本量。各原报告含更细分的任务子组、误差指标与案例，必要时以逐样本文件和固定协议为准。
