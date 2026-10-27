# 小规模图像压缩评测数据集：2026-10-26核对

目的：在ChartQA、OCRBench、AI2D之外增加不同能力维度，不因为小或分数高就选择数据集。
本轮只核对和推荐，没有下载数据、改动共享目录或启动新评测。

## 优先顺序

1. **MMStar全量1500题**：优先新增。六种能力各250题，原设计强调视觉依赖，适合检验是否真正保留图像证据，而不是主要依赖语言先验。官方单图选择题格式适合当前路径，但下载后仍需检查全部样本形态和与训练集的图像哈希重叠。
2. **POPE COCO adversarial全量3000题**：本地已有，答案短；补充对象存在性/幻觉控制。明确只跑adversarial时不能称为三策略完整POPE9000。随后可补random和popular，必须同时报告accuracy、F1、precision/recall和yes比例，不能只看总准确率。
3. **HallusionBench 1129题/346图像**：小且有成对控制设计，但需要核对文本/图像条件、成对/组级统计及判分流程，接入成本不等于题数。当前常用目录未发现原始数据，不声称已具备即跑条件。用户表中的HallB暂按HallusionBench理解，若原论文缩写另有所指，应重新对齐。
4. **MMMU validation 900题**：本地已有，但当前不能直接全量压缩。按parquet实际非空image字段核对，857题单图，43题多图（24题两图、5题三图、8题四图、6题五图）。需要多图适配，或把857题明确标注为单图子集，不得偷偷排除后仍称900全量。
5. **ScienceQA-IMG test2017题**：本地已有，适合作为低视觉预算敏感度控制；3B上已经做过，不宜作为新增方法有效性的唯一证据。

## 本地split与规模

本地共享目录只读：`/home/YinPeiran/datasets/lmms_eval_datasets/`。

| 数据集/split | 题数 | 本地状态 | 对当前研究的用途 |
|---|---:|---|---|
| MMStar | 1500 | 常用共享及本人目录未发现 | 首选新增多能力视觉依赖测试 |
| HallusionBench | 1129 | 常用共享及本人目录未发现 | 配对幻觉/视觉错觉诊断，需先核对协议 |
| MMMU validation | 900 | 有 | 专业推理，但43多图题不兼容当前单图binding |
| POPE random/popular/adversarial | 每组3000，共9000 | 有，`POPE/Full/` | 短答案对象幻觉对照；同图多题不算独立图像 |
| ScienceQA-IMG test | 2017 | 有 | 小预算控制，不单独承担方法结论 |
| AI2D test | 3088 | 有，已在跑 | 图表推理，沿用已统一的评分协议 |
| OCRBench test | 1000 | 有，已在跑 | 当前最小OCR主测试，保留而非重复另开 |
| OK-VQA val2014 | 5046 | 有 | 知识问答，与视觉压缩损失混杂，不优先 |
| TextVQA validation | 5000 | 有 | OCR细节；3B已全量，新轮须统一原生软分 |
| DocVQA DocVQA/validation | 5349 | 有 | 长视觉序列/文档压力测试，不与InfographicVQA合并计数 |
| GQA testdev balanced instructions | 12578 | 有，图像表398行 | 一般视觉关系测试，但不算小；图像表不是问题数量 |
| VQAv2 validation | 214354 | 有 | 大规模，一开始不值得优先全量 |

DocVQA目录同时含InfographicVQA；本表5349只指DocVQA validation，不把不同任务合成8150题。
`Avg`、`Avg2`、各能力组平均是汇总列，不是额外数据集。跨数据集平均还需先统一分数尺度，不能把OCRBench原始分数直接与百分比相加。

## 运行边界

- 本轮不训练、不自动增加GPU任务，当前3.5双卡队列不修改。
- 新增数据首先与2560张训练清单做图像哈希查重，固定split、prompt、选项提取和生成设置。
- 先按统一平台入口完成8条smoke，再对选定版本全量；不再增加500张开发集。
- 第一轮只比较dense、旧MLP、当前相关性＋保护及一两个外部参照；无需把所有弱分支重跑一遍。
- 原始图像内容、问题与成对组信息保留；bootstrap按图像/控制组而不是每题独立重采样。
- 小数据集只代表成本较低，不保证能识别1pp级的提升，需报告配对区间。

## 官方来源

- MMStar项目：https://mmstar-benchmark.github.io/ ，官方数据：https://huggingface.co/datasets/Lin-Chen/MMStar ，1500题及六能力设计。
- MMMU官方：https://github.com/MMMU-Benchmark/MMMU ，validation900题；多图43题是本地parquet核对结果，不是从全数据集比例推算。
- POPE官方：https://github.com/RUCAIBox/POPE ，原论文：https://aclanthology.org/2023.emnlp-main.20.pdf 。本地三split题数分别由parquet元数据确认。
- HallusionBench官方会议页：https://cvpr.thecvf.com/virtual/2024/poster/29422 ，确认346图像/1129问题及控制组设计。GitHub原仓库本轮抓取未成功，因此不宣称已经核对其全部评测代码。
