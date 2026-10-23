# 可学习组合实验：原文、官方实现与适配边界

核查日期：2026-10-21。

## ReConV

- 原文：https://arxiv.org/html/2610.17412v1 ，§3.1–3.3、§4.2、附录实现。
- 论文中的 `TokenDrop` 链接是无有效主机的链接，直接打开失败；Hugging Face 尚未收录，按论文标题、ReConV 和 TokenDrop 搜索也未找到能确认属于作者的公开实现。因此当前实现标为“依据论文适配”，不能称为调用官方代码或完整复现。
- 原文机制：摘要库跨注意力、按 attention logits Top-1 路由、位置条件两层重构、正交残差评分、InfoNCE 重构目标。原文允许传递原始向量或残差。
- 原文 MLLM 设置为 CLIP＋Vicuna 的 LLaVA，两层压缩器，558K 对齐预训练＋665K 指令微调；第二阶段包含 LLM LoRA。并非仅训练 Projector，也非原生 Qwen2.5-VL。

## Braco

- 原文：https://arxiv.org/html/2609.35232v1 。
- 官方仓库：https://github.com/zrrraa/Braco 。
- 已核查的 HEAD：`05ef563f7720ca29bd5f58e081e141489d955bea`。
- 逐行检查文件：https://github.com/zrrraa/Braco/blob/05ef563f7720ca29bd5f58e081e141489d955bea/src/braco/compressor.py ，仓库 Apache-2.0。
- 实现有正交 DCT、低频截断、可选小 IDCT、基底坐标嵌入及 sparsemax 残差池化；接受方形 N² 或带 CLS 的 N²+1，官方预设输出预算 4/9/16/25。不能原封不动用于 Qwen2.5-VL 动态矩形网格 K128。
- 官方实现的所有预算默认 coefficients 模式，README 允许手动 IDCT；不能把论文中预算相关规则假定成所有官方训练命令的默认值。
- 论文的 Qwen2.5-3B 行属于其跨语言主干实验，不能当作我们的原生 Qwen2.5-VL-3B 同协议基线。

## 本轮最小适配

四组：pool_adapter、summary_random、summary_residual、spectral_residual。只训练新压缩模块；视觉编码器、原生 merger、LLM 全冻结。K128，学习摘要组分配 64 摘要＋64 原始 Token。

后三组参数结构、参数量、初始化种子和训练损失相同，随机控制也训练重构器。频域组只将摘要初值从空间池化改为一半池化＋一半矩形 DCT 截断后的小 IDCT；先对齐粗空间网格，并按 sqrt(K_base/N) 缩放以保持常量特征幅值。

与原文差异明确记录：输入是 post-native-merger 的 LLM 宽特征；单次压缩；动态矩形网格；重构使用归一化正交残差比例评分，消除原始范数的直接影响；选中项采用有偏 STE 代理梯度；InfoNCE 采用单图内所有原始 Token 作负样本、最多 128 锚点。没有多卡大规模预训练、LoRA、PPE 修改或学习正交旋转，不能把该最小实现命名为完整 OrthoRes 或论文复现结果。

pool_adapter 参数更少，单独报告实际参数量。四组使用相同训练数据、样本顺序、320 个 optimizer step、累积 8、学习率 2e-4、原生动态分辨率；摘要组三组是真正参数量匹配的结构对照。

数据为本地 TextVQA **train** 2048 张、ScienceQA-IMG **train** 512 张，按图像内容哈希选择一道问题。排除全部 ChartQA test 与 TextVQA validation 的 4675 张独立图像，再做训练源之间去重。ChartQA 本地只有 test，因此只做迁移评估，不用 test 或 validation 补训练。

评测沿用已有固定 200 张开发与 800 张留出图像，训练完全排除这些图像；这些图像之前已经评测过，不能称为针对新假设的全新独立确认集。最终固定 step320 checkpoint，避免按留出集挑选 checkpoint。报告原生任务得分及相同样本上的配对区间；保存断点恢复、Token 数、位置、mask、标签和 native 生成验收。
