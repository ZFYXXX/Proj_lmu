# Qwen 外部对照修正与重跑

本轮修正 Qwen3.5 对照的文本迁移，并核对官方 Qwen 选择内核。保持原生动态分辨率、post-merger K128 和统一平台评分。只评测，不训练。原始 27 组、旧 checkpoint 和执行快照保留；新增独立 binding，不改变旧训练模块的实现哈希。不自动提交或推送 Git。

## 官方 Qwen 代码提供的迁移依据

| 来源 | 已读代码的实际做法 | 本轮接入 |
|---|---|---|
| [MMTok Qwen2.5](https://github.com/Ironieser/MMTok/tree/main/mmtok/qwen) | 在原生 vision merger 后选择；原始视觉分支按4个 patch 平均并逆转 window 顺序；独立编码 `Question: ...`；文本温度0.01；保留原坐标并同步删 mask/embedding | Qwen3.5 从本机原生 vision tower 捕获同阶段状态，按实际 merger 大小池化；该架构没有2.5同一窗口重排，不能照搬逆置换；恢复专用问题构造和0.01温度 |
| [EADP Qwen](https://github.com/SJTU-DeepVisionLab/EADP/tree/e1a08801461c181871b9ff1cb0803b8e9966f083/Qwen_vl) | 使用 merger 后 LLM 空间视觉/文本向量；alpha=.5、beta=2；发布路径将输入补方并统一1008×1008，通过VLMEvalKit运行 | 恢复从裸 user 文本重新分词；保留官方负余弦、BF16归一化顺序和选择器；动态矩形网格仍按真实H/W处理 |
| [PruneSID Qwen](https://github.com/ZhengyaoFang/PruneSID/blob/4ef17d96b390027f31243bc7e1d27bdee77792a5/prunesid/prunesid_qwen/modeling_qwen2_vl.py) | merger后PCA/NMS；预算按N换算；K≤150按语义组顺序输出，位置须按同一索引重排 | 保留已有修正内核和顺序，不随意改为源排序；在真实9B上比对官方内核，并检验旧/新binding逐Token分数一致后复用旧全量 |
| [AnchorPrune Qwen](https://github.com/MULTI-cau/AnchorPrune/blob/0e0098b591133433f787e870383ca7ad406e2486/anchorprune/qwen.py) | 单独提取问题；在LLM空间计算图文相关性；注意力统计要处理非CLS视觉塔与merger/window对应关系 | 参考其“显式问题来源＋网格映射”的接入原则，未引入新的选择算法 |
| [Bifocal 发布代码](https://github.com/EIT-NLP/Bifocal/tree/e50d76571ac933e641d8259f0f5e6c80953d9d85) | 公开可运行入口以LLaVA为主；query.py按词拆分、每词子词均值；完整方法包含Decoder Stage2 | 恢复词级输入，移除平台已知答案格式提示；Qwen使用原生视觉attention平均received mass，明确标注Stage1组件 |

MMTok本地来源为已有下载目录，未伪造Git commit；清单记录具体文件SHA256。新获取的Bifocal快照位于自己的 research_sources 目录。没有修改其他人的目录。

## Qwen 与 LLaVA 的结构边界

- Qwen的K以原生vision merger后的Token数计算；不把LLaVA固定576 patch的预算数或CLS分数直接搬过来。
- 图像H/W来自 `image_grid_thw / spatial_merge_size`，保留矩形动态网格。
- Qwen3.5使用其原生四轴position输入（物理序列轴＋三轴M-RoPE）和混合attention cache；只缩短物理序列，保留原始视觉与文本坐标，decode delta补偿删除长度。
- Qwen3.5的线性attention也受序列顺序影响，不能认为保留原M-RoPE就能任意重排Token。PruneSID的特殊顺序是官方算法行为，须单独留痕。
- 新生成binding复用已验收的native compaction；不替换整套Qwen forward。每个请求检验placeholder、特征数量、逐层长度和首个decode坐标。
- EADP原发布的固定1008分辨率/默认位置生成协议，与当前统一比较协议仍有区别。本轮是恢复可追溯的Qwen算法/文本规则，不把它写成原论文全设置复现。

## 改动与验收

新增 `qwen35_reference_projector`，修正MMTok、Bifocal、EADP的问题输入；旧 `qwen35_transferred_projector` 和训练head实现哈希不变。

CPU覆盖文本构造、答案隔离、词级聚合、精确预算、真实小型Qwen3.5生成、原始前缀恢复、dense identity、缓存与M-RoPE。Qwen2.5旧binding测试使用4.57运行时；Qwen3.5测试使用隔离5.19运行时，不能混用两个模型API的测试环境。

真实9B gate对三个任务各一图检查：

1. EADP/MMTok/PruneSID与固定SHA的官方选择内核比较选中索引及顺序；独立构造官方文本向量作对照。
2. Bifocal词向量与官方query.py直接计算完全一致。
3. 四种方法各自验证真实K128、混合缓存和decode位置。
4. 四种方法identity分支逐步生成分数和生成Token与dense完全一致。
5. PruneSID新旧binding逐步生成分数与Token完全一致后，允许复用旧全量。

Gate通过后，每个受影响单元先跑同一8题smoke，再跑完整数据集，不根据smoke准确率调参。

## 执行矩阵和恢复

- 新重跑：MMTok、Bifocal Stage1、EADP × ChartQA2500、OCRBench1000、AI2D3088，共9组19,764条正式请求。
- 复用：dense、PruneSID及四个已训练版本的旧全量；新表保留结果来源。
- 两GPU按单元取队列，每卡最多一个本任务子进程，不修改其他用户进程。
- 任一gate、数量、位置或评分完整性验收失败停止新任务；保留错误和旧attempt，修复实现后使用新执行快照/run ID。
- 每组有complete标记，续跑校验配置和已完成记录；协调进程有文件锁。

输出：`outputs/eval/qwen35_vl/reference_corrections_k128_20261027_v2/`。
冻结源码：`/home/ZhouFengyu/fastv-repro/worktrees/qwen35-reference-20261027-v2/`。
运行入口：`scripts/eval/lmms_eval/run_qwen35_reference.py`，命令为prepare、launch、pipeline、gpu-gate和report。

这轮回答“修正官方Qwen规则后，原有对照排名变化多少”；不训练新自方法，也不依据新测试结果调整对照超参数。

## 启动验收记录

2026-10-27实际完成：隔离Transformers5.19下58项相关测试通过；Qwen2.5原Transformers4.57环境18项通过。四个旧训练head的实现哈希与严格CPU加载保持一致。

两个GPU各自的真实9B gate均通过：三个任务的EADP/MMTok/PruneSID各9次官方选集/顺序比较一致，Bifocal词向量与官方函数一致；四种方法identity的生成分数/Token与dense一致；PruneSID新旧binding的生成分数/Token一致。

协调进程PID266187已启动；GPU0进入Bifocal Stage1的ChartQA2500题，GPU1进入MMTok的ChartQA2500题，两边smoke已通过且已写出正式逐请求预测。其余任务由队列接续。最新进度以输出目录的manifest、worker_gpu0/1和各单元complete记录为准。本段为启动记录，不代表9组已经完成。
