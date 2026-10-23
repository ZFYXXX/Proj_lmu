# MMSU-final：A / B / C / E / H 冗余实验

所有新增代码位于本目录。通过运行时 hook、现有 `LMMSEvalAdapter` 和临时 worker bridge
开展实验，不修改 `mllm_comp` 主框架、模型权重、任务 YAML、问题、选项、答案或评分器。
用户提到的 `mmsufinalj` 已确认指现有 `mmsufinal`。

## 环境与数据

- 完整或 merged Qwen2.5-Omni 3B / 7B checkpoint，包含顶层 `thinker_config`。
  不支持仅 LoRA adapter 或仅 thinker 的导出目录。
- 项目原 GPU 环境：Python 3.10、Transformers **4.57.3**、PyTorch CUDA、lmms-eval、
  accelerate、qwen-omni-utils；四个进程各加载一个模型副本、每卡 batch size 1。
- 安装本目录 `requirements.txt` 中的分析依赖。无需新的神经网络 checkpoint。
- `MMSU_JSONL` 指向 AudioMCQ 的 MMSU.jsonl；`MMSUFINAL_DATASET_DIR` 指向包含
  `data/*.parquet` 和 `_audio_cache/` 的 mmsu 目录。前者覆盖仅发生在 worker 内，
  不改动任务文件；后者沿用任务已有的环境变量。
- 若不设置 `MMSU_JSONL`，读取仓库 mmsufinal.yaml 中的数据路径。移到服务器时应显式设置。
- `mmsufinal` 原生贪心生成和 letter scorer 保持不变，包括少数五选项记录。

## 四 GPU 一键运行

从任何目录调用均可，脚本会定位项目根目录：

```bash
export PYTHON_BIN=/absolute/path/to/gpu/environment/bin/python
export CHECKPOINT_PATH=/absolute/path/to/full-or-merged-Qwen2.5-Omni
export MMSU_JSONL=/absolute/path/to/AudioMCQevaluator/MMSU.jsonl
export MMSUFINAL_DATASET_DIR=/absolute/path/to/mmsu
export CUDA_VISIBLE_DEVICES=0,1,2,3
export ANALYSIS_RUN_DIR=/absolute/path/to/new-run-directory

# 只打印实际条件、推理次数和分组；不加载模型、不创建输出。
bash scripts/analysis/audio_redundancy/run_mmsufinal_4gpu.sh --plan-only

# 正式全量；推荐在自己的 tmux 会话中运行。
bash scripts/analysis/audio_redundancy/run_mmsufinal_4gpu.sh
```

首次运行的输出目录必须是新目录。若 native worker 在评分结果和全部逐题记录落盘后
退出异常，可以用完全相同参数和 `--resume` 继续已有输出目录；启动器会检查原生
`*_results.json`、逐题覆盖率以及校准特征，再跳过已经完整的阶段。未完成的阶段
不会被误判为成功，也不会追加到不完整的逐题日志；不完整的校准或正式评估须另建输出目录，
中断的 CPU 码本拟合则可以复用校准特征重新拟合。
所有入口（单项脚本、四卡 `all` 脚本、直接 Python 调用）共用一个不加载 GPU adapter 的
监督进程，各阶段在独立 Python 子进程执行。默认开启 `faulthandler`，便于记录原生崩溃栈。
默认 66 个唯一干预，五项共用一个原始评估基线。B/E 共用一次校准捕获和分层码本：

- 校准捕获：5,000 次原始生成；
- 正式评估：5,000 × (1 原始 + 66 干预) = 335,000 次生成；
- 合计 **340,000 次生成**。四 GPU 分片不改变总次数。

码本拟合在两轮 GPU 评估之间运行于 CPU；PCA 使用 PyTorch float64、固定单线程，
聚类与转移统计使用 `--fit-threads`（默认 4）。此时 GPU 空闲是预期行为。
聚类使用 `threadpool_limits(--fit-threads)` 在运行时设置线程数；仅在 shell 设置
`OMP_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1` 不会把默认 4 线程聚类改成单线程。
3B、默认 7 个 stage、每音频每 stage 32 帧时，校准特征原始 float16 体积约 3.1 GB，
实际 NPZ 文件大小受压缩率影响。默认只为题号 0/1/2 保存详细图所需数组。

成功生成正式报告后，启动器还会自动创建 `portable/`，作为供远端下载的精简分析包。
原始逐题 JSONL、校准 NPZ、码本权重和全部示例数组仍留在运行目录。
推荐传 `--example-doc-ids 0`，减少远端本身的示例数组与图表数量。

## 先做四卡小样本接线验证

小样本必须减小 K，否则每个训练折的帧数可能不足。这里的指标不能当作正式结果：

```bash
export ANALYSIS_RUN_DIR=/absolute/path/to/new-smoke-directory
bash scripts/analysis/audio_redundancy/run_mmsufinal_4gpu.sh \
  --limit 8 --layers 0,31 --ranks 8 --keep-ratios 0.5 \
  --codebook-sizes 8 --codebook-pca-dim 4 --mel-ranks 8 \
  --bootstrap-repeats 100
```

`--layers 0,31` 适用于 32 层音频 encoder；其他深度按 checkpoint 选择。
无 FlashAttention 的验证环境可传 `--attention-backend eager`，正式比较应固定后端。

单 GPU 小样本可直接用 Python 入口（四卡脚本始终固定四进程）：

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m scripts.analysis.audio_redundancy.launch \
  --checkpoint "$CHECKPOINT_PATH" --mmsu-jsonl "$MMSU_JSONL" \
  --output-dir /absolute/path/to/new-single-gpu-smoke \
  --num-processes 1 --limit 4 --layers 0,31 --ranks 8 \
  --keep-ratios 0.5 --codebook-sizes 8 --codebook-pca-dim 4 --mel-ranks 8
```

## 独立运行与扩展

单项入口每次只评估指定的 A、B、C、E 或 H，并各自生成独立报告和 `portable/`。
不同运行之间基线不共用；B 与 E 单独运行时还会分别捕获校准特征、拟合码本。
单项脚本在输出目录旁自动写 `.launch.log`，成功后将 `portable/` 打成同名前缀的
`.portable.tar.gz`；从远端下载这个压缩包即可，无须复制整个运行目录。

**A/C/H** 的正式评估、报告在两个独立进程运行；**B/E** 和 **all** 使用校准、CPU 码本拟合、
正式评估、报告四个独立进程。若 GPU 阶段的启动进程在输出落盘后发生 SIGSEGV，监督进程会
检查原生结果和所有条件的逐题覆盖率；校准阶段还要检查全部特征文件。只有这些产物完整才继续，
并保留 `recovered_launcher_exit.json`，其副本也进入精简分析包。拟合或报告阶段失败会停止，
不会把崩溃静默记成成功。阶段隔离和显式清理不等于保证驱动或第三方原生库永不崩溃。

旧运行目录可设为 `ANALYSIS_RUN_DIR`，用完全相同参数加 `--resume` 跳过完整阶段；
不完整的码本会重新拟合。不完整的校准或正式评估仍须使用新目录。
B/C/E 的真实 token 压缩在 worker 内使用可写 runner 代理，并在正常结束或异常时恢复原 facade，
避免给主框架的 frozen runner 直接赋值。全部实验的 worker 在退出前显式销毁分布式进程组。
同一节点上使用同四张 GPU 和同一个 `MAIN_PROCESS_PORT` 时，请串行运行。

C 的旧日志若在第一题出现 `FrozenInstanceError: cannot assign to field 'generate'`，
应先同步本目录的新版代码，然后使用新的 C 输出目录重跑。后面的“没有原生聚合结果”是该异常
造成的连锁错误；仅添加 `--resume` 不能补齐这次未完成的评估。

### B/E 在 CPU PCA 拟合中的 LU / SVD 段错误

若校准的 5,000 题已完成，随后显示 `starting fit phase`，且崩溃栈为
`calibration.py → PCA.fit_transform → _randomized_range_finder → scipy.linalg.lu`，
这是 B/E 共用 CPU 拟合路径中的原生崩溃。较早的 QR 补丁只绕开 LU；远端后续日志已确认
同一 PCA 仍在 `scipy.linalg._decomp_svd.svd` 崩溃，不能把旧 QR 补丁视为完整解决方案。

当前 PCA 使用项目已有的 `torch.pca_lowrank`：在 CPU 上以 float64、固定单线程计算，
显式中心化，过采样 10 维（受矩阵尺寸限制），7 次幂迭代。PCA 不再调用 sklearn PCA
或 SciPy 的 LU/QR/SVD；没有失败后回退到这些例程的分支。局部 CPU 随机种子固定为原
`seed`，完成或 Python 异常退出后恢复随机状态及线程数，不初始化 CUDA 随机状态。
基底与均值仍保存为 float32，并统一通过 `(hidden - mean) @ components.T` 投影。
保留 PCA 维数、KMeans 参数、两折划分和 MMSU 评分；数值后端、计算精度与随机算法实现
改变会影响拟合结果，不承诺与旧码本逐位相同，正式比较应统一使用一个版本。

PCA 开始前会打印 `backend=torch.pca_lowrank device=cpu dtype=float64 pca_threads=1`，
完成后打印 `PCA complete`，然后才出现 `fit KMeans ... threads=4`（或指定线程数）。
`models/metadata.json` 记录拟合算法及线程设置；拟合开始前另写 `models/fit_environment.json`，
保留 NumPy/SciPy/scikit-learn/PyTorch 版本、已加载的 BLAS/OpenMP 库及 PyTorch 构建信息，
即使原生崩溃也可读取该文件。两个小型 JSON 均进入精简分析包。

此修复不改变启动参数或 `analysis.json` 的计划格式。对于在上述拟合步骤失败的 B/E，
同步新版代码后，把 `ANALYSIS_RUN_DIR` 设回原目录，以**原来的完整参数加 `--resume`**
调用单项脚本即可：校验并跳过完整校准，重新拟合不完整码本，再运行正式评估、报告和打包。
不要删除已有校准特征，也不要把旧目录当新目录启动。若希望先只验证 CPU 拟合，可使用
Python 启动入口，保留原计划的参数并添加 `--resume --phase fit`；通过后去掉 `--phase fit`
继续。单项 shell 包装脚本是完整评估入口，只有生成精简包后才显示“实验完成”。
改变 `--fit-threads` 等参数会触发已有的冻结计划不匹配检查，不能直接与旧目录的
`--resume` 混用。本修复保留原计划参数，PCA 内部单线程与聚类线程预算分别记录；
聚类预算为单线程/四线程时，均有禁止 SciPy LU/QR/SVD 的完整拟合回归测试。
完整旧码本仍按已有恢复规则复用；若要把已完成的旧运行改用 PyTorch PCA，应另建运行目录。

单独运行 **C** 时 `needs_calibration=false`，只执行 evaluation/report，不调用此 PCA
或 KMeans；C 的 frozen runner 修复继续生效。`--experiments all` 仍因包含 B/E 而执行拟合。
PyTorch PCA 绕开已定位的 SciPy 分解路径，但仍使用原生数值库；KMeans 也仍使用
scikit-learn。真实远端库构建和四卡环境仍须验证，不能由本地测试推断原生库永不崩溃。

| 单项 | 唯一干预条件 | 完整 5,000 题预期生成次数 | 校准 |
|---|---:|---:|---|
| A | 11 | 60,000 | 无 |
| B | 16 | 90,000 | 5,000 次生成、两折拟合 |
| C | 24 | 125,000 | 无 |
| E | 3 | 25,000 | 5,000 次生成、两折拟合 |
| H | 16 | 85,000 | 无 |

五次独立运行合计 385,000 次生成；一次性 `all` 由于共用基线及 B/C 的四个
uniform 条件，只需 340,000 次生成。单项运行方便分批执行、诊断和分别下载。

```bash
# A 观察默认五个 encoder 深度及 P_in/P_out，只在 P_in 做任务干预。
bash scripts/analysis/audio_redundancy/run_experiment_4gpu.sh A

# A 在指定层逐层干预，每次只改一层，其他层保持原生。
bash scripts/analysis/audio_redundancy/run_experiment_4gpu.sh A \
  --layers 0,8,16,23,31 --intervention-stages E_l0,E_l8,E_l16,E_l23,E_L,P_in,P_out

# 所有层观察，多个码本粒度；计算量和磁盘需求相应增加。
bash scripts/analysis/audio_redundancy/run_experiment_4gpu.sh E \
  --layers all --codebook-sizes 256,512,1024

# 选定组合仍共享基线和重复条件。
bash scripts/analysis/audio_redundancy/run_mmsufinal_4gpu.sh --experiments A,C,H

# 重新生成已有完整运行的报告，无须再推理。
"$PYTHON_BIN" -m scripts.analysis.audio_redundancy.report --run-dir "$ANALYSIS_RUN_DIR"
```

## 实验定义

| 实验 | 观测 | 默认任务干预及对照 |
|---|---|---|
| A | E_l/P_in/P_out 原始与去均值全局 SVD；0.5/1/2 秒局部 SVD | P_in 的 k=16/32/64/128/256 PCA 重建、同维数随机子空间、恒等 |
| B | 每层交叉拟合 unit；一阶 Markov predictive entropy、实际 surprisal、预测准确率 | P_in 按高熵、高 surprisal、反向低熵分配边界；同预算 uniform pooling |
| C | L2 与 cosine 变化量、分段长度和实际 tokens/s | 75/50/25/10% 预算；L2/cosine 动态、uniform pooling、uniform sampling、随机边界；cosine 段均值广播 |
| E | 分层 run 长度、cluster 占用、transition entropy、量化误差、顺序打乱参考 | 相同 run 内原 hidden 平均后 collapse；精确匹配 token 数的 uniform pooling；run 均值广播 |
| H | 去均值 log-Mel 奇异谱与残差平方范数保留率 | 均值 + rank-4/8/16/32/64 残差；匹配 MSE 的时间/频率 Gaussian smoothing；恒等 |

### A：表示维度

对有效 token 的 hidden 矩阵 SVD，不包含 padding。Effective Rank 使用归一化奇异值；
PR 和 k90/k95 使用奇异值平方。去均值矩阵可达到的最大秩是 min(T−1,d)，图中给出
effective rank / 该上限。零方差记录为空，不伪造 rank=1。

全局指单条完整音频；局部窗口从各硬块起点开始、互不重叠，尾窗可以较短，实际 token
数及 rank capacity 一并保存。局部不跨硬块。它不是重叠滑窗估计。

PCA 基底由当前音频自身计算，重建回 T×d、保留原均值；此实验回答单音频子空间可替代性，
不是一个跨音频共享、已经部署的低维瓶颈。随机子空间使用相同有效 k 和原均值。
默认干预 P_in；可用 --intervention-stages 指定捕获层，不改变 hidden 形状和 token 数。

### B/E：离散状态与预测器

按唯一 audio_path 随机分成两折，分配在读取任何任务表现前冻结。对评估折 f，码本、
PCA 和 Markov 参数只使用另一折。所有 5,000 题都有严格 audio-disjoint 的预测器。
这是两折交叉拟合协议，不是用全体测试音频训练一个共享码本，也不是外部数据训练方案。

每条校准音频每个 stage 抽一个至多 32 帧的块内连续片段；按合法起点均匀抽样。
每层使用相同 K 和训练协议，分别学习中心，不在不同表示空间中强用同一组中心。
流程为 hidden 单位范数化 → 训练折 PCA（默认 32 维，不 whitening）→ MiniBatchKMeans。
PCA 使用 `torch.pca_lowrank`（CPU float64，过采样 10，7 次迭代，固定单线程），
保存 float32 均值/基底供统一投影，不经过 SciPy 的矩阵分解路径。
最多随机抽 8,192 个校准帧拟合 PCA/聚类；Markov 转移使用该折所有缓存连续片段。
这些限制都写入 models/metadata.json。量化误差在此归一化 PCA 空间中报告。

预测器是一阶 p(c_t|c_{t−1})，加性平滑默认 0.1，每个硬块起点重置为训练折边缘分布。
同时报告熵 −Σp log2 p 和真实 surprisal −log2 p(c_t)。不把置信但预测错误当作信息少。
音频 encoder 的双向上下文仍存在，因此这里不宣称在线信号的因果熵。

B 的高熵/高 surprisal 倾向在该位置之前保留边界，low_entropy 为反向分配对照。
E collapse 使用 run 内原始 hidden 均值，不用 cluster centroid 替换，避免混入量化替换损伤。
run 在硬块处重置；顺序打乱参考仅在各块内部打乱，不改变 cluster 占用。

### C/B/E：公平预算与时间

每条音频预算 round(ratio×T)，至少每个硬块一个输出 token；若该下限超过请求预算，
同组所有方法统一使用相同可实现预算，显式记录 budget_clamped_by_hard_chunks。
原生池化已经跨块的孤立 token 不与其他 token 合并，沿用已有时间映射约定。

动态方案保留硬边界后，选变化分数最大的 B−块数 个内部边界，分成恰好 B 段。
uniform 根据块长度分配精确 B 个连续 bin。所有 pooling 是段内原 token 的一次均值。
uniform sampling 保留相同 bin 的中心原 token；随机方案仅随机边界，池化算子不变。

所有分段覆盖每个原 token 恰好一次，输出保持时间顺序。P_in 减长复用现有
compact_audio_request 和原生合法化接口，统一使用 compact position policy。
不增加模型未训练过的 duration token；保存时长元数据不意味着模型能读取时长。
广播控制把段内均值写回原长度，分别检验段内变化和真正时间减长的影响。

### H：输入联合低秩

直接在原生有效 log-Mel 上操作，不从波形反演。沿时间去除每个 Mel bin 的均值，
保留前 k 个残差奇异方向，再加回均值；总矩阵秩最多 k+1。输入维度和长度不变。
所称“保留率”是 log-Mel 残差平方范数，不是波形能量。

时间/频率反射边界 DCT Gaussian smoothing 在原矩阵上分别搜索 sigma，使 MSE 匹配对应 rank-k
重建。若单轴平滑的最大可达误差不足，则记录 matched=false、实际误差和误差差值，
不会声称已匹配；报告汇总未匹配样本数。所有 Mel 拼图使用相同颜色范围。
通过正交 DCT 和 Parseval 等式搜索 MSE，避免长音频的大 sigma 时域卷积开销；
无限平滑的极限为该轴均值，sigma=null 表示这一极限。

## 输出及解释

```text
analysis.json                       冻结的条件、音频两折分配、预算与源数据 ID
calibration/features/doc*.npz        无标签的校准片段
calibration/coverage.json            校准捕获完整性
models/fold*_stage_K*.npz            交叉拟合 PCA/码本/Markov 参数
models/metadata.json                 拟合数据规模、空间和折信息
models/fit_environment.json          数值库版本、BLAS/OpenMP 路径及 PyTorch 构建信息
evaluation/samples.rank*.jsonl       原始逐题预测、各层观测、耗时
evaluation/conditions.rank*.jsonl    配对干预、实际预算、预测及原生得分
evaluation/example*.npz             示例熵、rank 曲线、unit、分段和 Mel
evaluation/coverage.json             每个条件与全体目标 ID 精确覆盖检查
report/summary.json                 CI、容忍度、恒等检查、预算匹配比较
report/paired_task_results.csv       总体/感知推理/语言副语言/细任务结果
report/matched_budget_comparisons.csv 动态/熵/collapse 对匹配 uniform 的优势
report/rank_by_audio.csv             每音频每层全局和局部 rank
report/units_by_audio.csv            每音频每层每 K 的离散统计
report/*.png, *.pdf                  所选实验的 performance 图、rank/熵热图及示例
report/README.zh.md                  中文结果表与解释限制
portable/                             可直接复制到本地的精简分析包
```

`portable/` 包含 `summary.json`、参数和条件的 `plan.json`、完整覆盖率、
`baseline.csv.gz`、`condition_scores.csv.gz`、已有的分题型/逐音频 rank/离散统计
CSV.gz、匹配预算对照和 PNG 总图；示例图只保留一个音频。逐题压缩表保留条件、
准确率、预测文本、实际保留率、重建误差及耗时，足以在本地重新做多数配对比较。
它不包含校准原特征、码本权重和全部示例 NPZ；需要重新拟合或重画任意样本时
仍须访问远端原始运行目录。也可对已完成的旧运行执行
`python -m scripts.analysis.audio_redundancy.export --run-dir "$ANALYSIS_RUN_DIR"`
单独生成此目录。

原生 lmms-eval 输出中的 accuracy 对应未干预基线；干预成绩必须看 conditions 和 report。
报告拒绝缺题、重复题、错配题或不同实际 token 数的“匹配预算”比较。

默认按音频做 1,000 次配对 bootstrap，给出原始正确→错误、错误→正确、95% 双侧区间。
“满足容忍度”要求 ΔAcc 的单侧 95% 下界 ≥ −1 pp，可用 --tolerance-pp 修改。
细任务区间样本较少，且所有区间未做多重比较校正，不把筛选出的最优条件当作预注册结论。
恒等条件预测文本应完全一致；若不一致，报告 interpretation_valid=false，先排查接线或数值后端。

耗时分开记录原始生成、共享观测、每条件准备和生成。代码为分析而复用原始表示／SVD；
这些诊断耗时不能直接解释为独立部署加速。B/C/E 不减少 audio encoder FLOPs；A/H 不减少 token 数。

## 本地验证

```bash
python -m pytest scripts/analysis/audio_redundancy/test_audio_redundancy.py -q
bash -n scripts/analysis/audio_redundancy/run_mmsufinal_4gpu.sh
bash -n scripts/analysis/audio_redundancy/run_experiment_4gpu.sh
```

测试覆盖低秩与样本上限、局部硬边界、匹配预算、熵/surprisal、交叉拟合隔离、原生小型
Qwen 音频 encoder 的 hook/恢复、真实 frozen runner 的 B/C/E 压缩上下文、Mel 对照、覆盖检查、
五项与 all 的 SIGSEGV 恢复/停止条件、四进程计划及五组图表的完整报告生成。
另外以禁止 SciPy LU/QR/SVD 的回归测试验证 B/E 拟合和中断码本重建；将 PyTorch PCA
与精确 SVD 子空间对比，检查非有限输入、随机状态/线程恢复，并验证独立 C 无校准与拟合。
真实完整 checkpoint 的四 GPU 冒烟与正式成绩必须在目标 GPU 环境运行，CPU 测试不替代它们。
