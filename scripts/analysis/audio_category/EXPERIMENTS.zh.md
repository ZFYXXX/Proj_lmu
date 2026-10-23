# 当前六组本质属性实验

旧 task=sound/music/speech 三分类与 dataset 来源识别已取消。本文件对应当前实现。
这里借鉴用户给定文献的分析问题，不声称复现 HEAR、DCASE、MERT、MARBLE 或 HuBERT 的官方评测。

| 实验 | 当前目标 / 读出 | 干预 | 实际输出 |
| --- | --- | --- | --- |
| S1 Sound Event Identity | QA 显式词证据推断粗事件类别；各层有效 token mean pooling；相同配置 linear + 一层 MLP | +6 dB gain、20 dB SNR 白噪声、250 ms circular shift；离线 token 压缩 | Accuracy、Macro-F1、增强 prediction consistency、增强准确率、层曲线 |
| S2 Temporal Event / Onset | waveform 的正向 spectral flux 和 onset_detect 弱 onset；对原生 token 时间点赋二元标签，线性分类 | 250/500/1000 ms chunk shuffle；离线 token 压缩 | frame Accuracy、Macro-F1、AUPRC、event F1、匹配 onset 定位 MAE；drift、Sound 原参考 QA ΔAcc |
| M1 Tonal / Pitch–Harmony | CQT chroma 12 维回归、dominant pitch-class 分类、WORLD dominant F0 回归、clip key 分类、clip chroma 回归 | -2/-1/+1/+2 semitone pitch shift；离线 token 压缩 | Pitch-class / weak-key Accuracy、chroma correlation、log2 F0 R²；chroma 循环移位等变相关、直接不变相关、drift、Music 原参考 QA ΔAcc |
| M2 Rhythm / Beat / Tempo | beat_track 弱 beat 标签；BPM 回归；onset-envelope 自相关峰值作为周期性目标 | 保音高 stretch rate 0.8/1.2；half-beat segment shuffle；离线 token 压缩 | Beat F1/AUPRC、BPM R²/MAE、4% tempo accuracy、允许半/倍速的 tempo accuracy、周期性 R²、drift、QA ΔAcc |
| SP1 Phonetic / Linguistic | ASR + MFA phone intervals；原生 token 按区间赋 phone；silence 单列；线性 phone 分类 | 离线 token 压缩，原训练 probe 固定 | Phoneme Accuracy/Macro-F1、非静音 Accuracy、折叠序列 PER、held-out CCA correlation、预测标签 NMI、ΔPER |
| SP2 Prosody | WORLD voiced log2 F0、log RMS energy 独立回归头；aligned words/full duration 的 speaking rate；ARPA vowel stress 分类 | WORLD 重合成对照、pitch flatten、局部 energy normalization、stretch 0.8/1.2；离线 token 压缩 | F0/energy/rate R²、correlation、MAE；stress Accuracy；drift、Speech 与 prosody-sensitive QA ΔAcc |

## 共用协议

完整冻结 Qwen2.5-Omni，使用原生 MMAU prompt、选项、生成配置和 mmaunew3_accuracy scorer。
C 是第一层 encoder 输入；E_000… 为各层输出；P_in/P_out 为 projector 两侧。
特征不包括文字、参考答案、文件名、dataset 信息。音素等标签只供 probe 监督。
为覆盖完整 waveform，按每次输入时长扩展 feature extractor 的音频容量，并验证 token
时间覆盖；这项处理长度设置与可能截断长音频的默认评估不同，记录于 runtime.json。

encoder 时间坐标按每个音频分块的真实起点及 stride-2 convolution 计算；projector 时间坐标
按 native kernel=2/stride=2/floor 平均池化取配对中心。保存的是名义时间中心，不是
attention 的完整感受野边界。没有按 hidden 长度均匀铺满整段音频的近似。

按原音频路径做外层 5 折 KFold；同一音频的所有 token、变体及压缩表示都在同一折。
不同层复用相同随机种子与符合标签条件的音频分组。缺失标签会使不同属性使用不同子集。
没有原始长录音/说话人 provenance，因此不能宣称严格的说话人或原录音独立划分。

StandardScaler 仅在外层训练帧拟合。Ridge alpha=1，分类用 balanced class weights。
S1 另有固定 hidden=128 的一层 MLP。每个训练 clip 最多取 256 个等距有效帧，
避免长音频支配训练量；评估使用测试 clip 全部有效帧。超参数不使用测试成绩选择。
当前没有 bootstrap probe 区间和 permutation p 值；不要把上一版实验的统计协议用于本版。

## 弱标签的含义

- S1：6 个粗事件类别是 QA 推断的弱标签，不是 AudioSet 多标签真值；多源冲突与否定自动排除，
  可在审阅 JSONL 中修正。未审阅比例与排除原因保留在结果中。
- S2：flux onset 是声学突变代理，不等于语义事件完整 onset/offset。未实现语义 offset 标注。
- M1：CQT chroma 包含混合音高，argmax 只是主导 pitch-class；WORLD 在复调音乐上只是近似。
  key 是模板算法预测，不能当作人工调性真值。chroma 对 octave 本身不敏感；
  本实验不能仅凭 chroma 曲线证明某层形成 octave-invariant harmony 理解。
- M2：BPM/beat 来自 tracker，可能有半速/倍速和拍点错误。少于 3 个 beat 的 clip 不做 BPM 目标
  或 beat-phase shuffle；beat/no-beat 帧标签仍记录算法输出。周期性目标是波形 flux 的自相关峰，
  不是把表征的自相关数值冒充人工节拍真值。另在 music_periodicity.csv 中输出各层中心化
  embedding 在弱 beat 周期 0.5/1/2 倍延迟处的自相关，不额外训练读出头。
- SP1：采用实际音素区间，默认 ASR/MFA 均为弱监督。PER 对预测逐帧标签 run-length collapse、
  删除 silence，再与参考音素序列计算 edit distance / reference phones。相邻重复音素和对齐偏差
  会影响这个解码代理；它不是 CTC 训练或完整 ASR 的 PER/WER。
  CCA 使用训练帧中的最多 4,096 个样本，训练标准化与最多 32 维 PCA，再拟合第一 canonical pair，
  在已见 phone 类别的测试帧计算相关。NMI 是预测 phone 与参考 phone 的 NMI，不是 I(X; phone)。
- SP2：F0 仅在 voiced teacher 帧评价，不跨 unvoiced gap 插值；energy 是 log RMS。
  speaking rate 明确定义为 aligned lexical tokens / full clip seconds（含停顿），并非 syllables/sec。
  ARPA stress 反映词典重音弱标注，不等同于句子中的真实强调或韵律重音。

## 指标、阈值与干预解释

onset/beat 训练正样本为距离 teacher 时间 ≤40 ms 的 token。Ridge 决策阈值固定为 0，
不在测试集上选阈值；事件提取取正分局部峰。onset 匹配容差 50 ms，beat 70 ms。
使用一对一匹配，重复预测记 FP；定位 MAE 只对匹配事件计算，同时报告 TP/FP/FN。
这些阈值与 HEAR/DCASE 官方完整流程不同，分数不能直接与其榜单比较。

Sound circular shift 会在拼接点产生新的边界，噪声也可能掩蔽事件；这里的增强一致性不等于
人工确认 event identity 完全不变。Music pitch-shift 等变指标对 clip chroma 预测循环平移
±1/±2 个 pitch-class bin 后与变体预测比较。这个分析与直接余弦 drift 是不同指标。

time-stretch rate=0.8 表示长度变为 1/0.8；rate=1.2 表示长度变为 1/1.2。
token drift 在 changed_time × rate 的时间轴上插值对齐。其他干预使用原时间轴；
shuffle 的该 drift 有意包含顺序改变，不做逆置换恢复。

pitch flatten 使用 WORLD 将 voiced F0 设为该 clip 的中位数，保留 voiced/unvoiced 状态、
spectral envelope 和 aperiodicity。另有相同处理链的原 F0 重合成对照；
报告同时给出 flatten 相对 clean、相对 world_resynthesis 的 QA 差值，仍不能保证绝无合成伪影。

QA 比较始终保留原问题和参考答案。移调、变速、打乱事件顺序后，正确答案可能改变。
因此列名为 intervened_original_reference_accuracy，差值只表示对原参考答案的敏感性，
不是经过重新标注的反事实正确率。prosody 子集由原始技能标签选择：
Dissonant Emotion Interpretation、Phonemic Stress Pattern Analysis、
Emotion State summarisation、Emotion Flip Detection。

离线压缩使用相同的 clean-trained probe。uniform_drop 均匀保留原生 token；
mean_pool 合并相邻 token。对帧任务插值回同一原时间轴，维持相同评价集合。
它能比较时间信息/属性读出的保留程度；不能推断真实部署压缩器的 QA 准确率或速度。
mean pooling 后再全局 mean 的 clip 表征可能保持不变，这类 S1 不敏感不能解释成“所有压缩都安全”。

## 方法参考与工具文档

- [HEAR：通用音频表征评估](https://proceedings.mlr.press/v176/turian22a.html)
- [HEAR tasks：scene / timestamp 任务与 onset 口径](https://hearbenchmark.com/hear-tasks.html)
- [MERT：CQT musical teacher](https://arxiv.org/abs/2306.00107)
- [MARBLE：音乐表征多任务评估](https://arxiv.org/abs/2306.10548)
- [Pasad 等：逐层语音表征分析](https://arxiv.org/abs/2107.04734)
- [2026 prosodic/phonemic layer-wise analysis](https://www.jstage.jst.go.jp/article/pjsai/JSAI2026/0/JSAI2026_1YinA46/_article/-char/en)
- [MFA 3.x align 与模型/词典接口](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/alignment.html)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [WORLD Python wrapper](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder)
