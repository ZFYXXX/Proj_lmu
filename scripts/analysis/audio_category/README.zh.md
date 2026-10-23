# MMAU test-mini：Sound / Music / Speech 本质属性实验

本目录已替换旧的三类任务分类、dataset 来源分类和技能 QA 汇总实验。
当前入口运行 S1、S2、M1、M2、SP1、SP2。数据仅使用 mmaunew3_test_mini：
Sound 333、Music 334、Speech 333，共 1,000 条；完整/merged Qwen2.5-Omni 3B/7B 冻结。

详细协议、指标和边界见 [EXPERIMENTS.zh.md](EXPERIMENTS.zh.md)。

## 环境与路径

沿用项目 GPU 环境：Transformers 4.57.3、支持 dataset scripts 的 datasets 3.x、
qwen-omni-utils、完整模型权重和可用的 FlashAttention。额外依赖：

~~~bash
export PYTHON_BIN=/absolute/path/to/gpu/python
"$PYTHON_BIN" -m pip install -r scripts/analysis/audio_category/requirements.txt
export CHECKPOINT_PATH=/absolute/path/to/Qwen2.5-Omni
export MMAU_MINI_JSON=/absolute/path/to/AudioMCQevaluator/mmau-test-mini.json
export MMAU_MINI_AUDIO_ROOT=/absolute/path/to/AudioMCQevaluator
~~~

MMAU_MINI_AUDIO_ROOT 是 test-mini-audios 的父目录。
需要运行项目根目录下的 Python 模块；两个 sh 入口会自动定位根目录，可从任意目录调用。
无 FlashAttention 时显式追加 --attention-backend eager；长音频显存会增加。

## S1 事件标签

默认使用问题/正确答案中的显式声源词，生成带证据的弱标签。
粗类别为 human_action、animal、machine、impact、ambient、music。
多类别冲突、否定和没有明确词证据的样本排除，不强行归类。
标签不输入 encoder，问题/答案也不拼入 probe 特征。

可先导出审阅文件，逐行修改 event_label、evidence、reviewed、provenance：

~~~bash
"$PYTHON_BIN" -m scripts.analysis.audio_category.labels \
  --mini-json "$MMAU_MINI_JSON" --audio-root "$MMAU_MINI_AUDIO_ROOT" \
  --output /absolute/path/to/event_labels.jsonl
~~~

正式运行传入 --event-labels /absolute/path/to/event_labels.jsonl。
自定义文件可以只覆盖部分 ID，未覆盖的 Sound 样本仍使用规则弱标签。
reviewed 只表示人工审阅了该行，不自动将 QA 推断标签升级为音频人工真值。

## SP1 音素标签：ASR → MFA

SP1 必须有真正的 phone 时间区间。模型不会用字符、单词时间戳或 QA 答案冒充音素。
提供自动流程：faster-whisper 转写 → MFA 3.x 强制对齐 → phone/word JSONL。
英文默认 small.en + english_us_arpa；其他语言必须选择匹配的 ASR 语言、词典和 acoustic model。
MMAU 没有逐片段语言/转写真值，自动转写和对齐需抽查，多说话人重叠会降低可信度。

MFA 需要 Kaldi，建议独立 conda 环境。以下是用户在服务器执行的安装与准备命令：

~~~bash
"$PYTHON_BIN" -m pip install -r scripts/analysis/audio_category/requirements-speech.txt
conda create -n mmau-mfa -c conda-forge "montreal-forced-aligner>=3,<4"
conda run -n mmau-mfa mfa model download dictionary english_us_arpa
conda run -n mmau-mfa mfa model download acoustic english_us_arpa

"$PYTHON_BIN" -m scripts.analysis.audio_category.prepare_speech \
  --mini-json "$MMAU_MINI_JSON" --audio-root "$MMAU_MINI_AUDIO_ROOT" \
  --output-dir /absolute/path/to/mmau_speech_labels \
  --mfa-bin /absolute/path/to/conda/envs/mmau-mfa/bin/mfa \
  --device cuda --device-index 0
~~~

输出 speech_alignment.jsonl，以及可检查的 transcripts.json、corpus 和 aligned。
ASR 模型名会下载模型；使用本地路径可避免重复下载。语音准备阶段使用单独的指定设备，
完成后再启动双 GPU 主实验；不会同时加载 ASR 和完整 Qwen 模型争用同一张卡。

已有标注可通过 --transcripts-jsonl 提供每行 {id, transcript}，或通过 --textgrid-dir
导入按原 doc_index 四位数字命名的 TextGrid，如 0007.TextGrid。
TextGrid 必须包含 phones 和 words tier。直接提供等价 JSONL 也可以：

~~~json
{"id":"真实MMAU样本ID","phones":[{"start":0.1,"end":0.3,"label":"AA1"}],"words":[{"start":0.1,"end":0.6,"label":"example"}],"provenance":"ASR+MFA","language":"en","weak_labels":true}
~~~

时间单位是秒，相对完整音频起点。每个 tier 的区间必须有序且不重叠。
SP1 使用去重音数字的音素；SP2 stress 使用 ARPA 元音末尾 0/1/2。
其他音素集没有这个标记时，stress 分支会明确缺失，不猜测重音。
没有任何 phone 标注时，包含 SP1 的正式运行会提前报错；--dry-run 会显示缺失数量。
部分缺失会在各 probe 的 coverage/excluded 中记录。

## 双 GPU 正式运行

~~~bash
bash scripts/analysis/audio_category/run_2gpu.sh \
  --event-labels /absolute/path/to/event_labels.jsonl \
  --speech-labels /absolute/path/to/mmau_speech_labels/speech_alignment.jsonl \
  --output-dir runs/analysis/audio_category/intrinsic_2gpu_001
~~~

默认 GPU 0、1，各加载完整模型并按样本分片。全量六组实验时每卡 500 条原始样本。
每条原音频和各个适用干预分别执行原生答案生成；会比旧实验产生更多次 GPU 推理。
若所有干预均有效，全量六组实验最多生成 7,001 次答案。
所有所选 encoder 层一次采集，C、P_in、P_out 始终保留。GPU 结束后运行 CPU probe。
本地 mini 有 66 条音频超过 30 秒；本入口按完整 waveform（含变速后音频）扩展
feature extractor 的最大长度，并检查实际捕获时长。该长度策略记录在 runtime.json，
不会静默沿用可能截断音频的 30 秒容量，因此不能与截断配置的 QA 成绩直接混用。
--gpus 2,3 可更换两张卡。--output-dir 必须是新目录，不自动续跑不完整分片。

先检查计划：

~~~bash
bash scripts/analysis/audio_category/run_2gpu.sh \
  --speech-labels /absolute/path/to/mmau_speech_labels/speech_alignment.jsonl \
  --dry-run
~~~

先做小规模接线验证；缺少数据的 probe 会输出 insufficient_data：

~~~bash
bash scripts/analysis/audio_category/run_2gpu.sh \
  --experiments S1,S2,M1,M2,SP2 --limit 12 --layers 0,1 \
  --output-dir runs/analysis/audio_category/intrinsic_smoke_001
~~~

没有准备音素标签时，可显式选择 S1,S2,M1,M2,SP2；这不会被报告为已经评估 SP1。
SP2 的 F0/energy 仍能运行，speaking-rate/stress 需要真实 word/phone 标注。
--interventions none 只采集原始音频，用于诊断；正式干预实验保留默认 all。

## 特征压缩与资源

默认 --keep-ratios 1,0.5,0.25，--compression-method uniform_drop。
对每层原生 token 均匀保留指定比例，冻结 clean-trained probe，在相同原时间坐标评估。
也支持 --compression-method mean_pool，连续 token 分组求均值。
帧任务通过压缩 token 中心插值回原时间轴；clip 任务对保留 token 求均值。
均匀 mean pooling 可能几乎保持全局均值，因此 S1 clip probe 对这种压缩不敏感是预期现象。

这是一项真实减少离线表征 token 数的读出实验，不修改完整模型推理，不测加速，
不输出“压缩模型 MMAU ΔAcc”。MMAU ΔAcc 来自 waveform 干预。
如需评估某个部署压缩器，必须另接其实际前向执行与时间 provenance，不能套用这里的数值。

全层原始 token 特征保存为 float16，均值为 float32，时间坐标为 float64。
磁盘占用随音频总时长、层数和 hidden 维度增长，可能达到数十至数百 GB。
干预只落盘 clip 特征和与原音频的 token drift，避免为每种干预重复保存全层序列。
每次 CPU 读取一个 domain 的一个表征位置；默认每个训练 clip 最多等距取 256 个有效帧，
测试 clip 使用全部有效帧。可通过 --max-train-frames 调整，并在论文中报告该设置。
S1 MLP 默认一层 128 hidden、最多 300 次迭代；其余任务使用固定 alpha=1 的 Ridge。
这些参数不依据 test-mini 测试折成绩调节。

## 输出

- plan.json、selected.json：参数、原样本序号、标签快照、覆盖率。
- shard-*/NNNN/clean.npz：各层原生 token 向量、物理时间、clip 均值。
- shard-*/NNNN/targets.npz、labels.json：弱目标、算法来源、事件和音素标签。
- shard-*/NNNN/<variant>.npz：干预后的各层 clip 均值。
- shard-*/samples.jsonl：原生答案、原生 scorer 细节、干预结果、逐层 drift。
- report/probe_metrics.csv：各 probe、层、head、保留比例的指标与相对 clean 差值。
- report/probe_predictions.jsonl：每个折外 clip 的逐帧/clip 预测、时间、保留 token 数、未知类别。
- report/probe_interventions.csv：S1 增强一致性、M1 chroma 移调关系的逐 clip 结果。
- report/probe_intervention_summary.csv：上述指标按 probe、层、head 和变体汇总，保留有效 n。
- report/qa_interventions.csv：成对原参考答案准确率差值、95% paired bootstrap 区间。
- report/representation_drift.csv：逐样本、干预、层的 clip/token 余弦距离。
- report/representation_drift_summary.csv：按 domain、干预、层汇总的平均距离。
- report/music_periodicity.csv：表征在弱 beat 周期的 0.5/1/2 倍物理延迟处的中心化自相关。
- report/folds.json：每个读出头使用的音频级训练/测试 ID。
- report/summary.json、status.json：完整指标、缺失标签、被跳过干预和协议状态。
- report/figures：六组逐层图，以及干预 QA 图，PNG/PDF。
- complete.json：采集/报告完成，以及各 probe 是否有足够数据；不等价于所有分支均有效。

~~~bash
"$PYTHON_BIN" -m scripts.analysis.audio_category.report --run-dir /absolute/path/to/run
"$PYTHON_BIN" -m pytest -q scripts/analysis/audio_category/test_audio_category.py
~~~

CPU 重建不重新加载 Qwen。合成测试图不代表真实模型成绩。

2026-09-23 本地验证：13 项 CPU 测试覆盖原生小型编码器时间映射、分组划分、
事件匹配、真实信号算法/变换、TextGrid 音素读取、长音频容量及六组实验完整合成报告。
另核对真实 mini 全部 1,000 条原生任务顺序、500/500 双卡分片，以及三类各一条真实
音频的全部适用干预。S1 默认规则当前标注 123/333 条，其余明确排除。
未运行完整 Qwen GPU 推理或 ASR/MFA 模型下载与强制对齐。MLP 在合成测试中出现迭代
上限警告；正式结果的 folds.json 会记录每个 MLP 是否到达上限，便于判断拟合状态。
