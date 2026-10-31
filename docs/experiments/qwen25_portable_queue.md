# Qwen2.5-VL-7B 可迁移串行评测入口

入口：`scripts/eval/lmms_eval/qwen25_portable_queue.py`。在已有 GPU 作业中运行；
模型与数据只读，输出、任务配置和缓存写入指定目录。运行前需要安装本项目及匹配的
lmms-eval 环境；P6000 本次验证使用 Transformers 4.57.3。不同版本不能仅凭版本号
保证一致，迁移后应记录 lmms-eval 代码版本并先用有限样本验证。
P6000 当前依赖还包括 Python 3.10、PyTorch 2.8.0+cu128、datasets 3.6.0、
pyarrow 25.0.0、accelerate 1.14.0、qwen-vl-utils 0.0.14、lmms-eval 0.7.2。
当前实际 lmms-eval 是一个独立目录中的 editable 安装，并非本仓库已提交的完整源码；
该目录的代码与本机另一份 lmms-eval 不完全一致，且没有可识别的 Git commit。
因此仅同步本项目 Git 和写相同版本号并不足以保证复现；需要另外确认 LMU 使用了
与 P6000 实际评测一致的 lmms-eval 源码，并安装本项目的 lmms-eval 模型插件。
P6000 实际导入的部分文件 SHA-256（用于识别版本，不代表覆盖了整个依赖树）：

```text
lmms_eval/utils.py                           dfb88f4e34d6f4bc36805100c2e6dce57bbb7648187f565fa587b25704c796d7
lmms_eval/tasks/mmmu/utils.py                eeec16447100cba9c9036300ea5a092b771f17399b48901a313c4c0c9f82d30b
lmms_eval/tasks/ai2d/utils.py                9dd3501d7ff3a61f77c6486e600140e0d9c92e2eaa2a3fbfdd56218f0b42e8f1
lmms_eval/tasks/refcoco/utils_rec.py         7826aa20c976507c5c262f2be13aac2c355d50f13317ab24344b0d9f6a3bb560
lmms_eval/tasks/hallusion_bench/evaluate_hb.py cf45027a0d7c9c61b7a962409e4f5a4b06a1c8a3e6b7fdbf3e2f4c3bc630b5e9
```

## 输入

- `--data-root`：包含 `GQA/`、`VQAv2/`、`ai2d/` 等目录的数据根目录。
  目前要求各目录内的 parquet 组织方式与 P6000 一致；分片缺失会报错。
- `--model`：本地完整 Qwen2.5-VL-7B 权重目录。
- `--reference-identity`：训练时保存的 `model_identity.json`。
  目标机器模型路径不同时，程序会读取整个模型并重新计算哈希，确认与训练基座一致后
  生成目标机器上的身份记录，绝不跳过或伪造身份检查。
- `--checkpoint METHOD=DIR`：训练选择器的权重目录，可重复提供。
  方法名为 `old_mlp`、`visual_selector`、`image_context_selector`、
  `question_context_selector`、`protected_selector`。Git 默认不包含这些权重或身份文件，
  要单独复制；无需复制它们的优化器状态用于推理。
- `--output`：新的可写运行目录。`--cache-dir` 可选，用于指定共用的可写缓存目录。

下面是命令模板。替换尖括号内的路径后，在已获 GPU 的作业环境里运行；
实际分区、时限等资源参数由 LMU 作业配置决定。

```bash
python scripts/eval/lmms_eval/qwen25_portable_queue.py \
  --data-root /hkfs/work/workspace/scratch/tum_yvc3016-comp/mllm_dataset/lmms_lab_dataset \
  --model /hkfs/work/workspace/scratch/tum_yvc3016-comp/workspace/ckpts/ckpts-lmm/Qwen2.5-VL-7B-Instruct \
  --output '<可写目录>/qwen25_run_001' \
  --reference-identity '<权重目录>/model_identity.json' \
  --checkpoint 'old_mlp=<权重目录>/old_mlp' \
  --checkpoint 'visual_selector=<权重目录>/visual_selector' \
  --checkpoint 'image_context_selector=<权重目录>/image_context' \
  --checkpoint 'question_context_selector=<权重目录>/question_context' \
  --checkpoint 'protected_selector=<权重目录>/protected' \
  --methods dense eadp prunesid mmtok old_mlp visual_selector image_context_selector question_context_selector protected_selector \
  --tasks gqa vqav2_val scienceqa_img mmmu hallusionbench textvqa \
          refcoco_bbox_rec_testA refcoco_bbox_rec_testB refcocog_bbox_rec_test \
  --mmmu-single-image-only
```

示例中的引号用来保护 `+`、空格等字符；尖括号是占位符，不是实际目录名。
默认 GPU 是当前进程的 `CUDA_VISIBLE_DEVICES` 中第一张；也可用 `--gpu` 显式指定。

`scripts/eval/lmms_eval/lmu_qwen25_queue.sbatch` 是单节点、4×A100、8 小时的
Slurm 模板，作业名以 28 开头并禁用邮件。它先对 LMU 模型做一次完整内容校验，随后
按数据集启动四个独立 worker：GPU0 跑 GQA，GPU1 跑 VQAv2，GPU2 跑
ScienceQA/TextVQA，GPU3 跑 MMMU/HallusionBench/三个 Grounding 切分。每张卡内部
按固定顺序跑九个方法，避免方法并行造成同一基准或输出目录的竞争；四个 worker 使用
独立运行目录和日志，完成后生成顶层 `parallel_status.json`、`parallel_summary.json`
和 `parallel_report.md`。顶层汇总记录 Slurm job/node、主机和可见 GPU，
`logs/gpu_usage.log` 每分钟记录四卡利用率、显存和功耗。提交前需按模板中的必需环境变量设置
解释器、数据、模型、输出、模型身份和五个已训练方法的权重路径。先设置 `EVAL_LIMIT=2`
在已申请的 GPU 环境中调试；正式全量需取消该变量并使用新的 `RUN_DIR`。不能用
调试目录续跑全量。模板目前只通过本机 shell 语法检查，尚未在 LMU 执行或验证；
尤其要先确认 ChartQA 路径，不能仅替换两个路径就假定环境已可复现。
模板列出 9 个待补充的评测切分，并用 `--mmmu-single-image-only` 排除多图题；
已在 Qwen2.5-VL-7B 跑过的 ChartQA、OCRBench、AI2D、MMStar、POPE、OK-VQA、
RefCOCO+ testA/testB 和 DocVQA 不在默认队列中。DocVQA 只确认部分方法已有结果，
为避免重复，默认也不补其他方法；如需补组需另行明确指定。
TextVQA 和 ScienceQA 的旧全量记录是 3B，不当作 7B 结果，因此列入待补充清单。
HallusionBench 只生成并按分片保存完整的官方裁判输入，标记为待评分，不调用外部服务；
四卡运行后可运行 `python scripts/eval/lmms_eval/export_hallusion_for_judge.py <运行目录>/worker_3`
导出各方法的 JSONL。受限样本检查必须显式加 `--allow-limited`，不能作为成绩。
官方评分需要另行配置和运行裁判，导出不等于已经完成评分。后续若明确同意发送
题目、参考答案和预测到外部服务，可在配置 API 凭据后运行
`python scripts/eval/lmms_eval/score_hallusion_official.py <运行目录>/hallusion_judge_input --confirm-external-judge`；
该命令不由 GPU 队列自动调用。它调用当前安装的上游裁判，要求 API 请求成功才继续，
并按方法保存裁判缓存和单独的分数文件。两阶段生成与导出路径已在 P6000 有限样本上
完成验证；外部裁判未调用。
MMMU 子集的原始数量及排除数量会写入每个任务结果。

## 全量和测试

不提供 `--limit` 时，对选中任务的所有分片进行完整评测。先加
`--limit 2 --first-shard-only` 做最小流程检查；该模式仍先检查分片集合是否齐全，
但只加载首个分片、每个任务只生成两条，结果不得当作全量成绩。
两种模式分别使用新的输出目录，不能互相续跑。

每个方法按 dense、其余方法的顺序执行；每个数据集再按 `--chunk-size`（默认 512）
分片。完成的分片写入 `cells/<method>/<task>/chunk_<起点>/complete.json`，
失败立即停止并保留尝试目录和日志。原参数不变时重跑同一命令并追加 `--resume`
可跳过已完成分片；中断的分片从头重跑，不是逐样本续跑。分片结束后调用数据集原生
聚合函数处理全部逐样本结果，不直接平均分片得分。该新版续跑流程及九种方法入口已在
P6000 有限样本上完成验收；这只证明代码路径可运行，不代表 LMU 环境或全量成绩已经验证。
完成后 `summary.json` 保存原始分数及相对同任务 dense 的百分比，`report.md` 是表格。
在 dense 得分为 0 的小样本场景中比例标为缺失，不除以零。

## 当前已知限制

- 目前 LMU 提供的目录清单中没有 ChartQA 路径；ChartQA 不在默认待补充列表中。
- MMTok 的运行记录尚未提供显式首个 decode 坐标验证，需进一步核对。
- HallusionBench 正式 aAcc/qAcc/fAcc 使用上游外部裁判，需要配置可用服务与凭据；
  生成阶段不计算正式得分。上游调用失败时可能把失败转成 unclear，后续接入裁判时
  应检查请求状态，避免静默误评分。
- P6000 的 MMMU val 按上游完整 prompt（含选项）检查有 861 道单图、39 道多图，
  没有无图题。此前只检查 question 字段而漏掉选项中图像，所说的 13 道无图并不正确。
  `--mmmu-single-image-only` 会在原生预处理后筛选单图子集并记录排除数量，该成绩
  不是 MMMU 官方全量。已确定多图按每图最多 128 个视觉 Token、无图走原模型。PruneSID 的两图及三图
  小样本已完成真实生成，队列允许 dense 与 PruneSID 全量执行；其他方法的多图绑定
  和端到端验证尚未完成，包含这些方法的 MMMU 全量请求仍会被拒绝。不能只凭小样本
  声称整套方法均已完成 MMMU 适配。
- P6000 的检查不能验证 LMU 上数据文件是否存在、格式是否相同、存储配额、
  GPU 型号及作业调度；复制后仍需要用 LMU 的路径运行一次有限样本检查。
