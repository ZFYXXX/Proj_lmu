# Qwen2.5-Omni 分模态训练入口

本目录提供与参考工程对应的八个分模态薄启动脚本，以及覆盖四个执行族的八种 active
音频压缩方法和六种 target 候选方法的 SFT 薄入口：

| 输入模态 | 全参数 SFT | LoRA |
| --- | --- | --- |
| 图像 | `train_img_sft.sh` | `train_img_lora.sh` |
| 音频 | `train_aud_sft.sh` | `train_aud_lora.sh` |
| 视频画面 | `train_vid_sft.sh` | `train_vid_lora.sh` |
| 视频画面与内嵌音轨 | `train_av_sft.sh` | `train_av_lora.sh` |

| 执行族 | 方法 | 生命周期 | 全参数 SFT 入口 |
| --- | --- | --- | --- |
| Projector Token | Segmentwise Top-K | active | `train_aud_sft_segmentwise_topk.sh` |
| Projector Token | LTBM | active | `train_aud_sft_ltbm.sh` |
| Projector Token | SA-MAP | active | `train_aud_sft_sa_map.sh` |
| Projector Token | FastSLM HTA | target | `train_aud_sft_fastslm_hta.sh` |
| Projector Token | UniAvg | target | `train_aud_sft_uniform_average_pooling.sh` |
| Projector Token | UniSamp | target | `train_aud_sft_uniform_sampling.sh` |
| Projector Token | UnSeg | target | `train_aud_sft_unsupervised_segmentation.sh` |
| Encoder Token | A-ToMe | active | `train_aud_sft_atome.sh` |
| Encoder Token | Encoder Attention TopK | active | `train_aud_sft_encoder_topk.sh` |
| Encoder Token | Progressive Attention Pruning | active | `train_aud_sft_progressive_pruning.sh` |
| LLM Token | Audio-Pyramid | active | `train_aud_sft_audio_pyramid.sh` |
| LLM Token | FastAdaSP | target | `train_aud_sft_fastadasp.sh` |
| LLM Token | HeadRouter | target | `train_aud_sft_headrouter.sh` |
| LLM Operation | SPADE Operation-Only | active | `train_aud_sft_spade_operation_only.sh` |

这些脚本都只选择对应的 `configs/train/qwen2_5_omni/train_aud_sft_<method>.yaml`，不复制算法实现或训练参数。是否已经完成完整 Swift checkpoint/Manifest、目标规模收敛或质量实验，必须按下文和验证状态逐方法判断，不能从“存在脚本”推导。

基线与八个 active 方法的验证范围见
[`Omni 状态`](../../../docs/support/qwen25_omni/status.zh.md)。六个候选仍保持
`target`/`declared`，可按实际机制合同进入训练；其已有数值测试、HTA 选定组合的真实
联合训练与完整恢复，以及其余场景的验证边界，统一见
[`六个候选方法状态`](../../../docs/support/qwen25_omni/audio_compression/candidate_methods.zh.md)。

其他六个常规模态入口已经通过配置和启动计划契约检查，在完成目标规模实跑前不声明
训练质量验证通过。压缩训练的逐断点验收分别见
[`Projector-Level Segmentwise 手册`](../../../docs/support/qwen25_omni/audio_compression/debugging/projector_level.zh.md)
和 [`Encoder-Level A-ToMe 手册`](../../../docs/support/qwen25_omni/audio_compression/debugging/encoder_level.zh.md)。

脚本只负责选择对应 YAML，并设置模型、数据集、输出目录、分布式和 W&B 运行环境。模态参数位于 YAML 的 `backbone.config`，训练参数位于 `train.args`；脚本不重复保存这些数值，也不依赖 `_common.bash`。

```text
shell entry -> mllm-comp train -> MSSwiftTrainAdapter -> swift sft
```

## 使用

先检查最终 Swift 命令：

```bash
bash scripts/train/qwen2_5_omni/train_img_lora.sh --plan_only

bash scripts/train/qwen2_5_omni/train_aud_sft_segmentwise_topk.sh --plan_only
bash scripts/train/qwen2_5_omni/train_aud_sft_atome.sh --plan_only

bash scripts/train/qwen2_5_omni/train_aud_sft_audio_pyramid.sh --plan_only

bash scripts/train/qwen2_5_omni/train_aud_sft_headrouter.sh --plan_only

DATASET_PATH=/data/train_video.jsonl \
bash scripts/train/qwen2_5_omni/train_vid_lora.sh --plan_only
```

正式运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
DATASET_PATH=/data/train_video.jsonl \
bash scripts/train/qwen2_5_omni/train_vid_lora.sh

DATASET_PATH=/data/train_audio.jsonl \
OUTPUT_DIR=/data/runs/audio_sft_compressed \
bash scripts/train/qwen2_5_omni/train_aud_sft_atome.sh
```

脚本支持覆盖 `MODEL_PATH`、`DATASET_PATH`、`OUTPUT_DIR`、`PYTHON_BIN`、分布式环境变量以及 `WANDB_PROJECT`/`WANDB_MODE`。图像和音频入口保留参考工程的数据集默认路径；视频和音视频入口要求显式设置 `DATASET_PATH`。

需要修改 `max_pixels`、`use_audio_in_video`、采样率、视频预算或帧数时，修改对应 YAML 的 `backbone.config`。需要修改 batch、学习率、冻结策略或 LoRA 参数时，修改 `train.args`。额外的命令行覆盖仍可直接追加到脚本调用末尾。
