# FastSLM HTA 的 Qwen2.5-Omni 接入说明

> 生命周期：`target`。当前实现与配置可供显式验证，但尚未进入正式 active registry；
> 激活缺口见[六个候选方法状态](candidate_methods.zh.md)。

## 实现范围

Plugin 名为 `fastslm_hta_audio`，默认接在 Qwen2.5-Omni 的
`audio_encoder_output_pre_projector`：原生 `avg_pooler -> ln_post` 之后、
`audio_tower.proj` 之前。

研究配置也可选择 `audio_projector_output_post_projector`，配置方式见
[执行位置说明](extension.zh.md#3-projector-token-executor)。以下流程和参数量描述默认配方；
选择投影后时先执行原生 projector，再由 HTA 处理 LLM 宽度的特征，不重复投影。
`num_attention_heads=0` 会按实际宽度自动推导：尽量保持 Encoder 的单头宽度，无法整除时
使用单头。显式指定的头数保持原值，不能整除目标宽度时明确拒绝；实际头数写入运行记录。

实现对应论文 arXiv:2601.06199v3 第 3.1 节、图 2、公式 (1)-(4) 和官方
`FastSLM-8BD3/models/model.py`：

1. Stage 1 用 80 个可学习 query 对当前音频特征做带正弦位置编码的 cross-attention；
2. 两个双卷积 Downsampler 分别以 stride 2 降采样，Stage 2/3 各用 80 个 query；
3. 拼接三级 query 表征，再用 50 个最终 query 做 Semantic Distillation；
4. 两个 Attention Block 让压缩 token 重新关注原始与两级下采样特征，恢复细节；
5. Qwen 原生 audio projector 继续完成到 LLM hidden width 的投影。

HTA 本体包含 52,861,440 个可训练参数。论文表 9 的约 56M 还包含从 1280 到
Qwen3-4B hidden width 2560 的线性 projector；本接入复用 Qwen2.5-Omni 已有 projector，
不重复创建该层。

## Qwen token-rate 适配

论文和官方代码在每个 30 秒窗口处理 1500 个 50 Hz Whisper encoder token，并输出
50 个 token，即约 1.67 token/s。当前 target site 位于 Qwen 的两帧 average pooler
之后，输入为 25 Hz，因此等价默认值是：

```text
window_tokens = 750
output_query_tokens = 50
keep_ratio = 50 / 750 = 1 / 15
```

每个 audio unit 先按 750 token 切成互不重叠窗口；全局 `keep_count` 按窗口有效长度
成比例分配，并严格输出该数量。窗口顺序和窗口内 query 位置顺序保持不变。若
`keep_ratio=1` 的自定义小型配置触发 identity，框架直接返回原 tensor。

## 权重与训练

这是可训练压缩器，不是 training-free token selector。Eval 配置必须通过
`FASTSLM_HTA_CHECKPOINT_PATH` 指向训练好的权重。loader 支持：

- 官方完整 `FastSLM` PyTorch state dict（自动提取 `encoder.compressor*`、
  `encoder.stage*` 和 `encoder.out_attn*`）；
- HTA-only PyTorch state dict；
- 安装了 `safetensors` 时的 HTA safetensors 文件。

由于 Qwen 使用 750/375/188 的位置 buffer，而官方 checkpoint 使用
1500/750/375，位置 buffer 会按 Qwen 配置重新生成；所有可训练 tensor 在 strict
模式下必须完整且形状一致。

Train 配置把 HTA 注册为 Thinker 子模块 `_mllm_comp_fastslm_hta_audio`，发生在
optimizer 创建之前。当前训练配置采用与项目 7B LoRA 基线一致的 rank 8、alpha 32、
学习率 2e-5；原论文使用 rank 16、alpha 64。配置通过
`modules_to_save` 让 HTA 全参数训练与保存。若 PEFT 把该子模块包装为 trainer
module，controller 会跟随注册后的 wrapper，确保 forward 使用实际可训练副本。

## 入口

```bash
# 训练
scripts/train/qwen2_5_omni/train_aud_sft_fastslm_hta.sh

# 使用官方或平台训练得到的 HTA 权重评测
MODEL_PATH=/path/to/Qwen2.5-Omni-3B \
FASTSLM_HTA_CHECKPOINT_PATH=/path/to/Stage3_FastSLM.pt \
  scripts/eval/lmms_eval/qwen25_omni_1gpu_aud_fastslm_hta.sh
```

Qwen backbone 使用 `MODEL_PATH`/`QWEN25_OMNI_MODEL_PATH`，HTA 独立使用
`FASTSLM_HTA_CHECKPOINT_PATH`。若 full-SFT checkpoint 同时包含两者，可以把两个变量指向
同一路径；脚本不会回退到某次带时间戳的本地 run。

训练数据与三阶段数据组织仍由调用方提供。单个 YAML 复现 HTA 结构、压缩率和论文
LoRA 超参数，但不会凭空构造论文使用的短语音预训练、长语音适配和指令微调三套数据。
