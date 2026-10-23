# 验证和激活能力

> 类型：Guide
>
> 文档层级：3 / 平台维护者
>
> 目标结果：为一个精确能力选择足够且必要的证据，并决定是否可以激活或扩大声明范围
>
> 适用范围：backbone、compressor、analysis、integration 与精确 SupportUnit
>
> 当前状态：Active

验证的目标不是运行最多的测试，而是回答一个明确的支持声明是否成立。Catalog 管理
内置 capability 的 `future/target/active`；compression SupportUnit 独立管理
`algorithm/variant/backbone/modality/phase` 格子的证据生命周期。两者都不允许从部分
通过推导未验证的笛卡尔积支持。

## 1. 验证对象和声明

开始前把声明写成可证伪的一句话，例如：

```text
ToMe image variant 在 Qwen2.5-VL 的 training forward 中能够真实缩短 token，
完成一次 finite-loss optimizer step，并保持 M-RoPE/layout 合法。
```

不要使用“ImageLLM 已支持”“训练可用”这样的宽泛对象。至少固定：

- capability kind；
- algorithm/variant/binding；
- backbone 和 checkpoint；
- target modality；
- execution phase；
- inference、train step、reload、quality 或 efficiency 目标；
- 预期 invariant 和产物。

## 2. 最小证据层级

Compression SupportUnit 的状态顺序为：

```text
declared
  -> kernel_verified
  -> cpu_contract_verified
  -> real_checkpoint_verified
  -> train_verified（声明完整训练生命周期时）
  -> active
```

稳定证据类别包括：

| EvidenceKind | 回答的问题 |
| --- | --- |
| `kernel` | 算法、预算、索引和布局数学是否正确 |
| `cpu_contract` | binding/provider 生命周期和接口是否成立 |
| `real_checkpoint` | 真实模型中是否命中目标位置并保持 invariant |
| `train_step` | 是否完成可追溯的一步压缩感知 optimizer step |
| `train_backward` | 目标路径 backward 和梯度语义是否成立 |
| `checkpoint_reload` | 保存重载后是否维持同一合同 |
| `quality_baseline` | 是否有同协议质量对照 |
| `efficiency_baseline` | 是否有同协议效率对照 |

Active inference 通常要求 kernel、CPU contract 和真实 checkpoint。训练支持独立表达：
`train_step_verified` 不等于完整 `train_verified`，后者还要求 backward 与 reload 证据。

## 3. 完整验收流程

### O：算法和配置

验证论文/设计步骤、deterministic 小输入、exact budget、identity、边界长度、多 media
unit、非法配置和布局映射。Definition/registry 测试要证明选择语义唯一，但 schema 或
`--plan_only` 不能替代模型执行。

### E：真实模型接入

先运行 CPU provider/executor contract，再用固定最小 checkpoint 检查：

- 目标 hook/target site 确实命中；
- token、结构或 operation 按配置改变；
- placeholder、mask、position、grid、M-RoPE、DeepStack、AnyRes、KV cache 等适用
  invariant 成立；
- generate/prefill/decode 生命周期符合声明；
- activation 卸载后清理动态 hook 和请求状态；可回滚替换恢复原模块，已物化的静态删层保留最终结构并随 runtime 释放模型；
- runtime metadata 和 Manifest 能解释变化。

真实最小运行使用 `tests/integration/*_gpu_smoke.py` 或 `scripts/diagnostics/` 中的正式
入口。只选择能改变本次结论的最小测试集合，不重复运行无关矩阵。

### T：训练和重载

只有声明训练能力时才要求：

- finite loss；
- 目标值路径 finite non-zero gradient；
- optimizer step；
- gradient-checkpoint replay 不重复决策；
- checkpoint 保存；
- reload 后再次满足相同 compression 和模型合同。

结构 operation 还要验证导出范围、辅助文件、原始 checkpoint + YAML 重应用或独立
结构 checkpoint 的明确 reload 契约。

### Q/P：质量和效率

质量和 profile 不是算法正确性的前置门禁，但要声明研究收益时必须完成。Baseline 和
compressed 必须使用同一 checkpoint、task、样本、generation/train 协议和环境。
Debug 采集不作为正式效率结果。

### Integration 和 Human 验收

lmms-eval/ms-swift 的正式候选还要确认原生日志、结果、cache/resume 和输出文件存在，
同时请求确实经过平台 runtime。独立人工复核的操作步骤见
[Integration 验证 Runbook](../support/integration/validation_runbook.zh.md)。

交给 Human 验收时，应提供候选 commit、实际命令和配置、Manifest 与原生结果路径、
已通过的 invariant、尚未覆盖的范围，以及验证者需要执行的最小命令。Agent dirty tree
可以用于开发 smoke，但不能作为正式候选。

## 4. 激活、记录和边界

通过证据必须记录 artifact、source commit 和项目当前要求的 provenance；失败证据可以
保留用于诊断，但不能满足 gate。支持事实按以下归属维护：

- catalog 或精确 SupportUnit 记录当前声明，`mllm-comp discover` 从代码投影，不另维护 Active 名称或数量表；
- 最小配置和对应回归测试保护新声明的真实行为；
- `support/*/status.zh.md` 记录 checkpoint、运行条件、证据出处与限制，引用查询入口获取精确状态；
- 用户操作或限制发生变化时更新 Guide，达到新里程碑时更新 `docs/progress.md`，两者不复制完整支持矩阵。

以下不能作为激活依据：

- 仅有算法文件或 YAML；
- `plan_only`、schema 或 definition validator；
- 另一个 backbone、模态或 phase 的通过记录；
- 没有 provenance 的口头运行结论；
- 单样本结果被外推为完整 benchmark 或收敛；
- 失败后复用混有旧 artifact 的输出目录。

如果证据失败，保持原状态并按[运行时排错](3_troubleshoot_runtime.zh.md)定位；不要通过
降低 gate、回退 baseline 或扩大容差来制造通过结论。
