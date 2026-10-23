# 量化与 Token / Operation 联合执行设计

本文给后续实现者定义量化接入范围与验收边界。**当前没有可执行的量化算法或量化 executor。** 下述内部 fake-QAT、SPEED-Q 组件与完整配方均是设计，不是已开放的配置或支持声明。本文沿用 2026-09-09 已核对的官方源码依据，没有新增量化实验。

目标是在同一次模型运行中，让量化与框架 Token / Operation 共同 forward、backward、保存和恢复。校准可以是专用过程，量化训练仍须使用同一个已组合的学生模型。训练和评测继续由 ms-swift / lmms-eval 拥有，不另建训练器或运行框架。

## 1. 已有基础与仍缺失的接点

| 当前实现 | 后续量化可以复用的部分 | 仍须完成或遵守的边界 |
| --- | --- | --- |
| [共同实验入口](../../mllm_comp/experiments.py)、[恢复入口](../../mllm_comp/restoration.py) | `run_experiment` 记录有效配置、独立运行目录与结构化结果；`restore_config` 区分实际训练产物、完整权重、HTA 模块、PEFT 和诊断参数产物 | 恢复入口尚不识别量化结构与表示，不能把量化 checkpoint 按普通浮点模型恢复后声称保留量化 |
| [RuntimeCompressionDispatcher / CompressionSession](../../mllm_comp/pipeline/runtime_session.py) | 同一 session 区分 PREPARE 与 FORWARD / GENERATE，动态阶段依赖准备阶段；activation 在安装时构造并重新读取物理句柄，能看到先前结构修改 | 复用现有依赖、失败清理和逆序卸载。量化的模块替换及状态加载尚未实现，不把它编码成 TokenAction 或 LayoutDelta |
| [Omni Swift 配置桥](../../mllm_comp/integrations/ms_swift/adapter.py)、[训练插件](../../mllm_comp/integrations/ms_swift/training_plugin.py) | Omni payload 传递实际 `stages` 与研究扩展；loader 返回后，同一模型先执行 PREPARE，再安装动态阶段，早于 tuner / optimizer 设置；一个 forward 桥拥有请求生命周期 | 不再存在 Omni 全局单阶段限制。ImageLLM Swift 路径仍限单阶段；动态 Omni 阶段仍拒绝 packing / padding-free。新量化 executor 必须接入这一实际准备路径，不能只放宽验证 |
| [OperationApplier](../../mllm_comp/pipeline/operation_apply.py)、[Omni 模型适配](../../mllm_comp/backbones/qwen25_omni/implementation.py) | 已持久化 `mllm_comp_operations`，含阶段名、原始 / 最终层数及层映射；复合模型和 Thinker 配置保留同一事实；本地 Thinker-only 权重使用原生 Thinker 类加载 | 推理恢复会禁用已物化的 Operation，避免二次删层；量化目标仍须在当前真实结构上解析，不能拿原始层序号直接索引最终模型 |
| [模型运行时加载](../../mllm_comp/runtime/loader.py) | 一个 model / processor 归属与统一压缩激活 | 当前先 `load_model` 读取权重，再激活。推理 / 分析加载带量化子模块、buffer 的 checkpoint，仍缺少在权重读取前构造对应结构的接点 |
| [压缩定义](../../mllm_comp/pipeline/definition.py) | 现有 method、算法、variant、binding、phase 校验及研究注册 | `quantization` 仍是保留词汇，启用会报错；没有量化目标解析、校准、执行及表示恢复实现。注册一个名字或出现配置字段不等于可运行 |

这些已有生命周期能力解决了早期设计中的共同 phase、结构变换前捕获模块引用、Omni 单阶段训练和物化 Operation 重复恢复问题。它们不是量化支持证据。真实能力与验证条件仍以 `discover` 返回的精确声明及运行产物为准；尚无 executor 的量化不能通过研究扩展标记变为可执行。

## 2. 在同一个模型上完成准备与训练

不设全框架固定的 Token → Operation → KV → Quant 顺序。首个代表组合选择现有 Qwen2.5-Omni 音频 Token 方法、静态 LLM 删层和保留层 MLP 的有限权重量化；这是待验证组合，不能由单方法成功推断。

对这个组合，准备顺序明确如下：

1. 根据有效配置与输入产物，区分原始模型、已物化结构的推理权重和完整训练 resume。确认量化表示及当前训练阶段。
2. 新训练加载原始模型后应用所选静态 Operation；推理恢复按最终结构构造，禁用已物化的对应阶段。保留源层身份到最终层位置的映射。
3. 在剩余真实模块中解析量化目标，再替换选定 Linear、初始化或恢复量化状态。方法定义数学，基座适配解析目标与安装模块；不让算法猜测不同基座的属性路径。
4. 构造 HTA 等需要新增参数的压缩模块。动态 Token 接点在相关模块替换后安装，使其引用最终模块。若某方法先有 hook，再替换其目标模块，即使复用了 Parameter，旧 hook 也不会自动转移。
5. 完成所选后端 / tuner 的最终冻结与解冻配置，才建立参数组及分布式包装。同一个学生模型交给 ms-swift；optimizer、scheduler、数据循环和保存仍属于后端。有明确不同学习率需求时，复用后端已有参数组选项。
6. 开始训练阶段。阶段转换需要明确新的冻结范围和是否重建 optimizer；不能在正在优化的模型中替换参数后继续使用旧参数组。

现有 session 的依赖和安装时解析足以承接这条具体准备链，不需要再建立一个通用 runner。结构编辑不可逆，session 失败清理不能保证还原被删掉的权重；失败运行应释放该模型并结束。若以后选择动态 Operation，准备阶段安装机制，forward 才作其输入相关决策，不能将它当作静态删层。

量化所需记录只包含实际消费者需要的事实：方法与源码版本、目标路径和结构映射、当前数值表示、量化参数 / buffer、冻结范围、校准条件及恢复阶段。复用有效配置、checkpoint 和 Manifest；不要用任意 metadata 开关传递隐藏执行指令，也不增加无人读取的签名或平行配置格式。

## 3. 内部首个方法：固定量化参数的 W8 fake-QAT

第一版限定为选定的普通 `nn.Linear`：逐输出通道、对称 W8、仅权重、固定量化参数。通道轴为输出轴 0；明确保存所选整数范围、量化开关和阶段。暂不同时接入激活量化、KV 量化、打包 kernel、任意 PEFT Linear 或全部视觉层。scale / zero-point 不作为可训练 Parameter。

优先复用 PyTorch 2.8.0 的 `torch.ao.nn.qat.Linear.from_float`、`FakeQuantize` 和 `PerChannelMinMaxObserver`。准备时从当前权重初始化一次量化范围，随后关闭 observer，保持 fake-quant 开启。forward 使用量化 / 反量化权重执行浮点 `F.linear`，浮点主权重通过 STE 获得梯度。这是 QAT 数值行为，不能宣称已经拥有低比特计算 kernel 或实际模型存储收益。权重更新后可能超出固定范围并被裁剪，这是该有限方法的定义。[PyTorch QAT Linear](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/ao/nn/qat/modules/linear.py#L33-L97)、[FakeQuantize](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/ao/quantization/fake_quantize.py#L169-L308)

先验证 FP32 数值路径，再验证当前模型需要的 BF16 / autocast 与 FP32 量化 buffer。仅在固定依赖的实际失败表明需要时，在该方法内部增加 dtype 转换；不为假设中的兼容性包装所有 Linear，不升级整套环境。

选择保留层 MLP 作为第一批目标，可先避开 Token 评分依赖的 Attention 模块，但仍需确认实际 forward 调用了量化 Linear。Parameter 复用、梯度非零和 optimizer 中存在该 Parameter 必须同时成立；仅看到模块类名改变不能证明联合训练成功。

## 4. 外部量化：SPEED-Q 组件与完整配方分开

### 4.1 先接外部 QLinear / BilevelQuantizer 组件

首条外部路径是在上述准备链中复用选定源码版本的 `QLinear` / `BilevelQuantizer`，让它们参与同一个学生模型的 forward / backward，再由 ms-swift 进行普通监督训练。名称必须表明是“使用 SPEED-Q 量化组件的变体”，不能将其称为完整 SPEED-Q 复现。

已核对的官方实现给出以下约束：

| 官方实现事实 | 组件适配要求 |
| --- | --- |
| `convertModelToQuant` 递归替换 Linear 并转交原 Parameter，排除规则按叶子名称；其默认 `clsq-n2f3` 没有对应 QLinear 分支 | 框架解析最终模块路径，显式给出 `quant_type`、`q_group_size`、`qq_group_size`、`compute_dtype`；不继承全模型遍历与默认排除规则。[qlinear.py](https://github.com/antgroup/SPEED-Q/blob/main/quantization/qlinear.py#L9-L135) |
| QLinear 在 forward 中量化 / 反量化权重，再执行浮点 Linear；两级分组依赖形状整除 | 安装前核对实际输入宽度和两级组形状。保持模拟量化表示标记，不报告打包低比特速度或显存收益。[QLinear](https://github.com/antgroup/SPEED-Q/blob/main/quantization/qlinear.py#L80-L135) |
| Bilevel 的 STE 不向 scale / zero 传梯度；它们是 buffer，`find_params` 会重算并重建二级量化器 | 不将这些 buffer 加入 optimizer；验证重放、设备迁移、保存键和恢复时子模块构造，不能从“有 STE”推断恢复正确。[bilevel_quantizer.py](https://github.com/antgroup/SPEED-Q/blob/main/quantization/bilevel_quantizer.py#L5-L174) |
| 原工程 requirements 固定 torch 2.1.0、transformers 4.37.2、peft 0.10.0 等 | 与当前 provider 依赖不同，不安装其整套 requirements 覆盖统一环境；只验证选定组件在现有环境中的实际兼容性。[requirements.txt](https://github.com/antgroup/SPEED-Q/blob/main/requirements.txt) |

组件失败时修正具体适配边界，或保留该变体未支持状态；不能绕过其量化 forward，也不能因此删除“外部量化与框架压缩同模型训练”的需求。

### 4.2 完整 SPEED-Q 是另一项配方实现

完整配方包含 ViT 校准产物、projector 对齐、后续 QAT、教师损失和冻结策略。它当前依赖未纳入本框架基座的 InternVL；不能为满足示例名称直接引入其整套训练工程。官方示例还要求 `quant_vit_path`，不能假设该产物已随代码提供。[官方 README](https://github.com/antgroup/SPEED-Q/blob/main/README.md)、[stage3 配方](https://github.com/antgroup/SPEED-Q/blob/main/stage3_internvl3_1b_2bit_qat.sh)

| 配方阶段 | 必须明确的模型与状态 |
| --- | --- |
| ViT 校准 / clip | 声明使用原模型分布还是已启用指定 Token / Operation 的分布，保存校准条件、结构身份及范围；专用校准过程可独立，但输入必须经过所声明的真实机制 |
| Projector 对齐 | 学生量化 ViT 与指定框架压缩共同执行；教师保持配方要求的独立状态，不能被学生的安装代码一并改写；记录本阶段的可训练参数 |
| 后续 QAT | 同一学生保留量化与框架机制；使用后端损失扩展点实现所需监督 / 蒸馏，保存主权重、量化状态、压缩状态及完整训练阶段 |

官方 `KDTrainer` 用同一输入调用教师和学生，并逐位置比较 logits。Token 删除或合并后，必须按监督目标 token 的真实身份建立位置对应，处理 causal shift；不能截断教师 logits 或仅将张量补成同形状。可参考现有 [image_training.py](../../mllm_comp/backbones/image_training.py) 的标签修复，但它不是通用蒸馏实现。中间特征合并后若没有明确对应关系，不设置默认广播损失。[multi_trainer.py](https://github.com/antgroup/SPEED-Q/blob/main/internvl/train/multi_trainer.py#L14-L62)

原配方还包含额外的 embedding / lm_head W4 处理、冻结及教师统计等步骤；组件变体不应静默复制这些操作，完整移植也不应省略后继续使用完整方法名。[官方训练脚本](https://github.com/antgroup/SPEED-Q/blob/main/internvl/train/internvl_chat_quant_finetune.py#L1054-L1259)

## 5. 校准、重放与请求状态

校准条件属于有效配方：输入来源、使用哪一模型结构、启用了哪些 Token / Operation、采集位置、观察器何时冻结。SPEED-Q 的 `run_clip` 捕获 LLM 第一层的输入后逐层处理文本校准数据，这里的层属于模型结构，与增量 decode 阶段不同；该流程并不是完整多模态模型的通用校准 API。若目标是压缩后分布，应从实际组合模型采集，不能仅调用原脚本后推断框架机制已经参与。[autoclip.py](https://github.com/antgroup/SPEED-Q/blob/main/quantization/autoclip.py#L111-L207)、[clip_utils.py](https://github.com/antgroup/SPEED-Q/blob/main/quantization/clip_utils.py#L46-L48)

首个内部方法在准备后冻结 observer，因此 forward 与 gradient-checkpoint replay 不推进统计。后续若加入在线激活 observer，必须区分一次正式请求与反向重放，避免移动统计更新两次。不能仅用 `torch.is_grad_enabled()` 判定，因为不同 checkpoint 模式的初次前向语义不同；优先复用已有请求 / 重放标识，只有方法确实需要时才扩展接点。

SPEED-Q 的参数重算不是移动平均 observer。相同权重下重算仍须验证动态子模块、autograd 引用及 buffer 的实际行为。Token 决策按已有重放机制复用，量化持久状态按方法保存；请求临时决策和 KV 不默认跨实验持久化。

## 6. 保存与恢复：区分推理加载和完整训练 resume

量化组合 checkpoint 需要保存最终结构及 Operation 映射、浮点主权重或实际量化表示、量化模块参数 / buffer / 开关、HTA 等持久模块状态、有效配方与训练阶段。继续训练还需要后端原有 optimizer / scheduler / scaler / RNG / step 等状态；使用哪些文件和分片格式由后端决定，不由框架猜测一套固定文件名。

两条恢复路径有不同的原生顺序：

| 目标 | 当前可复用顺序 | 量化仍需补齐的部分 |
| --- | --- | --- |
| eval / analyze 加载完整模型产物 | `restore_config` 从运行的有效配置和真实输出选择 checkpoint，按最终结构加载；禁用已物化 Operation，HTA 在原位置构造并加载对应状态；动态阶段按目标选择 FORWARD / GENERATE | 选定基座必须能在 checkpoint 权重 / buffer 读取前构造量化模块，再加载完整量化状态、安装动态机制。现有 post-load 激活不能自动覆盖此边界 |
| Swift 完整训练 resume | 保留原始模型来源和原 pipeline，在 loader 返回后重建相同 Operation / 压缩模块，早于 tuner / optimizer 构建；随后由 Swift Trainer 恢复 checkpoint 权重和训练状态 | 将量化模块及其参数 / buffer 纳入这个已有准备步骤，使它们在 Trainer 读取 resume 状态前存在；验证参数组身份和状态恢复。这个路径不必为了量化再独立加载一次 checkpoint |

完整 resume 不能同时把产物设为 `train.checkpoint_in` 再执行原始删层，也不能启用会提前替换模型来源的 `resume_only_model=true`。当前 [Swift 集成说明](framework_integration.zh.md) 与 [恢复使用指南](../guides/1_run_compression_experiment.zh.md) 已说明这一区别。量化推理的“加载前构造”仍是真实缺口；不能把已经实现的 Operation 恢复重新列为缺失。

`restore_config` 当前对完整权重、HTA、PEFT、selected-parameter diagnostic 有具体边界：HTA 不是通用量化模块；PEFT 可走原生 Swift resume，但当前没有通用 PEFT eval / merge 路径；仅诊断参数不能冒充完整权重。后续量化只增加所选实际表示的识别和恢复，不能将任意 `.safetensors` 判为量化 checkpoint。

产物表示必须清楚区分：

- **fake-QAT checkpoint**：主权重仍浮点，forward 保留量化 / 反量化模块，具有继续该训练阶段所需的量化状态。
- **反量化浮点导出**：显式计算量化 / 反量化权重后放回浮点模块，用于对应数值的浮点推理；不是默认可继续 QAT 的 checkpoint。
- **整数 / 打包低比特产物**：必须有选定格式的 loader 和实际 kernel；仅保存整数张量不足以声明框架可运行。

PyTorch `qat.Linear.to_float()` 使用浮点主权重，不会自动烘焙 fake-quant 效果。SPEED-Q 的 `save_fake_quant.py` 可保存统计并导出量化 / 反量化后的浮点权重，也不能单独证明完整训练状态恢复或低比特 kernel 存在。[PyTorch Linear 转换](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/ao/nn/qat/modules/linear.py#L89-L97)、[SPEED-Q 导出脚本](https://github.com/antgroup/SPEED-Q/blob/main/save_fake_quant.py)

不要用 `strict=False` 静默丢弃量化 buffer，也不要只载入浮点 Linear 权重后绕过量化推理。阶段转换可能有意重建 optimizer，它应记为新阶段，不能与恢复同一阶段的 resume 验收混在一起。

## 7. KV 的存储和读取语义独立于权重量化

权重量化改变计算数值，不自动改变 KV 的物理格式。首个 weight-only 组合继续使用原生浮点 DynamicCache。decode 专属机制不进入普通 `use_cache=False` 的 teacher-forced 训练；若一个训练步骤确实包含生成，则由该生成请求拥有它的生命周期。

分别记录四件事：Token 逻辑布局、Attention 读取的 KV 索引、物理 KV 存储、KV 数值表示。当前 [DyCoKe 实现](../../mllm_comp/backbones/dycoke_video.py) 先把完整 K / V 追加到原生缓存，下游层再读取 `index_select` 后的子集；它保留完整物理缓存，不能据此报告持久 KV 字节数减少。其支持边界要求当前 provider 的完整 DynamicLayer，不包含 sliding / compiled / 量化缓存。

后续若选择真实 KV 量化，需在具体缓存实现中明确 append 的逻辑位置、物理表示与 scale 元数据，以及 Attention 在何处选索引、何处反量化。若 kernel 要求完整连续缓存，不能推断它兼容当前稀疏读取。待选择真实实现后补必要协议，不预建万能 Cache 类，也不用 LayoutDelta 代替缓存存储语义。

## 8. 最小代表验证与开放支持的条件

以下是未来量化实施的验收计划，**尚未执行**。现有 [Operation + Token 生命周期集成](../../tests/integration/test_qwen25_omni_llm_token_cpu.py)、[Swift session 集成](../../tests/integration/test_ms_swift_compression_session_cpu.py) 和 [恢复直接测试](../../tests/unit/test_restoration.py) 可以提供真实准备、更新和恢复的测试基础，不能替代量化验证。

1. **真实 Tensor 联合交接**：小模型同时执行删层、Token gather / merge 与量化 Linear；确认被删层不在目标和 optimizer、量化模块实际被调用、监督位置正确、有限 loss 与预期梯度出现、optimizer 更新了选定参数。内部方法与外部组件各完成一次，复用行为断言。
2. **必要的重放检查**：选择实际使用 checkpoint 的一个代表路径，对比开启 / 关闭重放的 loss、梯度和量化状态，确认 Token 决策和 observer 没有错误推进。不复制方法的全部独立数学测试。
3. **一次完整后端生命周期**：固定一个已支持真实基座及小数据，组合训练至少发生一次参数更新，保存，从框架入口恢复继续更新，再执行恢复模型推理。检查量化表示、Operation 结构、动态 Token、模块权重和 optimizer 状态都保留；三个单方法分别成功不替代此项。
4. **仅在纳入范围时增加 KV 验证**：如果实际实现 DyCoKe + weight-only，验证完整物理缓存、稀疏读取、追加位置及跨请求清理；未选定 KV 量化格式时不增空测试、不放宽声明。
5. **仅在接完整配方时增加蒸馏验证**：使用发生真实序列变化的学生和独立冻结的教师，核对 causal target 对齐；等长 fixture 不足以证明 Token 与蒸馏的交互正确。

通过后只为具体算法、variant、binding、phase 和已执行条件登记证据，分别说明工程可用性、质量与效率；fake-QAT 功能通过不代表速度提升。BF16 / autocast、SPEED-Q 在统一环境中的组件兼容性、量化状态重放及真实组合恢复都仍待验证，不能写成已发现的现有产品回归。

## 9. 源码依据与实现范围

外部方案依据来自已审阅的 `antgroup/SPEED-Q` 官方 `main` 文件，内部方案依据来自 PyTorch `v2.8.0`。本文保留原审阅链接，未重新联网核实其后续变化；实施时应按仓库现有第三方依赖登记方式固定实际采用的源码版本。

近期实现顺序是内部有限 W8 fake-QAT、同一边界上的外部 QLinear 组件变体，再根据实际研究需求决定完整 SPEED-Q 配方和具体 KV 量化。当前不新增 InternVL、不整体替换统一环境、不承诺任意压缩组合或框架外独立模型导出，也不为本文预先添加空接口。
