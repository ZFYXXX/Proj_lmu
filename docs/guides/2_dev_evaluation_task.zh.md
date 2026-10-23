# 开发 Evaluation Task

> 类型：Guide
>
> 文档层级：2 / 压缩研究人员
>
> 目标结果：在项目固定版本的 lmms-eval 中增加可复现 task，并通过平台 eval 入口得到指标与 Manifest
>
> 适用范围：新增 benchmark、dataset variant、prompt、post-processing 或 metric
>
> 当前状态：Active

Evaluation task 的语义归 lmms-eval 所有。平台 YAML 只选择 task、backbone、压缩和
运行参数，不复制 dataset path、split、prompt、generation kwargs 或 metric。先阅读
固定 submodule 中的 `third_party/frameworks/lmms-eval/docs/guides/task_guide.md`，本指南
只规定与 MLLM-Comp-Platform 的集成和复现边界。

## 1. 扩展目标与准备

先检查固定版本中是否已有可复用 task 或 template。新 task 需要明确：

- benchmark 名称、版本和许可；
- dataset path/config、split 和本地数据约定；
- image/video/audio/text 输入映射；
- prompt/chat 消息结构；
- output type 与 generation kwargs；
- target、post-processing、metric 和 aggregation；
- sample identity 与失败处理；
- 与已有论文或官方评测协议的差异。

任务开发在 `third_party/frameworks/lmms-eval` 子模块中完成。Git 操作从该目录执行，
提交 lmms-eval 变更后再由父仓库更新 submodule pointer；不要把 task 实现复制到平台
integration。

## 2. 最小 task

典型文件结构：

```text
third_party/frameworks/lmms-eval/lmms_eval/tasks/<task>/
├── <task>.yaml
├── utils.py
└── README.md                 # 数据准备或协议复杂时
```

YAML 至少需要精确声明 dataset、split、output type、输入转换、target、generation 和
metric。复杂处理放在 `utils.py`，保持样本转换和评分函数可独立测试。

开发阶段先用 lmms-eval 原生命令检查发现和少量样本：

```bash
cd third_party/frameworks/lmms-eval
python -m lmms_eval --tasks list
python -m lmms_eval --model <model> --tasks <task> --limit 1 \
  --output_path /new/output/path
```

回到父仓库后再增加一个平台 eval 配置，只写：

```yaml
eval:
  backend: lmms_eval
  tasks: [<task>]
```

task 已拥有的 dataset 和 prompt 不在平台配置中重复声明。

## 3. 完整集成流程

### 数据与样本语义

固定 dataset config、split、媒体字段和 sample ID。需要本地数据或 HPC 路径时，在
task/environment 中提供明确配置；平台同步不得覆盖有效的服务器 task YAML。长期文档
记录数据准备规则，不记录某次运行的个人绝对路径。

### Prompt、输出和指标

分别测试：

- `doc_to_text` 或 chat messages；
- image/video/audio 输入；
- `doc_to_target`/choice；
- generation kwargs；
- response normalization；
- sample metric；
- group/task aggregation。

同一 task 的 prompt 或评分协议发生实质变化时，应更新 task version 或名称，不能让旧
结果与新协议共用一个身份。

### 平台调用

使用已激活的平台注册模型，通过 `mllm-comp eval` 运行 `--sample_limit 1`。确认：

1. lmms-eval 实际发现新 task；
2. 请求经过平台 model plugin，而非上游同名 wrapper；
3. 媒体和 chat/simple 语义正确；
4. 原生结果包含有限指标；
5. 平台 Manifest 记录 task、integration、checkpoint 和 sample count；
6. compressed 模式下仍使用同一 task 协议。

### 提交与父仓库更新

在 lmms-eval 子模块内提交 task、测试和文档，再在父仓库提交新的 gitlink。父仓库
Manifest 和正式配置引用固定 commit；不要依赖浮动上游分支。

## 4. 验收与边界

最低证据包括：

- task YAML 可以被发现；
- 转换和 metric 的确定性测试；
- 一个真实样本的原生 lmms-eval 结果；
- 平台 plugin 路由证据；
- baseline Manifest；
- 如用于压缩实验，同协议 compressed Manifest；
- 与官方协议的差异和当前未覆盖 split。

`--tasks list` 或 YAML 加载成功不证明数据、prompt 和 metric 正确；单样本结果也不证明
完整 benchmark 可复现。正式运行见[运行评测](1_run_evaluation.zh.md)；用于压缩对比时
还应遵循[运行压缩实验](1_run_compression_experiment.zh.md)中的协议。
