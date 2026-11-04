# Projector 实验工作流

日常入口只有 `bash workflow.sh`。修改方法/数据集组合用实验清单；修改机器路径用机器配置。
P6000 负责本地整理、开发和测试；LMU 的拉取、资源申请和任务提交由用户手动执行。
当前修改保持在本地，完成检查后再按发布安排同步 GitHub。
模型前向、压缩接口、原生 prompt/评分和共享队列仍复用 MLLM-Comp，不再为每个组合复制启动器。

| 内容 | 唯一维护位置 |
|---|---|
| 方法别名、参数、checkpoint 角色 | `configs/workflow/methods.json` |
| 数据集文件模式、原生 task YAML、主指标 | `configs/workflow/tasks.json` |
| 本次哪些方法跑哪些数据集、run ID、是否复用 dense | `configs/workflow/experiments/*.json` |
| P6000 / LMU 的 Python、模型、数据、权重、输出路径 | `configs/workflow/profiles/*.json` |
| 新算法的实现与一次框架注册 | `mllm_comp/compressors/` |
| 统一调度与快照 | `mllm_comp/workflow/` |
| 原生评测、缓存、共享分片、结果复用 | `scripts/eval/lmms_eval/` 的公共 Python 模块 |
| 历史运行说明与审计证据 | `docs/experiments/`、`outputs/` |

## 1. 选方法和数据集

复制 `configs/workflow/experiments/example.json`，改 `run_id` 和 `method_tasks`：

```json
{
  "schema_version": 1,
  "run_id": "qwen25_my_experiment_v1",
  "reference_only": true,
  "chunk_size": 1024,
  "method_tasks": {
    "importance_fps": ["mmstar", "scienceqa_img", "pope", "gqa"],
    "importance_fps_pool": ["mmstar", "scienceqa_img", "pope", "gqa"]
  }
}
```

这是精确的单元清单，不会扩展成方法×所有数据集。`reference_only=true` 只复用 dense，
缺少合格 dense 时先报告原始分数，不偷偷增加 dense 推理。
需要重新跑 dense 时设为 `false`，并为每个任务显式添加 dense 单元。
已经完成且协议一致的结果由队列复用；更换 run ID 不代表必须重跑全部参照。

R1/T2 补充自然图像任务的八个单元已写入 `r1_t2_general.json`。
已有的 43 单元补全清单也可以直接传入：
`--experiment configs/eval/lmms_eval/qwen25_r1t2_completion.json`。

## 2. P6000 开发与小样本验证

只读查看组合和检查路径，不占 GPU：

```bash
bash workflow.sh plan --profile configs/workflow/profiles/p6000.json \
  --experiment configs/workflow/experiments/r1_t2_general.json
bash workflow.sh check --profile configs/workflow/profiles/p6000.json \
  --experiment configs/workflow/experiments/r1_t2_general.json
```

修改算法时补有针对性的测试；只修改清单不必新增测试。
工作流相关检查：

```bash
/home/ZhouFengyu/fastv-repro/mllm-comp-310/bin/python -m pytest \
  tests/unit/test_project_workflow.py tests/unit/test_portable_resident_queue.py -q
```

在空闲 GPU 上用同一评测内核执行每任务两个请求：

```bash
bash workflow.sh smoke --profile configs/workflow/profiles/p6000.json \
  --experiment configs/workflow/experiments/r1_t2_general.json --limit 2
```

GPU 编号在 profile 的 `gpus` 中选择；本地所选 GPU 有其他计算进程时拒绝启动。
smoke 自动生成独立 run ID，不导入全量结果，不会把 smoke 分数写成全量结果。
`run` 使用同样入口执行全量；P6000 日常只做 smoke。

## 3. GitHub 发布

小改动先保留工作区，完成一组可运行改动后再发布。用 `git status --short` 和
`git diff --stat` 审查，显式选择源码、配置、测试和文档加入发布提交，不使用 `git add .`。
当前目录已有之前尚未提交的 R1/T2、resident queue 和任务修正文件，发布前需一起审查，
避免只上传新的入口却漏掉它依赖的文件。此工作流不会自动 commit 或 push。

发布到 `lmu` remote 的版本应包含 `workflow.sh`、`WORKFLOW.md`、`mllm_comp/`、
公共 `scripts/`、`configs/` 和依赖声明。模型、checkpoint、数据、缓存、全量输出和
本地 `archive/` 不进入 Git。现有小型结果复用包单独保留在 `artifacts/eval/`；
它是核验过的预测/分数证据，不包含图片或模型。

LMU 在已分配的 CPU 交互节点拉取发布版本。源代码通过 GitHub；推理权重用已有
HF checkpoint 包。现有共享环境不需重复安装，模型和数据只读复用。

## 4. LMU 提交

只需维护 `configs/workflow/profiles/lmu.json`，以后换路径不编辑 Python 或 sbatch。
新版本使用独立 checkout，不覆盖运行中任务的源码。从仓库根目录：

```bash
bash workflow.sh plan --profile configs/workflow/profiles/lmu.json \
  --experiment configs/workflow/experiments/r1_t2_general.json
bash workflow.sh prepare --profile configs/workflow/profiles/lmu.json \
  --experiment configs/workflow/experiments/r1_t2_general.json
```

`prepare` 只生成源码快照和 `job.sbatch`，不会调用 `squeue`、`sbatch` 或连接 LMU。
脚本声明 1 节点、4 GPU、32 CPU、23h50m，任务名前缀 28，mail-type NONE，wandb 离线。
由用户在 LMU 确认自己的配额和已有任务后，手动执行输出的 `sbatch /absolute/path/to/job.sbatch`。
`sbatch` 申请 GPU 资源，无需再提前申请 GPU 交互节点。登录节点仅做提交和查看。
所有推理和模型内容校验在分配到的计算节点执行。

四个 GPU 各运行一个常驻模型进程，从共享队列领取未完成分片。
没有把 GQA / VQAv2 等数据集固定分配给某张卡。任务接近结束时只有少量分片，
GPU 仍可能短暂空闲；不会保证任意时刻都满载。

显式调用 profile 中的 Python，不依赖当前终端激活哪个环境。
每次运行保存 `source/` 快照、方法/任务配置、文件哈希、Git commit 与未提交状态、
checkpoint 哈希、Slurm ID、GPU 使用率、逐样本预测/trace、原始分数及 dense 归一化分数。
运行导入 `source/`，后续开发不会改变已提交任务使用的代码。

达到时限后，检查失败原因，然后用 `prepare --resume` 生成相同配置并手动再次提交；完成的分片会跳过。
源码、配置或权重改变时需新 run ID。推理失败记录保留，不能删除失败标记冒充恢复成功。

## 5. 查看进度和结果

```bash
bash workflow.sh status --run-dir /absolute/path/to/run
bash workflow.sh report --run-dir /absolute/path/to/run
```

`status` 列出完成单元、剩余单元、已完成分片及失败路径；也能读取已有共享队列的
`plan.json`。详细日志在 `logs/worker_N.log` 和每个分片的 `worker.log`。
`report.md`、`scores.csv` 和 `summary.json` 使用原生评分；HallusionBench 延迟评分，
MMMU 为单图子集，VQAv2 test-dev 只生成官方提交文件，不能伪造分数。

## 6. 新增方法、训练和归档规则

已有算法增加参数组合：在 `methods.json` 增加别名，在实验清单里选它，无需新启动器。
全新算法：实现一次框架接口和注册，再增加 catalog 条目；不能仅靠改方法名字得到算法。
新任务复用现有数据格式时增加 task YAML 和 `tasks.json` 条目；特殊数据格式仍需适配读取。
新的 backbone 适配需要独立实现，目前这个入口固定 Qwen2.5-VL-7B。

标准 SFT 训练继续使用框架的 `mllm-comp train --config <训练配置>`。
冻结基座的聚合 MLP 适应性训练通过 `workflow.sh train-plan|train` 使用
`configs/workflow/training/anchor_adaptation.json`，复用原有训练数据、紧凑输入和
答案损失实现。配置选择训练组，后台队列等待空闲 GPU；详见
[适应性训练记录](docs/experiments/anchor_adaptation_20261103.md)。

历史专项 Python 脚本如果被其他脚本、测试或配置导入，先保留作为内部依赖。
31 个无活动代码引用的旧 Shell、恢复和拆队列入口已移到
`archive/legacy_entrypoints_20261103/`，该目录被 Git 忽略。
校验和及原路径见 `docs/maintenance/entrypoint_archive_20261103.json`。
恢复时按清单移回未被占用的原路径；已跟踪旧文件也可从历史 Git 版本找回。
其他 worktree、权重、数据和实验结果没有移动或删除。

## 本次整理的验收记录

- 11 个方法的生成配置与整理前的发布快照逐项一致。
- 工作流/共享队列 15 项 CPU 测试通过，包含四个真实子进程的设备映射、
  共享队列无重复领取、稀疏配置、源码快照、恢复约束和只生成作业而不提交的检查。
  原有 MLP、训练保存恢复、位置及距离分组的 30 项 CPU 回归通过，共 45 项。
- 真实旧 step320 MLP checkpoint 在当前工作区 CPU 加载成功，128×3584 输入的
  输出有限且重复一致；旧实现兼容哈希未改变，HF 推理包的 SHA256SUMS 全部通过。
- ScienceQA-IMG、AI2D 各一条样本完成原生任务的数据准备和配置生成。
- 31 个归档文件均与原 SHA256 一致，活动代码/配置/测试无指向已归档入口的引用。
- 生成的四卡 sbatch 和根入口通过 Bash 语法检查。
- 本次整理时两张本地卡均有计算进程，没有新增 GPU smoke、全量推理或 LMU 提交。
  四进程调度测试不代表四卡 A100 的实机测试。

如果回归发现问题，优先从本地可恢复归档或冻结源码恢复对应文件。
需要 GitHub 版本时，先取到独立目录比较，再恢复具体文件；保留当前未提交方法和历史产物。
