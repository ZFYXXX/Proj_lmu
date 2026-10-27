# 仓库清理记录

2026-10-27 将 426 个旧的、可选音频/Omni/LMMs-Eval 实验入口移出主工作区，文件共 1,719,378 字节。工作区清理前有 1,695 个 tracked 文件；本次不改动 `mllm_comp/`、测试、依赖、共享数据、模型、实验结果或冻结中的 Qwen3.5 worktree。12 个 Omni 训练配置因被测试动态引用而保留。

被移出的原文件完整保存在：

`/home/ZhouFengyu/fastv-repro/archive/repository_cleanup_20261026_v1/`

归档内 `manifest.json` 记录每个相对路径、SHA256、字节数及移动状态。需要恢复时，按 manifest 将对应文件移回仓库原相对路径；先确认目标路径不存在，再核对 SHA256。此次是可恢复归档，不是永久删除。

本轮测试以清理前的 `cleanup_full_profile_baseline_20261026.xml` 为参照，清理后的结果记录在 `outputs/maintenance/cleanup_full_profile_after_20261027.xml`。该完整测试集含既有失败；验收标准是清理后没有新增失败、动态收集到的测试仍存在，不要求把既有失败描述成通过。

本地 `outputs/` 是忽略目录，测试 XML、归档计划和 3.5 运行结果不会上传到 GitHub。仓库只保存源码、必要文档和本次维护工具。
