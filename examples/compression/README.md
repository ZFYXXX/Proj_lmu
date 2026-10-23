# Compression definition examples

本目录保存 compression pipeline 定义示例。真实的 `analyze`、`eval`、`train`
命令会在运行前自动校验已启用的 stage；runtime activation 对直接调用执行同一校验，
因此不再提供独立的 `dry_run` 命令。

Qwen2.5-Omni 示例在模型路径环境变量已解析、数据和依赖可用时，可交给与其 phase
一致的真实命令。配置省略输出字段，由实验入口生成独立运行目录。
组合配置仍需满足各方法的实际执行约束；单阶段示例通过不代表任意组合已获验证。

只需运行 definition 回归测试时：

```bash
python -m pytest tests/unit/test_compression_definitions.py -q
```
