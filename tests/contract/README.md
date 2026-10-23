# 契约测试

`test_framework_api_contract.py` 使用已安装的 lmms-eval 和 ms-swift，检查平台模型入口、
原生 loader 和 SFT 入口仍可接入。依赖或插件变化、发布候选验证时运行：

```bash
python -m pytest tests/contract -q
```

缺少依赖会跳过，跳过不构成通过。CLI 参数、请求结果、训练保存和恢复的行为由
`tests/unit/` 与 `tests/integration/` 对应测试承担；这里不重复合成启动参数矩阵。
