# 原生源码训练

阶段：Phase 1a 之后的 reserved 能力，当前未激活。

规则：

- 仅在可学习压缩方法确实需要训练时激活；
- 条件允许时，优先在进程内导入源码树的训练入口；
- 可以使用子进程加文档化 import hook 或 patch 作为后备方案；
- baseline 复现可以直接调用源码脚本；
- compressed training 必须记录压缩是否注入真实的 training forward 路径。
