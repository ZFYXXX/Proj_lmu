# 测试 Fixtures

小型合成 fixture 和 manifest 示例可以放在此目录。数据集、checkpoint 和大型输出应保留在 Git 跟踪的源码之外。

仅保留有测试或运行文档使用的 fixture。测试生成的文件使用 pytest `tmp_path`，不写入
本目录；训练与评测的常规配置维护在 [`configs/`](../../configs/README.md)。

`native_image.py` 提供原生图像小模型及公共训练检查；`qwen_video.py` 和
`qwen25_omni_audio.py` 共享视频输入和音频文件构造。保留小型替身用于固定数值或异常注入，
不要在夹具中重写原生 Attention、Decoder 或 Cache 的执行循环。

合成报告只保留当前消费者所需的输入。报告显示、字段解析和失败落盘测试使用最小样本，
不为每个方法维护一份完整的成功报告。方法特有的预算、位置和数值期望保持独立，
不能调用被测实现计算；基座的 MRoPE、DeepStack 和结构 token 行为由相应原生测试验证。
