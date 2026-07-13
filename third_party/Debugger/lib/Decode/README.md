# Decode Sources

本目录是 D 模块实现目录。
负责人：玉珏。

当前文件：

- `Decoder.cpp`
- `Reporter.cpp`

对应公共接口：

- `third_party/Debugger/include/Debugger/Decode/Decoder.h`
- `third_party/Debugger/include/Debugger/Decode/Reporter.h`

实现重点：

- 原始字节流解码
- `recordKind` 分派
- `TrackedOpTable` 关联查询
- runtime metadata 和静态 metadata 拼接
- 文本报告 / 结构化报告渲染

最终需要展示的指标，见：

- `third_party/Debugger/include/Debugger/Decode/README.md`
