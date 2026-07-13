# Frontend Sources

本目录是 A 模块实现目录。
负责人：华师。

当前文件：

- `Bridge.cpp`

对应公共接口：

- `third_party/Debugger/include/Debugger/Frontend/Bridge.h`

实现重点：

- Python 选项到 `DebugCompileRequest` 的归一化
- `KernelDebugMetadata` 挂载
- `DebugBufferPlan` 构造
- `DebugRuntimeMetadata` 整理
- `TransferEngine` 接线和 `hiddenArgValue` 透传
- 从 launcher backend / stream 推导 `TransferEngineOptions`
- 在 `BufferMeta` 缺字段时用 artifacts 做归一化补齐

联调对象：

- B 的 `KernelDebugMetadata`
- F 的 `TransferEngine`
