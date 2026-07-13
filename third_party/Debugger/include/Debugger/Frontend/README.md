# Frontend

本目录对应 A 模块：Python 前端与参数穿透。
负责人：华师。

这个目录就是 A 模块的主要开发目录。A 的公共接口在这里定义，A 的实现入口在 `third_party/Debugger/lib/Frontend/`。

核心文件：

- `Bridge.h`

模块目标：

- 把 Python 前端选项、compile request、launch request 统一成 debugger 可消费的结构。
- 把 B 输出的 `KernelDebugMetadata` 挂到编译产物上。
- 把 F 提供的 control block 指针作为隐藏参数 `__debug_ctrl_ptr` 透传到 kernel launch。
- 把运行时 dynamic tensor/buffer 信息整理成 `DebugRuntimeMetadata`，供 F 和 D 使用。

上游输入：

- Python 侧 debug 选项
- kernel 名称、backend、target
- B 输出的 `KernelDebugMetadata`
- F 提供的 `TransferEngine`
- launch 时的 tensor / buffer 实参

对下游输出：

- `DebugCompileRequest`
- `DebugKernelArtifacts`
- `DebugLaunchRequest`
- `PreparedDebugLaunch`
- 隐藏参数值 `hiddenArgValue`

本模块负责接线和整理的字段：

- 编译/运行开关：
  - `enabled`
  - `recordLevel`
  - `exportMode`
  - `recordCapacity`
  - `captureMemoryEvents`
  - `captureFullValues`
- kernel 级上下文：
  - `kernelName`
  - `backendName`
  - `targetName`
- launch 期上下文：
  - `kernelId`
  - `hiddenArgValue`
  - `DebugBufferPlan`
  - `TransferEngineOptions`
  - `streamHandle`
- 运行期 dynamic tensor / buffer 元数据：
  - tensor `argumentIndex`
  - tensor `logicalName`
  - `dtype`
  - `shape`
  - `stride`
  - `layout`
  - `bufferId`
  - `baseAddress`
  - `sizeBytes`
  - buffer `bufferName`
  - buffer `alignment`

推荐真实 launcher 入口：

- `prepareOwnedLaunch()`
  - 由 A 按 `BufferMeta.backendKind + streamHandle` 直接创建
    `TransferEngine`
  - 在 caller 未填全 `BufferMeta` 时，自动从 `DebugKernelArtifacts` 补齐
    `protocolVer / recordLevel / exportMode / kernelId / backendKind`
