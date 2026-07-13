# Runtime

本目录对应 F 模块：缓冲与数据异步导出。
负责人：闫明。

这个目录就是 F 模块的主要开发目录。F 的公共接口在这里定义，F 的实现入口在 `third_party/Debugger/lib/Runtime/`。

核心文件：

- `BufferLayout.h`
- `TransferEngine.h`

模块目标：

- 为一次 debug run 生成 host 侧运行时上下文。
- 分配并管理 control block / ring buffer。
- 初始化 `RingBufferHeader`。
- 返回 `__debug_ctrl_ptr` 的设备地址。
- 在 kernel 结束后把原始字节流导出给 D。
- 维护 runtime tensor / buffer 注册表，供 D 做动态实例解释。

上游输入：

- A 传入的 `BufferMeta`
- A 整理的 `DebugBufferPlan`
- A 整理的 `DebugRuntimeMetadata`
- C 写入的 ring buffer

对下游输出：

- `DebugLaunchContext`
- `hiddenArg()`
- `DebugExportedRun`
- `createTransferEngine()`
- `resolveTransferDriverKind()`
- `makeTransferEngineOptions()`

本模块负责维护的运行时上下文和动态数据：

- 运行上下文：
  - `runId`
  - `deviceId`
  - `kernelId`
  - `recordLevel`
  - `exportMode`
  - `backendKind`
- buffer 计划：
  - `recordCapacity`
  - `recordSize`
  - `payloadBytes`
  - `payloadOffset`
  - `totalBytes`
- buffer 注册信息：
  - `bufferId`
  - `bufferName`
  - `baseAddress`
  - `sizeBytes`
  - `alignment`
- launch tensor 实例信息：
  - `argumentIndex`
  - `logicalName`
  - `dtype`
  - `shape`
  - `stride`
  - `layout`
  - `bufferId`
  - `baseAddress`
  - `sizeBytes`
- 导出结果：
  - `rawBuffer`
  - `runtimeMetadata`

本模块与 C 的对齐点：

- `DebugBufferPlan.recordSize` 必须和协议 record 尺寸一致。
- `initHeader()` 负责把 `capacity / recordSize / payloadOffset` 写进 header。
- `hiddenArg()` 返回的就是 kernel 看到的 `__debug_ctrl_ptr`。
- `DebugLaunchContext.streamHandle` 用于把 F 的 H2D / D2H 操作和实际 kernel
  launch stream 对齐；A 未接线前允许保持 `0`。

真实后端入口：

- `createTransferEngine()`
- `createTransferEngine(BackendKind, streamHandle)`
- `TransferEngineOptions.driverKind`
- `TransferEngineOptions.streamHandle`
- `resolveTransferDriverKind()`
- `makeTransferEngineOptions()`

独立开发：

- 可以直接在 `third_party/Debugger/lib/Runtime/` 目录单独配置和构建 F 模块
- 独立构建入口：`third_party/Debugger/lib/Runtime/CMakeLists.txt`
- standalone 目标：`FlagTreeDebuggerRuntimeStandalone`
