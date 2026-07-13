# Common

本目录定义 debugger 的二进制协议和跨模块共享的最小公共类型。

核心文件：

- `Protocol.h`

关键结构：

- `RingBufferHeader`
- `BufferMeta`
- `RecordHeader`
- `SummaryRecord`
- `MemoryEventRecord`
- `FullValueRefRecord`

协议约束：

- 所有 runtime record 固定为 `32B`。
- `RecordHeader` 固定携带：
  - `recordKind`
  - `opId`
  - `logicalInstanceId`
- `BufferMeta` 固定携带：
  - `runId`
  - `deviceId`
  - `kernelId`
  - `protocolVer`
  - `recordLevel`
  - `exportMode`
  - `backendKind`

谁依赖这里：

- C 模块按照这些结构在 GPU 上构造 record。
- F 模块按照这些结构初始化 control block 和导出 buffer。
- D 模块按照这些结构解码 host 侧原始字节流。

修改要求：

- 修改 `Protocol.h` 等于修改全模块协议。
- 修改 record 字段、枚举值、结构体大小之前，必须同时检查 C/F/D 三个模块。
