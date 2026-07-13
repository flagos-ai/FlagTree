# Runtime Sources

本目录是 F 模块实现目录。
负责人：闫明。

当前文件：

- `BackendAdapter.h`
- `BufferLayout.cpp`
- `TransferEngine.cpp`

对应公共接口：

- `third_party/Debugger/include/Debugger/Runtime/BufferLayout.h`
- `third_party/Debugger/include/Debugger/Runtime/TransferEngine.h`

实现重点：

- backend adapter 选择与可用性校验
- backend kind -> transfer driver 映射
- control block / ring buffer 分配
- header 初始化
- host / device buffer 生命周期
- runtime buffer/tensor 注册信息管理
- 同步/异步导出

需要维护的运行期上下文和动态元数据，见：

- `third_party/Debugger/include/Debugger/Runtime/README.md`
