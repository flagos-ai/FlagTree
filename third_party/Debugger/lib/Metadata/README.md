# Metadata Sources

本目录是 B 模块实现目录。
负责人：华师。

当前文件：

- `Passes.cpp`
- `TrackedOpTable.cpp`

对应公共接口：

- `third_party/Debugger/include/Debugger/Metadata/Passes.h`
- `third_party/Debugger/include/Debugger/Metadata/TrackedOpTable.h`

实现重点：

- scope 校验
- `op_id / scope_id / kernel_id` 分配
- `TrackedOpTable` 构建
- metadata 序列化 / 反序列化

需要特别保证的字段：

- `debugKernelId == BufferMeta.kernelId`
- `TrackedOpEntry` 中的 `dtype / shape / stride / layout / access* / alignment*` 与编译期 IR 一致
