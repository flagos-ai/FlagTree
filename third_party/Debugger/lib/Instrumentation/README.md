# Instrumentation Sources

本目录是 C 模块实现目录。
负责人：颜臻。

当前文件：

- `Collectors.cpp`
- `Passes.cpp`
- `RecordBuilder.cpp`
- `Writer.cpp`

对应公共接口：

- `third_party/Debugger/include/Debugger/Instrumentation/Collectors.h`
- `third_party/Debugger/include/Debugger/Instrumentation/Passes.h`
- `third_party/Debugger/include/Debugger/Instrumentation/RecordBuilder.h`
- `third_party/Debugger/include/Debugger/Instrumentation/Writer.h`

实现重点：

- summary collector 选择
- memory event 插桩
- full value payload 引用
- `logical_instance_id` 生成
- GPU 侧 record 构造
- 通过 `__debug_ctrl_ptr` 写 ring buffer

需要采集和写入的指标，见：

- `third_party/Debugger/include/Debugger/Instrumentation/README.md`
