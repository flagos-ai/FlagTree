# Metadata

本目录对应 B 模块：作用域解析与编译期元数据记录。
负责人：华师。

这个目录就是 B 模块的主要开发目录。B 的公共接口在这里定义，B 的实现入口在 `third_party/Debugger/lib/Metadata/`。

核心文件：

- `Passes.h`
- `TrackedOpTable.h`

模块目标：

- 校验 `collect begin/end` 配对。
- 为 scope 分配稳定 `scope_id`。
- 为被跟踪 op 分配稳定 `op_id`。
- 为 kernel 分配稳定 `kernel_id`。
- 构建 `KernelDebugMetadata` 与 `TrackedOpTable`。
- 把 compile-time 静态语义输出给 C 和 D。

上游输入：

- A 插入的 debug marker
- TTIR / MLIR 中保留下来的源码位置信息
- 被跟踪 op 的编译期 IR 语义

对下游输出：

- `KernelDebugMetadata`
- `TrackedOpTable`
- `scope_id`
- `op_id`
- `debugKernelId`
- 可序列化的 metadata json

本模块需要收集和导出的编译期数据：

- 标识与源码映射：
  - `scope_id`
  - `op_id`
  - `kernel_id`
  - `source_loc`
  - `triton_statement`
  - `mlir_op_name`
  - `inline_call_path`
- result 语义：
  - `valueKind`
  - `dtype`
  - `elementDtype`
  - `shape`
  - `stride`
  - `layout`
  - `encoding`
  - `addrSpace`
  - `rank`
  - `elementBits`
  - `vecWidth`
- operand 语义：
  - `operandIndex`
  - `operandRole`
  - `producerOpId`
  - `isConstant`
  - `isPredicate`
  - `isKernelArgument`
  - `constantValueRepr`
  - operand 对应的 `dtype / shape / stride / layout / addrSpace`
- memory op 静态信息：
  - `isMemoryOp`
  - `opCategory`
  - `role`
  - `addrSpace`
  - `accessType`
  - `accessBytes`
  - `alignmentRequired`
  - `hasMask`
  - `maskDtype`
  - `cacheModifier`
  - `evictionPolicy`
  - `isVolatile`
  - `boundaryCheckPolicy`
  - `paddingSemantics`

`opCategory` 和 `role` 仅用于描述内存行为语义。非 memory op 保留字段但写空字符串，
避免把普通 compute op 误分类为 memory 语义。

对齐要求：

- `KernelDebugMetadata.debugKernelId` 必须和运行期 `BufferMeta.kernelId` 一致。
- C 和 D 只消费 `op_id`，不应自行生成另一套 id。

正式解析入口：

- `parseTrackedOpTableFromJson()`
- `parseKernelDebugMetadataFromJson()`
