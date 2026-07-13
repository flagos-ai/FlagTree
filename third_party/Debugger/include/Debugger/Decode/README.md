# Decode

本目录对应 D 模块：Host 解码与报告。
负责人：玉珏。

这个目录就是 D 模块的主要开发目录。D 的公共接口在这里定义，D 的实现入口在 `third_party/Debugger/lib/Decode/`。

核心文件：

- `Decoder.h`
- `Reporter.h`

模块目标：

- 读取 F 导出的原始字节流。
- 按 `Protocol.h` 的布局解码 record。
- 用 `BufferMeta + DebugRuntimeMetadata + KernelDebugMetadata / TrackedOpTable` 恢复语义。
- 输出最终文本报告或后续结构化报告。

上游输入：

- `DebugExportedRun`
- `KernelDebugMetadata`
- `TrackedOpTable`
- `Protocol.h`

对下游输出：

- `DecodedDebugRun`
- 文本报告

本模块需要消费并解释的静态数据：

- `kernelId`
- `scope_id`
- `op_id`
- `source_loc`
- `triton_statement`
- `mlir_op_name`
- `dtype / shape / stride / layout`
- `addrSpace`
- `accessType`
- `accessBytes`
- `alignmentRequired`
- `hasMask`
- `boundaryCheckPolicy`

本模块需要消费并解释的运行时数据：

- `runId`
- `deviceId`
- `backendKind`
- `recordLevel`
- `exportMode`
- `RingBufferHeader`
- `recordKind`
- `logical_instance_id`
- 所有 summary / memory / full-value record 字段
- `DebugRuntimeMetadata` 中的 buffer 与 tensor 实例信息

报告侧需要最终还原/展示的指标：

- 标识类：
  - `run_id`
  - `kernel_id`
  - `backend`
  - `device`
  - `op_id`
  - `source_loc`
- 语义类：
  - `dtype_in / dtype_out`
  - `shape`
  - `stride`
  - `layout`
  - `addr_space`
  - `access_type`
  - `access_bytes`
  - `alignment_required`
- 数值类：
	  - `nan_count`
	  - `inf_count`
	  - `zero_count`
	  - `mean`
	  - `min`
	  - `max`
	  - `l2_norm`
	  - `element_count`
	  - 样本值（后续扩展）
	  - `denom_near_zero_count`（敏感 op 后续扩展）
	  - `neg_sqrt_count`（敏感 op 后续扩展）
- 内存类：
  - `LAST_ALIGNED_ADDR`
  - `BASE_ALIGNED_ADDR`
  - `FIRST_ADDR`
  - `LAST_ADDR`
  - `MIN_ADDR`
  - `MAX_ADDR`
  - `ACTIVE_LANE_COUNT`
  - `ADDRESS_SPAN_BYTES`
  - `alignment_ok`
  - `offset`
  - 局部地址快照
- 动态实例类：
  - launch 期 tensor `shape / stride / layout`
  - `bufferId`
  - buffer 边界信息

报告格式：

- `Triton Statement Records` 是 Triton 语句级主视图，按源码语句展示 result
  和 operand 的运行时捕获值。
- 语句级文本报告用 `[result x]` 标注当前语句结果，用 `<operand x>` 标注参数；
  若 operand 来自前序 result，则只输出 `<operand x>: [result x]` 引用。重名
  result 会按出现顺序标成 `[result x:001]`、`[result x:002]`。
- IR op 级动态记录单独放在 `IR Op Log Records` 视图中，并在自动导出时
  写入同 stem 的 `_op_log.txt` / `_op_log.json` 文件；JSON 中对应字段名为
  `op_log`。
- 每个 `op_id` 只展示一次编译期静态元数据；同一个 op 的动态记录按
  `logical_instance_id` 聚合。文本报告中 `instances` 是对齐轴，`summary` 与
  `address_summary` 的每个指标都按这个顺序输出数组；多个 instance 时会按最长
  单元格补空格，方便横向比对。
- `IR Op Log Static Only Ops` 列出有 `op_id` 和静态元数据、但没有 runtime
  record 的 op。这些 op 通常用于 producer/context 分析，例如 pointer-producing
  `tt.splat` / `tt.addptr`；它们不会重复写无意义的动态 summary。
- `record_count` 表示写入 debug ring buffer 的 record slot 数，不是 tensor
  元素数。
- `element_count` 表示静态 tensor lane 数，不代表 mask 后实际 active lane 数。
- memory address summary 会在对应 op 下聚合成与 `summary` 同级的
  `address_summary` 块，包含 `status / first_addr / last_addr / min_addr /
  max_addr / active_lane_count / address_span_bytes`。每一列与 `instances`
  中的同一 `logical_instance_id` 对齐。未形成 summary 的 fallback memory event
  仍保留在 `memory_events_by_instance` 下，便于后续扩展。
- 自动导出时主 `.txt` 和 `.json` 文件用于 Triton 语句级报告；IR op 级报告
  额外生成 `_op_log.txt` 和 `_op_log.json`。只有显式开启 raw record export
  时才会额外生成 `_raw_records.txt`。
- `Runtime Inventory` 是补充索引，用于查看完整 launch 资源信息。
- `Static Op Catalog` 默认不展示；完整静态 op 表已经在 JSON 中保留，文本
  报告只在调用方显式设置 `ReportOptions.includeStaticOpCatalog` 时输出。
- `Aggregates` 默认不展示；仅在调用方显式设置 `ReportOptions.includeAggregates`
  时输出，用于内部调试汇总。
- 当前 device summary lowering 会先把浮点值转成 f32 后统计；f64 精度敏感
  场景需要后续 dtype-preserving collector 支持。

正式解码入口：

- `decodeExportedRun()`
- `renderTextReport()`
