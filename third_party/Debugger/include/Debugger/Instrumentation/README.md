# Instrumentation

本目录对应 C 模块：GPU 插桩采集。
负责人：颜臻。

这个目录就是 C 模块的主要开发目录。C 的公共接口在这里定义，C 的实现入口在 `third_party/Debugger/lib/Instrumentation/`。

核心文件：

- `Passes.h`
- `Collectors.h`
- `RecordBuilder.h`
- `Writer.h`

模块目标：

- 读取 B 输出的 `op_id` 和静态语义。
- 在 GPU 上对目标 op 做 summary / memory-event 插桩。
- 在 GPU 上直接构建协议规定的数据块。
- 通过 `__debug_ctrl_ptr` 把数据块写入 F 管理的 ring buffer。

上游输入：

- A/F 提供的隐藏参数 `__debug_ctrl_ptr`
- B 提供的 `scope_id / op_id / kernel_id / TrackedOpTable`
- `Protocol.h` 中规定的 record 布局

对下游输出：

- `SummaryRecord`
- `MemoryEventRecord`
- `FullValueRefRecord`
- 写入到 ring buffer 的原始 record 流

本模块直接依赖的编译期输入数据：

- `scope_id`
- `op_id`
- `kernel_id`
- `source_loc`
- `triton_statement`
- result / operand 的：
  - `dtype`
  - `shape`
  - `stride`
  - `layout`
  - `addrSpace`
- memory op 的：
  - `accessType`
  - `accessBytes`
  - `alignmentRequired`
  - `hasMask`
  - `boundaryCheckPolicy`

本模块需要在运行期插桩计算和写入的指标：

- record 基础标识：
  - `recordKind`
  - `op_id`
  - `logical_instance_id`
- 数值摘要指标：
	  - `nan_count`
	  - `inf_count`
	  - `zero_count`
	  - `mean`
	  - `min`
	  - `max`
	  - `l2_norm`
	  - `element_count`
	  - 样本值 / sample values（后续扩展）
	- 敏感操作辅助指标：
	  - `denom_near_zero_count`（后续扩展）
	  - `neg_sqrt_count`（后续扩展）
- 内存侧动态指标：
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
- 执行上下文：
  - `logical_instance_id`
  - CTA 级上下文
  - warp / lane 级上下文
- 全量值导出相关：
  - `payloadOffset`
  - `payloadLength`
  - 输入/输出切片
  - 关键中间结果窗口

说明：

- `element_count` 是静态 tensor lane 数，不代表 mask 后实际 active lane 数。
- 当前 device summary lowering 会先把浮点值转成 f32 后统计；f64 精度敏感场景
  需要后续 dtype-preserving collector 路径。
- 内存地址动态采集由 `addr_level` 单独控制：`0` 不插入动态地址记录，仅保留静态
  memory metadata；`1` 插入地址 summary 记录；`2` 预留给 full lane dump。
- 内存地址动态采集通过 debugger 专用 IR
  `flagtree_debug.capture_memory_address` 表达，不修改 Triton 原生
  `tt.ptr_to_int` 语义。当前 CANN9 TTIR lowering 对可反向切片的 pointer 形态
  生成 lane-wise 地址摘要，支持的基础形态包括
  `tt.addptr(tt.splat(base), offsets)`、嵌套 `tt.addptr`、`tt.bitcast` 以及简单
  reshape/broadcast/expand_dims。完整摘要要求 offset 可证明为连续 lane offset，
  mask 为空、全 true，或形如 `offsets < limit` 的 prefix mask。匹配成功时每个
  memory op 写出
  `FIRST_ADDR / LAST_ADDR / MIN_ADDR / MAX_ADDR / ACTIVE_LANE_COUNT /
  ADDRESS_SPAN_BYTES`；匹配失败时只写保守的 `LAST_ALIGNED_ADDR` 或
  `BASE_ALIGNED_ADDR` fallback。新增后端时必须验证或适配
  `flagtree_debug.capture_memory_address`，否则可以只保留 B 模块静态 memory
  metadata，暂不启用动态 address event。

本模块不重复写入、而是依赖 B/F/D 关联恢复的内容：

- `run_id`
- `deviceId`
- `backendKind`
- `kernelId`
- `source_loc`
- `dtype / shape / stride / layout`
- `bufferId`
- buffer 边界信息

正式 record 构造入口：

- `buildSummary*Record()`
- `buildMemoryEventRecord()`
- `buildFullValueRefRecord()`
- `initializeRingBufferStorage()`
- `append*Record()`
