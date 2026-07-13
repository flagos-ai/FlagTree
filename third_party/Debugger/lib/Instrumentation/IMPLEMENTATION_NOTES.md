# C 模块实现说明

**负责人**：颜臻
**模块路径**：`third_party/Debugger/lib/Instrumentation/` + `third_party/Debugger/include/Debugger/Instrumentation/`
**参考文档**：`debugger分工.md` §3.3、§6.2

---

## 1. 模块职责回顾

根据分工文档，C 模块的核心目标是：

1. 读取 B 模块标注的被跟踪 op 集合（`op_id`、`is_memory_op`、`record_level`、`addr_level` 等 IR attribute）
2. 对每个被跟踪 op 决定需要哪些 record 类型（summary / memory event / full value）
3. 在 GPU 上统计数值摘要指标并构建协议 record（Phase 1 必选：`nan_count`、`inf_count`、`zero_count`、`mean`、`min`、`max`、`l2_norm`、`element_count`；`addr_level=1` 时对支持的 Triton pointer 链采集 `first/last/min/max/active_lane_count/address_span_bytes` 地址摘要，无法匹配时退回 base/last aligned address）
4. 通过 `__debug_ctrl_ptr` 写入 F 模块管理的 ring buffer

**单独开发期替代方案**（分工文档 §3.3 明确允许）：
- 消费外部传入的 `op_id`，不在 C 内部固化 `op_id` 生成逻辑
- 写记录逻辑依赖抽象的 `RecordSink`，可使用线性 buffer 或 F 的正式 ring buffer
- 优先验证 summary / memory event 的**格式和字段语义**，再接入完整设备运行路径

---

## 2. 代码文件一览

| 文件 | 性质 | 说明 |
|------|------|------|
| `third_party/Debugger/include/Debugger/Instrumentation/Passes.h` | 接口 | Pass 入口声明 |
| `third_party/Debugger/include/Debugger/Instrumentation/Collectors.h` | 接口 | collector 规格 + `SummaryStats` + 计算函数声明 |
| `third_party/Debugger/include/Debugger/Instrumentation/RecordBuilder.h` | 接口 | host-side record 构造辅助函数声明 |
| `third_party/Debugger/include/Debugger/Instrumentation/Writer.h` | 接口 | ring buffer 操作 + `RecordSink` 抽象类 + 工厂函数 |
| `third_party/Debugger/lib/Instrumentation/Passes.cpp` | 实现 | `InsertInstrumentationPass` |
| `third_party/Debugger/lib/Instrumentation/Collectors.cpp` | 实现 | collector 规格表 + host-side 统计计算 |
| `third_party/Debugger/lib/Instrumentation/RecordBuilder.cpp` | 实现 | record 构造辅助函数 |
| `third_party/Debugger/lib/Instrumentation/Writer.cpp` | 实现 | ring buffer 操作 + `LinearAppendSink` + `RingBufferSink` |
| `third_party/Debugger/test/unittest/InstrumentationTest.cpp` | 测试 | 14 个单元测试 |

---

## 3. 各功能实现详解

### 3.1 `InsertInstrumentationPass`（`Passes.cpp`）

**职责**：MLIR Pass，对 ModuleOp 做一遍遍历，识别需要插桩的 op 并在 IR 上打 attribute 标记。

#### 3.1.1 Pass 整体流程

```
runOnOperation()
  ├─ [幂等性检查] 若 module 已有 flagtree.debug.instrumentation_inserted → 直接返回
  ├─ module.walk(所有嵌套 op)
  │   ├─ 读取 op_id（优先 flagtree.debug.op_id，回退 op_id）
  │   ├─ op_id == 0 或缺失 → 跳过
  │   ├─ 读取 RecordLevel（从 op 向上逐层查找）
│   ├─ 判断 hasSummary = 结果类型是否支持 summary collector
│   ├─ 判断 hasMemoryEvent = addr_level > 0 && isMemoryLikeOp(op) && 存在 memory pointer operand
  │   ├─ 判断 hasFullValueRef = (hasSummary && level == LEVEL_TENSOR_FULL)
  │   └─ 在 op 上设置以下 attribute：
  │       flagtree.debug.instrumented          = true
  │       flagtree.debug.record_kinds          = ["summary"?, "memory_event"?, "full_value"?]
  │       flagtree.debug.summary_collectors    = ["nan_count", "inf_count", ...]  (如有 summary)
│       flagtree.debug.memory_event_kind     = "ADDRESS_SUMMARY"/"LAST_ALIGNED_ADDR"/"BASE_ALIGNED_ADDR" (如有 memory event)
  │       flagtree.debug.full_value_ref        = true                            (如有 full value)
  ├─ [若无任何 op 被标记] → 返回
  ├─ 在 module 上设置 flagtree.debug.instrumentation_inserted = true
  └─ 遍历所有 FunctionOpInterface
      └─ 含已标记 op 的函数 → 设置：
          flagtree.debug.hidden_arg                     = "__debug_ctrl_ptr"
          flagtree.debug.logical_instance_id_formula   = "pid0 + pid1 * num_programs0 + pid2 * num_programs0 * num_programs1"
```

#### 3.1.2 op 分类策略

| op 类型 | hasSummary | hasMemoryEvent | 生成 record |
|---------|-----------|---------------|------------|
| 普通计算 op（有结果） | true | false | SummaryRecord |
| store / atomic（无结果） | false | true | MemoryEventRecord |
| load（有结果 + 是 memory op） | true | true | SummaryRecord + MemoryEventRecord |

`isMemoryLikeOp` 判断逻辑：
1. 优先读取 B 显式标注的 `flagtree.debug.is_memory_op` / `is_memory_op` bool attribute
2. 若无，则根据 op 名称字符串匹配：`load`、`store`、`atomic`、`async_copy`

#### 3.1.3 RecordLevel 发现

`getRecordLevel(op)` 从当前 op 向上逐级遍历到 ModuleOp，查找 `flagtree.debug.record_level` 或 `debug_record_level` attribute（兼容 B 模块的不同命名风格）。默认值为 `LEVEL_SUMMARY`。

`getAddrLevel(op)` 使用同样的逐级查找规则读取 `flagtree.debug.addr_level` 或
`debug_addr_level`。默认值为 `0`，表示不插入动态 address event；`1` 表示插入
地址 summary；`2` 预留给 full lane dump。

#### 3.1.4 幂等性保护

新增逻辑：`runOnOperation()` 开头检查 `flagtree.debug.instrumentation_inserted` 是否已设，若是则立即返回。这保证 pass 可以在 pipeline 中安全重复执行（例如在联调期被多次调用）。

#### 3.1.5 当前阶段与最终目标的关系

当前 pass 在 metadata-only 编译路径只输出 IR attribute；在 hidden-arg ABI 打开时会插入并降低 `flagtree_debug.record_*` / `flagtree_debug.capture_memory_address`，生成实际的 GPU 侧归约计算指令和 ring buffer 写入指令。

---

### 3.2 `RecordSink` 抽象与实现（`Writer.h` / `Writer.cpp`）

**职责**：提供抽象的 record 写入接口，使 C 模块的 record 写入逻辑与底层存储（线性 buffer 或 F 的 ring buffer）解耦。

#### 3.2.1 接口定义

```cpp
// third_party/Debugger/include/Debugger/Instrumentation/Writer.h
class RecordSink {
public:
  virtual ~RecordSink() = default;
  virtual RecordWriteResult writeSummary(const SummaryRecord &record) = 0;
  virtual RecordWriteResult writeMemoryEvent(const MemoryEventRecord &record) = 0;
  virtual RecordWriteResult writeFullValueRef(const FullValueRefRecord &record) = 0;
  virtual uint32_t recordCount() const = 0;
};
```

`recordCount()` 返回**成功写入**的 record 数（仅 `WRITTEN` 状态）。

#### 3.2.2 `LinearAppendSink`（线性 append sink）

**实现文件**：`third_party/Debugger/lib/Instrumentation/Writer.cpp`，匿名 namespace 内的 `LinearAppendSink` 类。

**存储布局**：

```
[slot 0: 32 bytes][slot 1: 32 bytes][slot 2: 32 bytes]...
   ↑ offset 0         ↑ offset 32       ↑ offset 64
```

无 `RingBufferHeader` 前缀。每条 record 固定 `kDefaultRecordSize`（32 bytes）步长。

**写入流程**（`writeRaw` 私有方法）：
1. 校验 `recordSize == kDefaultRecordSize`，否则返回 `INVALID_ARGUMENT`
2. 计算 `offset = count_ * 32`，检查是否超出 `sizeBytes_`
3. 超出则 `overflowCount_++`，返回 `OVERFLOW`
4. 否则 `memcpy`，`count_++`，返回 `WRITTEN`

**适用场景**：单元测试。caller 传入 `vector<uint8_t>` buffer，写完后直接将 buffer 字节 cast 回具体 record struct 验证字段值，无需 ring buffer header 解析开销。

**工厂函数**：
```cpp
std::unique_ptr<RecordSink> createLinearAppendSink(void *buffer, size_t sizeBytes);
```

#### 3.2.3 `RingBufferSink`（联调期 ring buffer sink）

**实现文件**：同上，匿名 namespace 内的 `RingBufferSink` 类。

**实现方式**：每个 `write*` 方法直接委托给已有的 `appendSummaryRecord` / `appendMemoryEventRecord` / `appendFullValueRefRecord`，这三个函数内部调用 `appendRecordToRingBuffer`，具备完整的：
- `write_idx` 原子递增（host-side 模拟）
- `overflow_count` 计数
- `RB_FLAG_OVERFLOW` 标志位设置

联调期将 `ctrlPtr` 替换为 F 模块分配的设备地址，即可复用相同的写入语义。

**工厂函数**：
```cpp
std::unique_ptr<RecordSink> createRingBufferSink(void *ctrlPtr, size_t bufferSize);
```

---

### 3.3 `SummaryStats` 与 host-side 统计计算（`Collectors.h` / `Collectors.cpp`）

**职责**：提供在 CPU 上计算 summary 指标的能力，用于：
1. 单元测试中验证指标计算公式是否正确
2. 作为 GPU 侧实现的参考规范（设备侧归约的预期结果）

#### 3.3.1 `SummaryStats` 结构体

```cpp
struct SummaryStats {
  uint64_t nanCount    = 0;  // 元素中 NaN 的个数
  uint64_t infCount    = 0;  // 元素中 ±Inf 的个数
  double   mean        = 0.0; // 有限值的算术均值（无有限值时为 0.0）
  double   min         = 0.0; // 有限值的最小值（无有限值时为 0.0）
  double   max         = 0.0; // 有限值的最大值（无有限值时为 0.0）
  uint64_t elementCount = 0;  // 元素总数（含 NaN 和 Inf）
};
```

字段语义与分工文档 §3.3 "B. C 需要直接统计并写入的数值摘要指标"完全对应。

#### 3.3.2 `computeSummaryStatsF32` 实现

```
遍历 float 数组：
  isnan(v)  → nanCount++
  isinf(v)  → infCount++
  else      → sum += (double)v, 更新 min/max, finiteCount++

结束后：
  elementCount = count（总数，含异常值）
  finiteCount > 0 时：mean = sum / finiteCount, min = minVal, max = maxVal
  finiteCount == 0 时：mean/min/max 保持 0.0（空数组或全 NaN/Inf）
```

**关键实现细节**：
- 内部累加使用 `double`，避免 float 单精度累加误差
- `min` / `max` 初始值为 `+∞` / `-∞`，保证任何有限值都能正确更新
- NaN/Inf 均不参与 mean/min/max 计算（对应文档中 `MEAN_FINITE`、`MIN_FINITE`、`MAX_FINITE` 的语义）

`computeSummaryStatsF64` 实现逻辑相同，无需 float→double 转换。注意：这是
host-side 参考实现；当前 device-side summary lowering 仍会先转成 f32 再做归约，
因此 f64 精度敏感场景需要后续 dtype-preserving collector 路径补齐。

#### 3.3.3 `writeSummaryRecordsToSink` 实现

```cpp
void writeSummaryRecordsToSink(uint32_t opId, uint64_t logicalInstanceId,
                               const SummaryStats &stats, RecordLevel level,
                               RecordSink &sink);
```

**实现流程**：
1. 调用 `getEnabledCollectors(level)` 获取当前 level 下启用的 collector 列表
2. 按列表顺序，对每个 `CollectorKind` 从 `SummaryStats` 中取对应字段：
   - `NAN_COUNT` / `INF_COUNT` / `ELEMENT_COUNT` → `buildSummaryU64Record(opId, instanceId, kind, value)`
   - `MEAN_FINITE` / `MIN_FINITE` / `MAX_FINITE` → `buildSummaryF64Record(opId, instanceId, kind, value)`
3. 调用 `sink.writeSummary(record)` 写入

**Phase 1 下启用的 6 个 collector 及写入顺序**：
```
NAN_COUNT → INF_COUNT → ZERO_COUNT → MEAN_FINITE → MIN_FINITE → MAX_FINITE → L2_NORM → ELEMENT_COUNT
```

此顺序由 `kSummaryCollectorSpecs` 静态表决定，与 `getEnabledCollectors` 遍历顺序一致。

**端到端语义**：`writeSummaryRecordsToSink` 把"host-side 统计计算 → record 构造 → sink 写入"三步串联，镜像了 GPU kernel 执行时"归约 → 构建 record header/payload → 写 ring buffer"的逻辑流程。

---

### 3.4 已有代码（本次未修改）

以下功能在本次修改前已完整实现，本次工作直接复用：

#### `RecordBuilder.cpp`（完整）
- `buildSummaryU64Record` / `buildSummaryF32Record` / `buildSummaryF64Record`
- `buildMemoryEventRecord`
- `buildFullValueRefRecord`

每个函数填充 `RecordHeader`（`recordKind`、`opId`、`logicalInstanceId`）和 payload 字段，保证 record 大小严格等于 32 bytes（由 `Protocol.h` 的 `static_assert` 约束）。

#### `Writer.cpp` ring buffer 操作（完整）
- `computeLogicalInstanceId(pid0, pid1, pid2, num_programs0, num_programs1)`
  实现公式：`pid0 + pid1 * num_programs0 + pid2 * num_programs0 * num_programs1`
- `makeRingBufferHeader` / `initializeRingBufferStorage`
- `appendRecordToRingBuffer`（含 overflow 计数和 `RB_FLAG_OVERFLOW` 标志）

#### `Collectors.cpp` 规格表（完整）
- `kSummaryCollectorSpecs` 静态表，定义 Phase 1 的 6 个 collector
- `getSummaryCollectorSpecs` / `getCollectorName` / `getEnabledCollectors` / `isCollectorEnabledAtLevel` / `isKnownCollector`

---

## 4. 单元测试覆盖（14 个 test case）

测试文件：`third_party/Debugger/test/unittest/InstrumentationTest.cpp`

| 测试名 | 对应测试矩阵 | 验证内容 |
|--------|------------|---------|
| `BuildsSummaryRecords` | 原有 | record builder 字段正确性 |
| `BuildsMemoryAndFullValueRecords` | 原有 | memory event / full value record 字段 |
| `ComputesLogicalInstanceId` | C-3 | `logical_instance_id` 公式：`pid0 + pid1*n0 + pid2*n0*n1` |
| `CollectorLookupMatchesPhase1Set` | 原有 | Phase 1 启用 6 个 collector |
| `RingBufferWriterWritesAndTracksOverflow` | C-4 | ring buffer overflow 计数和 FLAG |
| `RejectsMismatchedRecordSize` | 原有 | 不匹配 record size 返回 INVALID_ARGUMENT |
| `ComputeSummaryStatsF32_Mixed` | **新增** | 混合数组（有限+NaN+Inf）统计正确 |
| `ComputeSummaryStatsF32_Empty` | **新增** | 空数组边界处理 |
| `ComputeSummaryStatsF32_AllNaN` | **新增** | 全 NaN 时 mean/min/max 为 0.0 |
| `ComputeSummaryStatsF64_Finite` | **新增** | F64 全有限值统计正确 |
| `LinearAppendSink_WritesAndOverflows` | **新增（C-4）** | LinearAppendSink overflow 计数，raw buffer 字节验证 |
| `LinearAppendSink_MixedRecordTypes` | **新增** | 混合写入不同 record 类型，RecordKind 字段正确 |
| `RingBufferSink_WritesAndOverflows` | **新增（C-4）** | RingBufferSink 通过 RecordSink 接口触发 overflow |
| `WriteSummaryRecordsToSink_Phase1Core` | **新增（C-1）** | 端到端：统计→8条record→sink，opId/instanceId/collectorKind/value 全部正确 |

---

## 5. 模块内数据流

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                    C 模块（单独开发期）                  │
                    │                                                         │
  B 的 IR 标注      │   InsertInstrumentationPass                             │
  (op_id,          ─→  ┌───────────────────────────────────────────┐         │
   is_memory_op,   │   │ 遍历 op → 读 op_id / is_memory_op /       │         │
   record_level)   │   │ record_level → 决定 record 类型 →         │         │
                    │   │ 打 attribute：instrumented / record_kinds / │        │
                    │   │ summary_collectors / memory_event_kind /  │         │
                    │   │ hidden_arg / logical_instance_id_formula  │         │
                    │   └───────────────────────────────────────────┘         │
                    │                                                         │
  实际数据（测试）  │   computeSummaryStatsF32/F64                            │
  float/double ──→ │   ┌──────────────┐   SummaryStats                      │
  data[]            │   │ 统计 NaN/Inf │ ──────────────→                      │
                    │   │ 有限值 sum   │                 writeSummaryRecordsToSink
                    │   │ min / max   │                 ┌─────────────────┐   │
                    │   └──────────────┘                │ 按 collector 顺序│   │
                    │                                   │ buildSummary*Record  │
                    │                                   │ sink.writeSummary() │  │
                    │                                   └────────┬────────┘   │
                    │                                            │            │
                    │                          ┌─────────────────┴──────────┐ │
                    │                          │        RecordSink          │ │
                    │                          │  ┌──────────────────────┐  │ │
                    │                          │  │ LinearAppendSink     │  │ │
                    │                          │  │ （单独开发期测试用）  │  │ │
                    │                          │  └──────────────────────┘  │ │
                    │                          │  ┌──────────────────────┐  │ │
                    │                          │  │ RingBufferSink       │  │ │
                    │                          │  │ （联调期 → F 模块）  │  │ │
                    │                          │  └──────────────────────┘  │ │
                    │                          └────────────────────────────┘ │
                    └─────────────────────────────────────────────────────────┘
```

---

## 6. 与其他模块的接口约定

### C → B（上游输入）

C 模块消费以下 B 输出的 IR attribute（`InsertInstrumentationPass` 读取）：

| Attribute 名 | 回退名 | 类型 | 含义 |
|---|---|---|---|
| `flagtree.debug.op_id` | `op_id` | IntegerAttr | B 分配的稳定 op 标识 |
| `flagtree.debug.is_memory_op` | `is_memory_op` | BoolAttr | 是否为 memory op |
| `flagtree.debug.record_level` | `debug_record_level` | IntegerAttr / StringAttr | 采集级别（1=SUMMARY, 2=TENSOR_FULL） |
| `flagtree.debug.addr_level` | `debug_addr_level` | IntegerAttr / StringAttr | 动态地址采集级别（0=关闭，1=summary，2=full lane 预留） |

### C → F（下游输出）

C 向函数打的 attribute，F 模块在接线隐藏参数时需读取：

| Attribute 名 | 值 | 含义 |
|---|---|---|
| `flagtree.debug.hidden_arg` | `"__debug_ctrl_ptr"` | 隐藏参数名称 |
| `flagtree.debug.logical_instance_id_formula` | `"pid0 + pid1 * num_programs0 + pid2 * num_programs0 * num_programs1"` | 设备侧 instance id 计算公式 |

`RingBufferSink` 生成的 record bytes 格式严格符合 `Protocol.h` 冻结的结构体布局，F 模块可直接导出。

### C → D（间接输出）

`writeSummaryRecordsToSink` 写出的 `SummaryRecord` 字节流中，`header.op_id` 与 B 的 `TrackedOpTable` 主键一致，D 模块可通过 `op_id` 回查静态元数据。

---

## 7. 尚未实现的内容（待后续阶段）

以下内容属于联调期或后续阶段任务，当前正确地保持为 stub：

1. **`addr_level=2` full lane dump**：当前只实现 `addr_level=1` 地址摘要；全量 lane address/value dump 仍需单独的 payload ABI。

2. **更复杂 pointer 链的数据流分析**：当前地址摘要采用有界反向切片，覆盖常见 `tt.addptr(tt.splat(base), offsets)` 和 prefix mask 形态；跨循环 iter_arg、复杂 select/where、非连续 offset、非 prefix mask、非等价 reshape 等形态仍会退回 fallback。

3. **P1-Optional 指标**：`denom_near_zero_count`、`neg_sqrt_count`、有限值样本快照等，由 B 识别敏感 op 类型后 C 插入专用 collector（分工文档 §3.6.2）。

4. **host buffer offset 关联增强**：报告中的 `alignment_ok` 和 buffer offset 依赖 F 的 buffer registry；地址摘要中的每个地址边界后续可进一步关联到具体 buffer/range。
