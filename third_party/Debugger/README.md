# FlagTree Debugger

`third_party/Debugger` 是 FlagTree debugger 的独立可选模块，包含编译期插桩、
运行期导出、record 解码和报告生成等实现。跨模块接口和协议定义以
`third_party/Debugger/include/Debugger` 为准。

本文面向 debugger 使用者和维护者，说明功能目的、用户接口、输出格式、可采集
指标、运行示例以及简要设计架构。

## 编译开关

Debugger 由 CMake option `FLAGTREE_ENABLE_DEBUGGER` 控制，默认开启。容器内可用
以下命令分别构建启用或禁用版本：

```bash
FLAGTREE_ENABLE_DEBUGGER=ON bash build.sh --rebuild
FLAGTREE_ENABLE_DEBUGGER=OFF bash build.sh --rebuild
```

禁用时不会生成或编译 Debugger dialect、passes、runtime、native binding 和测试
target；普通 `import triton` 与非 debugger kernel 保持可用。禁用版本调用
`triton.enable_debug()` 或 `tl.debug_collect_start/end` 会明确提示使用
`-DFLAGTREE_ENABLE_DEBUGGER=ON` 重新构建。

可运行示例位于 `third_party/Debugger/examples/`，精简的 FlagGems 回归样例位于
`third_party/Debugger/samples/`。生成的 kernel、报告、cache 和复制的第三方源码
不进入仓库。

## 目的

FlagTree debugger 用于在 Triton kernel 内采集指定代码区域的运行期调试信息。
它将编译期静态信息与运行期动态记录关联起来，用于定位 kernel 内部 operation
的数值状态、内存访问状态和 full dump 数据。

## 用户接口

### Python 配置接口

用户在 Python 侧通过 `triton.runtime.debugger` 配置输出目录、record 容量和导出
选项，并通过 `triton.enable_debug(...)` 开启 debugger 编译和运行流程。

```python
import triton
from triton.runtime import debugger

debugger.configure(
    output_dir="/tmp/flagtree_debugger_example",
    record_capacity=4096,
    export_raw_records=False,
)
triton.enable_debug(level=1, addr_level=1)
```

常用接口：

- `debugger.configure(...)`：配置输出目录、record 容量和导出选项。
- `debugger.get_config()`：读取当前 debugger 配置。
- `debugger.reset_config()`：恢复默认配置。
- `triton.enable_debug(level=..., addr_level=...)`：开启后续 kernel 的 debugger
  pipeline。
- `triton.disable_debug()`：关闭 debugger pipeline。

### Triton JIT 采集接口

Python 侧 `triton.enable_debug(...)` 仅开启 debugger pipeline。实际采集范围由
Triton JIT 函数内的 `tl.debug_collect_start(...)` 和 `tl.debug_collect_end()` 界定。

```python
import triton
import triton.language as tl


@triton.jit
def kernel(x_ptr, y_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    tl.debug_collect_start(level=1, addr_level=1)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.abs(x)
    tl.store(y_ptr + offsets, y, mask=mask)
    tl.debug_collect_end()
```

采集等级：

- `level=1`：采集 summary 指标。
- `level=2`：采集 full tensor value。
- `addr_level=1`：采集 memory address summary。
- `addr_level=2`：采集 full memory address。

## 输出

### level 1 输出示例

level 1 通常生成主文本报告、主 JSON 报告和 IR op 级日志报告。例如：

```text
test_debug_abs_kernel_aiv_20260629_193456_497_run1.txt
test_debug_abs_kernel_aiv_20260629_193456_497_run1.json
test_debug_abs_kernel_aiv_20260629_193456_497_run1_op_log.txt
test_debug_abs_kernel_aiv_20260629_193456_497_run1_op_log.json
```

通用格式：

```text
<script>_<kernel>_<timestamp>_run<N>.txt
<script>_<kernel>_<timestamp>_run<N>.json
<script>_<kernel>_<timestamp>_run<N>_op_log.txt
<script>_<kernel>_<timestamp>_run<N>_op_log.json
```

字段说明：

- `<script>`：触发 kernel 的 Python 脚本名。
- `<kernel>`：Triton kernel 名称，可能包含后端或 kernel variant 后缀。
- `<timestamp>`：报告导出时间戳。
- `run<N>`：当前进程内的 debugger run 序号。
- 主 `.txt` / `.json`：Triton statement 级报告，面向源码语句查看。
- `_op_log.txt` / `_op_log.json`：Triton MLIR op 级报告，JSON 字段名为
  `op_log`。

### level 2 full dump 输出

level 2 在主报告之外生成 artifact 目录：

```text
<script>_<kernel>_<timestamp>_run<N>_artifacts/
  tensor_index.json
  op<id>_inst<id>_rec<id>_value.npy
  op<id>_inst<id>_rec<id>_memory_address.npy
```

字段说明：

- `tensor_index.json`：artifact 索引文件。
- `op<id>`：编译期分配的 operation id。
- `inst<id>`：operation 的逻辑实例 id。
- `rec<id>`：运行期 record slot index。
- `*_value.npy`：完整 tensor value。
- `*_memory_address.npy`：完整 memory address。

level 2 主报告会在对应 statement result 或独立捕获的 operand 下记录 artifact
文件名，不直接展开完整 tensor 数据；引用已有 result 的 operand 只显示
`[result ...]` 引用，不重复打印文件名。完整 dump 路径可在 `_op_log` 或
`tensor_index.json` 中查看。
例如：

```text
[result x]:
  instances: [0]
  summary:
    ...
  full_value_file: op3_inst0_rec10_value.npy
  address_summary(load from):
    ...
  memory_address_file: op3_inst0_rec11_memory_address.npy
```

可选输出：

- `*_raw_records.txt`：当 `debugger.configure(export_raw_records=True)` 时生成。
  该文件用于 debugger record 协议和 decoder 调试。

## 可采集指标

### 静态信息

静态信息来自编译期 metadata，通常包括：

- `kernel_id`
- `kernel_name`
- `op_id`
- `scope_id`
- MLIR operation 名称
- source location
- Triton statement
- 输入和输出 dtype
- shape
- stride 或 layout 信息，如果编译期可获得

### level 1 summary 指标

对 tensor-producing operation，level 1 可采集：

- `element_count`
- `nan_count`
- `inf_count`
- `zero_count`
- `min`
- `max`
- `mean`
- `l2_norm`

对 memory operation 或 pointer-related operation，`addr_level=1` 可采集：

- `first_addr`
- `last_addr`
- `min_addr`
- `max_addr`
- `active_lane_count`
- `address_span_bytes`

地址摘要是否存在取决于后端是否能够为对应 pointer pattern 生成合法 lowering。

### level 2 full dump 指标

level 2 导出完整数据：

- full tensor value：写入 `*_value.npy`。
- full memory address：写入 `*_memory_address.npy`。
- artifact metadata：写入 `tensor_index.json`。

run-level 信息包括：

- `record_level`
- `record_count`
- `overflow_count`
- `raw_buffer_bytes`
- `export_mode`
- `backend`
- `target`

如果 `overflow_count` 非零，表示 device record buffer 容量不足，报告可能不完整。
此时应增大 `record_capacity` 后重新运行。

## 运行示例

以下示例展示如何针对 `test.py` 进行调试和信息导出。

```python
import torch
import torch_npu
import triton
import triton.language as tl
from triton.runtime import debugger


debugger.configure(
    output_dir="/tmp/flagtree_debugger_example",
    record_capacity=4096,
    export_raw_records=False,
)
triton.enable_debug(level=1, addr_level=1)


@triton.jit
def debug_abs_kernel(x_ptr, y_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    tl.debug_collect_start(level=1, addr_level=1)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.abs(x)
    z = y + 1.0
    tl.store(y_ptr + offsets, z, mask=mask)
    tl.debug_collect_end()


n = 16
block = 16
x = torch.linspace(-8, 7, n, dtype=torch.float32, device="npu")
y = torch.empty_like(x)

debug_abs_kernel[(1,)](x, y, n, BLOCK_SIZE=block)
torch_npu.npu.synchronize()

expected = torch.abs(x) + 1.0
ok = torch.allclose(y.cpu(), expected.cpu())
runs = debugger.take_exported_runs()
print(f"output_allclose={ok}")
print(f"exported_runs={len(runs)}")
for run in runs:
    print(f"report_path={run.get('report_path')}")
    print(f"meta={run.get('meta')}")
```

`debugger.take_exported_runs()` 用于取得本次 debug 导出的信息，便于在脚本中打印
报告路径、metadata 等内容。如果采集 level 2 full dump，还会包含 artifact 目录。

### 运行命令

Ascend 后端运行时建议显式设置后端和目标 SoC 名称。输出目录由
`debugger.configure(output_dir=...)` 在 Python 脚本中指定。

```bash
export FLAGTREE_BACKEND=ascend
export TRITON_ASCEND_ARCH=Ascend910B4
python3 test.py
```

### 运行输出示例

level 1 debug 运行完成后，脚本输出可包含：

```text
output_allclose=True
exported_runs=1
report_path=/tmp/flagtree_debugger_example/test_debug_abs_kernel_aiv_20260630_002414_291_run1.txt
meta={'run_id': 1, 'device_id': 0, 'kernel_id': 4288825906, 'protocol_version': 2, 'record_level': 1, 'export_mode': 1, 'backend_kind': 4}
```

`exported_runs=1` 表示本次执行导出了一次 debugger run；
`report_path` 指向文本主报告。

level 1 输出目录通常包含：

```text
/tmp/flagtree_debugger_example/test_debug_abs_kernel_aiv_20260630_002414_291_run1.txt
/tmp/flagtree_debugger_example/test_debug_abs_kernel_aiv_20260630_002414_291_run1.json
/tmp/flagtree_debugger_example/test_debug_abs_kernel_aiv_20260630_002414_291_run1_op_log.txt
/tmp/flagtree_debugger_example/test_debug_abs_kernel_aiv_20260630_002414_291_run1_op_log.json
```

### 报告片段示例

level 1 主报告以 Triton statement 级视图为主，按源码语句展示 result 和
operand。IR op 级报告保留在 `_op_log.txt` / `_op_log.json` 中，用于查看
编译后 op 粒度的静态 metadata 和动态记录。

statement 级报告只展示源码语句相关信息，不展开 `op_id`、capture policy 等
IR 级实现细节。load 结果下的 `address_summary(load from)` 表示该结果从哪里读出；
store 语句下的 `address_summary(store to)` 表示该语句写到哪里。地址来自完整
pointer 表达式求值结果，例如 `y_ptr + offsets`；如果语句有 mask，
`active_lane_count` 和地址范围按 mask 后的 active lane 统计。

#### Triton 语句级报告示例

load 语句示例：

```text
Triton Statement Records
source_loc: loc("test.py":30:16)
statement_id: 30004
statement: x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
  [result x]:
    instances: [0]
    summary:
      element_count: [16 (U64)]
      nan_count    : [0 (U64)]
      inf_count    : [0 (U64)]
      zero_count   : [1 (U64)]
      mean         : [-0.5 (F32)]
      min          : [-8 (F32)]
      max          : [7 (F32)]
      l2_norm      : [18.5472 (F32)]
    full_value_file: op3_inst0_rec10_value.npy     # level 2 only
    address_summary(load from):
      status            : [complete]
      first_addr        : [0x...]
      last_addr         : [0x...]
      min_addr          : [0x...]
      max_addr          : [0x...]
      active_lane_count : [16]
      address_span_bytes: [64]
    memory_address_file: op3_inst0_rec11_memory_address.npy
  <operand x_ptr + offsets>:
    runtime: not captured
  <operand mask>:
    instances: [0]
    summary:
      element_count: [16 (U64)]
    full_value_file: op4_inst0_rec12_value.npy     # level 2 only
  <operand other>:
    constant_value: dense<0.000000e+00> : tensor<16xf32>
    runtime: not captured
```

计算语句示例：

```text
source_loc: loc("test.py":31:15)
statement_id: 31004
statement: y = tl.abs(x)
  [result y]:
    instances: [0]
    summary:
      element_count: [16 (U64)]
      nan_count    : [0 (U64)]
      inf_count    : [0 (U64)]
      zero_count   : [1 (U64)]
      mean         : [4 (F32)]
      min          : [0 (F32)]
      max          : [8 (F32)]
      l2_norm      : [18.5472 (F32)]
  <operand x>: [result x]
```

store 关联示例：

```text
source_loc: loc("test.py":32:12)
statement_id: 32004
statement: z = y + 1.0
  [result z]:
    instances: [0]
    summary:
      element_count: [16 (U64)]
      nan_count    : [0 (U64)]
      inf_count    : [0 (U64)]
      zero_count   : [0 (U64)]
      mean         : [5 (F32)]
      min          : [1 (F32)]
      max          : [9 (F32)]
      l2_norm      : [22.0907 (F32)]
  <operand y>: [result y]
  <operand rhs>:
    constant_value: dense<1.000000e+00> : tensor<16xf32>
    runtime: not captured

source_loc: loc("test.py":33:30)
statement_id: 33004
statement: tl.store(y_ptr + offsets, z, mask=mask)
  memory_access:
    instances: [0]
    address_summary(store to):
      status            : [complete]
      first_addr        : [0x...]
      last_addr         : [0x...]
      min_addr          : [0x...]
      max_addr          : [0x...]
      active_lane_count : [16]
      address_span_bytes: [64]
  <operand y_ptr + offsets>:
    runtime: not captured
  <operand z>: [result z]
```

#### Triton IR op 级报告示例

IR op 级报告以 `IR Op Log Records` 为主视图。每个 `op_id` 只展示一次编译期
静态元数据；同一个 op 的动态记录按 `logical_instance_id` 聚合，文本报告中
`instances` 是对齐轴，`summary` 与 `address_summary` 的每个指标都按这个顺序
输出数组。

计算 op 示例：

```text
op_id=5 scope_id=1
  static:
    mlir_op: arith.addf
    source_loc: loc("test.py":42:12)
    triton_statement: arith.addf
    dtype_in: arg0=tensor<16xf32>, arg1=tensor<16xf32>
    dtype_out: tensor<16xf32>
    shape: [16]

  instances: [0]
  summary:
    element_count: [16 (U64)]
    nan_count    : [0 (U64)]
    inf_count    : [0 (U64)]
    zero_count   : [0 (U64)]
    mean         : [5 (F32)]
    min          : [1 (F32)]
    max          : [9 (F32)]
    l2_norm      : [22.0907 (F32)]
```

store op 示例：

```text
op_id=8 scope_id=1
  static:
    mlir_op: tt.store
    role: store
    category: store
    source_loc: loc("test.py":45:30)
    triton_statement: tt.store
    dtype_in: arg0=tensor<16x!tt.ptr<f32>>, arg1=tensor<16xf32>, arg2=tensor<16xi1>
    dtype_out: tensor<16xf32>
    shape: [16]
    memory_semantics: addr_space=global access_type=store access_bytes=4 alignment_required=4 has_mask=true boundary_check_policy=<none>

  instances: [0]
  address_summary:
    status            : [complete]
    first_addr        : [0x...]
    last_addr         : [0x...]
    min_addr          : [0x...]
    max_addr          : [0x...]
    active_lane_count : [16]
    address_span_bytes: [64]
```

`IR Op Log Static Only Ops` 列出有 `op_id` 和静态元数据、但没有 runtime record
的 op。这些 op 通常用于 producer/context 分析，例如 pointer-producing
`tt.splat` / `tt.addptr`，不会重复写无意义的动态 summary。

## 设计架构

整体设计包含以下模块：

- Frontend/：保存 Python 配置，控制 debug mode，并为 kernel launch 准备运行期
  metadata 和 hidden argument。
- Metadata/：扫描 collect region 内的 IR operation，生成稳定的 `op_id` 和静态
  描述。
- Instrumentation/：根据 `level` 和 `addr_level` 插入 summary、full value 和 full
  address 记录逻辑。
- Runtime/：管理 debug buffer 和 control block，执行 post-kernel export，并将 raw
  buffer 交给 host decoder。
- Decode/：将 raw record 解码为 op/instance 结构，并处理 overflow、payload 和路径
  信息。
