# Debugger Include Tree

`third_party/Debugger/include/Debugger` 目录保存 debugger 的公共契约。这里定义的是跨模块都要遵守的接口，而不是某个人的临时实现。

目录划分：

- `Common/`：统一协议、record 布局、buffer header、运行时主键
- `Frontend/`：A 模块，Python 前端与 launch/ABI 接线，负责人华师
- `Metadata/`：B 模块，编译期作用域解析、`op_id` 分配、静态元数据，负责人华师
- `Instrumentation/`：C 模块，GPU 插桩、summary/memory event 记录，负责人颜臻
- `Runtime/`：F 模块，control block、ring buffer、导出与运行时上下文，负责人闫明
- `Decode/`：D 模块，解码与报告，负责人玉珏

对齐原则：

- 协议主键由 `kernel_id + op_id + logical_instance_id` 组成。
- 编译期静态信息统一由 B 输出到 `KernelDebugMetadata / TrackedOpTable`。
- 运行期动态数据统一由 C 写入 `Record`，由 F 导出，由 D 解码。
- 运行期 host 上下文和动态 tensor/buffer 信息统一由 A/F 通过 `DebugRuntimeMetadata` 传递。
- 不要在各模块内部各自重新定义一套 record 布局、buffer header 或 metadata schema。

并行开发入口：

- 统一公共头：`Debugger.h`
- 真实后端入口：`createTransferEngine()`
- 按 backend 选择后端入口：`createTransferEngine(BackendKind, streamHandle)`

Python 调试接口：

Debugger 默认通过 Python 侧接口开启，编译期会进入 debugger instrumentation
mode，运行期会为 kernel launch 准备 `__debug_ctrl_ptr` hidden arg，并在 kernel
结束后导出报告。

基本用法：

```python
import triton
import triton.language as tl
from triton.runtime import debugger

# 通常在 import 后配置并开启一次。后续哪些 Triton IR op 被记录，
# 由 @triton.jit 内部的 tl.debug_collect_start/end 控制。
debugger.configure(
    output_dir="/tmp/flagtree_debugger_manual",
    record_capacity=4096,
    export_raw_records=False,
)
triton.enable_debug(level=1, addr_level=0)


@triton.jit
def kernel(x_ptr, y_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    tl.debug_collect_start(level=1, addr_level=1)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.abs(x)
    tl.store(y_ptr + offsets, y, mask=mask)
    tl.debug_collect_end()


kernel[(grid,)](...)
```

容器内编译命令：

当前验证通过的构建方式是在 CANN9 容器内运行仓库根目录的 `build.sh`。脚本统一
配置 LLVM/clang、Python、CANN、glibc compatibility object 和 CMake 参数，不依赖
conda clang。

从 host 侧触发容器内完整 rebuild：

```bash
docker exec flagtree-cann9-quan /bin/bash -lc '
cd "${FLAGTREE_SOURCE_DIR:-/workspace/FlagTree}"
FLAGTREE_ENABLE_DEBUGGER=ON MAX_JOBS=16 bash build.sh --rebuild
'
```

如果已经 attach 到容器内部，执行等价命令：

```bash
cd "${FLAGTREE_SOURCE_DIR:-/workspace/FlagTree}"
FLAGTREE_ENABLE_DEBUGGER=ON MAX_JOBS=16 bash build.sh --rebuild
```

常用接口：

- `debugger.configure(...)`：设置 debugger 默认配置；未传字段保持当前值。
  支持字段包括：
  - `output_dir`：报告输出目录；传 `None` 可关闭文件导出。
  - `record_capacity`：ring buffer record 容量。
  - `export_mode`：导出模式，默认 `POST_KERNEL_EXPORT`。
  - `export_on_error`：kernel 报错时是否仍尝试导出。
  - `export_raw_records`：是否把 decoded raw records 额外写到 sidecar 文件。
- `debugger.get_config()`：查询当前默认配置。
- `debugger.reset_config()`：恢复默认配置。
- `triton.enable_debug(level=1, addr_level=0)`：开启进程级 debugger 模式。通常
  在 import 后调用一次；`level` 控制数值采集等级，`addr_level` 控制动态地址
  采集，默认 `0` 表示不插入地址采集。
- `tl.debug_collect_start/end`：在 `@triton.jit` 内界定实际采集范围。Python
  侧 enable 只开启 debugger pipeline，不会记录普通 PyTorch/torch_npu 语句。
  `tl.debug_collect_start(level=..., addr_level=...)` 可覆盖当前 region 的地址
  采集等级；不传 `addr_level` 时继承 `triton.enable_debug(...)` 的配置。
- `triton.disable_debug()`：关闭 debugger，并清理 launch hook。普通一次性脚本
  通常不需要调用；长进程、notebook 或测试套件中可用它避免影响后续 kernel。
- `debugger.take_exported_runs()`：取回本进程内已导出的 run 信息。
- `debugger.clear_exported_runs()`：清空本进程内已缓存的导出结果。

后端适配注意事项：

- summary record 的 device lowering 主要依赖通用 TTIR arithmetic/reduce/store。
- memory address event 依赖 debugger 专用
  `flagtree_debug.capture_memory_address` lowering；只有 `addr_level > 0` 才会插入
  该动态地址采集。当前 CANN9 路径在 `addr_level=1` 时会对可反向切片的
  `tt.addptr(tt.splat(base), offsets)` 指针链生成地址摘要：
  `first_addr / last_addr / min_addr / max_addr / active_lane_count /
  address_span_bytes`。该路径要求 offset 可证明为连续 lane offset，mask 为空、
  全 true，或形如 `offsets < limit` 的 prefix mask。无法匹配的指针/掩码形态会
  退回到单条 base/last aligned address 事件，保证 debugger 不破坏正常编译。
  新增后端时需要验证或重写
  `flagtree_debug.capture_memory_address` lowering。`addr_level=2` 预留给 full
  lane dump，当前未实现时报告不得伪装为已采集。

导出文件：

- 默认输出目录：`/tmp/flagtree_debugger_manual`。
- 主报告文件名包含脚本名、kernel 名、时间戳和 run id，例如
  `test_debug_abs_kernel_aiv_20260617_150006_507_run1.txt`。
- 主报告默认只包含整理后的 header 和文本报告，不直接 dump decoded raw records。
- 需要调试 raw record 时，先使用
  `debugger.configure(export_raw_records=True)`，再在进程初始化阶段
  `triton.enable_debug(level=1)`，会额外生成
  `*_raw_records.txt`。

查看报告：

```bash
cat /tmp/flagtree_debugger_manual/<report-file-stem>.txt
cat /tmp/flagtree_debugger_manual/<report-file-stem>_raw_records.txt
```
