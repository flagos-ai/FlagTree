# Sunrise Backend — TLE 使用文档

> 面向 flagtree Sunrise backend（GPUTarget 名 `tang`，torch device type `ptpu`）上使用
> Triton Language Extensions (TLE) 的用户指南。

---

## 1. 快速开始

```python
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

DEVICE = triton.runtime.driver.active.get_active_torch_device()  # ptpu

@triton.jit
def axpy_kernel(x_ptr, y_ptr, out_ptr, n, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    # 用 shared memory 暂存 x，再读回参与计算
    smem = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None,
                         scope=tle.gpu.smem, nv_mma_shared_layout=False)  # 注意 False
    ptrs = tle.gpu.local_ptr(smem, (tl.arange(0, BLOCK),))
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(ptrs, x)
    xs = tl.load(ptrs)
    tl.store(out_ptr + offs, xs * alpha + tl.load(y_ptr + offs, mask=mask, other=0.0), mask=mask)
```

---

## 2. 支持的 TLE 特性

TLE 官方（[FlagTree Wiki: TLE](https://github.com/flagos-ai/FlagTree/wiki/TLE)）将扩展分为
**三个层次**，面向不同技能画像的用户；本节按此三层归类 Sunrise 的支持情况：

- **TLE-Lite** — 轻量语义提示（"write once, run anywhere"）。用高层语义提示引导编译器
  启发式，向后兼容、跨平台，改动最小即可加速。面向算法工程师。
- **TLE-Struct** — 架构感知的结构化抽象（按 GPGPU/DSA 硬件拓扑族分类），显式描述
  层级化的存储/并行结构做深度调优。需要一定硬件知识。
- **TLE-Raw** — 原生直通，内联 vendor 私有代码，绕过通用中间层，最大化控制。面向专家。

> Lowering：TLE-Lite / TLE-Struct 经 FLIR 下降到 LLVM IR；TLE-Raw 经 vendor 私有流水下降。

### 2.0 Sunrise 支持矩阵（按三层归类）

**TLE-Lite**

| 特性 | 状态 | 用途 |
|---|---|---|
| `tle.load(..., is_async=True/False)` | ✅ | 异步/同步加载；`is_async=True` 走 async copy→shared→local_load |
| `tle.extract_tile(x, index, tile_shape)` | ✅ | 从大 tensor 抽取 tile |
| `tle.insert_tile(x, tile, index)` | ✅ | 把 tile 写回大 tensor |
| `tle.cumsum(x, axis, reverse)` | ✅ | CTA 级 exclusive 前缀和，返回 `(exclusive, total)` |
| `#@hint: shared_memory` 源码注释 | ✅ | 提示编译器把该 `tl.load` 走 shared memory async copy |
| `tle.pipe` / `pipe_reader` / `pipe_writer` | ❌ | 依赖 warp_specialize+mbarrier，S2 不支持（见第 3 节）|

**TLE-Struct**

| 特性 | 状态 | 用途 |
|---|---|---|
| `tle.gpu.alloc(...)` | ✅ | 分配 shared memory 缓冲（`buffered_tensor`）|
| `tle.gpu.local_ptr(buffer, indices)` | ✅ | 物化 shared memory 指针，供 `tl.load`/`tl.store` |
| `tle.memory_space(tensor, "shared_memory")` | ✅ | 给 tensor 指定内存空间 |
| `tle.gpu.copy(src, dst, shape)`（普通指针 / normcopy）| ✅ | 全局↔shared 拷贝 |
| `tle.gpu.copy`（`tensor_descriptor` / tmacopy）| ❌ | TMA 是 Hopper 专有（见第 3 节）|
| `tle.gpu.warp_specialize` / `tle.gpu.pipeline` | ❌ | 依赖 Hopper warpgroup（见第 3 节）|

**TLE-Raw**

| 特性 | 状态 | 用途 |
|---|---|---|
| `tle.raw` / `dsl_region` | ❌ | 裸 DSL region（高级用法）|

### 2.1 `tle.gpu.alloc` — Shared Memory 分配

```python
smem = tle.gpu.alloc(
    [M, N],                       # shape
    dtype=tl.float32,
    layout=None,
    scope=tle.gpu.smem,           # shared memory
    nv_mma_shared_layout=False,   # ⚠️ Sunrise 必须传 False
)
```

> **⚠️ 关键限制**：`nv_mma_shared_layout` **必须为 `False`**。`True` 是 NVIDIA Hopper
> 的 MMA shared layout，Sunrise 不支持。传 `False` 走 SwizzledShared layout，与 Sunrise
> 的 shared memory 下沉匹配。

### 2.2 `tle.gpu.local_ptr` — Shared 指针

```python
# 全视图（覆盖整个 buffer）
ptrs = tle.gpu.local_ptr(smem)

# 带索引（每个索引张量 shape 需一致，个数 == buffer rank）
row = tl.broadcast_to(tl.arange(0, M)[:, None], (M, N))
col = tl.broadcast_to(tl.arange(0, N)[None, :], (M, N))
ptrs = tle.gpu.local_ptr(smem, (row, col))

vals = tl.load(ptrs)     # 从 shared 读
tl.store(ptrs, vals)     # 往 shared 写
```

- `buffer` 必须来自 `tle.gpu.alloc`。
- 索引张量的个数必须等于 buffer 的 rank；否则编译报错。

### 2.3 `tle.load` — 异步加载

```python
# is_async=True：编译期改写为 async copy → shared → local_load
x = tle.load(x_ptr + offs, mask=mask, other=0.0, is_async=True)

# is_async=False：等价于普通 tl.load
x = tle.load(x_ptr + offs, mask=mask, other=0.0, is_async=False)
```

### 2.4 `tle.extract_tile` / `insert_tile` — Tile 操作

```python
# 从 512x512 抽取 index=[1,1] 处的 128x128 tile
tile = tle.extract_tile(x, index=[1, 1], tile_shape=[128, 128])

# 把 tile 写回 index=[1,1]
z = tle.insert_tile(x, tile, index=[1, 1])
```

契约要求（违反会在编译期报 `ValueError`/`CompilationError`）：
- `tile_shape` 的 rank 必须等于源 tensor 的 rank；
- 源各维必须能被对应 tile 维整除；
- `index` 不能越界（tile grid 范围内）；
- `insert_tile`：源与 tile 的 rank、element type 必须一致。

**两条下沉路径**（自动选择，用户无感）：
- **静态 index + CTA-tile 对齐** → 寄存器置换（最快，无 shared/barrier）；
- **动态 index 或非对齐** → SMEM 中继路径（用 shared memory + barrier）。

### 2.5 `tle.cumsum` — CTA 级前缀和

```python
# exclusive[i] = sum(x[:i]); total = sum(x)
exclusive, total = tle.cumsum(x, axis=0, reverse=False)
```

- 返回 **元组** `(exclusive_sum, total_sum)`，语义是 **exclusive**（排他），与上游
  `tl.cumsum`（inclusive、无 total）不同。
- 主要用于 TopK selector 类算法。支持 `reverse=True`。

### 2.6 `tle.gpu.copy` — 内存拷贝（仅 normcopy）

```python
# 全局 → shared
tle.gpu.copy(a_ptrs, smem, [M, N])
# shared → 全局
tle.gpu.copy(smem, c_ptrs, [M, N])
```

> **⚠️ 限制**：Sunrise 只支持 **normcopy**（src/dst 是普通指针 tensor ↔ `buffered_tensor`）。
> 若传入 `tl.tensor_descriptor`（TMA 描述符）会走 tmacopy，**Sunrise 不支持**（TMA 是
> Hopper 专有）。

### 2.7 `#@hint: shared_memory` — 源码注释提示

```python
@triton.jit
def kernel(x_ptr, ...):
    ...
    x = tl.load(x_ptr + offs, mask=mask)  #@hint: shared_memory
    ...
```

- 注释必须写在 `tl.load` **所在行**（按行号匹配）。
- 编译器会把该 load 改写成 async copy → shared → local_load 链路。

---

## 3. 不支持的特性（及原因）

以下特性依赖 **NVIDIA Hopper 专有硬件** 或 **多卡/cluster 硬件**，Sunrise 无对应能力，
**不支持**。使用会在编译期报错或明确失败：

| 特性 | 所属层 | 不支持原因 |
|---|---|---|
| `tle.gpu.pipe`（pipe/reader/writer）| TLE-Lite | 依赖 NVWS dialect + Hopper mbarrier |
| `tle.gpu.warp_specialize` | TLE-Struct | 依赖 Hopper warpgroup + mbarrier |
| `tle.gpu.pipeline`（tile-style 流水）| TLE-Struct | 依赖 warp specialization 基础设施 |
| `tle.gpu.copy` 的 **tmacopy**（TMA 描述符）| TLE-Struct | TMA 引擎（Hopper 专有）|
| WGMMA 相关 | TLE-Struct | Hopper Tensor Core 专有指令 |
| **distributed**：`device_mesh` / `remote` / `distributed_barrier` / `distributed_dot` / `reshard` / `shard_id` | (Distributed) | 需 cluster/多卡远程内存硬件 |

> 这些在其他（NVIDIA Hopper）后端可用，但**不要**在 Sunrise 上使用。

---

## 4. 使用限制速查（Checklist）

在 Sunrise 上写 TLE kernel 时，务必遵守：

1. ✅ `tle.gpu.alloc` 一律传 **`nv_mma_shared_layout=False`**。
2. ✅ `tle.gpu.copy` 只用**普通指针**参数，不要用 `tensor_descriptor`（TMA）。
3. ✅ `local_ptr` 的索引张量个数 == buffer rank，shape 一致。
4. ✅ `extract_tile`/`insert_tile`：tile rank == 源 rank，源各维被 tile 整除，index 不越界。
5. ✅ `tle.cumsum` 返回的是 `(exclusive, total)` 元组，注意是 **exclusive** 语义。
6. ❌ 不要使用 pipe / warp_specialize / pipeline / TMA / WGMMA / distributed。
7. ⚠️ `tle.cumsum` fastpath 要求 `threadsPerWarp==32 && numWarps<=32`（Sunrise 默认满足）；
   否则自动退回串行 fallback（结果正确，仅较慢）。

---

## 5. 自动调块（AABS，可选）

`FLAGTREE_AABS` 是自动调整 autotune 配置中 BLOCK_SIZE 的开关（默认 **开启**）：

```bash
export FLAGTREE_AABS=1                 # 开启
export TRITON_PRINT_AUTOTUNING=1       # 查看调整日志（可选）
```

Sunrise 分支会保证调整后的 dot BLOCK 满足硬件下界（M≥8, N≥8, K≥16，对齐
`min_dot_size`）。仅在使用 `@triton.autotune` 的 kernel 上生效。

---

## 6. 常见错误与排查

| 现象 | 可能原因 | 处理 |
|---|---|---|
| `alloc` 后编译失败 / layout 报错 | `nv_mma_shared_layout=True` | 改为 `False` |
| `tle.gpu.copy` 编译/下沉失败 | 传了 `tensor_descriptor`（tmacopy） | 改用普通指针（normcopy）|
| `extract_tile` 报 `ValueError` | rank 不匹配 / 不整除 / index 越界 | 检查 tile_shape 与 index |
| dot 相关 kernel 报 `min_dot_size` assert | BLOCK 被调得过小 | 开 `FLAGTREE_AABS=1` 或手动保证 M≥8/N≥8/K≥16 |
| 使用 pipe/warp_specialize/TMA 报错 | 用了不支持的 Hopper 特性 | 见第 3 节，改用支持的等价路径 |

---

## 7. 验证 / 参考示例

Sunrise 后端的 TLE 端到端测试位于 `third_party/sunrise/python/test/`：

| 测试 | 覆盖特性 |
|---|---|
| `01-vector-add.py` | 基础 vector add |
| `02-tle-local_ptr.py` | `alloc` + `local_ptr` + `copy` |
| `03-tle-extract_insert_tile.py` | `extract_tile`/`insert_tile`（对齐 + 非对齐 + 动态 index）|
| `04-tle-hint-shared-memory.py` | `#@hint: shared_memory` |
| `05-tle-copy-normcopy.py` | `tle.gpu.copy`（双向 normcopy）|
| `06-tle-cumsum.py` | `tle.cumsum`（int/float，forward/reverse）|
| `07-tle-load.py` | `tle.load(is_async=True/False)` |
| `08-tle-dot-local-ptr.py` | local_ptr staging + `tl.dot`（GEMM）|
| `09-tle-negative-contract.py` | 契约错误（负例）|

运行：

```bash
cd third_party/sunrise/python/test
pytest 02-tle-local_ptr.py -v -s
```

---

## 8. TopK 性能优化案例（Radix-Select Kernel）

TLE 版 TopK 的原始教程实现见 `python/tutorials/tle/03-topk.py`；经过本节所述优化后的版本
落地在 `third_party/sunrise/python/perf/01-topk.py`（`topk_kernel_radix_triton` /
`triton_radix_topk`），在 Sunrise 设备上相对原始 radix kernel 有数倍加速。本节记录性能
数据与采用的优化手段，供后续 kernel 调优参考。

### 8.1 性能数据

测试脚本：

```bash
python3 ./third_party/sunrise/python/perf/01-topk.py
```

对比 **Triton-RadixSelect**（radix-select kernel，本节优化对象）与 **Triton-TopK**
（`tl.topk` + `tl.bitonic_merge` 流式方案）两条路径。

#### float16

| M | N | K | Triton-RadixSelect (ms) | Triton-TopK (ms) | Speedup (TopK / RadixSelect) |
| --- | --- | --- | --- | --- | --- |
| 64 | 128 | 8 | 0.004373 | 0.002843 | 0.65x |
| 64 | 1024 | 32 | 0.006846 | 0.009075 | 1.33x |
| 64 | 8192 | 128 | 0.032572 | 0.087926 | 2.70x |
| 128 | 32768 | 256 | 0.140760 | 0.575338 | 4.09x |

#### float32

| M | N | K | Triton-RadixSelect (ms) | Triton-TopK (ms) | Speedup (TopK / RadixSelect) |
| --- | --- | --- | --- | --- | --- |
| 64 | 128 | 8 | 0.006771 | 0.003249 | 0.48x |
| 64 | 1024 | 32 | 0.010113 | 0.013637 | 1.35x |
| 64 | 8192 | 128 | 0.042933 | 0.193781 | 4.51x |
| 128 | 32768 | 256 | 0.203795 | 1.024013 | 5.02x |

> Speedup 列 = Triton-TopK 耗时 / Triton-RadixSelect 耗时：>1 表示 RadixSelect 更快，
> <1 表示流式 TopK 更快。

### 8.2 采用的优化手段

本次优化针对 RadixSelect kernel
（`topk_kernel_radix_triton`）做了以下几项改动：

1. **增大 `RADIX_BITS`，减少访存轮数**
   fp16（16 bit）的轮数从 4 轮降到 2 轮，fp32（32 bit）从 8 轮降到 4 轮，总访存量随轮数近似线性下降。

2. **增加 shared memory 使用量，缓存 key 以减小 L2 上的重复访存事务**
   新增 Stage 1.5：把尽可能多的 tile 一次性以 order-preserving key读入 shared memory，只有超出缓存容量的尾部
   tile 才回退到 global memory 重新读取。

3. **拆成两个 2 的幂次方 buffer，绕开 `tle.gpu.alloc` 的 shape 限制**
   两块shared memory把可用缓存容量提升到~96 KB。

4. **先写 shared memory，再合并成一次连续写回 global memory**
   直接向 global memory 做非连续（uncoalesced）写会在 STCU 写接口上严重串行化。优化后（Stage 3）改为先用 shared memory
   原子计数器把候选值/索引分散写到 shared memory 的 staging buffer，再做**一次连续、完全 coalesced** 的 global memory 写出，正好写出K 个元素。

### 8.3 参考实现位置

- 优化前（原始教程，供对比）：`python/tutorials/tle/03-topk.py`
- 优化后（perf 版本）：
  `third_party/sunrise/python/perf/01-topk.py`

---

## 9. References（参考资料）

- **TLE 架构设计与三层分类（官方 Wiki）**：
  [FlagTree Wiki — TLE](https://github.com/flagos-ai/FlagTree/wiki/TLE)
- **Sunrise TLE 端到端测试**：`third_party/sunrise/python/test/01~09`
  （见第 7 节）。
- **Sunrise TLE 性能测试**：`third_party/sunrise/python/perf/01`
  （见第 8 节）。
