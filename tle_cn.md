# TLE 架构设计

## 1. 引言

Triton 是一种 Python DSL 形式的算子编程语言。它基于 Block 的编程理念，屏蔽了存储层级、Layout、流水线、同步等硬件细节，并通过编译器优化实现较高性能的算子。Triton 的这些优点吸引了大量开发者，形成了庞大的社区和生态。

但近年来 Triton 的进一步发展遇到一些困难：

- 在 DSA 和新 GPU 架构上的适配进展较慢。
- 相比 TileLang 等新兴语言，Triton 在细粒度控制存储层级和并行粒度方面缺少抽象，导致部分场景性能落后。

针对这些问题，我们提出 TLE（Triton Language Extentions），从三个层级扩展 Triton，满足不同层次用户对算子编程语言的需求。

## 2. 观察和解决方案

我们对业界主流 DSL（Triton、TileLang、cuTile）进行了分析，并总结了理想的语言设计。

### 2.1 Pythonic

三种语言都基于 Python 语法，说明开发者更倾向于使用类 Python 的语法编写算子，即使只用到 Python 的一个子集。

### 2.2 Tile 编程

三种语言都支持 Block 级编程，本质上当前的 Block 编程就是在 Global Memory 上做 Tiling。cuTile 更进一步支持多层次 Tiling，这为统一多种内存层级架构的编程语言提供了可能。

而 Triton 缺少 Tile 和 Slice 的概念，程序员只能在 Global Memory 做 Tiling，制约了语言进一步发展。

TileLang 与 Triton 类似，也未显式提供 Tiling 原语；此外，TileLang 除了 copy 和 gemm 外，缺少高层 Tensor Ops，在 GPU 上编程有不便；再加上没有自动 Vectorize，导致要充分利用 SIMD 需要额外扩展大量 SIMD Ops。

### 2.3 存储层级抽象

为解决“存储墙”问题，现代硬件通常具备多级存储。

- Triton/cuTile 只暴露两层：Global Memory 和 Local Tensor。
- TileLang 直接暴露硬件原生存储层级，缺少抽象。

问题：

- 暴露层级过少，意味着编译器需要承担 Tiling 与 Buffer Promotion。
- 直接暴露原生层级，会显著降低可移植性。

倾向方案：

- 程序员只做 Tiling，不显式指定存储层级。
- 编译器负责 Buffer Promotion。
- 允许程序员提供 hints，tile size 作为超参数。

这样既保持兼容性，也为进一步性能优化留出空间。

### 2.4 并行性抽象

- Triton/cuTile 只暴露 Block 级并行，Block 内并行由编译器控制。
- TileLang 允许程序员显式控制 Block 内并行（Parallel、Vectorize），表达能力更强，但可复用性下降。

### 2.5 分布式抽象

现有语言都未覆盖跨 Block、跨 Node 通信，限制了通算融合能力（已有外部探索如 Triton Distributed、TileScale）。

### 2.6 理想的语言设计

- Level 1：类 Numpy/PyTorch 算法级编程。用户无需关心实现细节，由编译器负责硬件映射和通信。
- Level 2：类 cuTile 的 Tile 级编程 + 分布式描述。用户显式做 Tiling 和 Sharding，存储层级、并行与通信由编译器负责，并可提供硬件/场景 hints。
- Level 3：硬件特定扩展（存储层级、Thread Binding、Vectorize 等）。该层限制在特定 region，定义好与 Level 2 的交互，编译器仅做基础优化。

细节建议：

- 引入 Tile 语义，避免程序员手写地址计算。
- 不限制 Tensor Shape 必须为 2 的幂。

开放问题：还有哪些好的设计？

## 3. 架构设计

### 3.1 架构概览

TLE 位于 AI 生态中间层：

- 上层通过图编译器与算子库承接 AI 框架。
- 下层对接各类硬件 Runtime。

> 暂时无法在飞书文档外展示此内容

TLE 分为三层：

- **TLE-Lite**：对 Triton 的轻量扩展。特性兼容多后端，对原 Triton kernels 只需少量修改即可获得较大性能收益。面向算法工程师和快速优化场景。
- **TLE-Struct**：按硬件架构聚类抽象（如 GPGPU、DSA）提供扩展，满足更深入优化需求。需要一定硬件理解。
- **TLE-Raw**：提供最直接硬件控制，可使用厂商原生编程语言追求极致性能。面向性能优化专家。

Lowering 路径：

- TLE-Lite / TLE-Struct：通过 FLIR 最终 Lowering 到 LLVM IR。
- TLE-Raw：通过对应语言编译管线（如厂商私有编译器）Lowering 到 LLVM IR。
- 最终统一 Link 成完整 kernel，供 Runtime 加载执行。

### 3.2 TLE-Lite

- 设计哲学：一次编写，到处运行。
- 核心理念：通过高层语义 hint（而非强制约束）引导编译器做启发式优化，强调向后兼容，在不破坏 Triton 编程范式前提下，以最小侵入实现跨平台性能提升。

#### 3.2.1 内存管理

##### 3.2.1.1 `tle.load`

`tl.load` 扩展，支持异步 hint：

```python
x = tle.load(..., is_async=True)
```

#### 3.2.2 Tensor 切片

##### 3.2.2.1 `tle.extract_tile`

使用子 tile 的 shape 将输入 tensor 切分为子 tile 网格，并提取指定坐标子 tile。

- GPU：支持 register 与 shared_memory 提取。

```python
# x is [4, 4]
# z is [2, 2]
# 将 x 切分为 shape=[2, 2] 的子 tile 网格，取 [0, 0] 子 tile
z = x.extract_tile(index=[0, 0], shape=[2, 2])
```

##### 3.2.2.2 `tle.insert_tile`

使用子 tile 的 shape 将输入 tensor 切分为子 tile 网格，并更新指定坐标子 tile。

- GPU：支持 register 与 shared_memory 更新。

```python
# x is [4, 4], y is [2, 2], z is [4, 4]
# 将 x 切分为 shape=[2, 2] 子 tile，使用 y 更新 [0, 0] 子 tile，返回完整 [4, 4]
z = x.insert_tile(y, index=[0, 0])
```

#### 3.2.3 流水线

##### 3.2.3.1 `tle.pipeline_group`

Hint 类扩展。

自动分 Stage：

```python
for yoff in tl.range(0, ynumel, YBLOCK, num_stages=2):
    Q = tl.load(...)
    K = tl.load(...)
    KT = tl.trans(K)
    V = tl.dot(Q, KT)
```

手动分 Stage：

```python
for yoff in tle.range(
    0,
    ynumel,
    YBLOCK,
    num_stages=2,
    pipe_stages=[0, 0, 1] if LOAD_TRANS else [0, 1, 1],
    pipe_orders=[0, 1, 2],
    executors=[0, 0, 0] if ONE_CORE else [0, 0, range(1, 31)],
):
    # Warp-spec 或异构单元
    with tle.pipeline_group(0):
        Q = tl.load(...)
        K = tl.load(...)
    with tle.pipeline_group(1):
        KT = tl.trans(K)
    with tle.pipeline_group(2):
        V = tl.dot(Q, KT)
```

#### 3.2.4 分布式

Triton 分布式 API 包含四个核心部分：设备网格定义、分片规格描述、重分片（集合通信）、远程访问（点对点通信）。

##### 3.2.4.1 设备网格

`tle.device_mesh` 定义物理设备拓扑，是所有分布式操作的上下文基础。

```python
class device_mesh:
    def __init__(self, topology: dict):
        """
        初始化 DeviceMesh。

        Args:
            topology (dict): 描述硬件层级的字典。
                             键为层级名称，值为整数(1D)或元组列表(多维)。
        """
        self._physical_ids = ... # 内部存储：扁平化的物理 ID 列表 (0..N-1)
        self._shape = ...        # 当前视图形状，例如 (2, 2, 4, 2, 2, 4)
        self._dim_names = ...    # 当前维度名称

    @property
    def shape(self):
        """返回当前 Mesh 的逻辑形状。"""
        return self._shape

    @property
    def ndim(self):
        """返回维度数量。"""
        return len(self._shape)

    def flatten(self):
        """将 Mesh 展平为 1D，常用于 Ring 通信。"""
        return self.reshape(prod(self._shape))

    def __getitem__(self, key):
        """
        支持切片，返回 Sub-Mesh。
        支持标准 slice 和整数索引。
        """
        return sub_mesh

    def __repr__(self):
        return f"DeviceMesh(shape={self._shape}, names={self._dim_names})"


# 定义复杂硬件层级
topology = {
    # 跨节点层级 (2x2 = 4 nodes)
    "node": [("node_x", 2), ("node_y", 2)],
    # 节点内 GPU (4 devices)
    "device": 4,
    # GPU 内 Cluster (2x2)
    "block_cluster": [("cluster_x", 2), ("cluster_y", 2)],
    # Cluster 内 Block (4 blocks)
    "block": 4,
}

# mesh.shape -> (2, 2, 4, 2, 2, 4)
# 总大小 = 256
mesh = tle.device_mesh(topology=topology)
```

##### 3.2.4.2 分片规格

`tle.sharding` 用于声明 Tensor 在 Device Mesh 上的分布状态：

- `splits`：每个 Tensor 维度如何在 Mesh 上切分。
- `partials`：Tensor 是否为 Partial Sum。
- 未指明 Mesh 轴默认 Broadcast。

符号：

- `tle.S(axis)`：Split。
- `tle.B`：Broadcast/Replicate。
- `tle.P(axis)`：Partial，需要在指定 axis 上 Reduce。

```python
def sharding(tensor, splits, partials):
    """
    Annotation：仅标记状态，不直接产生命令，指导编译器后续优化/检查。
    """
    return tensor

# 在 cluster 上 split axis0、device 上 split axis1，在 block 维度 partial
x_shard = tle.sharding(
    mesh,
    split=[["cluster_x", "cluster_y"], "device"],
    partial=["block"],
)

# 定义 sharded_tensor
x = tle.make_sharded_tensor(x_ptr, sharding=x_shard, shape=[4, 4])
```

##### 3.2.4.3 同步

复杂分布式算子中（如 Ring-AllReduce、行列独立流水线）通常只需对子网格同步，而不是整个 Cluster。全局同步会引入额外等待。

```python
def distributed_barrier(mesh):
    """
    若传入 sub_mesh，仅同步该子网格内设备。
    子网格外设备应视为 No-Op（或由编译器保证控制流不进入）。
    """
    pass
```

##### 3.2.4.4 远程访问

`tle.remote` 获取其他设备上 Tensor 分片句柄，对应点对点通信或直接内存访问（RDMA/NVLink Load）。

```python
def remote(tensor, shard_id, scope):
    """
    获取指向特定设备分片的 Remote Tensor 句柄。

    :param tensor: 逻辑分布式 Tensor（已被 tle.sharding 标记）
    :param shard_id: tuple，目标设备在 Device Mesh 中的坐标
    :return: RemoteTensor，可执行 load/store 等操作
    """
```

##### 3.2.4.5 重分片

`tle.reshard` 是集合通信入口。编译器比较源 spec 与目标 spec，自动插入通信原语。

```python
def reshard(tensor, spec):
    """
    Action：将 Tensor 转换到新的分布状态。

    常见转换:
    1. [ ] -> [S]: Scatter
    2. [S] -> [ ]: Gather
    3. [P] -> [ ]: Reduce
    4. [B] -> [S]: Local Slice (无通信)
    5. [S] -> [B]: All-Gather
    6. [P] -> [B]: All-Reduce
    7. [B] -> [P]: 错误
    """
```

##### 3.2.4.6 分布式 GEMM

在 NVIDIA Hopper（H100）及更新架构中，引入了 Thread Block Cluster，可通过 DSMEM 在 Block 间进行高速低延迟交换。

`tle.distributed_dot` 旨在利用该能力，实现跨 Block 矩阵乘法，并屏蔽 DSMEM 屏障和搬运细节。

```python
def distributed_dot(a, b, c=None):
    """
    在当前 Thread Block Cluster 范围内执行分布式矩阵乘法。

    行为取决于输入 a/b 在 Cluster Mesh 上的 Sharding Spec。

    Args:
        a (Tensor): 左操作数，需带 Cluster 级 Sharding。
        b (Tensor): 右操作数，需带 Cluster 级 Sharding。
        c (Tensor, optional): 累加器。

    Returns:
        Tensor: 计算结果，分布状态由输入推导。
    """
```

开放问题：还需要哪些分布式原语？

#### 3.2.5 API 说明与实战示例

##### 3.2.5.1 `tle.load`

- Signature: `tle.load(ptr, mask=None, other=None, is_async=False)`
- 用途：保持 `tl.load` 语义不变，同时增加异步调度 hint。
- 实践建议：
  - 对后续会被多次复用的全局内存数据，优先尝试 `is_async=True`。
  - 在边界 tile 上显式写 `mask/other`，避免读取到未定义值。

示例：带边界保护的异步加载

```python
offs = base + tl.arange(0, BLOCK)
mask = offs < n_elements
x = tle.load(x_ptr + offs, mask=mask, other=0.0, is_async=True)
```

示例：异步加载与计算重叠

```python
for k in tl.range(0, K, BK, num_stages=2):
    a = tle.load(a_ptr + k * stride_a, is_async=True)
    b = tle.load(b_ptr + k * stride_b, is_async=True)
    acc = tl.dot(a, b, acc)
```

##### 3.2.5.2 `tle.extract_tile` 与 `tle.insert_tile`

- `extract_tile`：从大 tile 中提取子 tile 视图。
- `insert_tile`：把处理后的子 tile 写回大 tile。
- 典型场景：对子区域做激活、量化/反量化、归一化等局部变换，避免手写地址计算。

示例：在寄存器中对子 tile 做后处理

```python
# x: [4, 4]
sub = x.extract_tile(index=[1, 0], shape=[2, 2])  # rows [2:4], cols [0:2]
sub = tl.maximum(sub, 0.0)  # 对子 tile 做 ReLU
x = x.insert_tile(sub, index=[1, 0])
```

##### 3.2.5.3 `tle.pipeline_group`

- 通过 `tle.pipeline_group(stage_id)` 把操作显式标记到不同 stage。
- 适用于需要“可控 stage 划分”的场景（而不是完全依赖启发式自动分组）。

示例：load-transform-matmul 手动分 stage

```python
for k in tle.range(0, K, BK, num_stages=2, pipe_stages=[0, 0, 1], pipe_orders=[0, 1, 2]):
    with tle.pipeline_group(0):
        a = tl.load(a_ptr + k * stride_a)
        b = tl.load(b_ptr + k * stride_b)
    with tle.pipeline_group(1):
        bt = tl.trans(b)
    with tle.pipeline_group(2):
        acc = tl.dot(a, bt, acc)
```

##### 3.2.5.4 `tle.device_mesh` + `tle.sharding` + `tle.reshard`

- 推荐流程：
  1. 用 `tle.device_mesh` 定义拓扑。
  2. 用 `tle.sharding` 标注分布状态。
  3. 用 `tle.reshard` 做分布状态转换。
  4. 计算 kernel 保持在逻辑 tensor 视角。

示例：先按 device 切分，再 all-gather 到完整视图

```python
mesh = tle.device_mesh({"node": 2, "device": 4})
x_spec = tle.sharding(mesh, split=["device"], partial=[])
x = tle.make_sharded_tensor(x_ptr, sharding=x_spec, shape=[M, K])

# [S] -> [B]（device 轴上的 all-gather）
x_full = tle.reshard(x, spec=tle.sharding(mesh, split=[], partial=[]))
```

##### 3.2.5.5 `tle.shard_id`

- Signature: `tle.shard_id(mesh, axis)`
- 含义：返回当前 program 在指定 mesh 轴上的坐标。
- `axis` 可以是轴名称（如 `"node"`、`"device"`、`"cluster_x"`）或轴序号。
- 典型用途：构造 ring 交换、分阶段 all-reduce、cluster 协同计算中的对端 shard ID。

示例：查询当前 program 在 node/device 轴上的坐标

```python
mesh = tle.device_mesh({"node": 2, "device": 4})
node_rank = tle.shard_id(mesh, "node")      # 0..1
device_rank = tle.shard_id(mesh, "device")  # 0..3
```

##### 3.2.5.6 `tle.remote` + `tle.distributed_barrier`

- `tle.remote`：显式读取/写入远端分片。
- `tle.distributed_barrier`：仅同步传入 mesh/sub-mesh 对应的设备集合。

示例：读取相邻 shard（ring 风格交换）

```python
node_rank = tle.shard_id(mesh, "node")
device_rank = tle.shard_id(mesh, "device")
next_device = (device_rank + 1) % mesh.shape[1]
remote_x = tle.remote(x, shard_id=(node_rank, next_device), scope=mesh)
tle.distributed_barrier(mesh)
neighbor_vals = tl.load(remote_x)
```

### 3.3 TLE-Struct

- 设计哲学：架构感知，精细调优。
- 核心理念：按硬件拓扑将后端聚类为 GPGPU、DSA 等，暴露通用层次化并行与存储结构。允许开发者显式定义计算/数据的结构化映射（如 Warp Group 控制、流水线编排），在抽象层面解耦算法逻辑与硬件物理实现。

#### 3.3.1 GPU

##### 3.3.1.1 内存管理

###### 3.3.1.1.1 `tle.gpu.memory_space`

指定 Tensor 的 `memory_space`：

```python
x = ...
x = tle.gpu.memory_space(x, "shared_memory")
```

###### 3.3.1.1.2 `tle.gpu.alloc`

分配内存：

```python
a_smem = tle.gpu.alloc(
    [XBLOCK, YBLOCK],
    dtype=tl.float32,
    layout=None,
    scope=tle.gpu.storage_kind.smem,
)
```

###### 3.3.1.1.3 `tle.gpu.local_ptr`

获取内存指针：

```python
# 取 a_smem[0, :] 的 pointers: [(0, 0), (0, 1), ..., (0, YBLOCK-1)]
a_smem_ptrs = tle.gpu.local_ptr(
    a_smem,
    indices=(tl.broadcast(0, [YBLOCK]), tl.arange(0, YBLOCK)),
)
```

- Signature: `tle.local_ptr(buffer, indices=None) -> tl.tensor | tl.ptr`
- Purpose: 在 shared memory buffer 上构建任意形状 pointer view，用于 `tl.load/tl.store`。
- Parameters:
  - `buffer`: 由 `tle.alloc` 返回的 buffered_tensor（SMEM/TMEM）。
  - `indices`: 可选整数 tensor 元组，长度必须等于 `rank(buffer)`，且每个 tensor 形状相同；若省略/传 `None`，由后端按 full indices 语义处理。
- Semantics:
  - 当显式传入 `indices` 时，输出 pointer tensor 形状等于 indices 的公共形状。
  - 对输出形状中每个逻辑索引 `(i0, i1, ...)`，指针对应 `buffer[indices0(i0,...), indices1(i0,...), ...]`。
  - 当 `indices=None` 时，返回覆盖整个 `buffer` 的 full-view 指针（rank>0 返回 `shape(buffer)` 的 pointer tensor，rank=0 返回标量 pointer）。
  - 返回指针位于 shared memory 地址空间（LLVM addrspace=3）。索引需为整数类型（i32/i64 等，lowering 时规约到 i32）。
  - 线性化为 row-major（最后一维最快）。共享内存布局/encoding 跟随 buffer memdesc。

Example 1：1D slice

```python
smem = tle.alloc([BLOCK], dtype=tl.float32, scope=tle.smem)
# Slice [offset, offset + SLICE)
idx = offset + tl.arange(0, SLICE)
slice_ptr = tle.local_ptr(smem, (idx,))
vals = tl.load(slice_ptr)
```

Example 2：K 维切片（矩阵）

```python
smem_a = tle.alloc([BM, BK], dtype=tl.float16, scope=tle.smem)
# Slice (BM, KW), KW 是 K 维子切片
rows = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, KW))
cols = tl.broadcast_to(tl.arange(0, KW)[None, :] + k_start, (BM, KW))
a_slice = tle.local_ptr(smem_a, (rows, cols))
a_vals = tl.load(a_slice)
```

Example 3：任意 gather view

```python
smem = tle.alloc([H, W], dtype=tl.float32, scope=tle.smem)
# 每行取偏移列
rows = tl.broadcast_to(tl.arange(0, H)[:, None], (H, SLICE))
cols = tl.broadcast_to(1 + tl.arange(0, SLICE)[None, :], (H, SLICE))
gather_ptr = tle.local_ptr(smem, (rows, cols))
out = tl.load(gather_ptr)
```

###### 3.3.1.1.4 `tle.gpu.copy`

内存拷贝：

```python
tle.gpu.copy(a_ptrs + ystride_a * yoffs[None, :], a_smem, [XBLOCK, YBLOCK])
```

###### 3.3.1.1.5 `tl.load/tl.store/tl.atomic*` for `tle.local_ptr`

`tle.local_ptr` 返回的 Shared Memory 指针可直接用于：

- `tl.load`
- `tl.store`
- `tl.atomic_add/and/cas/max/min/or/xchg/xor`

#### 3.3.2 DSA

##### 3.3.2.1 内存管理

###### 3.3.2.1.1 `tle.dsa.alloc`

分配内存（Ascend）：

```python
a_ub = tle.dsa.alloc(
    [XBLOCK, YBLOCK],
    dtype=tl.float32,
    layout=tle.dsa.ascend.NZ,
    scope=tle.dsa.ascend.UB,
)
```

###### 3.3.2.1.2 `tle.dsa.copy`

内存拷贝：

```python
tle.dsa.copy(a_ptrs + ystride_a * yoffs[None, :], a_smem, [XBLOCK, YBLOCK])
```

###### 3.3.2.1.3 `tle.dsa.local_load`

内存加载：

```python
aval = tle.dsa.local_load(a_smem)
```

#### 3.3.3 Struct API 组合示例

##### 3.3.3.1 Shared Memory 预取流水（`alloc` + `copy` + `local_ptr`）

适用于同一 tile 会被多次复用的场景。

```python
# 1) 分配 SMEM tile
a_smem = tle.gpu.alloc([BM, BK], dtype=tl.float16, scope=tle.gpu.storage_kind.smem)

# 2) GMEM -> SMEM
tle.gpu.copy(a_ptrs, a_smem, [BM, BK])

# 3) 构建本地指针视图并加载
rows = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, BK))
cols = tl.broadcast_to(tl.arange(0, BK)[None, :], (BM, BK))
a_ptr_local = tle.gpu.local_ptr(a_smem, (rows, cols))
a_tile = tl.load(a_ptr_local)
```

##### 3.3.3.2 基于 `local_ptr` 的 Shared Memory 原子操作

适用于直方图、分桶统计、radix-select 计数等模式。

```python
bins = 256
counts = tle.gpu.alloc([bins], dtype=tl.int32, scope=tle.gpu.storage_kind.smem)
idx = tl.arange(0, BLOCK) % bins
count_ptr = tle.gpu.local_ptr(counts, (idx,))
tl.atomic_add(count_ptr, 1)
```

##### 3.3.3.3 DSA 本地缓冲流程（`dsa.alloc` + `dsa.copy` + `dsa.local_load`）

适用于暴露专用本地缓冲空间的 DSA 后端。

```python
a_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, layout=tle.dsa.ascend.NZ, scope=tle.dsa.ascend.UB)
tle.dsa.copy(a_ptrs, a_ub, [BM, BK])
a_val = tle.dsa.local_load(a_ub)
```

### 3.4 TLE-Raw

- 设计哲学：原生透传，极致掌控。
- 核心理念：打破 DSL 抽象边界，支持内联厂商原生代码。通过厂商私有编译管线直接生成目标指令，绕过通用编译器中间层开销，为专家用户提供对指令调度、寄存器分配与底层同步原语的强控制能力。

> 暂时无法在飞书文档外展示此内容

开放问题：Raw 接入语言是否仅限于 Python DSL？

#### 3.4.1 语言扩展

##### 3.4.1.1 MLIR

```python
from typing import Annotated
from mlir import ir
from mlir.dialects import arith, nvvm, tensor
import triton.language as tl
from triton.experimental.flagtree.edsl import dialect
import triton.experimental.flagtree.language as fl

# 1. 方言声明
@tle.raw.language(name="mlir")
# 2. 硬件约束
@tle.hardware_constraint(threads_dim=1, sync_scope="block")
# 3. 函数实现
def vector_add_tile(
    x: Annotated[ir.RankedTensorType, "tensor<1024xf32>"],
    y: Annotated[ir.RankedTensorType, "tensor<1024xf32>"],
    output: Annotated[ir.RankedTensorType, "tensor<1024xf32>"]
):
    tidx = nvvm.ThreadIdXOp(ir.IntegerType.get_signless(32)).res
    bidx = nvvm.BlockIdXOp(ir.IntegerType.get_signless(32)).res
    bdimx = nvvm.BlockDimXOp(ir.IntegerType.get_signless(32)).res
    idx = arith.addi(arith.muli(bidx, bdimx), tidx)
    idx = arith.index_cast(ir.IndexType.get(), idx)
    xval = tensor.extract(x, [idx])
    yval = tensor.extract(y, [idx])
    result = arith.addf(xval, yval)
    tensor.insert(result, output, [idx])

@tle.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = tl.zeros_like(x)

    # 4. 函数调用
    tle.call(
        vector_add_tile,
        args=[x, y, output],
        hardware={
            "threads": (BLOCK_SIZE,),
        },
        layout={
            x: {"space": "shared", "order": [0]},
            y: {"space": "shared", "order": [0]},
            output: {"space": "shared", "order": [0]},
        }
    )
    tl.store(output_ptr + offsets, output, mask=mask)
```

## 4. 示例和评估

### 4.1 SparseMLA

当前已在 RTX 5060Ti 与 H800 上针对 DSA 中 SparseMLA 算子做优化和测试。

- TileLang 版本：`v0.1.7`
- 示例代码：<https://github.com/flagos-ai/FlagTree/blob/triton_v3.5.x/python/tutorials/tle/01-sparse-mla.py>

核心 kernel（节选）：

```python
@triton.jit
def triton_sparse_mla_fwd(
    q,
    kv,
    indices,
    sm_scale: tl.constexpr,
    output,
    lse,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kvb, stride_kvg, stride_kvn, stride_kvd,
    stride_tb, stride_tg, stride_tm, stride_tt,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lb, stride_lh, stride_lm,
    B: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    TD: tl.constexpr,
    DP: tl.constexpr,
    TDP: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    VG: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
    is_causal: tl.constexpr
):
    i_b, i_sq, i_gbh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_g, i_bh = i_gbh // G, i_gbh % G
    q_base = q + i_b * stride_qb + i_sq * stride_qm + i_gbh * (BH * stride_qh)
    tq_base = q_base + D * stride_qd
    kv_base = kv + i_b * stride_kvb + i_g * stride_kvg
    tkv_base = kv_base + D * stride_kvd
    t_base = indices + i_b * stride_tb + i_sq * stride_tm + i_g * stride_tg
    o_base = output + i_b * stride_ob + i_sq * stride_om + i_gbh * (BH * stride_oh)
    l_base = lse + i_b * stride_lb + i_sq * stride_lm + i_gbh * (BH * stride_lh)

    offs_h = tl.arange(0, BH)
    offs_d = tl.arange(0, DP)
    offs_td = tl.arange(0, TDP)
    offs_od = tl.arange(0, DP)
    offs_t = tl.arange(0, BK)
    mask_h = i_bh * BH + offs_h < G
    mask_d = offs_d < D
    mask_td = offs_td < TD
    mask_od = mask_d

    q_ptr = q_base + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_msk = mask_h[:, None] & mask_d[None, :]
    q_blk = tl.load(q_ptr, q_msk, other=0.0)

    tq_ptr = tq_base + offs_h[:, None] * stride_qh + offs_td[None, :] * stride_qd
    tq_msk = mask_h[:, None] & mask_td[None, :]
    tq_blk = tl.load(tq_ptr, tq_msk, other=0.0)

    max_log = tl.full([BH], float('-inf'), dtype=tl.bfloat16)
    sum_exp = tl.full([BH], 1.0, dtype=tl.float32)
    acc = tl.zeros([BH, DP], dtype=tl.float32)

    log_scale: tl.constexpr = sm_scale * 1.44269504
    max_col = i_sq if is_causal else SQ - 1
    NK = tl.cdiv(K, BK)

    for ck in tl.range(NK, num_stages=0):
        if ck * BK <= max_col:
            t_ptr = (BK * ck + offs_t) * stride_tt
            t_msk = t_ptr < K
            t_ptr += t_base
            kv_ids = tl.load(t_ptr, t_msk, other=-1)
            mask_ids = (kv_ids <= max_col) & (kv_ids >= 0)

            kv_ptr = kv_base + offs_d[:, None] * stride_kvd + kv_ids[None, :] * stride_kvn
            kv_msk = mask_d[:, None] & mask_ids[None, :]
            kv_blk = tle.load(kv_ptr, kv_msk, other=0.0, is_async=True)

            tkv_ptr = tkv_base + offs_td[:, None] * stride_kvd + kv_ids[None, :] * stride_kvn
            tkv_msk = mask_td[:, None] & mask_ids[None, :]
            tkv_blk = tl.load(tkv_ptr, tkv_msk, other=0.0)

            qk = tl.dot(tq_blk, tkv_blk, out_dtype=tl.float32)
            qk = tl.dot(q_blk, kv_blk, qk, out_dtype=tl.float32) * log_scale
            qk = tl.where(mask_ids[None, :], qk, float('-inf'))

            new_max = tl.maximum(max_log, tl.max(qk, axis=1))
            exp_qk = tl.math.exp2(qk - new_max[:, None])
            sum_qk = tl.sum(exp_qk, axis=1)
            alpha = tl.math.exp2(max_log - new_max)
            sum_exp = sum_exp * alpha + sum_qk
            acc = acc * alpha[:, None]
            acc = tl.dot(exp_qk.to(tl.bfloat16), kv_blk.trans(), acc, out_dtype=tl.float32)

            max_log = new_max.to(tl.bfloat16)

    out_vals = acc / sum_exp[:, None]
    o_ptr = o_base + offs_h[:, None] * stride_oh + offs_od[None, :] * stride_od
    o_msk = mask_h[:, None] & mask_od[None, :]
    tl.store(o_ptr, out_vals.to(q_blk.dtype), o_msk)

    fin_log = max_log + tl.math.log2(sum_exp.to(tl.float32))
    l_ptr = l_base + offs_h * stride_lh
    l_msk = mask_h
    tl.store(l_ptr, fin_log.to(q_blk.dtype), l_msk)
```

性能对比（TFLOPS）：

| 设备 | 理论算力 | Triton | TileLang | TLE | TLE over Triton |
| --- | ---: | ---: | ---: | ---: | ---: |
| H800 | 800 | 165.5 | **355.0** | 210.6 | 1.27x |
| H20 | - | 81.0 | **110.2** | 93.2 | 1.15x |
| RTX 5060Ti | - | 30.7 | Not supported | **32.8** | 1.07x |

### 4.2 MoeAlignBlockSize

利用 `tle-struct` 中 shared memory 相关扩展，可对标实现 `vllm/sglang` 的 `moe_align_block_size`，实现性能提升。

- 示例代码：<https://github.com/flagos-ai/FlagTree/blob/triton_v3.5.x/python/tutorials/tle/02-moe_align_block_size.py>

#### 4.2.1 RTX 5060 Ti

| num_tokens | triton | triton_atomic | **tle_atomic_fused [ours]** | **tle_cluster_fused [ours]** | sglang_cuda | **加速比（sglang_cuda / min(tle_atomic_fused, tle_cluster_fused)）** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.0348 | 0.0302 | 0.0323 | **0.0097** | 0.0138 | 1.42x |
| 512 | 0.0369 | 0.0301 | 0.0240 | **0.0117** | 0.0138 | 1.18x |
| 1024 | 0.0369 | 0.0313 | 0.0179 | **0.0117** | 0.0139 | 1.19x |
| 2048 | 0.0368 | 0.0313 | 0.0158 | **0.0131** | 0.0138 | 1.05x |
| 4096 | 0.0369 | 0.0301 | **0.0138** | 0.0143 | 0.0148 | 1.07x |
| 8192 | 0.0369 | 0.0313 | **0.0138** | 0.0164 | 0.0179 | 1.30x |
| 16384 | 0.0369 | 0.0301 | **0.0158** | 0.0205 | 0.0240 | 1.52x |
| 32768 | 0.0389 | 0.0322 | **0.0179** | 0.0301 | 0.0312 | 1.74x |
| 65536 | 0.0430 | 0.0374 | **0.0225** | 0.0486 | 0.0507 | 2.25x |
| 163840 | 0.0609 | 0.0512 | **0.0384** | 0.1036 | 0.1001 | 2.61x |

#### 4.2.2 H800

| num_tokens | triton | triton_atomic | **tle_atomic_fused [ours]** | **tle_cluster_fused [ours]** | sglang_cuda | **加速比（sglang_cuda / min(tle_atomic_fused, tle_cluster_fused)）** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.0260 | 0.0408 | 0.0445 | **0.0133** | 0.0160 | 1.20x |
| 512 | 0.0262 | 0.0399 | 0.0315 | **0.0140** | 0.0162 | 1.16x |
| 1024 | 0.0274 | 0.0401 | 0.0239 | **0.0158** | 0.0163 | 1.03x |
| 2048 | 0.0509 | 0.0422 | 0.0226 | **0.0169** | 0.0173 | 1.02x |
| 4096 | 0.0265 | 0.0412 | 0.0200 | **0.0177** | 0.0187 | 1.06x |
| 8192 | 0.0476 | 0.0416 | **0.0192** | 0.0211 | 0.0230 | 1.20x |
| 16384 | 0.0548 | 0.0441 | **0.0219** | 0.0256 | 0.0286 | 1.31x |
| 32768 | 0.0443 | 0.0441 | **0.0221** | 0.0358 | 0.0401 | 1.81x |
| 65536 | 0.0361 | 0.0481 | **0.0273** | 0.0561 | 0.0645 | 2.36x |
| 163840 | 0.0509 | 0.0626 | **0.0451** | 0.1177 | 0.1323 | 2.93x |

#### 4.2.3 H800 真实数据（`build/gems/moe_topk_ids.pt`）

- 运行配置：`num_tokens=163840`、`num_experts=512`、`block_size=16`、`source=real`。

| num_tokens | num_experts | block_size | triton | triton_atomic | **tle_atomic_fused [ours]** | **tle_cluster_fused [ours]** | sglang_cuda | **加速比（sglang_cuda / min(tle_atomic_fused, tle_cluster_fused)）** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 163840 | 512 | 16 | 0.0471 | 0.0535 | **0.0387** | 0.0750 | 0.1467 | 3.79x |

#### 4.2.4 RTX 5060 Ti 真实数据（`build/gems/moe_topk_ids.pt`，本机实测）

- 运行配置：`num_tokens=163840`、`num_experts=512`、`block_size=16`、`source=real`。
- 运行命令：
  `conda run -n flagtree python python/tutorials/tle/02-moe_align_block_size.py --skip_correctness --real_data build/gems/moe_topk_ids.pt --num_experts 512 --block_size 16`

| num_tokens | num_experts | block_size | triton | triton_atomic | **tle_atomic_fused [ours]** | **tle_cluster_fused [ours]** | sglang_cuda | **加速比（sglang_cuda / min(tle_atomic_fused, tle_cluster_fused)）** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 163840 | 512 | 16 | 0.0507 | 0.0395 | **0.0261** | 0.0532 | 0.1060 | 4.06x |

### 4.3 TopK

利用 `tle-struct` 中 shared memory 相关扩展，可实现 radix-select-based TopK，在大 N、小 K 的 MoE 场景取得性能提升。

- 示例代码：<https://github.com/flagos-ai/FlagTree/blob/triton_v3.5.x/python/tutorials/tle/03-topk.py>

#### 4.3.1 RTX 5060 Ti (`tle-topk-radix-vs-torch`)

| M | N | K | Triton-RadixSelect | Torch-TopK | **加速比（Torch / Triton-RadixSelect）** |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 128 | 8 | **0.008192** | 0.010240 | 1.25x |
| 64 | 1024 | 32 | **0.008192** | 0.020480 | 2.50x |
| 64 | 8192 | 128 | **0.026624** | 0.059392 | 2.23x |
| 128 | 32768 | 256 | **0.124928** | 0.192512 | 1.54x |

#### 4.3.2 H800 (`tle-topk-radix-vs-torch`)

| M | N | K | Triton-RadixSelect | Torch-TopK | **加速比（Torch / Triton-RadixSelect）** |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 128 | 8 | **0.008384** | 0.017536 | 2.09x |
| 64 | 1024 | 32 | **0.010688** | 0.024304 | 2.27x |
| 64 | 8192 | 128 | **0.029952** | 0.057184 | 1.91x |
| 128 | 32768 | 256 | **0.092256** | 0.117856 | 1.28x |

### 4.4 TopK Selector

TopK Selector 性能使用 `python/tutorials/tle/deepseek_v32/01-topk_selector.py`（`plot_name=tle-radix-topk-selector`）评估。

#### 4.4.1 RTX 5060 Ti（本机实测）

- 运行参数：本机（GeForce RTX 5060 Ti）执行 `--skip_correctness --warmup 10 --rep 80`。

| batch | seq_len | topk | Torch-TopK | Triton-Radix | TileLang | TLE-Radix | **加速比（Torch-TopK / TLE-Radix）** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 4096 | 128 | 0.038912 | 0.039456 | 0.020480 | **0.015808** | 2.46x |
| 64 | 8192 | 256 | 0.088624 | 0.053248 | 0.028672 | **0.023936** | 3.70x |
| 64 | 32768 | 1024 | 0.158272 | 0.131616 | 0.073728 | **0.062912** | 2.52x |
| 64 | 32768 | 2048 | 0.163264 | 0.133120 | 0.075776 | **0.065536** | 2.49x |

#### 4.4.2 H800（`tle-radix-topk-selector`）

| batch | seq_len | topk | Torch-TopK | Triton-Radix | TileLang | TLE-Radix | **加速比（Torch-TopK / TLE-Radix）** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 4096 | 128 | 0.045728 | 0.054256 | **0.017200** | 0.017472 | 2.62x |
| 64 | 8192 | 256 | 0.097344 | 0.072512 | 0.020960 | **0.020928** | 4.65x |
| 64 | 32768 | 1024 | 0.125008 | 0.176768 | 0.043088 | **0.041856** | 2.99x |
| 64 | 32768 | 2048 | 0.125072 | 0.179264 | 0.044256 | **0.041984** | 2.98x |
