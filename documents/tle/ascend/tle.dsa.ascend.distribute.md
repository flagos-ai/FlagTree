# tle.dsa.ascend.distribute 接口文档

## 1. 概述

本文档介绍 TLE DSA Ascend 后端的分布式通信相关 OP，分为 Host 侧与 Device 侧两部分：

- **Host 侧**：负责通信域的初始化与对称内存的管理。
- **Device 侧**：提供 kernel 内的远端访问、同步与索引排布原语。

> 当前 Ascend 版仅覆盖“卡间”（NPU 间）一层，同步粒度固定为“全部 rank”，无法做到核间（单卡内跨核）通信与子组同步。与社区 GPU 版的差异详见第 4 节。

---

## 2. Host 侧 OP

### 2.1 init_communicator

**背景**：

在进行分布式通信前，需要先进行通信的网络构建。`init_communicator` 就是将你需要通信的卡和进程进行绑定，一个进程一张卡，并初始化通信域，包括通信协议、通信域最大大小、通信端口号。绑定和初始化完成就可以得到本卡的 rank 号，在通信域上创建需要通信的“对称内存”。

**接口说明**：

```python
def init_communicator(ip_port=None, ash_size=None, engine_type=None):
```

**入参说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `ip_port` | `str` | 通信的端口号。默认是本机的 8666：`tcp://127.0.0.1:8666` |
| `ash_size` | `int` | 创建当前通信域的大小，每张卡上的每个进程都会给本卡创建 `ash_size` 大小的通信域（Ascend 上是 HCCL 的 `buf_size` 大小），默认是 1 个 G |
| `engine_type` | `str` | 通信协议。如果只是 intra-node 模式，Ascend 上是 MTE3；如果需要 inter-node 模式，则是 ROCE/RDMA |

**返回值**：

无返回值

**约束说明**：

- 需要 `torch.distributed` 算子库和 shmem 库

**示例**：

```python
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton.experimental.tle as tle

def test_tle_d2d_barrier(self):
    grid = (N, )

    tle.init_communicator()
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    x = (torch.arange(N, dtype=torch.float32, device="npu") + rank * 1000).clone()
    device_dptr = tle.create_dist_tensor(x)
    device_dptr.copy_(x)
    output = torch.zeros(N, dtype=torch.float32, device="npu")

    _runtime_verify(output, device_dptr, grid, rank, world_size)
    tle.cleanup_communicator(device_dptr)
```

### 2.2 create_dist_tensor

**背景**：

创建需要通信的对称内存，每张卡上都会进行 `alloca`，分配在 GM 上。通过传入 `buf_tensor` 判断创建的对称内存 buffer 大小，并用全局变量 `_g_mem_count` 进行记录。

**接口说明**：

```python
def create_dist_tensor(buf_tensor):
```

**入参说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `buf_tensor` | `torch.Tensor` | 传入 tensor，并根据 `tensor.dtype` 和 `numel` 计算需要的对称内存大小，并创建 |

**返回值**：

创建成功，返回对称内存的指针。

**约束说明**：

- `buf_tensor` must be a tensor

**示例**：

```python
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton.experimental.tle as tle

def test_tle_d2d_barrier(self):
    grid = (N, )

    tle.init_communicator()
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    x = (torch.arange(N, dtype=torch.float32, device="npu") + rank * 1000).clone()
    device_dptr = tle.create_dist_tensor(x)
    device_dptr.copy_(x)
    output = torch.zeros(N, dtype=torch.float32, device="npu")

    _runtime_verify(output, device_dptr, grid, rank, world_size)
    tle.cleanup_communicator(device_dptr)
```

### 2.3 cleanup_communicator

**背景**：

释放传入的对称内存，会通过全局变量 `_g_mem_count` 判断是不是最后一个对称内存，`True` 就执行 `aclshmem_finalize` 去释放整个通信域网络。

**接口说明**：

```python
def cleanup_communicator(peer_mem):
```

**入参说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `peer_mem` | 指针 | 通过 `create_dist_tensor` 创建的对称内存 buf 地址 |

**返回值**：

无

**约束说明**：

- 无

**示例**：

```python
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton.experimental.tle as tle

def test_tle_d2d_barrier(self):
    grid = (N, )

    tle.init_communicator()
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    x = (torch.arange(N, dtype=torch.float32, device="npu") + rank * 1000).clone()
    device_dptr = tle.create_dist_tensor(x)
    device_dptr.copy_(x)
    output = torch.zeros(N, dtype=torch.float32, device="npu")

    _runtime_verify(output, device_dptr, grid, rank, world_size)
    tle.cleanup_communicator(device_dptr)
```

---

## 3. Device 侧 OP

### 3.1 MeshConfig

**背景**：

- 社区：将整个通信网络切分成多个 team，层级分为 node、device（rank）、cluster、block。提供给 `device_mesh` 的入参。
- DSA：对于 `backend.ascend` 只实现 device 间（rank 间）的切分，不支持 cluster、block 等。提供给 `device_mesh` 的入参。
- TODO：teamOp 实现，node 层级。

**接口说明**：

无

**入参说明**：

不涉及

**返回值**：

不涉及

**约束说明**：

- 不支持 cluster、block

**示例**：

```python
topology = {
    # 节点内 GPU (4 devices)
    "device": 4,
}
```

### 3.2 device_mesh

**背景**：

`device_mesh`：以 rank/device 为单位的通信域大小。区别于社区实现：社区目前可以通过 `device_mesh` 配置 `[node, device, cluster, block]` 四个维度的通信域，DSA 目前只支持 device，TODO 实现 node。

**接口说明**：

```python
class device_mesh:
    def __init__(self, topology: MeshConfig)

    @property
    def device_count(self) -> int:
        return self._mesh_config.device

    @property
    def shape(self) -> tuple:
        return (self.device_count, )

    def __repr__(self):
        return f"DeviceMesh(device={self.device_count})"
```

**入参说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `topology` | `MeshConfig` | `MeshConfig` 初始化的通信域配置 |

**返回值**：

`device_mesh` 类的一个实例

**约束说明**：

- 不支持 cluster、block

**示例**：

```python
DEVICE_MESH = tle.device_mesh(tle.MeshConfig(device=2))

# 输出 DEVICE_MESH：
# type      : device_mesh
# repr      : DeviceMesh(shape=(4), names=('device'))
# shape     : (4)
# ndim      : 1
# dim_names : ('device')
# phys_ids  : (0, 1, 2, 3)
# launch    : (4) ('device')
# size      : 8
```

### 3.3 remote

**背景**：

`tle.remote` 是 TLE 的远端访问入口：给定目标分片（`shard_id` / `rank_id`），返回一个指向远端对称内存的指针，之后配合 `tl.load` / `tl.store` 完成实际数据搬运。

**接口说明**：

```python
def remote(
    tensor,
    shard_id=None,
    scope=None,
    space: str = None,
    dtype: tl.dtype = None,
    offset: int = None,
    _semantic=None,
):
```

**入参说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `tensor` | 指针 `tl.tensor` | 必填，指向本 rank 对称堆内存的指针；不允许 block 指针（`assert not is_block() and is_ptr()`） |
| `shard_id` | `int` | 必填，对端 rank（不是 mesh 坐标 tuple），经 `_convert_elem_to_ir_value(..., require_i64=False)` 转为 i32 |
| `scope` | - | 仅做 isinstance 校验，不参与坐标线性化（与 NVIDIA 版不同） |
| `space` | `str` | 默认 `None`，只支持 `device_mesh` 类实例 |
| `dtype` | `tl.dtype` | 选填，仅校验 float16 / float32 / bfloat16 / int8~64 / uint8~64 |
| `offset` | `int` | 必须为 `None`，不支持 |

**返回值**：

返回一个 `tl.tensor` 远端指针，它不搬运数据。

**约束说明**：

- `offset` 不支持，`space` 只支持 device，其余会 assert
- `scope` 必须是 `device_mesh` 的实例

**示例**：

```python
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton.experimental.tle as tle

DEVICE_MESH = tle.device_mesh(tle.MeshConfig(device=2))   # Ascend 版：只支持 device 层

# 读邻居 rank 的对称内存
@triton.jit()
def _barrier_d2d_kernel(out_ptr, device_dptr, mesh: tl.constexpr):
    pid = tl.program_id(0)
    local_rank = tle.shard_id(mesh, 'device', device_dptr=device_dptr)  # aclshmem_my_pe
    peer = (local_rank + 1) % mesh.shape[0]

    remote_mem = tle.remote(device_dptr, space="device", dtype=tl.float32, shard_id=peer)
    val = tl.load(remote_mem + pid)      # 偏移在调用点加，不用 remote(offset=...)
    tl.store(out_ptr + pid, val)
    tle.distributed_barrier(mesh=mesh, device_dptr=device_dptr, space="device")
```

### 3.4 shard_id

**背景**：

分布式 kernel 需要知道“我是几号 rank（PE）”，才能确定当前对称内存是在哪张卡上，因为 shmem 特点就是每张卡的对称内存用的 ptr 是同一个偏移。Ascend 版 `shard_id` 签名与社区 `distributed.py::shard_id()` 对齐，但实现简单得多：目前只支持 device 轴，返回当前 PE 编号。

**接口说明**：

语义：查询当前 kernel 所在 NPU 的 PE 编号（0 ~ rank_size-1）。

```python
@builtin
def shard_id(mesh=None, axis=-1, device_dptr=None, _semantic=None):
    """Return current shard coordinate. Currently only ``device`` axis is supported."""
    # mesh 仅做类型校验；axis 归一化后调用 create_get_rank(axis)
    return tl.tensor(_semantic.builder.create_get_rank(axis), tl.int32)
```

**入参说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `mesh` | `device_mesh` | 仅做 isinstance 校验，不参与计算 |
| `axis` | `str` / `int` | 默认 `-1`。str 只允许 `"device"`（内部归一化为 0）；int 只允许 0 或 -1；其余值触发 assert（TODO 未实现） |
| `device_dptr` | 指针 `tl.tensor` | 仅为签名对齐保留：给出时校验必须是指针 tensor，但不会被使用（rank 来自 SHMEM 运行时，不需要通信上下文） |

**返回值**：

标量 `tl.int32` tensor：当前 PE / rank 编号。

**约束说明**：

- `axis` 为 str 时只支持 `"device"`；为 int 时只支持 0 / -1（即一维 device mesh 的唯一轴）；其他轴（`"node"`、cluster 轴等）assert 失败：`"TODO: axis=... not yet implemented"`

**示例**：

```python
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton.experimental.tle as tle

DEVICE_MESH = tle.device_mesh(tle.MeshConfig(device=2))   # Ascend 版：只支持 device 层

# 读邻居 rank 的对称内存
@triton.jit()
def _barrier_d2d_kernel(out_ptr, device_dptr, mesh: tl.constexpr):
    pid = tl.program_id(0)
    local_rank = tle.shard_id(mesh, 'device', device_dptr=device_dptr)  # aclshmem_my_pe
    peer = (local_rank + 1) % mesh.shape[0]

    remote_mem = tle.remote(device_dptr, space="device", dtype=tl.float32, shard_id=peer)
    val = tl.load(remote_mem + pid)      # 偏移在调用点加，不用 remote(offset=...)
    tl.store(out_ptr + pid, val)
    tle.distributed_barrier(mesh=mesh, device_dptr=device_dptr, space="device")
```

### 3.5 distributed_barrier

**背景**：

多 rank 协同（如先写对端对称内存、再做 GEMM）需要全组同步。Ascend 版 barrier 是一个外部函数调用：生成 `extern_call`，目标符号 `aclshmem_barrier_all`（库 `libshmem_device`），即 ACL SHMEM 的全 PE 屏障。签名与 NVIDIA 版对齐，但 `mesh` / `space` / `group_kind` / `barrier_kind` / `order` / `index` 等参数均只用于校验或占位，实际同步范围固定为所有 rank（无子组 barrier）。

**接口说明**：

```python
@builtin
def distributed_barrier(mesh=None, device_dptr=None, space=None,
                        group_kind=None, barrier_kind=None, order=None,
                        index=None, _semantic=None):
    """Distributed barrier across all ranks. Currently only ``device`` level is supported."""
```

**入参说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `mesh` | `device_mesh` | 仅校验类型；不影响同步范围 |
| `device_dptr` | 指针 `tl.tensor` | 非 None 时必须是指针 tensor；实际未使用 |
| `space` | `str` / `None` | 只允许 `None` 或 `"device"`，其余 assert |
| `group_kind` / `barrier_kind` / `order` | - | 仅为签名对齐，完全不适用 |
| `index` | `int` | 非 None 时必须是 int，未使用 |

**返回值**：

无

**约束说明**：

- 同步范围固定是全部 rank，暂不支持 mesh 切片 / 子组 barrier（与 NVIDIA 版的 sub-mesh / FlagCX 组 barrier 不同）

**示例**：

```python
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton.experimental.tle as tle

DEVICE_MESH = tle.device_mesh(tle.MeshConfig(device=2))   # Ascend 版：只支持 device 层

# 读邻居 rank 的对称内存
@triton.jit()
def _barrier_d2d_kernel(out_ptr, device_dptr, mesh: tl.constexpr):
    pid = tl.program_id(0)
    local_rank = tle.shard_id(mesh, 'device', device_dptr=device_dptr)  # aclshmem_my_pe
    peer = (local_rank + 1) % mesh.shape[0]

    remote_mem = tle.remote(device_dptr, space="device", dtype=tl.float32, shard_id=peer)
    val = tl.load(remote_mem + pid)      # 偏移在调用点加，不用 remote(offset=...)
    tl.store(out_ptr + pid, val)
    tle.distributed_barrier(mesh=mesh, device_dptr=device_dptr, space="device")
```

### 3.6 swizzle2d_Nz

**背景**：

allgather 类通信把本 rank 的一块本地数据（如 GEMM 的 M×K 块，Ascend Nz 分形格式）按通信 tile 切块写到各对端的对称内存。朴素的“行优先 + 顺序 rank”映射会让连续迭代集中打同一个对端，造成通信热点。

**接口说明**：

```python
@triton.jit
def swizzle2d_Nz(iter_id, rank_size, data_row_shape, data_col_shape,
                 tile_row_shape, tile_col_shape, comm_npu_split=1):
    """Ascend Nz-format 2D swizzle for communication tiles."""
    # 迭代号 → (rank 组, 组内数据 tile) 交错分解，rank 再经 rank_stride 旋转
    return data_row_idx, data_col_idx, rank_idx, comm_row_size, comm_col_size
```

**入参说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `iter_id` | - | 当前迭代 id，通常为遍历搬运核 id |
| `rank_size` | - | 参与通信的 rank 数量 |
| `data_row_shape` | - | 待通信数据的行数 |
| `data_col_shape` | - | 待通信数据的列数 |
| `tile_row_shape` | - | 通信 tile 的行尺寸 |
| `tile_col_shape` | - | 通信 tile 的列尺寸 |
| `comm_npu_split` | - | 默认为 1 |

**返回值**：

5 元组 `[data_row_idx, data_col_idx, rank_idx, comm_row_size, comm_col_size]`：

| 元素 | 说明 |
|------|------|
| `data_row_idx` | 本次要发送的数据块的行 tile 序号 |
| `data_col_idx` | 列 tile 号 |
| `rank_idx` | 目标 rank |
| `comm_row_size` | 尾块处理的有效行数 |
| `comm_col_size` | 尾部收缩后的有效列数 |

**约束说明**：

- 纯算术函数，无硬件/编译约束，但要求输入尺寸为正（cdiv 语义）

**示例**：

```python
import triton.language as tl
import triton.experimental.tle as tle

for k in range(pid, comm_num_m_blocks * comm_num_k_blocks * rank_size, ncore):
    block_id_m, block_id_k, target_rank, comm_row_shape, comm_col_shape = tle.swizzle2d_Nz(
        k, rank_size, actual_block_size_m, K, COMM_BLOCK_SIZE_M, COMM_BLOCK_SIZE_K)

    remote_ptr = tle.remote(peer_mem_ptr, target_rank)      # swizzle 出的对端 rank
    ...  # 计算 local 源地址 a_ptrs 与对端目的地址 remote_ptrs
    a = tl.load(a_ptrs, mask=(comm_offs_k[None, :] < K) & comm_msk_m, other=0.0)
    tl.store(remote_ptrs, a, mask=...)                       # put 到对端对称内存
```

### 3.7 gemm_swizzle2d_Nz

**背景**：

allgather 完成后，GEMM 计算按 M×N tile 网格展开；朴素行优先调度相邻 block 的列地址跳跃大，对 Ascend Nz 格式数据的片上缓存（L2/CMEM）不友好。把扁平 block id 映射为（行 tile，列 tile），按 `swizzle_offset`（默认 7）个列 tile 分组，并且奇数列组反转行序（之字形 / zigzag 走向），让相邻迭代尽量复用同一批 A 行 / B 列，提高数据局部性。

**接口说明**：

```python
@triton.jit
def gemm_swizzle2d_Nz(iter_id, data_row_shape, data_col_shape,
                      tile_row_shape, tile_col_shape, swizzle_offset=7):
    """Ascend Nz-format 2D swizzle for GEMM compute tiles."""
    # n 方向按 swizzle_offset 分组，行方向在奇数组反转
    return data_row_idx, data_col_idx
```

**入参说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `iter_id` | - | 当前扁平 block_id（通常来自 `for block_id in range(pid, total, ncore)`） |
| `data_row_shape` | - | 计算问题的总行数 |
| `data_col_shape` | - | 总列数 |
| `tile_row_shape` | - | 计算 tile 行尺寸 |
| `tile_col_shape` | - | 计算 tile 列尺寸 |
| `swizzle_offset` | - | 默认 7 个为一组 |

**返回值**：

2 元组 `(data_row_idx, data_col_idx)`：该 block 负责的 M 方向 tile 序号和 N 方向 tile 序号。

**约束说明**：

- 纯算术函数，无硬件/编译约束
- `swizzle_offset` 默认 7 是经验值

**示例**：

```python
import triton.language as tl
import triton.experimental.tle as tle

for block_id in range(pid, num_tiles_m * num_loops_n * rank_size, ncore):
    block_id_m, block_id_n = tle.gemm_swizzle2d_Nz(
        block_id,
        rank_size * BLOCK_SIZE_M * num_tiles_m,   # 总行数（allgather 后）
        N,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
    rank_idx = block_id_m // num_tiles_m          # 由行 tile 反推该数据来自哪个 rank
    ...
    a = tl.load(peer_mem_ptr + ..., mask=...)     # 从本 rank 对称内存读 allgather 到的数据
```

---

## 4. 与社区 GPU Op 对比

社区 GPU 版是一套从核内到跨节点全覆盖的分布式原语；Ascend 版当前只覆盖“卡间”（NPU 间）一层，且同步粒度固定为“全部 rank”，无法做到核间（单卡内跨核）通信与子组同步，需要做 teamOp。

### 4.1 通信域的支持粒度

| 通信域 | GPU 实现 | Ascend 实现 |
|--------|----------|-------------|
| 核内 | thread / warp / block | 不支持 |
| 卡内 cluster | `space="cluster"`，remote 经过 cluster 访问对端 CTA 的 shared memory | 不支持 |
| 卡内 block | grid launch 同步 | 不支持 |
| 卡间 | `space="device"` | 支持 |
| 节点间 | `space="node"` | TODO |

### 4.2 Device API 差异

**device_mesh**

- GPU 版：完整四层拓扑（node / device / block_cluster / block），多维命名轴、`physical_ids`、launch 映射、切片（`__getitem__`）、reshape / flatten；mesh 本身是通信/同步范围的“坐标系”。
- Ascend 版：仅对应社区的实现，仅 device 一维（`shape=(device_count,)`）；node / block_cluster / block 传值即 assert；无切片/变形；mesh 仅作签名兼容，不参与任何计算（只做 isinstance 校验）。TODO：后续需要加入 node 一维。

**remote**

- GPU 版三条路径：cluster（DSMEM 远端指针 / buffered_tensor 标记 + `gpu.local_ptr` 物化）、device（FlagCX 对端指针，offset 必填且生效）、node（注册内存 put/get）；`shard_id` 支持 int / mesh 坐标 tuple（经 scope 行优先线性化，再经 `physical_ids` 映射回全局 rank）/ 运行时标量 int32。
- Ascend 版单条路径：`space` 只支持 `"device"`；语义是 SHMEM 对称指针交换（`tle.symm_at` → `aclshmem_ptr_<T>`）；tensor 只能是标量指针；`shard_id` 只能是对端 rank（无坐标 tuple）；`offset` 必须为 `None`（HIVM 层 addptr 丢失，偏移要调用点手动加）；dtype 仅白名单校验、不参与构造。
- 关键语义差异：GPU 版 cluster 路径的 remote 是“卡内核间寻址”（一个 cluster 里的 CTA 互相访问 shared memory）；Ascend 版没有核间等价物，remote 永远是“卡间”对称内存寻址。

**shard_id**

- GPU 版：`axis="device"` → 运行时查询本卡编号（`get_device_id`）；`axis="node"` → `world_rank // n_pes`；其余 cluster/block 轴由 program_id 按 launch mesh 分解得到。返回的是 device_mesh 上的坐标。
- Ascend 版：axis 只允许 `"device"` / 0 / -1，一律降级为 `tle.get_rank` → `aclshmem_my_pe`（当前 PE 编号）；mesh / device_dptr 只校验不使用，只返回第几张卡。

**distributed_barrier**

- GPU 版按 mesh 形态支持四种维度的同步。
- Ascend 版：无分派，恒等于 `aclshmem_barrier_all`（全 rank 屏障）；其余参数（group_kind / barrier_kind / order）完全忽略。即：只有“卡间全同步”一种粒度，既没有子组，也没有核内/核间粒度。

### 4.3 Ascend 独有（GPU 版无对应）

- `swizzle2d_Nz` / `gemm_swizzle2d_Nz`：`@triton.jit` 纯索引排布函数，服务于 Nz 分形格式下“通信 tile 排布”与“GEMM 计算 tile 排布”（提升局部性），属于优化的性能 OP。

### 4.4 GPU 独有 API

- `reshard` 和 `distributed_dot`：raise NotImplementedError，未实现。
- `sharding(mesh, split, partial)`：构造绑定 mesh 的分片次序 shardingSpec。
- `make_sharded_tensor(handle, sharding, shape)`：把已有的 tensor 句柄包装成 shardingTensor 并校验。

用来切分数据，让每个 mesh 维度知道当前的数据在整个通信域中的第几块。`sharding`、`shard_id` 等不进行分配显存，不拷贝，不做通信操作。

假如存在 `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]`：

```text
rank0 : [0, 1, 2, 3]
rank1 : [4, 5, 6, 7]
rank2 : [8, 9, 10, 11]
rank3 : [12, 13, 14, 15]
```

`0, 4, 8, 12` 不知道这块数据在整个通信域中属于哪一块，需要 `sharding` 确定如何组装，`make_sharded_tensor` 通知对应 rank 上的数据。

代码示例：

```python
MESH = tle.device_mesh(...)                        # 4 张卡，维度为 device
spec = tle.sharding(mesh, split=(S("device"), B))  # S 切，B 不切
```

即：有个 2 维张量，第 0 维（行）沿着 device 这根轴（4 张卡）切开；第 1 维（列）不切，每张卡都持有全部列。

```python
# rank0:
x0_local = <指向[1...8]指针>
x0 = tle.make_sharded_tensor(x_local, spec, (4, 4))
```

AscendC shmem 中：每张卡数据和通信量都是开发人员在 host 端和 device 端开发时确认的，需要写入对应的 slot 中，而不是组装。shmem 中需要通信对象一定是在对称内存中。

---

## 5. TODO

1. **team**：待实现。
2. **Signal / wait / fence OP**：待合入。
3. **InterNode 节点间通信**：待实现。
4. **device_mesh**：待扩展 GM->GM IntraNode mode。
