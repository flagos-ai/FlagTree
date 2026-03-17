# TLE Architecture Design

## 1. Introduction

Triton is an operator programming language in the form of a Python DSL. It follows a block-based programming model that abstracts away hardware details such as memory hierarchy, layout, pipelining, and synchronization, while achieving strong operator performance through compiler optimization. These advantages have attracted a large developer community and ecosystem.

In recent years, however, Triton has faced growth challenges:

- Adaptation to DSA platforms and new GPU architectures has progressed slowly.
- Compared with emerging languages like TileLang, Triton lacks abstractions for fine-grained control of memory hierarchy and parallel granularity, which can lead to weaker performance in some cases.

To address these issues, we propose TLE (Triton Language Extentions), which extends Triton across three levels to meet urgent needs from users with different skill profiles.

## 2. Observations and Proposed Solutions

We analyzed mainstream DSLs in the industry (Triton, TileLang, and cuTile) and summarized a target language design.

### 2.1 Pythonic

All three are Python-syntax-based DSLs, indicating that developers prefer Python-like syntax for kernel development, even if only a subset of Python is available.

### 2.2 Tile Programming

All three support block-level programming. In essence, current block programming mainly performs tiling on global memory. cuTile goes further by supporting multi-level tiling, making it possible to design a unified language across multiple memory hierarchy architectures.

Triton, however, does not explicitly model tile/slice concepts, so users can only tile at the global memory level, limiting further language evolution.

TileLang is similar to Triton in that it does not provide explicit tiling primitives. In addition, except for copy and GEMM, it lacks higher-level tensor ops, which makes GPU programming less convenient. Without automatic vectorization, utilizing SIMD hardware well often requires adding many SIMD-specific ops.

### 2.3 Memory Hierarchy Abstraction

To address the memory wall, modern hardware uses multi-level memory hierarchies.

- Triton/cuTile expose only two levels: global memory and local tensor.
- TileLang directly exposes native hardware memory hierarchy without abstraction.

Problems:

- Exposing too few levels pushes tiling and buffer promotion work to the compiler.
- Directly exposing native hierarchy significantly hurts portability.

Preferred direction:

- Developers perform tiling, but do not explicitly select memory levels.
- Compiler performs buffer promotion.
- Developers may provide hints; tile sizes are treated as hyperparameters.

This keeps portability while leaving room for further optimization.

### 2.4 Parallelism Abstraction

- Triton/cuTile expose only block-level parallelism, and intra-block parallelism is fully compiler-controlled.
- TileLang lets developers explicitly control intra-block parallelism (Parallel and Vectorize), improving expressiveness but reducing portability and reuse across hardware.

### 2.5 Distributed Abstraction

None of these languages directly covers cross-block or cross-node communication, which limits compute-communication fusion (ongoing external work includes Triton Distributed and TileScale).

### 2.6 Ideal Language Design

- Level 1: Numpy/PyTorch-like algorithm-level programming. Users focus on algorithm logic only; compiler handles hardware mapping and communication.
- Level 2: cuTile-like tile-level programming plus distributed descriptions. Users explicitly provide tiling and sharding, while compiler handles memory hierarchy, parallelism, and communication, with optional hardware/scenario hints.
- Level 3: Hardware-specific extensions (memory hierarchy, thread binding, vectorize, etc.). This level is confined to specific regions with explicit interaction contracts with Level 2. Compiler performs only essential optimizations.

Detailed principles:

- Tile semantics to avoid manual address arithmetic.
- Do not require tensor shapes to be powers of two.

Open question: what other strong design ideas should be added?

## 3. Architecture Design

### 3.1 Architecture Overview

TLE sits in the middle layer of the AI software stack:

- Upstream: serves AI frameworks through graph compilers and operator libraries.
- Downstream: integrates with various hardware runtimes.

> Content not available outside Feishu document yet.

TLE is split into three layers:

- **TLE-Lite**: lightweight extension over Triton. Features are backend-compatible, and only small changes to existing Triton kernels are needed to gain significant speedups. Targets algorithm engineers and fast optimization workflows.
- **TLE-Struct**: architecture-clustered abstractions (e.g., GPGPU, DSA) for deeper performance tuning. Requires moderate hardware knowledge.
- **TLE-Raw**: direct hardware control, including vendor-native programming languages for maximum performance. Targets expert performance engineers.

Lowering paths:

- TLE-Lite and TLE-Struct lower to LLVM IR via FLIR.
- TLE-Raw lowers to LLVM IR via language-specific pipelines (e.g., vendor private compilers).
- All parts are finally linked into a complete kernel loaded/executed by runtime.

### 3.2 TLE-Lite

- Design philosophy: write once, run anywhere.
- Core idea: use high-level semantic hints (instead of hard constraints) to guide compiler heuristics. Keep backward compatibility and achieve cross-platform speedups with minimal code changes.

#### 3.2.1 Memory Management

##### 3.2.1.1 `tle.load`

Extension of `tl.load` with async hint support:

```python
x = tle.load(..., is_async=True)
```

#### 3.2.2 Tensor Slicing

##### 3.2.2.1 `tle.extract_tile`

Split input tensor into a sub-tile grid using a child-tile shape and extract tile at specified coordinates.

- GPU: supports extraction from registers and shared memory.

```python
# x is [4, 4]
# z is [2, 2]
# Split x into shape=[2, 2] sub-tiles and return tile at [0, 0]
z = x.extract_tile(index=[0, 0], shape=[2, 2])
```

##### 3.2.2.2 `tle.insert_tile`

Split input tensor into a sub-tile grid using child-tile shape and update tile at specified coordinates.

- GPU: supports updates in registers and shared memory.

```python
# x is [4, 4], y is [2, 2], z is [4, 4]
# Split x into shape=[2, 2] sub-tiles, update tile [0, 0] with y,
# and return full updated [4, 4] tensor
z = x.insert_tile(y, index=[0, 0])
```

#### 3.2.3 Pipeline

##### 3.2.3.1 `tle.pipeline_group`

Hint-style extension.

Automatic stage partitioning:

```python
for yoff in tl.range(0, ynumel, YBLOCK, num_stages=2):
    Q = tl.load(...)
    K = tl.load(...)
    KT = tl.trans(K)
    V = tl.dot(Q, KT)
```

Manual stage partitioning:

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
    # Warp specialization or heterogeneous units
    with tle.pipeline_group(0):
        Q = tl.load(...)
        K = tl.load(...)
    with tle.pipeline_group(1):
        KT = tl.trans(K)
    with tle.pipeline_group(2):
        V = tl.dot(Q, KT)
```

#### 3.2.4 Distributed

Triton distributed API has four core parts: device mesh definition, sharding specification, resharding (collective communication), and remote access (point-to-point communication).

##### 3.2.4.1 Device Mesh

`tle.device_mesh` defines physical device topology and serves as the context foundation for distributed operations.

```python
class device_mesh:
    def __init__(self, topology: dict):
        """
        Initialize DeviceMesh.

        Args:
            topology (dict): Hardware hierarchy description.
                             Keys are hierarchy names; values are int (1D)
                             or tuple lists (multi-dimensional).
        """
        self._physical_ids = ... # Internal flattened physical IDs (0..N-1)
        self._shape = ...        # Current logical shape, e.g. (2, 2, 4, 2, 2, 4)
        self._dim_names = ...    # Current dimension names

    @property
    def shape(self):
        """Return logical mesh shape."""
        return self._shape

    @property
    def ndim(self):
        """Return number of dimensions."""
        return len(self._shape)

    def flatten(self):
        """Flatten mesh to 1D, typically for ring communication."""
        return self.reshape(prod(self._shape))

    def __getitem__(self, key):
        """
        Supports slicing and returns a sub-mesh.
        Supports standard slice and integer indexing.
        """
        return sub_mesh

    def __repr__(self):
        return f"DeviceMesh(shape={self._shape}, names={self._dim_names})"


# Define complex hardware hierarchy
topology = {
    # Cross-node hierarchy (2x2 = 4 nodes)
    "node": [("node_x", 2), ("node_y", 2)],
    # In-node GPUs (4 devices)
    "device": 4,
    # In-GPU cluster (2x2)
    "block_cluster": [("cluster_x", 2), ("cluster_y", 2)],
    # In-cluster blocks (4 blocks)
    "block": 4,
}

# mesh.shape -> (2, 2, 4, 2, 2, 4)
# total size = 256
mesh = tle.device_mesh(topology=topology)
```

##### 3.2.4.2 Sharding Specification

`tle.sharding` declares tensor distribution state on the device mesh:

- `splits`: how each tensor axis is partitioned on mesh axes.
- `partials`: whether tensor is partial-sum state.
- Unspecified mesh axes are treated as broadcast.

Symbols:

- `tle.S(axis)`: split.
- `tle.B`: broadcast/replicate.
- `tle.P(axis)`: partial; requires reduce on specified axis.

```python
def sharding(tensor, splits, partials):
    """
    Annotation only: marks tensor state, emits no direct code,
    but guides compiler checks and optimizations.
    """
    return tensor

# Split axis0 on cluster, axis1 on device, and partial on block axis
x_shard = tle.sharding(
    mesh,
    split=[["cluster_x", "cluster_y"], "device"],
    partial=["block"],
)

# Define a sharded tensor
x = tle.make_sharded_tensor(x_ptr, sharding=x_shard, shape=[4, 4])
```

##### 3.2.4.3 Synchronization

In complex distributed kernels (e.g., ring all-reduce or row/column-independent pipelines), only “same-row” or “same-column” blocks often need synchronization rather than the whole cluster. Global synchronization introduces unnecessary waiting.

```python
def distributed_barrier(mesh):
    """
    If sub_mesh is passed, synchronize only devices in this sub-mesh.
    Devices outside this sub-mesh should treat it as No-Op
    (or compiler guarantees control flow does not enter).
    """
    pass
```

##### 3.2.4.4 Remote Access

`tle.remote` obtains a handle for tensor data located on other devices. This maps to point-to-point communication or direct memory access (RDMA/NVLink load).

```python
def remote(tensor, shard_id, scope):
    """
    Get a RemoteTensor handle to a shard on a target device.

    :param tensor: logically distributed tensor (already marked by tle.sharding)
    :param shard_id: tuple coordinate in device mesh
    :return: RemoteTensor, supporting load/store and related ops
    """
```

##### 3.2.4.5 Resharding

`tle.reshard` is the entrypoint for collectives. Compiler compares source and target specs and inserts communication primitives automatically.

```python
def reshard(tensor, spec):
    """
    Action: transform tensor to a new distribution state.

    Typical transitions:
    1. [ ] -> [S]: Scatter
    2. [S] -> [ ]: Gather
    3. [P] -> [ ]: Reduce
    4. [B] -> [S]: Local slice (no communication)
    5. [S] -> [B]: All-gather
    6. [P] -> [B]: All-reduce
    7. [B] -> [P]: Error
    """
```

##### 3.2.4.6 Distributed GEMM

NVIDIA Hopper (H100) and newer architectures introduce Thread Block Cluster, allowing groups of CTAs to cooperate via DSMEM for high-bandwidth, low-latency exchange.

`tle.distributed_dot` is designed to use this feature so developers can write cross-block matrix multiplication without manually handling DSMEM barriers and data movement.

```python
def distributed_dot(a, b, c=None):
    """
    Execute distributed matrix multiplication within current
    Thread Block Cluster scope.

    Behavior depends on sharding specs of input tensors `a` and `b`
    over the cluster mesh.

    Args:
        a (Tensor): left operand with cluster-level sharding annotation.
        b (Tensor): right operand with cluster-level sharding annotation.
        c (Tensor, optional): accumulator.

    Returns:
        Tensor: result tensor with distribution inferred from inputs.
    """
```

Open question: what additional distributed primitives are needed?

#### 3.2.5 API Reference and Practical Examples

##### 3.2.5.1 `tle.load`

- Signature: `tle.load(ptr, mask=None, other=None, is_async=False)`
- Use case: Keep `tl.load` semantics while adding async scheduling hints.
- Practical guidance:
  - Use `is_async=True` for global-memory reads that are later reused in compute-heavy regions.
  - Keep `mask` and `other` explicit on boundary tiles to avoid undefined values.

Example: guarded async load for tail tiles

```python
offs = base + tl.arange(0, BLOCK)
mask = offs < n_elements
x = tle.load(x_ptr + offs, mask=mask, other=0.0, is_async=True)
```

Example: async load + compute overlap pattern

```python
for k in tl.range(0, K, BK, num_stages=2):
    a = tle.load(a_ptr + k * stride_a, is_async=True)
    b = tle.load(b_ptr + k * stride_b, is_async=True)
    acc = tl.dot(a, b, acc)
```

##### 3.2.5.2 `tle.extract_tile` and `tle.insert_tile`

- `extract_tile`: read a sub-tile view from a larger tile tensor.
- `insert_tile`: write a processed sub-tile back to a larger tile tensor.
- Typical use: local transforms (activation, quant/dequant, normalization) on sub-regions without manual pointer arithmetic.

Example: tilewise post-processing in registers

```python
# x: [4, 4]
sub = x.extract_tile(index=[1, 0], shape=[2, 2])  # rows [2:4], cols [0:2]
sub = tl.maximum(sub, 0.0)  # ReLU on the sub-tile
x = x.insert_tile(sub, index=[1, 0])
```

##### 3.2.5.3 `tle.pipeline_group`

- Use `tle.pipeline_group(stage_id)` to explicitly tag operations into stages.
- Useful when you need deterministic stage control (instead of fully heuristic grouping).

Example: staged load-transform-matmul

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

- Recommended workflow:
  1. Define topology with `tle.device_mesh`.
  2. Mark tensor layout with `tle.sharding`.
  3. Transform layout with `tle.reshard`.
  4. Keep compute kernels operating on logical tensor views.

Example: split-by-device input, then all-gather before compute

```python
mesh = tle.device_mesh({"node": 2, "device": 4})
x_spec = tle.sharding(mesh, split=["device"], partial=[])
x = tle.make_sharded_tensor(x_ptr, sharding=x_spec, shape=[M, K])

# [S] -> [B] on device axis (all-gather)
x_full = tle.reshard(x, spec=tle.sharding(mesh, split=[], partial=[]))
```

##### 3.2.5.5 `tle.shard_id`

- Signature: `tle.shard_id(mesh, axis)`
- Returns current program's coordinate on a mesh axis.
- `axis` can be a mesh-axis name (e.g. `"node"`, `"device"`, `"cluster_x"`) or an axis index.
- Typical use: build peer shard IDs for ring exchange, staged all-reduce, and cluster-cooperative kernels.

Example: query current program coordinates on node/device axes

```python
mesh = tle.device_mesh({"node": 2, "device": 4})
node_rank = tle.shard_id(mesh, "node")      # 0..1
device_rank = tle.shard_id(mesh, "device")  # 0..3
```

##### 3.2.5.6 `tle.remote` + `tle.distributed_barrier`

- `tle.remote` reads/writes explicit remote shards.
- `tle.distributed_barrier` synchronizes only the mesh/sub-mesh you pass in.

Example: remote read from neighbor shard (ring-like exchange)

```python
node_rank = tle.shard_id(mesh, "node")
device_rank = tle.shard_id(mesh, "device")
next_device = (device_rank + 1) % mesh.shape[1]
remote_x = tle.remote(x, shard_id=(node_rank, next_device), scope=mesh)
tle.distributed_barrier(mesh)
neighbor_vals = tl.load(remote_x)
```

### 3.3 TLE-Struct

- Design philosophy: architecture-aware, fine-grained tuning.
- Core idea: classify backends by hardware-topology families (e.g., GPGPU, DSA), expose common hierarchical parallel/storage structures, and let developers explicitly define structured compute/data mappings (e.g., warp-group control, pipeline scheduling). This decouples algorithm logic from hardware physical implementation at the abstraction level.

#### 3.3.1 GPU

##### 3.3.1.1 Memory Management

###### 3.3.1.1.1 `tle.gpu.memory_space`

Specify tensor `memory_space`:

```python
x = ...
x = tle.gpu.memory_space(x, "shared_memory")
```

###### 3.3.1.1.2 `tle.gpu.alloc`

Allocate memory:

```python
a_smem = tle.gpu.alloc(
    [XBLOCK, YBLOCK],
    dtype=tl.float32,
    layout=None,
    scope=tle.gpu.storage_kind.smem,
)
```

###### 3.3.1.1.3 `tle.gpu.local_ptr`

Get memory pointers:

```python
# pointers for a_smem[0, :]: [(0, 0), (0, 1), ..., (0, YBLOCK-1)]
a_smem_ptrs = tle.gpu.local_ptr(
    a_smem,
    indices=(tl.broadcast(0, [YBLOCK]), tl.arange(0, YBLOCK)),
)
```

- Signature: `tle.local_ptr(buffer, indices=None) -> tl.tensor | tl.ptr`
- Purpose: Build arbitrary-shaped pointer views over shared memory buffer for `tl.load/tl.store`.
- Parameters:
  - `buffer`: buffered tensor returned by `tle.alloc` (SMEM/TMEM).
  - `indices`: optional tuple of integer tensors. Tuple length must equal `rank(buffer)`, and all tensors must have identical shapes. If omitted/`None`, backend treats it as full indices.
- Semantics:
  - If `indices` is provided: output pointer tensor shape equals common shape of index tensors.
  - For each logical output index `(i0, i1, ...)`, pointer value corresponds to `buffer[indices0(i0,...), indices1(i0,...), ...]`.
  - If `indices=None`: build full-view pointers over `buffer` shape (rank>0 returns pointer tensor with `shape(buffer)`, rank=0 returns scalar pointer).
  - Returned pointers live in shared-memory address space (LLVM addrspace=3). Indices must be integers (i32/i64, etc.; lowered to i32).
  - Linearization is row-major (last dimension fastest); shared-memory layout/encoding follows buffer memdesc.

Example 1: 1D slice

```python
smem = tle.alloc([BLOCK], dtype=tl.float32, scope=tle.smem)
# Slice [offset, offset + SLICE)
idx = offset + tl.arange(0, SLICE)
slice_ptr = tle.local_ptr(smem, (idx,))
vals = tl.load(slice_ptr)
```

Example 2: K-dimension tiling (matrix slice)

```python
smem_a = tle.alloc([BM, BK], dtype=tl.float16, scope=tle.smem)
# Slice (BM, KW), where KW is K-dimension slice
rows = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, KW))
cols = tl.broadcast_to(tl.arange(0, KW)[None, :] + k_start, (BM, KW))
a_slice = tle.local_ptr(smem_a, (rows, cols))
a_vals = tl.load(a_slice)
```

Example 3: arbitrary gather view

```python
smem = tle.alloc([H, W], dtype=tl.float32, scope=tle.smem)
# Take an offset column per row
rows = tl.broadcast_to(tl.arange(0, H)[:, None], (H, SLICE))
cols = tl.broadcast_to(1 + tl.arange(0, SLICE)[None, :], (H, SLICE))
gather_ptr = tle.local_ptr(smem, (rows, cols))
out = tl.load(gather_ptr)
```

###### 3.3.1.1.4 `tle.gpu.copy`

Memory copy:

```python
tle.gpu.copy(a_ptrs + ystride_a * yoffs[None, :], a_smem, [XBLOCK, YBLOCK])
```

###### 3.3.1.1.5 `tl.load/tl.store/tl.atomic*` for `tle.local_ptr`

Shared-memory pointers from `tle.local_ptr` can be directly used by:

- `tl.load`
- `tl.store`
- `tl.atomic_add/and/cas/max/min/or/xchg/xor`

#### 3.3.2 DSA

##### 3.3.2.1 Memory Management

###### 3.3.2.1.1 `tle.dsa.alloc`

Allocate memory (Ascend):

```python
a_ub = tle.dsa.alloc(
    [XBLOCK, YBLOCK],
    dtype=tl.float32,
    layout=tle.dsa.ascend.NZ,
    scope=tle.dsa.ascend.UB,
)
```

###### 3.3.2.1.2 `tle.dsa.copy`

Memory copy:

```python
tle.dsa.copy(a_ptrs + ystride_a * yoffs[None, :], a_smem, [XBLOCK, YBLOCK])
```

###### 3.3.2.1.3 `tle.dsa.local_load`

Load from local memory:

```python
aval = tle.dsa.local_load(a_smem)
```

#### 3.3.3 Struct API Cookbook

##### 3.3.3.1 Shared-memory staging (`alloc` + `copy` + `local_ptr`)

Use this pattern when data is reused across multiple math operations.

```python
# 1) Allocate SMEM tile
a_smem = tle.gpu.alloc([BM, BK], dtype=tl.float16, scope=tle.gpu.storage_kind.smem)

# 2) Copy GMEM -> SMEM
tle.gpu.copy(a_ptrs, a_smem, [BM, BK])

# 3) Build local pointer view and load
rows = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, BK))
cols = tl.broadcast_to(tl.arange(0, BK)[None, :], (BM, BK))
a_ptr_local = tle.gpu.local_ptr(a_smem, (rows, cols))
a_tile = tl.load(a_ptr_local)
```

##### 3.3.3.2 Shared-memory atomics with `local_ptr`

Useful for histogram, bucketization, and radix-select style counting.

```python
bins = 256
counts = tle.gpu.alloc([bins], dtype=tl.int32, scope=tle.gpu.storage_kind.smem)
idx = tl.arange(0, BLOCK) % bins
count_ptr = tle.gpu.local_ptr(counts, (idx,))
tl.atomic_add(count_ptr, 1)
```

##### 3.3.3.3 DSA local-buffer flow (`dsa.alloc` + `dsa.copy` + `dsa.local_load`)

Use this for DSA backends that expose dedicated local buffer spaces.

```python
a_ub = tle.dsa.alloc([BM, BK], dtype=tl.float16, layout=tle.dsa.ascend.NZ, scope=tle.dsa.ascend.UB)
tle.dsa.copy(a_ptrs, a_ub, [BM, BK])
a_val = tle.dsa.local_load(a_ub)
```

### 3.4 TLE-Raw

- Design philosophy: native passthrough and maximal control.
- Core idea: break DSL abstraction boundaries and support inlined vendor-native code. Target instructions are generated through vendor-private pipelines, bypassing general compiler middle layers and giving experts strong control over instruction scheduling, register allocation, and low-level synchronization primitives.

> Content not available outside Feishu document yet.

Open question: should Raw integration be limited to Python DSL only?

#### 3.4.1 Language Extensions

##### 3.4.1.1 MLIR

```python
from typing import Annotated
from mlir import ir
from mlir.dialects import arith, nvvm, tensor
import triton.language as tl
from triton.experimental.flagtree.edsl import dialect
import triton.experimental.flagtree.language as fl

# 1. Dialect declaration
@tle.raw.language(name="mlir")
# 2. Hardware constraints
@tle.hardware_constraint(threads_dim=1, sync_scope="block")
# 3. Function implementation
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

    # 4. Function call
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

## 4. Examples and Evaluation

### 4.1 SparseMLA

Optimization and tests have been conducted for SparseMLA in DSA on RTX 5060Ti and H800.

- TileLang version: `v0.1.7`
- Example code: <https://github.com/flagos-ai/FlagTree/blob/triton_v3.5.x/python/tutorials/tle/01-sparse-mla.py>

Core kernel (excerpt):

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

Performance comparison (TFLOPS):

| Device | Theoretical | Triton | TileLang | TLE | TLE over Triton |
| --- | ---: | ---: | ---: | ---: | ---: |
| H800 | 800 | 165.5 | **355.0** | 210.6 | 1.27x |
| H20 | - | 81.0 | **110.2** | 93.2 | 1.15x |
| RTX 5060Ti | - | 30.7 | Not supported | **32.8** | 1.07x |

### 4.2 MoeAlignBlockSize

With shared-memory extensions in `tle-struct`, it is possible to implement `vllm/sglang`-style `moe_align_block_size` and improve performance.

- Example code: <https://github.com/flagos-ai/FlagTree/blob/triton_v3.5.x/python/tutorials/tle/02-moe_align_block_size.py>

#### 4.2.1 RTX 5060 Ti

| num_tokens | triton | triton_atomic | **tle_atomic_fused [ours]** | **tle_cluster_fused [ours]** | sglang_cuda | **Speedup (sglang_cuda / min(tle_atomic_fused, tle_cluster_fused))** |
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

| num_tokens | triton | triton_atomic | **tle_atomic_fused [ours]** | **tle_cluster_fused [ours]** | sglang_cuda | **Speedup (sglang_cuda / min(tle_atomic_fused, tle_cluster_fused))** |
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

#### 4.2.3 H800 Real Data (`build/gems/moe_topk_ids.pt`)

- Runtime config: `num_tokens=163840`, `num_experts=512`, `block_size=16`, `source=real`.

| num_tokens | num_experts | block_size | triton | triton_atomic | **tle_atomic_fused [ours]** | **tle_cluster_fused [ours]** | sglang_cuda | **Speedup (sglang_cuda / min(tle_atomic_fused, tle_cluster_fused))** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 163840 | 512 | 16 | 0.0471 | 0.0535 | **0.0387** | 0.0750 | 0.1467 | 3.79x |

#### 4.2.4 RTX 5060 Ti Real Data (`build/gems/moe_topk_ids.pt`, Local Measurement)

- Runtime config: `num_tokens=163840`, `num_experts=512`, `block_size=16`, `source=real`.
- Runtime command:
  `conda run -n flagtree python python/tutorials/tle/02-moe_align_block_size.py --skip_correctness --real_data build/gems/moe_topk_ids.pt --num_experts 512 --block_size 16`

| num_tokens | num_experts | block_size | triton | triton_atomic | **tle_atomic_fused [ours]** | **tle_cluster_fused [ours]** | sglang_cuda | **Speedup (sglang_cuda / min(tle_atomic_fused, tle_cluster_fused))** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 163840 | 512 | 16 | 0.0507 | 0.0395 | **0.0261** | 0.0532 | 0.1060 | 4.06x |

### 4.3 TopK

With shared-memory extensions in `tle-struct`, radix-select-based TopK can improve performance in MoE scenarios with large N and small K.

- Example code: <https://github.com/flagos-ai/FlagTree/blob/triton_v3.5.x/python/tutorials/tle/03-topk.py>

#### 4.3.1 RTX 5060 Ti (`tle-topk-radix-vs-torch`)

| M | N | K | Triton-RadixSelect | Torch-TopK | **Speedup (Torch / Triton-RadixSelect)** |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 128 | 8 | **0.008192** | 0.010240 | 1.25x |
| 64 | 1024 | 32 | **0.008192** | 0.020480 | 2.50x |
| 64 | 8192 | 128 | **0.026624** | 0.059392 | 2.23x |
| 128 | 32768 | 256 | **0.124928** | 0.192512 | 1.54x |

#### 4.3.2 H800 (`tle-topk-radix-vs-torch`)

| M | N | K | Triton-RadixSelect | Torch-TopK | **Speedup (Torch / Triton-RadixSelect)** |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 128 | 8 | **0.008384** | 0.017536 | 2.09x |
| 64 | 1024 | 32 | **0.010688** | 0.024304 | 2.27x |
| 64 | 8192 | 128 | **0.029952** | 0.057184 | 1.91x |
| 128 | 32768 | 256 | **0.092256** | 0.117856 | 1.28x |

### 4.4 TopK Selector

TopK selector performance is evaluated with `python/tutorials/tle/deepseek_v32/01-topk_selector.py` (`plot_name=tle-radix-topk-selector`).

#### 4.4.1 RTX 5060 Ti (Local Measurement)

- Runtime: local benchmark (GeForce RTX 5060 Ti), `--skip_correctness --warmup 10 --rep 80`.

| batch | seq_len | topk | Torch-TopK | Triton-Radix | TileLang | TLE-Radix | **Speedup (Torch-TopK / TLE-Radix)** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 4096 | 128 | 0.038912 | 0.039456 | 0.020480 | **0.015808** | 2.46x |
| 64 | 8192 | 256 | 0.088624 | 0.053248 | 0.028672 | **0.023936** | 3.70x |
| 64 | 32768 | 1024 | 0.158272 | 0.131616 | 0.073728 | **0.062912** | 2.52x |
| 64 | 32768 | 2048 | 0.163264 | 0.133120 | 0.075776 | **0.065536** | 2.49x |

#### 4.4.2 H800 (`tle-radix-topk-selector`)

| batch | seq_len | topk | Torch-TopK | Triton-Radix | TileLang | TLE-Radix | **Speedup (Torch-TopK / TLE-Radix)** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 4096 | 128 | 0.045728 | 0.054256 | **0.017200** | 0.017472 | 2.62x |
| 64 | 8192 | 256 | 0.097344 | 0.072512 | 0.020960 | **0.020928** | 4.65x |
| 64 | 32768 | 1024 | 0.125008 | 0.176768 | 0.043088 | **0.041856** | 2.99x |
| 64 | 32768 | 2048 | 0.125072 | 0.179264 | 0.044256 | **0.041984** | 2.98x |
