# TLE Practical Guide (Beginner to Advanced)

This guide is self-contained and executable.
It targets three jobs:
1. write a working TLE kernel,
2. optimize it to high performance,
3. implement new TLE functionality in API/IR/lowering/pipeline and debug failures.

## 1. First-Run Quickstart

### 1.1 Environment Preflight
Run from repo root:

```bash
<py_exec> -V
<py_exec> -c "import torch, triton; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('triton', triton.__version__)"
<py_exec> -c "import torch; print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"
```

`<py_exec>` can be any of:
1. `python` (active shell env),
2. `/path/to/venv/bin/python`,
3. `conda run -n <env> python`.

If C++ bindings need rebuild, use your repo's actual build entrypoint.
Do not assume a specific script exists.

```bash
# Option A: project-provided build script (if present)
if [ -x ./build.sh ]; then
  ./build.sh
elif [ -x ./scripts/build.sh ]; then
  ./scripts/build.sh
fi

# Option B: editable python rebuild path (if your project uses setuptools/pyproject)
<py_exec> -m pip install -e .

# Option C: CMake/Ninja path (if your project is cmake-based)
ninja -C <build_dir>
```

If none of the above match your repo, define `<build_entrypoint>` explicitly in your task context.

### 1.2 Minimal End-to-End Script (Host + Kernel + Check)
Create and run this script directly.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def tle_axpy_kernel(x_ptr, y_ptr, out_ptr, n, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n

    smem = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    ptrs = tle.gpu.local_ptr(smem, (tl.arange(0, BLOCK),))

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(ptrs, x, mask=mask)
    z = tl.load(ptrs, mask=mask, other=0.0) * alpha + y
    tl.store(ptrs, z, mask=mask)
    tl.store(out_ptr + offs, tl.load(ptrs, mask=mask, other=0.0), mask=mask)


def main():
    torch.manual_seed(0)
    n = 4096
    block = 256
    alpha = 1.25

    x = torch.randn(n, device='cuda', dtype=torch.float32)
    y = torch.randn(n, device='cuda', dtype=torch.float32)
    out = torch.empty_like(x)

    grid = (triton.cdiv(n, block),)
    tle_axpy_kernel[grid](x, y, out, n, alpha, BLOCK=block)

    ref = x * alpha + y
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)
    print('PASS: correctness check')

    # Compiler artifact inspection (critical for debug/perf work)
    compiled = tle_axpy_kernel.warmup(x, y, out, n, alpha, BLOCK=block, grid=grid)
    ttgir = compiled.asm.get('ttgir', '')
    ptx = compiled.asm.get('ptx', '')
    print('TTGIR length:', len(ttgir))
    print('PTX length:', len(ptx))
    print('Has local pointers op:', 'tle.local_pointers' in ttgir)


if __name__ == '__main__':
    main()
```

Run:

```bash
<py_exec> /tmp/tle_axpy_quickstart.py
```

## 2. Current TLE Semantics Baseline (No External File Needed)

### 2.1 `local_ptr` Contract (Current Code)
API form:
```python
ptr = tle.gpu.local_ptr(buffer, indices)
```

Rules:
1. `buffer` must be a TLE buffered tensor from `tle.gpu.alloc`.
2. `indices` must be tuple/list (or Triton tuple) and cannot be empty.
3. Index count must equal buffer rank.
4. Index dtype must be integer.
5. Either all scalar indices or all tensor indices.
6. Tensor-index mode requires all index tensors to have identical shape.
7. Mixed scalar/tensor index usage is invalid.

### 2.2 Shared-Memory Pointer Semantics
1. Local pointers are shared-memory pointers in lowering semantics.
2. Load/store lowering must branch by pointer address space (shared vs global).

### 2.3 Local Pointer Pipeline Invariants
NVIDIA TTGIR pipeline local pointer segment:
1. `add_early_assign_memory_space`
2. `add_select_encodings`
3. `add_insert_local_pointer_barriers`
4. `add_optimize_local_pointer_loads`
5. `add_optimize_local_pointer_stores`

Do not reorder without proof and tests.

### 2.4 TLE->LLVM Legality Requirements
TLE conversion path includes:
1. legal `mlir::gpu::GPUDialect`,
2. legal `mlir::UnrealizedConversionCastOp`,
3. registered local pointer conversion patterns.

## 3. Kernel Authoring Patterns

### 3.1 1D Local Staging Pattern
Use for elementwise fusion and short reuse windows.

```python
smem = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
ptrs = tle.gpu.local_ptr(smem, (tl.arange(0, BLOCK),))
vals = tl.load(global_ptrs, mask=mask, other=0.0)
tl.store(ptrs, vals, mask=mask)
out = tl.load(ptrs, mask=mask, other=0.0)
```

### 3.2 2D Tile Pointer Pattern
Use when loading and slicing tiles.

```python
rows = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, BK))
cols = tl.broadcast_to(tl.arange(0, BK)[None, :], (BM, BK))
ptr = tle.gpu.local_ptr(tile_buf, (rows, cols))
sub = tl.load(ptr)
```

### 3.3 `copy` vs `load/store`
1. Use `tle.gpu.copy` for explicit transfer operations and descriptor/TMA flows.
2. Use `local_ptr + tl.load/store` for custom indexing and compute choreography.

### 3.4 Distributed Entry Pattern
```python
import triton.experimental.tle.language as tle

mesh = tle.device_mesh({"block_cluster": [("cluster_x", 2), ("cluster_y", 2)]})
sid = tle.shard_id(mesh, "cluster_x")
tle.distributed_barrier(mesh)
```

## 4. High-Performance Optimization Playbook

### 4.1 Parameter Priority (Most Impact First)
1. Tile sizes (`BLOCK_M`, `BLOCK_N`, `BLOCK_K` or 1D `BLOCK`).
2. `num_warps`.
3. `num_stages`.
4. Memory path choice (`copy` vs manual load/store).
5. Layout settings (`nv_mma_shared_layout`, swizzled layout choices).

### 4.2 One-Change Benchmark Loop
For each candidate:
1. Keep shape/seed/grid fixed.
2. Change one parameter only.
3. Run correctness check.
4. Run timed benchmark.
5. Capture TTGIR/PTX evidence.

Minimal timing skeleton:

```python
import time

def bench(fn, rep=50):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / rep
```

### 4.3 Stop Conditions
Stop tuning when one is true:
1. no measurable improvement for 3 consecutive single-parameter trials,
2. regression risk rises (correctness instability, brittle masking),
3. achieved target acceptance performance.

## 5. Debug Guide (Command-Level)

### 5.1 Fast Triage Order
1. Reproduce with smallest shape that still fails.
2. Confirm correctness mismatch vs Torch reference.
3. Dump TTGIR/PTX via `warmup(...).asm`.
4. Identify layer: API, verifier, lowering, or runtime behavior.

### 5.2 Useful Commands
Targeted tests first:

```bash
<py_exec> -m pytest python/test/tle/unit/test_tle_gpu_local_ptr.py -vv -s
<py_exec> -m pytest python/test/tle/integration/test_tle_local_store.py -vv -s
<py_exec> -m pytest python/test/tle/integration/test_tle_distributed.py -vv -s
```

Search relevant code quickly:

```bash
rg -n "def local_ptr\(|analyze_local_pointer_operation" python/triton/experimental/tle/language/gpu
rg -n "LocalPointersOp::verify|kSharedMemoryAddressSpace" third_party/tle/dialect/lib/IR/Ops.cpp
rg -n "TleSelectEncodings|TleInsertLocalPointerBarriers" third_party/tle/dialect/lib/Transforms
rg -n "add_early_assign_memory_space|add_select_encodings|add_insert_local_pointer_barriers" third_party/nvidia/backend/compiler.py
rg -n "populateLocalPointersOpToLLVMPatterns|UnrealizedConversionCastOp|GPUDialect" third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp
```

### 5.3 Symptom -> Likely Layer -> Action
1. Verifier error on pointer/index shape:
   - Layer: API/verifier.
   - Action: validate index count/type/shape contract in local_ptr call and verifier.
2. Compiles but wrong output:
   - Layer: kernel logic or lowering mismatch.
   - Action: reduce shape, isolate one tile, compare intermediate loads/stores.
3. Intermittent mismatch after local store/load:
   - Layer: ordering/barrier behavior.
   - Action: inspect barrier insertion path and simplify control flow.
4. No perf gain after local staging:
   - Layer: layout conversions / pipeline.
   - Action: count key TTGIR/PTX patterns before/after and verify traffic reduction.

## 6. Implementing New TLE Features (Concrete File Map)

Use this section when changing language semantics or compiler behavior.

### 6.1 Python API Layer
Typical files:
1. `python/triton/experimental/tle/__init__.py`
2. `python/triton/experimental/tle/language/__init__.py`
3. `python/triton/experimental/tle/language/gpu/core.py`
4. `python/triton/experimental/tle/language/gpu/semantic.py`

What to do:
1. expose API,
2. enforce argument contract and error messages,
3. add semantic checks and tests.

### 6.2 IR and Verifier Layer
Typical files:
1. `third_party/tle/dialect/include/IR/TleOps.td`
2. `third_party/tle/dialect/lib/IR/Ops.cpp`

What to do:
1. update op defs/types/attrs,
2. add/adjust verifier invariants,
3. keep diagnostics specific and actionable.

### 6.3 Lowering/Conversion Layer
Typical files:
1. `third_party/tle/dialect/lib/Conversion/TleToLLVM/LocalPointersOpToLLVM.cpp`
2. related conversion files under `third_party/tle/dialect/lib/Conversion/TleToLLVM/`.

What to do:
1. map op semantics to LLVM-compatible forms,
2. preserve address-space correctness,
3. handle shape/encoding consistency.

### 6.4 Transform and Pass Wiring
Typical files:
1. `third_party/tle/dialect/lib/Transforms/TleSelectEncodings.cpp`
2. `third_party/tle/dialect/lib/Transforms/TleInsertLocalPointerBarriers.cpp`
3. `third_party/nvidia/backend/compiler.py`
4. `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp`

What to do:
1. maintain pass ordering invariants,
2. ensure conversion target legality is correct,
3. ensure patterns are registered.

### 6.5 Test Coverage Placement
1. Unit semantics: `python/test/tle/unit/`
2. Integration behavior: `python/test/tle/integration/`
3. Backend-specific cases: `third_party/<backend>/python/test/`

Minimum required test additions for semantic changes:
1. one positive case,
2. one negative contract case,
3. one regression case that would fail without your fix.

## 7. Validation Matrix and Done Criteria

### 7.1 Validation Matrix
1. targeted unit tests for changed API/verifier path,
2. targeted integration tests for changed lowering path,
3. backend-specific tests if pass/codegen changed,
4. `ninja check-*` if C++ compiler components changed.

### 7.2 Done Criteria
A change is done only when:
1. behavior contract is explicit,
2. tests cover positive + negative + regression,
3. commands and outcomes are reproducible,
4. Fix Summary and Lessons Entry are completed,
5. residual risk and follow-up are listed.

## 8. API Surface Snapshot

### `triton.experimental.tle`
- `device_mesh`, `S`, `P`, `B`
- `sharding`, `ShardingSpec`
- `ShardedTensor`, `make_sharded_tensor`
- `reshard`, `remote`, `shard_id`, `distributed_barrier`, `distributed_dot`
- `language`, optional `raw`

### `triton.experimental.tle.language`
- `load`, `gpu`, `raw`

### `tle.gpu`
- `pipeline`, `alloc`, `copy`, `local_ptr`, `memory_space`
- `layout`, `shared_layout`, `swizzled_shared_layout`, `tensor_memory_layout`, `nv_mma_shared_layout`
- `scope`, `smem`, `tmem`, `buffered_tensor`, `buffered_tensor_type`

### `triton.experimental.tle.language.raw`
- `call`

### `triton.experimental.tle.raw`
- `dialect`, `Input`, `InOut`
