**TLE Architecture**

- **Purpose & Scope**
  - Extends Triton with explicit shared/tensor memory management, async/TMA data movement, and pipeline control optimized for NVIDIA Hopper-class GPUs for now (README.md).
  - Frontend APIs live under tle and lower into custom MLIR dialect + passes under tle.

- **Frontend DSL Layer (Python)**
  - `tle.language.core` overrides key `tl` builtins such as `load`, `alloc`, `copy`, `local_ptr`, and loop helpers to attach extra attributes (e.g., `"tt.load.async"`) and create `buffered_tensor` handles representing shared/tensor memory allocations (core.py). Pointer tensors are then consumed by standard `tl.load`/`tl.store` ops.
  - `tle.local_ptr(buffer, indices)` materializes arbitrary shared-memory pointer views from explicit index tensors. `indices` is a tuple of integer tensors (length == buffer rank) and all tensors must have identical shapes; that shape is the output pointer tensor shape.
  - GPU-specific helpers in gpu define layouts (`swizzled_shared_layout`, `nv_mma_shared_layout`, etc.), scopes (`smem`, `tmem`), and `buffered_tensor` semantics that wrap IR memdesc types while keeping Triton-style type checking.
  - Users import these symbols (e.g., `tle.alloc`, `tle.copy`, `tle.pipeline`) inside `@triton.jit` kernels to allocate SMEM tiles, launch async copies, or orchestrate staged loops.

- **tle.local_ptr details**
  - **Signature**: `tle.local_ptr(buffer, indices)` -> `tl.tensor` (pointer tensor)
  - **Purpose**: Build arbitrary-shaped pointer views over shared memory `buffer` for `tl.load`/`tl.store`.
  - **Parameters**:
    - `buffer`: `buffered_tensor` returned by `tle.alloc` (SMEM / TMEM).
    - `indices`: Tuple of integer tensors. Tuple length must equal `rank(buffer)`, and every tensor must have identical shapes.
  - **Semantics**:
    - Output pointer tensor shape equals the common shape of the `indices` tensors.
    - For each logical index `(i0, i1, ...)` in the output shape, the pointer value corresponds to `buffer[indices0(i0, i1, ...), indices1(i0, i1, ...), ...]`.
    - Returned pointers live in shared memory address space (LLVM addrspace=3). Indices must be integer (i32/i64, etc., reduced to i32 during lowering).
    - Linearization is row-major (last dimension fastest); shared memory layout/encoding follows the `buffer` memdesc.
    - `indices` also supports scalar values (not only tensors). With scalar indices, `tle.local_ptr` can be consumed by scalar `tl.load`/`tl.store` directly.
    - For scalar shared-memory reads/writes, no extra wrappers like `tl.full([1], ...)` + `tl.max(..., axis=0)` are required.

  - **Example 1: 1D slice**
    ```python
    smem = tle.alloc([BLOCK], dtype=tl.float32, scope=tle.smem)
    # Slice [offset, offset + SLICE)
    idx = offset + tl.arange(0, SLICE)
    slice_ptr = tle.local_ptr(smem, (idx,))
    vals = tl.load(slice_ptr)
    ```

  - **Example 2: K-dimension tiling (matrix slice)**
    ```python
    smem_a = tle.alloc([BM, BK], dtype=tl.float16, scope=tle.smem)
    # Slice (BM, KW), where KW is the K-dimension slice
    rows = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, KW))
    cols = tl.broadcast_to(tl.arange(0, KW)[None, :] + k_start, (BM, KW))
    a_slice = tle.local_ptr(smem_a, (rows, cols))
    a_vals = tl.load(a_slice)
    ```

  - **Example 3: arbitrary gather view**
    ```python
    smem = tle.alloc([H, W], dtype=tl.float32, scope=tle.smem)
    # Take an offset column per row
    rows = tl.broadcast_to(tl.arange(0, H)[:, None], (H, SLICE))
    cols = tl.broadcast_to(1 + tl.arange(0, SLICE)[None, :], (H, SLICE))
    gather_ptr = tle.local_ptr(smem, (rows, cols))
    out = tl.load(gather_ptr)
    ```

  - **Example 4: scalar shared-memory lookup**
    ```python
    RADIX = 16
    bins = tl.arange(0, RADIX)
    smem_counts = tle.alloc([RADIX], dtype=tl.int32, scope=tle.smem)
    smem_count_ptrs = tle.local_ptr(smem_counts, (bins,))

    # Build descending cumulative histogram.
    counts = tl.load(smem_count_ptrs)
    cumsum_desc = tl.cumsum(counts, axis=0, reverse=True)
    tl.store(smem_count_ptrs, cumsum_desc)
    tl.debug_barrier()

    # Scalar load through local_ptr.
    d = 7
    cum_d = tl.load(tle.local_ptr(smem_counts, (d,)))
    cum_next = tl.load(tle.local_ptr(smem_counts, (d + 1,)))
    ```

- **Semantic Validation**
  - `TLESemantic` in semantic.py runs alongside Triton’s semantic layer. It validates shapes, dtypes, and copy compatibility before lowering, providing early error messages and adapting constexpr inputs.
  - Semantic helpers call into custom builder hooks (exposed via the C++ bridge) to emit `LocalAllocOp`, `TMACopyOp`, etc., ensuring Python APIs map 1:1 to TTIR constructs.

- **Raw/EDSL Layer**
  - raw exposes a lightweight MLIR-based eDSL for writing dialect-specific intrinsics directly. Decorators like `@dialect(name="mlir")` build LLVM IR from Python ASTs via `EdslMLIRJITFunction`, enabling backend authors to prototype kernels or helper ops outside the high-level Triton syntax.
  - The raw runtime (`call()` helper) materializes `tle::DSLRegionOp` nodes whose bodies are later inlined by passes.

- **C++ Bridge & Dialect**
  - triton_tle.cc registers additional builder methods (creating encoding attributes, memdesc types, TMACopy ops, DSL regions) onto Triton’s `TritonOpBuilder`, and wires new passes plus raw IR helpers into Python via pybind11.
  - The MLIR dialect lives under dialect with IR definitions plus Analysis/Conversion/Transforms infrastructure mirroring upstream Triton conventions.

- **Pass & Lowering Pipeline**
  - Pass registrations are defined in Passes.td and surfaced to Python (`add_early_assign_memory_space`, `add_lower_async_load`, `add_lower_tma_copy`, `add_tle_convert_arg_to_memdesc`, `add_tle_dsl_region_inline`).
  - Key transformations:
    - **Early Assign Memory Space** rewrites tensors tagged with `tt.memory_space="shared_memory"` into explicit local alloc/store sequences and removes the attribute so later passes see concrete SMEM ops (TleEarlyAssignMemorySpace.cpp).
    - **Lower Async Load** looks for loads marked with `"tt.load.async"` (set by `tle.load`) and converts them into Hopper-style async copy + commit/wait chains feeding `LocalLoadOp`s, deduplicating redundant allocs (TleLowerAsyncLoad.cpp).
    - **Lower TMA Copy** lowers high-level `TMACopyOp` (emitted by `tle.copy` with tensor descriptors) into NVIDIA TMA intrinsics, handling both GM→SMEM and SMEM→GM directions with barrier management (TleLowerTmaCopy.cpp).
    - **Convert Arg To MemDesc** materializes memdesc-compatible operands/results inside DSL regions, inserting temporary local alloc/load sequences so generic Triton passes can reason about them (ConvertArgToMemDesc.cpp).
    - **DSL Region Inline** splices `tle::DSLRegionOp` bodies back into surrounding CFG blocks, replacing yields with branches once raw kernels are lowered (DSLRegionInline.cpp).

- **Backend Distribution**
  - Backend-specific logic currently targets NVIDIA (see nvidia and the use of `triton::nvidia_gpu` intrinsics inside passes). Other hardware backends can plug in by reusing the raw DSL + pass hooks and implementing their own lowering passes/encodings under `third_party/<backend>/backend/compiler.py`, similar to how HINTS are dispatched.
  - Pass wrappers exported from triton_tle.cc let each backend opt into only the passes it supports when assembling its pipeline (e.g., NVIDIA enabling TMA lowering while another backend might stop after memory-space tagging).

- **Testing & Examples**
  - Integration tests under tle (mentioned in the README) cover end-to-end kernels for pipeline loops, GEMM, and TMA copies to ensure Python APIs, semantic checks, and passes stay aligned.
  - Developers can run `python python/test/tle/run_tests.py` after modifying either the Python DSL or MLIR passes to catch regressions quickly.

- **Extending TLE**
  - New APIs should mirror the established pattern: add Python surface ops (with semantic validation) → expose necessary builder hooks → create/extend dialect ops → add lowering passes and register them for backends.
  - Keep layout/scope abstractions centralized in types.py so future hardware (e.g., tensor memory) can be toggled without touching user code, and document any new passes in Passes.td to keep the wiki aligned.

Potential next steps:
1. Add an English/Chinese doc under `docs/backend/tle/` summarizing this wiki for the official site.
2. Provide backend-specific pass pipeline examples to show how to combine the provided passes per target.
