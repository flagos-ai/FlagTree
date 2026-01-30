**TLE Architecture**

- **Purpose & Scope**  
  - Extends Triton with explicit shared/tensor memory management, async/TMA data movement, and pipeline control optimized for NVIDIA Hopper-class GPUs for now (README.md).  
  - Frontend APIs live under tle and lower into custom MLIR dialect + passes under tle.

- **Frontend DSL Layer (Python)**  
  - `tle.language.core` overrides key `tl` builtins such as `load`, `alloc`, `copy`, `local_ptr`, and loop helpers to attach extra attributes (e.g., `"tt.load.async"`) and create `buffered_tensor` handles representing shared/tensor memory allocations (core.py). Pointer tensors are then consumed by standard `tl.load`/`tl.store` ops.  
  - GPU-specific helpers in gpu define layouts (`swizzled_shared_layout`, `nv_mma_shared_layout`, etc.), scopes (`smem`, `tmem`), and `buffered_tensor` semantics that wrap IR memdesc types while keeping Triton-style type checking.  
  - Users import these symbols (e.g., `tle.alloc`, `tle.copy`, `tle.pipeline`) inside `@triton.jit` kernels to allocate SMEM tiles, launch async copies, or orchestrate staged loops.

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