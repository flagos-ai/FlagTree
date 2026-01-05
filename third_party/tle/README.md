# TLE (Tensor Language Extension)

<!-- flagtree tle -->

TLE is a language extension for Triton that exposes shared memory and pipeline compile hints for high-performance computing. This extension is specifically optimized for NVIDIA H100 devices. Note: The TLE extension for Ascend 910B devices is available in a separate repository.

## Features

- **Shared Memory Management**: `tle.alloc()` - Allocate shared/tensor memory with custom layouts
- **Data Movement**: `tle.copy()` - Efficient bidirectional copying between memory spaces
- **Local Operations**: `tle.local_load()`, `tle.local_store()` - Load/store from local memory
- **Pipeline Optimization**: `tle.pipeline()` - Hardware-aware pipeline iteration

## Memory Scopes & Layouts

- **Scopes**: `tle.smem` (shared memory), `tle.tmem` (tensor memory)
- **Layouts**: `tle.shared_layout`, `tle.swizzled_shared_layout`, `tle.tensor_memory_layout`, `tle.nv_mma_shared_layout`

## Quick Example

```python
import triton
import triton.language as tl
import triton.experimental.tle as tle

@triton.jit
def kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Allocate shared memory
    a_smem = tle.alloc([BLOCK_SIZE], dtype=tl.float32, scope=tle.smem)
    b_smem = tle.alloc([BLOCK_SIZE], dtype=tl.float32, scope=tle.smem)

    # Pipeline iteration for memory hiding
    for offset in tle.pipeline(0, n, BLOCK_SIZE, num_stages=2):
        # Copy to shared memory
        tle.copy(a_ptr + offset, a_smem, [BLOCK_SIZE])
        tle.copy(b_ptr + offset, b_smem, [BLOCK_SIZE])

        # Load and compute
        a_tile = tle.local_load(a_smem)
        b_tile = tle.local_load(b_smem)
        result = a_tile + b_tile

        tl.store(c_ptr + offset, result)
```

## Testing

```bash
cd python/test/tle
python run_tests.py
```

## Learn More

See integration examples in `python/test/tle/integration/`:
- `test_tle_pipeline_e2e.py` - Pipeline usage
- `test_tle_gemm.py` - GEMM with TLE
- `test_tle_tma_copy.py` - TMA operations
