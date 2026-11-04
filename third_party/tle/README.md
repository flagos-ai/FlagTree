# TLE (Triton Language Extensions)

<!-- flagtree tle -->

TLE is a language extension for Triton that exposes shared memory and pipeline compile hints for high-performance computing. This extension is specifically optimized for NVIDIA H100 devices. Note: The TLE extension for Ascend 910B devices is available in a separate repository.

## Features

- **Shared Memory Management**: `tle.alloc()` - Allocate shared/tensor memory with custom layouts
- **Data Movement**: `tle.copy()` - Efficient bidirectional copying between memory spaces
- **Local Operations**: `tle.local_ptr(buffer, indices=None)` + `tl.load/tl.store` - Access shared/tensor memory through pointer tensors (`indices=None` means full buffer view)
- **Pipeline Optimization**: `tle.pipeline()` - Hardware-aware pipeline iteration

## Memory Scopes & Layouts

- **Scopes**: `tle.smem` (shared memory), `tle.tmem` (tensor memory)
- **Layouts**: `tle.shared_layout`, `tle.swizzled_shared_layout`, `tle.tensor_memory_layout`, `tle.nv_mma_shared_layout`

## Quick Example

```python
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

@triton.jit
def kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Allocate shared memory
    a_smem = tle.alloc([BLOCK_SIZE], dtype=tl.float32, scope=tle.smem)
    b_smem = tle.alloc([BLOCK_SIZE], dtype=tl.float32, scope=tle.smem)
    # Full-view pointers (equivalent to passing explicit full indices)
    a_ptrs = tle.local_ptr(a_smem)
    b_ptrs = tle.local_ptr(b_smem)

    # Pipeline iteration for memory hiding
    for offset in tle.pipeline(0, n, BLOCK_SIZE, num_stages=2):
        # Copy to shared memory
        tle.copy(a_ptr + offset, a_smem, [BLOCK_SIZE])
        tle.copy(b_ptr + offset, b_smem, [BLOCK_SIZE])

        # Load and compute
        a_tile = tl.load(a_ptrs)
        b_tile = tl.load(b_ptrs)
        result = a_tile + b_tile

        tl.store(c_ptr + offset, result)
```

## Independent device barrier slots

FlagCX device barriers can synchronize corresponding CTA groups independently.
Create the context with enough runtime-owned slots and pass a CTA-uniform scalar
`int32` index together with its declared capacity:

```python
import torch

@triton.jit
def rendezvous(mem: tl.int64, comm: tl.int64, CAPACITY: tl.constexpr):
    tle.distributed_barrier(
        device_dptr=(mem, comm), space="device",
        index=1 + tl.program_id(0), barrier_count=CAPACITY,
    )

# Each participating process creates the same capacity. Slot 0 is reserved here
# for other synchronization; the 16 CTAs use distinct slots 1 through 16.
# `device` is this process's selected CUDA device.
tiles = 16
window = tle.allocate_dist_tensor((4096,), dtype=torch.uint8, device=device)
context = tle.create_dist_tensor(window, intra_barrier_count=tiles + 1)
mem, comm = context.get_packed_data()
rendezvous[(tiles,)](mem, comm, tiles + 1, num_warps=4,
                     launch_cooperative_grid=True)
```

Require `0 <= index < barrier_count <= context.intra_barrier_count`. Each slot
must have exactly one cooperating group per chip; all participants enter that
slot's rendezvous in the same order. The default group is one complete CTA.
Different slots do not synchronize other CTAs on the same chip. Ensure the
cooperative launch fits device residency so every required CTA can make progress.
Dynamic indices require explicit context handles; implicit/lite contexts are not
supported. Existing static-index calls retain their default behavior.

The runtime initializes slot state and retains it until
`tle.cleanup_communicator()`. Close the active context before creating a different
window or capacity. This state is separate from caller-owned payload scratch;
do not reset it between CUDA Graph replays.

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
