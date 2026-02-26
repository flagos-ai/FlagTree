# TLE (Triton Language Extension)

TLE is a language extension for Triton that exposes on-chip memory, pipeline compile hints and the accompanying calculation operations for high-performance computing. This extension is specifically optimized for Ascend 910B devices.

## Features

- **On-chip Memory Management**: `tle.alloc()` - Allocate memory on UB/L1/L0C
- **Data Movement**: `tle.copy()` - Efficient bidirectional copying between memory spaces
- **compute Operations**: `tle.npu_add()` - Addition on UB
- **Pipeline Optimization**: `tle.pipeline()` - Hardware-aware pipeline iteration

## Memory Scopes & Layouts

- **Scopes**: `tle.UB` (UB memory), `tle.L1` (L1 memory), `tle.L0C` (L0C memory)
- **Layouts**: `tle.ND`, `tle.NZ`

## Quick Example

```python
@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Allocate UB memory
    a_ub = tle.alloc([BLOCK_SIZE], dtype=tl.float32, layout=tle.ND, scope=tle.UB)
    b_ub = tle.alloc([BLOCK_SIZE], dtype=tl.float32, layout=tle.ND, scope=tle.UB)
    c_ub = tle.alloc([BLOCK_SIZE], dtype=tl.float32, layout=tle.ND, scope=tle.UB)

    # Tail block processing
    t0 = n_elements - block_start
    tail_size = tl.minimum(t0, BLOCK_SIZE)

    # Copy data from GM to UB
    tle.copy(x_ptr + offsets, a_ub, [tail_size])
    tle.copy(y_ptr + offsets, b_ub, [tail_size])

    # Addition
    tle.npu_add(a_ub, b_ub, c_ub)

     # Copy result back to GM
    tle.copy(c_ub, output_ptr + offsets, [tail_size])

```

## Testing

```bash
cd ascend/examples/tle/pytest_ut/
python3 test_vec_add.py
```

## Learn More

See other examples in `ascend/examples/tle/pytest_ut/`:
- `test_matmul.py` - GEMM implementation and pipeline usage