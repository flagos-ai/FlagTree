import torch
import torch_npu
import triton
import triton.language as tl
import test_common
import pytest

def naive_softmax(x):
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


kernels = {}

def softmax(x, stream):
    n_rows, n_cols = x.shape

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    y = torch.empty_like(x)

    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        num_programs = 32
        kernel = softmax_kernel
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE
    )
    return y


types = [
    (torch.float32, 'float32'),
    (torch.float16, 'float16'),
    (torch.bfloat16, 'bfloat16'),
]

shapes = [
    (1823, 781),
    (1823, 2),
    (1823, 4),
    (1823, -32),
    (1823, -100),
    (1823, -256),
]

map_for_64_t = {37: 31}

@pytest.mark.skip(reason="randomly failed")
@pytest.mark.parametrize('dtype, sigtype',types)
@pytest.mark.parametrize('M, N',shapes)
def test_softmax(dtype, sigtype, M, N):
    torch_npu.npu.utils.set_device(0)
    M = (-M)//torch.tensor(0,dtype=dtype).element_size() if M<0 else M
    N = (-N)//torch.tensor(0,dtype=dtype).element_size() if N<0 else N

    if sigtype == 'int64':
        M = map_for_64_t[M] if M in map_for_64_t else M
        N = map_for_64_t[N] if N in map_for_64_t else N

    device = torch.npu.current_device()
    stream = torch.npu.current_stream(device).npu_stream
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=dtype, device='npu')
    y_triton = softmax(x, stream)
    y_torch = torch.softmax(x, axis=1)
    test_common.validate_cmp(sigtype, y_triton, y_torch)
