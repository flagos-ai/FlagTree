import torch
import triton
import triton.language as tl


@triton.jit
def atomic_rmw_useanalysis_kernel(
    input_ptr,
    output_ptr,
    m_ptr,
    d_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base_idx = pid * 8

    term1 = 15.0 * 15.0
    term2 = 8.0 * (7.0 - base_idx)

    delta = term1 + term2
    sqrt_delta = tl.sqrt(delta)

    task_idx = tl.ceil((15.0 - sqrt_delta) / 2.0)
    task_idx_i32 = task_idx.to(tl.int32)

    block_start = task_idx_i32 * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    m_val = tl.load(m_ptr + offsets, mask=mask, other=0.0)
    d_val = tl.load(d_ptr + offsets, mask=mask, other=0.0)

    scaled = data - m_val
    p = tl.exp(scaled)

    result = p * (data * 2.0 - d_val)

    output_offsets = offsets
    tl.atomic_add(output_ptr + output_offsets, result, mask=mask)


def test_atomic_rmw_useanalysis():
    DEVICE = "npu"
    N = 1024
    BLOCK_SIZE = 128

    torch.manual_seed(42)
    input_data = torch.randn(N, dtype=torch.float32, device=DEVICE)
    m_data = torch.randn(N, dtype=torch.float32, device=DEVICE)
    d_data = torch.randn(N, dtype=torch.float32, device=DEVICE)
    output_data = torch.zeros(N, dtype=torch.float32, device=DEVICE)

    grid = (8, )

    atomic_rmw_useanalysis_kernel[grid](
        input_data,
        output_data,
        m_data,
        d_data,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    output_sum = output_data.abs().sum().item()

    if output_sum == 0:
        raise AssertionError("UseAnalysis bug detected: atomic_rmw dependencies were erased")
    else:
        print("  AtomicRMW UseAnalysis is working correctly.")
