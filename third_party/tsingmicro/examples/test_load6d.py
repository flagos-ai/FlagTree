import torch
import triton
import triton.language as tl
import pytest
import benchmark
benchmark.select_cpu_backend()

@triton.jit
def six_dim_load(
    T_ptr,
    output_ptr,
    B, C, D1, D2, D3, D4,
    stride_b, stride_c, stride_d1, stride_d2, stride_d3, stride_d4,
    N: tl.constexpr
):

    pid = tl.program_id(0)
    b = pid // (C * D1 * D2 * D3 * D4)
    remaining = pid % (C * D1 * D2 * D3 * D4)

    c = remaining // (D1 * D2 * D3 * D4)
    remaining %= (D1 * D2 * D3 * D4)

    d1 = remaining // (D2 * D3 * D4)
    remaining %= (D2 * D3 * D4)

    d2 = remaining // (D3 * D4)
    remaining %= (D3 * D4)

    d3 = remaining // D4
    d4 = remaining % D4


    off_b = b * N + tl.arange(0, N)[:, None, None, None, None, None]
    off_c = c * N + tl.arange(0, N)[None, :, None, None, None, None]
    off_d1 = d1 * N + tl.arange(0, N)[None, None, :, None, None, None]
    off_d2 = d2 * N + tl.arange(0, N)[None, None, None, :, None, None]
    off_d3 = d3 * N + tl.arange(0, N)[None, None, None, None, :, None]
    off_d4 = d4 * N + tl.arange(0, N)[None, None, None, None, None, :]


    mask_b = off_b < B  # Shape: [B,1,1,1,1,1]
    mask_c = off_c < C  # Shape: [1,C,1,1,1,1]
    mask_d1 = off_d1 < D1  # Shape: [1,1,D1,1,1,1]
    mask_d2 = off_d2 < D2  # Shape: [1,1,1,D2,1,1]
    mask_d3 = off_d3 < D3  # Shape: [1,1,1,1,D3,1]
    mask_d4 = off_d4 < D4  # Shape: [1,1,1,1,1,D4]


    final_mask = (
        mask_b & mask_c &
        mask_d1 & mask_d2 &
        mask_d3 & mask_d4
    )


    global_idx = (
        off_b * stride_b +
        off_c * stride_c +
        off_d1 * stride_d1 +
        off_d2 * stride_d2 +
        off_d3 * stride_d3 +
        off_d4 * stride_d4
    )


    data = tl.load(T_ptr + global_idx, mask=final_mask)
    tl.store(output_ptr + global_idx, data, mask=final_mask)

@pytest.mark.parametrize("N", [2, 4])
def test_triton_six_dim_load(N: int):
    shape = (N, N, N, 
            N, N, N)
    input = torch.arange(0, N**6, device='cpu',dtype=torch.float32).reshape(shape).contiguous()

    B, C, D1, D2, D3, D4 = input.shape
    output = torch.empty_like(input)

    strides = input.stride()
    stride_b, stride_c, stride_d1, stride_d2, stride_d3, stride_d4 = strides[0], strides[1], strides[2], strides[3], strides[4], strides[5]

    six_dim_load[(1, )](
        input,
        output,
        B, C, D1, D2, D3, D4,
        stride_b, stride_c, stride_d1, stride_d2, stride_d3, stride_d4,
        N=N
    )

    torch.testing.assert_close(input, output, rtol=0, atol=0)

if __name__ == "__main__":
    N = 2
    test_triton_six_dim_load(N)
