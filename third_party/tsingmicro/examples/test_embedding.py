import torch
import math

import triton
import triton.language as tl

import pytest
import benchmark

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def embedding_kernel(
    out_ptr,  # pointer to the output
    in_ptr,  # pointer to the input
    weight_ptr,  # pointer to the weights
    N: tl.constexpr,  # number of columns in X
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    out_ptr += pid * N
    in_ptr += pid

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)

    row_idx = tl.load(in_ptr)
    weight_ptr += row_idx * N
    embedding_weight = tl.load(weight_ptr + cols, mask, other=0.0)
    tl.store(out_ptr + cols, embedding_weight, mask)


class Embedding(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):

        assert not sparse, "Currently do not support sparse format"

        M = math.prod(indices.shape)
        N = weight.shape[-1]

        BLOCK_SIZE = triton.next_power_of_2(N)
        indices = indices.contiguous()
        weight = weight.contiguous()
        output = torch.empty((*indices.shape, N), device=indices.device, dtype=weight.dtype)

        output = output.to(DEVICE)
        indices = indices.to(DEVICE)
        weight = weight.to(DEVICE)
        embedding_kernel[
            M,
        ](output, indices, weight, N, BLOCK_SIZE)
        output = output.to("cpu")
        ctx.M = M
        ctx.N = N
        ctx.num_weights = weight.shape[0]
        ctx.padding_idx = padding_idx
        ctx.scale_grad_by_freq = scale_grad_by_freq
        ctx.sparse = sparse
        ctx.indices = indices

        return output


def embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    return Embedding.apply(weight, indices, padding_idx, scale_grad_by_freq, sparse)


@pytest.mark.parametrize("M, N, dtype", [  #
    (M, N, dtype) for M in [1152] for N in [2048] for dtype in [torch.float32]
])
def test_embedding(M, N, dtype, device='cpu'):
    torch.manual_seed(0)

    weight = torch.rand((M, N), dtype=dtype, device=device)
    indices = torch.randint(0, M, [M], dtype=torch.int32, device=device)

    triton_output = embedding(weight, indices)

    # pytorch
    torch_embedding = torch.nn.Embedding(M, N, _weight=weight)
    torch_output = torch_embedding(indices)

    # compare
    print(f"The maximum difference between torch and triton is "
          f"{torch.max(torch.abs(torch_output - triton_output))}")
    assert torch.allclose(triton_output, torch_output, atol=1e-5, rtol=0)


if __name__ == "__main__":
    # benchmark.select_cpu_backend()
    test_embedding(1151, 8192, torch.float32)
