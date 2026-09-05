import torch
import math

import triton
import triton.language as tl

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

        embedding_kernel[
            M,
        ](output, indices, weight, N, BLOCK_SIZE)

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


def test(M=1151, N=8192, dtype=torch.float32):
    torch.manual_seed(0)

    weight = torch.rand((M, N), dtype=dtype, device="cpu")
    indices = torch.randint(0, M, [M], dtype=torch.int32, device="cpu")

    # pytorch
    torch_embedding = torch.nn.Embedding(M, N, _weight=weight)
    torch_output = torch_embedding(indices)

    weight = weight.to(DEVICE)
    indices = indices.to(DEVICE)
    triton_output = embedding(weight, indices)

    triton_output = triton_output.to("cpu")

    print("triton_output:\n", triton_output)
    print("torch_output:\n", torch_output)

    # 验证结果一致性
    assert torch.allclose(torch_output, triton_output, atol=1e-6), "verification failure!\n"
    print("verification success!\n")


test()
