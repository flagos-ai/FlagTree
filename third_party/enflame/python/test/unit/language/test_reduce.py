import pytest
import torch

import triton
import triton.language as tl
import importlib.util
if importlib.util.find_spec("triton.backends.enflame") is None:
    import triton_gcu.triton

import numpy as np
import os
import torch_gcu

os.environ["ENFLAME_LOG_DEBUG_MOD"] = "TORCH_GCU/OP"


@triton.jit
def _max_kernel_reduce(INPUT, OUT, input_stride0, input_stride1, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       M: tl.constexpr, N: tl.constexpr):
    start_m = tl.program_id(0)
    I_block_ptr = tl.make_block_ptr(base=INPUT, shape=(M, N), strides=(input_stride0, input_stride1),
                                    offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, N), order=(1, 0))
    O_block_ptr = tl.make_block_ptr(base=OUT, shape=(M, ), strides=(1, ), offsets=(start_m * BLOCK_M, ),
                                    block_shape=(BLOCK_M, ), order=(0, ))
    i = tl.load(I_block_ptr)
    out = tl.max(i, 1)

    tl.store(O_block_ptr, out)
    I_block_ptr = tl.advance(I_block_ptr, (0, BLOCK_N))


@staticmethod
def tri_max_reduce(input, out, M, N):
    BLOCK_M = 32
    BLOCK_N = 16
    grid = (triton.cdiv(input.shape[0], BLOCK_M), 1, 1)
    num_warps = 4
    _max_kernel_reduce[grid](input, out, input.stride(0), input.stride(1), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, M=M, N=N,
                             num_warps=num_warps, num_stages=4)
    return


def dump_tensor(tensor):
    for a in tensor.tolist():
        npro = []
        for j in a:
            npro.append(round(j, 3))
        print("[", npro, "]")
    print("\n")


def cosine_similarity(ta, tb):
    assert ta.shape == tb.shape
    sum_a = np.square(ta).sum()
    sum_b = np.square(tb).sum()
    if sum_a == 0 or sum_b == 0:
        return 0.0
    else:
        return np.float64(np.sum(ta * tb) / np.sqrt(sum_a) / np.sqrt(sum_b))


@pytest.mark.parametrize('M, N', [(16, 16), (64, 16), (128, 32)])
def test_op(M, N, dtype=torch.float32):
    input = torch.empty((M, N), dtype=dtype, device="gcu").normal_(mean=0., std=0.5)
    out = torch.empty((M), dtype=dtype, device="gcu")

    ref_out, ref_out_index = torch.max(input, dim=1)
    # triton implementation
    tri_out = tri_max_reduce(input, out, M, N)
    print("gcu caculate done!")
    # compare
    assert torch.allclose(ref_out, out, atol=1e-2, rtol=0)

    # cos_sim = cosine_similarity(ref_out.cpu().numpy(), tri_out.cpu().numpy())
    # print("output cos similarity:  {}".format(cos_sim))
    print("ok")


test_op(16, 16, torch.float32)
test_op(64, 16, torch.float32)
test_op(128, 32, torch.float32)
