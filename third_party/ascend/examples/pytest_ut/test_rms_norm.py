# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import torch
import triton
import triton.language as tl
import torch_npu

@triton.jit
def _rms_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        y = x_hat * w
        # Write output
        tl.store(Y + cols, y.to(tl.float16), mask=mask)

# have to change the block_size
@torch.inference_mode()
def rms_norm(x, weight, eps, out=None):
    # allocate output, tl.store save y in tl.float16
    y = torch.empty_like(x, dtype=torch.float16) if out is None else out
    # reshape input data into 2D tensor
    x_arg = x.view(-1, x.shape[-1])
    M, N = x_arg.shape
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    BLOCK_SIZE = 128 * 2 * 2 * 2 * 2 * 2 * 2
    num_warps = 8
    # enqueue kernel
    kernel = _rms_norm_fwd_fused[(M,)](x_arg, y, weight, x_arg.stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return y, kernel


def _rms_norm(shape, datatype):
    x = torch.randn(shape[0], shape[1], dtype=datatype, device="npu")
    weight = torch.randn(shape[1], dtype=datatype, device="npu")
    y, kernel = rms_norm(x, weight, eps=1e-5)
    eps1 = 1e-5
    if datatype == torch.bfloat16 or datatype == torch.float16:
        x = x.to(torch.float32)
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps1)  # 计算均方根
    x_norm = x / rms  # 标准化
    y_ref = weight * x_norm
    y_ref = y_ref.to(torch.float16)
    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


def test_cases():
    _rms_norm((16, 1024), torch.float16)
    _rms_norm((16, 1024), torch.float32)
    _rms_norm((16, 1024), torch.bfloat16)
    _rms_norm((128, 3), torch.bfloat16)
    _rms_norm((128, 16), torch.bfloat16)
    _rms_norm((128, 37), torch.bfloat16)
    _rms_norm((128, 64), torch.bfloat16)
    _rms_norm((128, 781), torch.bfloat16)
    _rms_norm((128, 781), torch.bfloat16)
    _rms_norm((16, 1024), torch.float16)
    _rms_norm((16, 1024), torch.float32)
    _rms_norm((16, 1024), torch.bfloat16)

    _rms_norm((128, 128), torch.float16)
    _rms_norm((128, 128), torch.float32)
    _rms_norm((128, 128), torch.bfloat16)

    _rms_norm((1, 128), torch.float16)
    _rms_norm((1, 128), torch.float32)
    _rms_norm((1, 128), torch.bfloat16)

    _rms_norm((65535, 128), torch.float16)
    _rms_norm((65535, 128), torch.float32)
    _rms_norm((65535, 128), torch.bfloat16)

    _rms_norm((128, 3), torch.float16)
    _rms_norm((128, 3), torch.float32)
    _rms_norm((128, 3), torch.bfloat16)

    _rms_norm((128, 16), torch.float16)
    _rms_norm((128, 16), torch.float32)
    _rms_norm((128, 16), torch.bfloat16)

    _rms_norm((128, 37), torch.float16)
    _rms_norm((128, 37), torch.float32)
    _rms_norm((128, 37), torch.bfloat16)

    _rms_norm((128, 64), torch.float16)
    _rms_norm((128, 64), torch.float32)
    _rms_norm((128, 64), torch.bfloat16)

    _rms_norm((128, 781), torch.float16)
    _rms_norm((128, 781), torch.float32)
    _rms_norm((128, 781), torch.bfloat16)