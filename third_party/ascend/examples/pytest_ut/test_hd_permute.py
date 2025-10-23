import triton
import triton.language as tl
import torch
import torch_npu

X_SIZE: tl.constexpr = 4
Y_SIZE: tl.constexpr = 64
Z_SIZE: tl.constexpr = 32
NUMEL = X_SIZE * Y_SIZE * Z_SIZE


def torch_permute(x):
    return x.reshape((X_SIZE, Y_SIZE, Z_SIZE)).permute(1, 0, 2).reshape((X_SIZE * Y_SIZE * Z_SIZE))


@triton.jit
def triton_permute(output_ptr, input_ptr):
    x_index = tl.arange(0, X_SIZE * Y_SIZE * Z_SIZE)
    input_local = tl.load(input_ptr + x_index)
    output_local = input_local.reshape((X_SIZE, Y_SIZE, Z_SIZE)).permute(1, 0, 2).reshape((X_SIZE * Y_SIZE * Z_SIZE))
    tl.store(output_ptr + x_index, output_local)


def test_hd_permute():
    # 生成数据
    x = torch.randn(NUMEL).npu()
    # torch结果
    torch_res = torch_permute(x)
    # triton结果
    triton_res = torch.randn(torch_res.shape, dtype=torch_res.dtype).npu()
    triton_permute[1, 1, 1](triton_res, x)
    # 比较结果
    torch.testing.assert_close(triton_res, torch_res, rtol=1e-3, atol=1e-3)
