import torch
import triton
import triton.language as tl
import pytest


# Triton Conv2D Kernel实现
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_NI_HO_WO': 128, 'BLOCK_CI': 32, 'BLOCK_CO': 32}, num_warps=4),
        triton.Config({'BLOCK_NI_HO_WO': 256, 'BLOCK_CI': 64, 'BLOCK_CO': 32}, num_warps=8),
    ],
    key=[
        'in_n', 'weight_c', 'input_h', 'input_w', 'out_c', 'out_h', 'out_w', 'weight_h', 'weight_w', 'stride',
        'padding', 'groups'
    ],
)
@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    in_n,
    input_h,
    input_w,
    out_c,
    out_h,
    out_w,
    input_n_stride,
    input_c_stride,
    input_h_stride,
    input_w_stride,
    weight_n_stride,
    weight_c_stride,
    weight_h_stride,
    weight_w_stride,
    output_n_stride,
    output_c_stride,
    output_h_stride,
    output_w_stride,
    weight_c: tl.constexpr,
    weight_h: tl.constexpr,
    weight_w: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_NI_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_ni_ho_wo = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_group = tl.program_id(2)

    # 计算位置索引
    ni_ho_wo_offset = pid_ni_ho_wo * BLOCK_NI_HO_WO + tl.arange(0, BLOCK_NI_HO_WO)
    ni_ho_offset = ni_ho_wo_offset // out_w
    in_n_idx = ni_ho_offset // out_h
    out_h_idx = ni_ho_offset % out_h
    out_w_idx = ni_ho_wo_offset % out_w

    # 计算指针偏移
    out_per_group_c = out_c // groups
    output_c_offset = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)

    input_ptr += (input_n_stride * in_n_idx + input_c_stride * pid_group * weight_c)[:, None]
    weight_ptr += (weight_n_stride * output_c_offset + weight_n_stride * pid_group * out_per_group_c)[None, :]

    # 累加器初始化
    accum = tl.zeros((BLOCK_NI_HO_WO, BLOCK_CO), dtype=tl.float32)

    # 计算循环次数
    BLOCK_CI_COUNT = (weight_c + BLOCK_CI - 1) // BLOCK_CI

    for hwc in range(weight_h * weight_w * BLOCK_CI_COUNT):
        c = (hwc % BLOCK_CI_COUNT) * BLOCK_CI
        hw = hwc // BLOCK_CI_COUNT
        h = hw // weight_w
        w = hw % weight_w

        input_c_offset = c + tl.arange(0, BLOCK_CI)
        input_h_offset = h * dilation - padding + stride * out_h_idx
        input_w_offset = w * dilation - padding + stride * out_w_idx

        # 计算输入指针
        curr_input_ptr = (input_ptr + (input_c_stride * input_c_offset)[None, :] +
                          (input_h_stride * input_h_offset)[:, None] + (input_w_stride * input_w_offset)[:, None])

        # 计算权重指针
        curr_weight_ptr = (weight_ptr + (weight_c_stride * input_c_offset)[:, None] + (weight_h_stride * h) +
                           (weight_w_stride * w))

        # 掩码计算
        input_mask = ((in_n_idx < in_n)[:, None] & (input_c_offset < weight_c)[None, :] & (0 <= input_h_offset)[:, None]
                      & (input_h_offset < input_h)[:, None] & (0 <= input_w_offset)[:, None] &
                      (input_w_offset < input_w)[:, None])

        weight_mask = (input_c_offset < weight_c)[:, None] & (output_c_offset < out_per_group_c)[None, :]

        # 加载数据
        input_block = tl.load(curr_input_ptr, mask=input_mask)
        weight_block = tl.load(curr_weight_ptr, mask=weight_mask)

        # 矩阵乘法累加
        accum += tl.dot(input_block, weight_block, allow_tf32=False)

    # 处理偏置
    bias_ptr += (pid_group * out_per_group_c)[None, :] + output_c_offset[None, :]
    mask_bias = (output_c_offset < out_per_group_c)[None, :]
    bias = tl.load(bias_ptr, mask_bias).to(tl.float32)
    accum += bias

    # 计算结果存储位置
    output_ptr += ((output_n_stride * in_n_idx)[:, None] + (output_c_stride *
                                                            (pid_group * out_per_group_c + output_c_offset))[None, :] +
                   (output_h_stride * out_h_idx)[:, None] + (output_w_stride * out_w_idx)[:, None])

    # 输出掩码
    output_mask = ((in_n_idx < in_n)[:, None] & (output_c_offset < out_per_group_c)[None, :] &
                   (out_h_idx < out_h)[:, None] & (out_w_idx < out_w)[:, None])

    # 存储结果
    tl.store(output_ptr, accum, mask=output_mask)


# 计算输出尺寸的辅助函数
def conv2d_output_size(in_size, kernel_size, stride, padding, dilation):
    return (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


# Triton Conv2D函数封装
def triton_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    assert weight.ndim == 4, "Weights must be 4D"
    assert bias is None or bias.ndim == 1, "Bias must be 1D"
    assert input.shape[1] == groups * weight.shape[1], "Incompatible input and weights shape"
    assert bias is None or weight.shape[0] == bias.shape[0], "Incompatible weights and bias shape"

    # 统一参数格式
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

    in_n, _, input_h, input_w = input.shape
    out_c, weight_c, weight_h, weight_w = weight.shape

    # 计算输出尺寸
    out_h = conv2d_output_size(input_h, weight_h, stride[0], padding[0], dilation[0])
    out_w = conv2d_output_size(input_w, weight_w, stride[1], padding[1], dilation[1])

    # 准备输出张量
    output = torch.empty((in_n, out_c, out_h, out_w), device=input.device, dtype=input.dtype)

    # 如果没有偏置，创建零偏置
    if bias is None:
        bias = torch.zeros(out_c, device=input.device, dtype=input.dtype)

    # 计算网格大小
    grid = lambda META: (
        triton.cdiv(in_n * out_h * out_w, META['BLOCK_NI_HO_WO']),
        triton.cdiv(int(out_c // groups), META['BLOCK_CO']),
        groups,
    )

    # 调用kernel
    conv2d_kernel[grid](
        input,
        weight,
        output,
        bias,
        in_n,
        input_h,
        input_w,
        out_c,
        out_h,
        out_w,
        *input.stride(),
        *weight.stride(),
        *output.stride(),
        weight_c,
        weight_h,
        weight_w,
        stride[0],
        padding[0],
        dilation[0],
        groups,
    )

    return output


# 测试用例
@pytest.mark.parametrize("shape,kernel,groups", [
    ((1, 2, 5, 5), (1, 2, 3, 3), 1),
    ((2, 3, 9, 9), (1, 3, 3, 3), 1),
    ((32, 8, 8, 8), (32, 8, 2, 2), 1),
    ((18, 16, 4, 4), (16, 16, 2, 2), 1),
    ((9, 16, 4, 4), (128, 4, 2, 2), 4),
])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("bias", [True, False])
def test_triton_conv2d(shape, kernel, groups, stride, padding, dtype, bias):
    # 跳过float16测试如果设备不支持
    if dtype == torch.float16 and not torch.cuda.is_available():
        pytest.skip("CUDA not available for float16 test")

    # 创建输入数据
    torch.manual_seed(42)
    input = torch.randn(shape, dtype=dtype, device='cuda' if torch.cuda.is_available() else 'cpu')
    weight = torch.randn(kernel, dtype=dtype, device=input.device)

    # 创建偏置
    if bias:
        bias_tensor = torch.randn(kernel[0], dtype=dtype, device=input.device)
    else:
        bias_tensor = None

    # 计算PyTorch参考输出
    torch_out = torch.nn.functional.conv2d(input, weight, bias=bias_tensor, stride=stride, padding=padding, dilation=1,
                                           groups=groups)

    # 计算Triton输出
    triton_out = triton_conv2d(input, weight, bias=bias_tensor, stride=stride, padding=padding, dilation=1,
                               groups=groups)

    # 验证结果
    if dtype == torch.float32:
        rtol, atol = 1e-4, 1e-5
    else:
        rtol, atol = 1e-2, 1e-3

    torch.testing.assert_close(triton_out, torch_out, rtol=rtol, atol=atol)


if __name__ == "__main__":
    # 简单演示
    if torch.cuda.is_available():
        device = 'cuda'
        print("Running demo on CUDA")
    else:
        device = 'cpu'
        print("Running demo on CPU")

    # 创建测试数据
    input = torch.randn(1, 3, 16, 16, device=device)
    weight = torch.randn(6, 3, 3, 3, device=device)
    bias = torch.randn(6, device=device)

    # 运行Triton实现
    output = triton_conv2d(input, weight, bias=bias, stride=1, padding=1)
    print("Triton Conv2D output shape:", output.shape)

    # 运行PyTorch实现
    torch_output = torch.nn.functional.conv2d(input, weight, bias=bias, stride=1, padding=1)
    print("PyTorch Conv2D output shape:", torch_output.shape)

    # 比较结果
    print("Max difference:", (output - torch_output).abs().max().item())
