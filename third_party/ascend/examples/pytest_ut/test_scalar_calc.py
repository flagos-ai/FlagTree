import torch
import torch_npu
import triton
import triton.language as tl
import pytest
import test_common

### add
@pytest.mark.parametrize('param_list',
                         [
                             ['float32', 16]
                         ]
                         )
def test_scalar_add_calc(param_list):

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        idx = 0
        tmp0 = tl.load(in_ptr0 + idx)
        tmp1 = tmp0 + 2.0
        tl.store(out_ptr0 + idx, tmp1)

    def torch_func(x0):
        y = x0[0]
        y = y + 2.0
        return torch.tensor(y)

    dtype, N = param_list
    x0 = test_common.generate_tensor((N,), dtype).npu()
    y_ref = torch_func(x0)
    y_cal = test_common.generate_tensor((1,), dtype).npu()
    triton_kernel[1, 1, 1](y_cal, x0, N=N)
    test_common.validate_cmp(dtype, y_cal[0], y_ref)

### sub
@pytest.mark.parametrize('param_list',
                         [
                             ['float32', 16]
                         ]
                         )
def test_scalar_sub_calc(param_list):

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        idx = 0
        tmp0 = tl.load(in_ptr0 + idx)
        tmp1 = tmp0 - 2.0
        tl.store(out_ptr0 + idx, tmp1)

    def torch_func(x0):
        y = x0[0]
        y = y - 2.0
        return torch.tensor(y)

    dtype, N = param_list
    x0 = test_common.generate_tensor((N,), dtype).npu()
    y_ref = torch_func(x0)
    y_cal = test_common.generate_tensor((1,), dtype).npu()
    triton_kernel[1, 1, 1](y_cal, x0, N=N)
    test_common.validate_cmp(dtype, y_cal[0], y_ref)

### mul
@pytest.mark.parametrize('param_list',
                         [
                             ['float32', 16]
                         ]
                         )
def test_scalar_mul_calc(param_list):

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        idx = 0
        tmp0 = tl.load(in_ptr0 + idx)
        tmp1 = tmp0 * 2.0
        tl.store(out_ptr0 + idx, tmp1)

    def torch_func(x0):
        y = x0[0]
        y = y * 2.0
        return torch.tensor(y)

    dtype, N = param_list
    x0 = test_common.generate_tensor((N,), dtype).npu()
    y_ref = torch_func(x0)
    y_cal = test_common.generate_tensor((1,), dtype).npu()
    triton_kernel[1, 1, 1](y_cal, x0, N=N)
    test_common.validate_cmp(dtype, y_cal[0], y_ref)

### div
@pytest.mark.parametrize('param_list',
                         [
                             ['float32', 16]
                         ]
                         )
def test_scalar_div_calc(param_list):

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        idx = 0
        tmp0 = tl.load(in_ptr0 + idx)
        tmp1 = tmp0 / 2.0
        tl.store(out_ptr0 + idx, tmp1)

    def torch_func(x0):
        y = x0[0]
        y = y / 2.0
        return torch.tensor(y)

    dtype, N = param_list
    x0 = test_common.generate_tensor((N,), dtype).npu()
    y_ref = torch_func(x0)
    y_cal = test_common.generate_tensor((1,), dtype).npu()
    triton_kernel[1, 1, 1](y_cal, x0, N=N)
    test_common.validate_cmp(dtype, y_cal[0], y_ref)

### remf
@pytest.mark.parametrize('param_list',
                         [
                             ['float32', 16]
                         ]
                         )
def test_scalar_remf_calc(param_list):

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        idx = 0
        tmp0 = tl.load(in_ptr0 + idx)
        tmp1 = tmp0 % 2.0
        tl.store(out_ptr0 + idx, tmp1)

    def torch_func(x0):
        y = x0[0]
        y = y % 2.0
        return torch.tensor(y)

    dtype, N = param_list
    x0 = test_common.generate_tensor((N,), dtype).npu()
    y_ref = torch_func(x0)
    y_cal = test_common.generate_tensor((1,), dtype).npu()
    triton_kernel[1, 1, 1](y_cal, x0, N=N)
    test_common.validate_cmp(dtype, y_cal[0], y_ref)

### negf
@pytest.mark.parametrize('param_list',
                         [
                             ['float32', 16]
                         ]
                         )
def test_scalar_negf_calc(param_list):

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        idx = 0
        tmp0 = tl.load(in_ptr0 + idx)
        tmp1 = -tmp0
        tl.store(out_ptr0 + idx, tmp1)

    def torch_func(x0):
        y = x0[0]
        y = -y
        return torch.tensor(y)

    dtype, N = param_list
    x0 = test_common.generate_tensor((N,), dtype).npu()
    y_ref = torch_func(x0)
    y_cal = test_common.generate_tensor((1,), dtype).npu()
    triton_kernel[1, 1, 1](y_cal, x0, N=N)
    test_common.validate_cmp(dtype, y_cal[0], y_ref)

### cmpf
@pytest.mark.parametrize('param_list',
                         [
                             ['float32', 16]
                         ]
                         )
def test_scalar_cmpf_calc(param_list):

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        idx = 0
        tmp0 = tl.load(in_ptr0 + idx)
        tmp1 = (tmp0 > 0.5).to(tmp0.dtype)
        tl.store(out_ptr0 + idx, tmp1)

    def torch_func(x0):
        y = x0[0]
        y = (y > 0.5).to(y.dtype)
        return torch.tensor(y)

    dtype, N = param_list
    x0 = test_common.generate_tensor((N,), dtype).npu()
    y_ref = torch_func(x0)
    y_cal = test_common.generate_tensor((1,), dtype).npu()
    triton_kernel[1, 1, 1](y_cal, x0, N=N)
    test_common.validate_cmp(dtype, y_cal[0], y_ref)

### ceil
@pytest.mark.parametrize('param_list',
                         [
                             ['float32', 16]
                         ]
                         )
def test_scalar_ceil_calc(param_list):

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        idx = 0
        tmp0 = tl.load(in_ptr0 + idx)
        tmp1 = tl.math.ceil(tmp0)
        tl.store(out_ptr0 + idx, tmp1)

    def torch_func(x0):
        y = x0[0]
        y = torch.ceil(y)
        return torch.tensor(y)

    dtype, N = param_list
    x0 = test_common.generate_tensor((N,), dtype).npu()
    y_ref = torch_func(x0)
    y_cal = test_common.generate_tensor((1,), dtype).npu()
    triton_kernel[1, 1, 1](y_cal, x0, N=N)
    test_common.validate_cmp(dtype, y_cal[0], y_ref)

### floor
@pytest.mark.parametrize('param_list',
                         [
                             ['float32', 16]
                         ]
                         )
def test_scalar_floor_calc(param_list):

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        idx = 0
        tmp0 = tl.load(in_ptr0 + idx)
        tmp1 = tl.math.floor(tmp0)
        tl.store(out_ptr0 + idx, tmp1)

    def torch_func(x0):
        y = x0[0]
        y = torch.floor(y)
        return torch.tensor(y)

    dtype, N = param_list
    x0 = test_common.generate_tensor((N,), dtype).npu()
    y_ref = torch_func(x0)
    y_cal = test_common.generate_tensor((1,), dtype).npu()
    triton_kernel[1, 1, 1](y_cal, x0, N=N)
    test_common.validate_cmp(dtype, y_cal[0], y_ref)

### maximum(propagate_nan == tl.PropagateNan.ALL)
# setting propagate_nan=tl.PropagateNan.ALL to generate arith::MaximumFOp
@pytest.mark.parametrize('param_list',
                         [
                             ['float32', 16]
                         ]
                         )
def test_scalar_maximum_nanall_calc(param_list):

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        tl.static_assert(N > 1)
        tmp0 = tl.load(in_ptr0 + 0)
        tmp1 = tl.load(in_ptr0 + 1)
        tmp1 = tl.maximum(tmp0, tmp1, propagate_nan=tl.PropagateNan.ALL)
        tl.store(out_ptr0 + 0, tmp1)

    def torch_func(x0):
        y0 = x0[0]
        y1 = x0[1]
        y = torch.maximum(y0, y1)
        return torch.tensor(y)

    dtype, N = param_list
    x0 = test_common.generate_tensor((N,), dtype).npu()
    y_ref = torch_func(x0)
    y_cal = test_common.generate_tensor((1,), dtype).npu()
    triton_kernel[1, 1, 1](y_cal, x0, N=N)
    test_common.validate_cmp(dtype, y_cal[0], y_ref)

### maximum(propagate_nan == tl.PropagateNan.NONE)
# setting propagate_nan=tl.PropagateNan.NONE to generate arith::MaxNumFOp
@pytest.mark.parametrize('param_list',
                         [
                             ['float32', 16]
                         ]
                         )
def test_scalar_maximum_nannone_calc(param_list):

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        tl.static_assert(N > 1)
        tmp0 = tl.load(in_ptr0 + 0)
        tmp1 = tl.load(in_ptr0 + 1)
        tmp1 = tl.maximum(tmp0, tmp1, propagate_nan=tl.PropagateNan.ALL)
        tl.store(out_ptr0 + 0, tmp1)

    def torch_func(x0):
        y0 = x0[0]
        y1 = x0[1]
        y = torch.fmax(y0, y1)
        return torch.tensor(y)

    dtype, N = param_list
    x0 = test_common.generate_tensor((N,), dtype).npu()
    y_ref = torch_func(x0)
    y_cal = test_common.generate_tensor((1,), dtype).npu()
    triton_kernel[1, 1, 1](y_cal, x0, N=N)
    test_common.validate_cmp(dtype, y_cal[0], y_ref)

### minimum(propagate_nan == tl.PropagateNan.ALL)
# setting propagate_nan=tl.PropagateNan.ALL to generate arith::MinimumFOp
@pytest.mark.parametrize('param_list',
                         [
                             ['float32', 16]
                         ]
                         )
def test_scalar_minimum_nanall_calc(param_list):

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        tl.static_assert(N > 1)
        tmp0 = tl.load(in_ptr0 + 0)
        tmp1 = tl.load(in_ptr0 + 1)
        tmp1 = tl.minimum(tmp0, tmp1, propagate_nan=tl.PropagateNan.ALL)
        tl.store(out_ptr0 + 0, tmp1)

    def torch_func(x0):
        y0 = x0[0]
        y1 = x0[1]
        y = torch.minimum(y0, y1)
        return torch.tensor(y)

    dtype, N = param_list
    x0 = test_common.generate_tensor((N,), dtype).npu()
    y_ref = torch_func(x0)
    y_cal = test_common.generate_tensor((1,), dtype).npu()
    triton_kernel[1, 1, 1](y_cal, x0, N=N)
    test_common.validate_cmp(dtype, y_cal[0], y_ref)

### minimum(propagate_nan == tl.PropagateNan.NONE)
# setting propagate_nan=tl.PropagateNan.NONE to generate arith::MinNumFOp
@pytest.mark.parametrize('param_list',
                         [
                             ['float32', 16]
                         ]
                         )
def test_scalar_minimum_nannone_calc(param_list):

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        tl.static_assert(N > 1)
        tmp0 = tl.load(in_ptr0 + 0)
        tmp1 = tl.load(in_ptr0 + 1)
        tmp1 = tl.minimum(tmp0, tmp1, propagate_nan=tl.PropagateNan.NONE)
        tl.store(out_ptr0 + 0, tmp1)

    def torch_func(x0):
        y0 = x0[0]
        y1 = x0[1]
        y = torch.fmin(y0, y1)
        return torch.tensor(y)

    dtype, N = param_list
    x0 = test_common.generate_tensor((N,), dtype).npu()
    y_ref = torch_func(x0)
    y_cal = test_common.generate_tensor((1,), dtype).npu()
    triton_kernel[1, 1, 1](y_cal, x0, N=N)
    test_common.validate_cmp(dtype, y_cal[0], y_ref)

### extf
@pytest.mark.parametrize('param_list',
                         [
                             ['float16', 'float32', 16]
                         ]
                         )
def test_scalar_extf_calc(param_list):

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        idx = 0
        tmp0 = tl.load(in_ptr0 + idx)
        tmp1 = tmp0.to(tl.float32)
        tl.store(out_ptr0 + idx, tmp1)

    def torch_func(x0):
        y = x0[0]
        y = y.to(torch.float32)
        return torch.tensor(y)

    src_dtype, dst_dtype, N = param_list
    x0 = test_common.generate_tensor((N,), src_dtype).npu()
    y_ref = torch_func(x0)
    y_cal = test_common.generate_tensor((1,), dst_dtype).npu()
    triton_kernel[1, 1, 1](y_cal, x0, N=N)
    test_common.validate_cmp(dst_dtype, y_cal[0], y_ref)

### truncf
@pytest.mark.parametrize('param_list',
                         [
                             ['float32', 'float16', 16]
                         ]
                         )
def test_scalar_truncf_calc(param_list):

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        idx = 0
        tmp0 = tl.load(in_ptr0 + idx)
        tmp1 = tmp0.to(tl.float16)
        tl.store(out_ptr0 + idx, tmp1)

    def torch_func(x0):
        y = x0[0]
        y = y.to(torch.float16)
        return torch.tensor(y)

    src_dtype, dst_dtype, N = param_list
    x0 = test_common.generate_tensor((N,), src_dtype).npu()
    y_ref = torch_func(x0)
    y_cal = test_common.generate_tensor((1,), dst_dtype).npu()
    triton_kernel[1, 1, 1](y_cal, x0, N=N)
    test_common.validate_cmp(dst_dtype, y_cal[0], y_ref)

### exp
@pytest.mark.parametrize('param_list',
                         [
                             ['float32', 16]
                         ]
                         )
def test_scalar_exp_calc(param_list):

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        idx = 0
        tmp0 = tl.load(in_ptr0 + idx)
        tmp1 = tl.math.exp(tmp0)
        tl.store(out_ptr0 + idx, tmp1)

    def torch_func(x0):
        y = x0[0]
        y = torch.exp(y)
        return torch.tensor(y)

    dtype, N = param_list
    x0 = test_common.generate_tensor((N,), dtype).npu()
    y_ref = torch_func(x0)
    y_cal = test_common.generate_tensor((1,), dtype).npu()
    triton_kernel[1, 1, 1](y_cal, x0, N=N)
    test_common.validate_cmp(dtype, y_cal[0], y_ref)

### exp2
@pytest.mark.parametrize('param_list',
                         [
                             ['float32', 16]
                         ]
                         )
def test_scalar_exp_calc(param_list):

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        idx = 0
        tmp0 = tl.load(in_ptr0 + idx)
        tmp1 = tl.math.exp2(tmp0)
        tl.store(out_ptr0 + idx, tmp1)

    def torch_func(x0):
        y = x0[0]
        y = torch.exp2(y)
        return torch.tensor(y)

    dtype, N = param_list
    x0 = test_common.generate_tensor((N,), dtype).npu()
    y_ref = torch_func(x0)
    y_cal = test_common.generate_tensor((1,), dtype).npu()
    triton_kernel[1, 1, 1](y_cal, x0, N=N)
    test_common.validate_cmp(dtype, y_cal[0], y_ref)

### log
@pytest.mark.parametrize('param_list',
                         [
                             ['float32', 16]
                         ]
                         )
def test_scalar_log_calc(param_list):

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        idx = 0
        tmp0 = tl.load(in_ptr0 + idx)
        tmp0 = tl.abs(tmp0)
        tmp1 = tl.log(tmp0)
        tl.store(out_ptr0 + idx, tmp1)

    def torch_func(x0):
        y = x0[0]
        y = torch.abs(y)
        y = torch.log(y)
        return torch.tensor(y)

    dtype, N = param_list
    x0 = test_common.generate_tensor((N,), dtype).npu()
    y_ref = torch_func(x0)
    y_cal = test_common.generate_tensor((1,), dtype).npu()
    triton_kernel[1, 1, 1](y_cal, x0, N=N)
    test_common.validate_cmp(dtype, y_cal[0], y_ref)

### log2
@pytest.mark.parametrize('param_list',
                         [
                             ['float32', 16]
                         ]
                         )
def test_scalar_log2_calc(param_list):

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        idx = 0
        tmp0 = tl.load(in_ptr0 + idx)
        tmp0 = tl.abs(tmp0)
        tmp1 = tl.log2(tmp0)
        tl.store(out_ptr0 + idx, tmp1)

    def torch_func(x0):
        y = x0[0]
        y = torch.abs(y)
        y = torch.log2(y)
        return torch.tensor(y)

    dtype, N = param_list
    x0 = test_common.generate_tensor((N,), dtype).npu()
    y_ref = torch_func(x0)
    y_cal = test_common.generate_tensor((1,), dtype).npu()
    triton_kernel[1, 1, 1](y_cal, x0, N=N)
    test_common.validate_cmp(dtype, y_cal[0], y_ref)

### sin
@pytest.mark.parametrize('param_list',
                         [
                             ['float32', 16]
                         ]
                         )
def test_scalar_sin_calc(param_list):

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        idx = 0
        tmp0 = tl.load(in_ptr0 + idx)
        tmp1 = tl.sin(tmp0)
        tl.store(out_ptr0 + idx, tmp1)

    def torch_func(x0):
        y = x0[0]
        y = torch.sin(y)
        return torch.tensor(y)

    dtype, N = param_list
    x0 = test_common.generate_tensor((N,), dtype).npu()
    y_ref = torch_func(x0)
    y_cal = test_common.generate_tensor((1,), dtype).npu()
    triton_kernel[1, 1, 1](y_cal, x0, N=N)
    test_common.validate_cmp(dtype, y_cal[0], y_ref)

### cos
@pytest.mark.parametrize('param_list',
                         [
                             ['float32', 16]
                         ]
                         )
def test_scalar_cos_calc(param_list):

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        idx = 0
        tmp0 = tl.load(in_ptr0 + idx)
        tmp1 = tl.cos(tmp0)
        tl.store(out_ptr0 + idx, tmp1)

    def torch_func(x0):
        y = x0[0]
        y = torch.cos(y)
        return torch.tensor(y)

    dtype, N = param_list
    x0 = test_common.generate_tensor((N,), dtype).npu()
    y_ref = torch_func(x0)
    y_cal = test_common.generate_tensor((1,), dtype).npu()
    triton_kernel[1, 1, 1](y_cal, x0, N=N)
    test_common.validate_cmp(dtype, y_cal[0], y_ref)

### abs
@pytest.mark.parametrize('param_list',
                         [
                             ['float32', 16]
                         ]
                         )
def test_scalar_abs_calc(param_list):

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        idx = 0
        tmp0 = tl.load(in_ptr0 + idx)
        tmp1 = tl.abs(tmp0)
        tl.store(out_ptr0 + idx, tmp1)

    def torch_func(x0):
        y = x0[0]
        y = torch.abs(y)
        return torch.tensor(y)

    dtype, N = param_list
    x0 = test_common.generate_tensor((N,), dtype).npu()
    y_ref = torch_func(x0)
    y_cal = test_common.generate_tensor((1,), dtype).npu()
    triton_kernel[1, 1, 1](y_cal, x0, N=N)
    test_common.validate_cmp(dtype, y_cal[0], y_ref)

### erf
@pytest.mark.parametrize('param_list',
                         [
                             ['float32', 16]
                         ]
                         )
def test_scalar_erf_calc(param_list):

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        idx = 0
        tmp0 = tl.load(in_ptr0 + idx)
        tmp1 = tl.erf(tmp0)
        tl.store(out_ptr0 + idx, tmp1)

    def torch_func(x0):
        y = x0[0]
        y = torch.erf(y)
        return torch.tensor(y)

    dtype, N = param_list
    x0 = test_common.generate_tensor((N,), dtype).npu()
    y_ref = torch_func(x0)
    y_cal = test_common.generate_tensor((1,), dtype).npu()
    triton_kernel[1, 1, 1](y_cal, x0, N=N)
    test_common.validate_cmp(dtype, y_cal[0], y_ref)

### sqrt
@pytest.mark.parametrize('param_list',
                         [
                             ['float32', 16]
                         ]
                         )
def test_scalar_sqrt_calc(param_list):

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        idx = 0
        tmp0 = tl.load(in_ptr0 + idx)
        tmp0 = tl.abs(tmp0)
        tmp1 = tl.math.sqrt(tmp0)
        tl.store(out_ptr0 + idx, tmp1)

    def torch_func(x0):
        y = x0[0]
        y = torch.abs(y)
        y = torch.sqrt(y)
        return torch.tensor(y)

    dtype, N = param_list
    x0 = test_common.generate_tensor((N,), dtype).npu()
    y_ref = torch_func(x0)
    y_cal = test_common.generate_tensor((1,), dtype).npu()
    triton_kernel[1, 1, 1](y_cal, x0, N=N)
    test_common.validate_cmp(dtype, y_cal[0], y_ref)

### rsqrt
@pytest.mark.parametrize('param_list',
                         [
                             ['float32', 16]
                         ]
                         )
def test_scalar_rsqrt_calc(param_list):

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        idx = 0
        tmp0 = tl.load(in_ptr0 + idx)
        tmp0 = tl.abs(tmp0)
        tmp1 = tl.math.rsqrt(tmp0)
        tl.store(out_ptr0 + idx, tmp1)

    def torch_func(x0):
        y = x0[0]
        y = torch.abs(y)
        y = torch.rsqrt(y)
        return y.clone().detach()

    dtype, N = param_list
    x0 = test_common.generate_tensor((N,), dtype).npu()
    y_ref = torch_func(x0)
    y_cal = test_common.generate_tensor((1,), dtype).npu()
    triton_kernel[1, 1, 1](y_cal, x0, N=N)
    test_common.validate_cmp(dtype, y_cal[0], y_ref)

### tanh
@pytest.mark.parametrize('param_list',
                         [
                             ['float32', 16]
                         ]
                         )
def test_scalar_tanh_calc(param_list):

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        idx = 0
        tmp0 = tl.load(in_ptr0 + idx)
        tmp1 = tl.math.tanh(tmp0)
        tl.store(out_ptr0 + idx, tmp1)

    def torch_func(x0):
        y = x0[0]
        y = torch.tanh(y)
        return y.clone().detach()

    dtype, N = param_list
    x0 = test_common.generate_tensor((N,), dtype).npu()
    y_ref = torch_func(x0)
    y_cal = test_common.generate_tensor((1,), dtype).npu()
    triton_kernel[1, 1, 1](y_cal, x0, N=N)
    test_common.validate_cmp(dtype, y_cal[0], y_ref)

### sum
@pytest.mark.parametrize('param_list',
                         [
                             ['float32', 16]
                         ]
                         )
def test_scalar_sum_calc(param_list):

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        tmp0 = tl.load(in_ptr0 + tl.arange(0, N))
        tmp1 = tl.sum(tmp0, 0)
        tl.store(out_ptr0 + 0, tmp1)

    def torch_func(x0):
        y = torch.sum(x0, 0)
        return torch.tensor(y)

    dtype, N = param_list
    x0 = test_common.generate_tensor((N,), dtype).npu()
    y_ref = torch_func(x0)
    y_cal = test_common.generate_tensor((1,), dtype).npu()
    triton_kernel[1, 1, 1](y_cal, x0, N=N)
    test_common.validate_cmp(dtype, y_cal[0], y_ref)
