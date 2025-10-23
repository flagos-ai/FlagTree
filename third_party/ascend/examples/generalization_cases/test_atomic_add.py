import math
import pytest
import torch
import triton

import triton.language as tl

import test_common
from test_common import TestUtils
filtered_dtype = [dtype for dtype in TestUtils.full_dtype if dtype not in {'int64', 'bool'}]


@triton.jit
def atomic_add(in_ptr0, out_ptr0, out_ptr1, n_elements, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE
    index = offset + tl.arange(0, BLOCK_SIZE)[:]
    xmask = index < n_elements

    tmp0 = tl.load(in_ptr0 + (index), xmask)
    tmp1 = tl.load(out_ptr0 + (index), xmask)
    tl.atomic_add(out_ptr1 + (index), tmp0, xmask)
    tl.atomic_add(out_ptr1 + (index), tmp1, xmask)



@triton.jit
def atomic_add_broadcast(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    x = tl.load(x_ptr)  # x is scalar or 1D, no mask needed
    
    # Compute y indices
    y_offset = pid * BLOCK_SIZE
    y_indices = y_offset + tl.arange(0, BLOCK_SIZE)
    y_mask = y_indices < n_elements
    
    y_value = tl.load(y_ptr + y_indices, y_mask)
    # Atomic add: y += x (broadcasted)
    tl.atomic_add(out_ptr + y_indices, y_value, mask=y_mask)
    tl.atomic_add(out_ptr + y_indices, x, mask=y_mask)


# 定义不同测试场景的参数组合 (x_shape, y_shape, BLOCK_SIZE)
test_cases = [
    ((1, 1, 1, 1), (1, 1, 1, 4), 4),
    ((1, 1, 1, 3), (1, 5, 1, 3), 5),
    ((3,), (2, 3, 3, 3, 3), 81),
    ((3,), (2, 3, 3, 3), 27),
    ((3,), (2, 3, 3), 9),
    ((3,), (2, 3), 3),
]


def promote_dtype(x_dtype, y_dtype):
    """
    如果 y 的精度低于 x, 则提升 y 的精度以匹配 x。
    """
    # 如果两个数据类型一致，直接返回
    if x_dtype == y_dtype:
        return y_dtype
    
    # 构建类型的优先级列表（从低到高）
    priority = [
        torch.int8, torch.int16, torch.int32,
        torch.float16, torch.bfloat16, torch.float32
    ]

    # 查找两种类型在优先级列表中的位置
    x_priority = priority.index(x_dtype)
    y_priority = priority.index(y_dtype)

    # 如果y的优先级比x小，则提升到x的类型
    if y_priority < x_priority:
        return x_dtype
    else:
        return y_dtype


@pytest.mark.parametrize('x_dtype_str', filtered_dtype)
@pytest.mark.parametrize('y_dtype_str', filtered_dtype)
@pytest.mark.parametrize('x_shape, y_shape, BLOCK_SIZE', test_cases)
def test_atomic_add_broadcast_combined(x_dtype_str, y_dtype_str, x_shape, y_shape, BLOCK_SIZE):
    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)
    # 先构造 x0
    x0 = torch.full(x_shape, 83.0000, dtype=x_dtype).npu()

    y_raw_dtype = eval('torch.' + y_dtype_str)

    out_dtype = promote_dtype(x_dtype, y_raw_dtype)
    if out_dtype == torch.bfloat16:
        out_dtype = torch.float32

    # 构造y和out
    y = torch.full(y_shape, -105, dtype=y_raw_dtype).npu()
    out = torch.full(y_shape, 0, dtype=out_dtype).npu()

    # 保存副本用于验证
    x_temp = x0.clone()
    y_temp = y.clone()
    out_temp = out.clone()
    
    # 计算网格大小和元素总数
    n_elements = y.numel()
    grid = (n_elements // BLOCK_SIZE,)  # 自动计算需要的线程块数量
    
    # 调用 Triton 核函数
    atomic_add_broadcast[grid](
        x_ptr=x0,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # 验证结果：y += x (广播加法)
    expected = out_temp + y_temp + x_temp
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize('shape', TestUtils.test_shape2d + TestUtils.test_shape1d)
@pytest.mark.parametrize('x_dtype_str', filtered_dtype)
@pytest.mark.parametrize('y_dtype_str', filtered_dtype)
def test_atomic_add(x_dtype_str, y_dtype_str, shape):
    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)
    y_dtype = eval('torch.' + y_dtype_str)

    x0 = test_common.generate_tensor(shape, x_dtype_str).npu()
    x1 = test_common.generate_tensor(shape, y_dtype_str).npu()
    out_dtype = promote_dtype(x_dtype, y_dtype)
    if out_dtype == torch.bfloat16:
        out_dtype = torch.float32
    y = torch.full(x1.shape, 0, dtype=out_dtype).npu()

    # 保存副本用于验证
    x0_temp = x0.clone()
    x1_temp = x1.clone()
    y_temp = y.clone()
    
    if len(shape) == 2:
        n_elements = shape[0] * shape[1]
        atomic_add[shape[0], 1, 1](x0, x1, y, n_elements, BLOCK_SIZE=shape[1])
    elif len(shape) == 1:
        n_elements = shape[0]
        BLOCK_SIZE = min(1024, shape[0]) # 1024:限制最大线程块大小
        grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE # 向上取整
        atomic_add[grid_size, 1, 1](x0, x1, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    expected = y_temp + x1_temp + x0_temp
    torch.testing.assert_close(y, expected)


# 3d
@pytest.mark.parametrize('shape', TestUtils.test_shape3d)
@pytest.mark.parametrize('x_dtype_str', filtered_dtype)
@pytest.mark.parametrize('y_dtype_str', filtered_dtype)
def test_atomic_add_3d(x_dtype_str, y_dtype_str, shape):
    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)
    y_dtype = eval('torch.' + y_dtype_str)

    x0 = test_common.generate_tensor(shape, x_dtype_str).npu()
    x1 = test_common.generate_tensor(shape, y_dtype_str).npu()
    out_dtype = promote_dtype(x_dtype, y_dtype)
    if out_dtype == torch.bfloat16:
        out_dtype = torch.float32
    y = torch.full(x1.shape, 0, dtype=out_dtype).npu()

    # 保存副本用于验证
    x0_temp = x0.clone()
    x1_temp = x1.clone()
    y_temp = y.clone()

    n_elements = shape[0] * shape[1] * shape[2]
    atomic_add[1, 1, 1](x0, x1, y, n_elements, BLOCK_SIZE=shape[0] * shape[1] * shape[2])

    expected = y_temp + x1_temp + x0_temp
    torch.testing.assert_close(y, expected)


@triton.jit
def atomic_add_multi_d(in_ptr0, out_ptr0, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, MB: tl.constexpr, NB: tl.constexpr):
    offsets = tl.arange(0, XB) * (YB * ZB * MB * NB)
    if (YB * ZB * MB * NB) > 1:
        offsets = offsets[:, None] + tl.arange(0, YB)[None, :] * (ZB * MB * NB)
    if (ZB * MB * NB) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, ZB)[None, None, :] * (MB * NB)
    if (MB * NB) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, MB)[None, None, None, :] * NB
    if NB > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, NB)[None, None, None, None, :]
    
    tmp0 = tl.load(in_ptr0 + offsets)
    tl.atomic_add(out_ptr0 + offsets, tmp0)


# multi_d
@pytest.mark.shape_4d_5d
@pytest.mark.parametrize('shape', [
    (2, 4, 8, 4),
    (8, 4, 2, 4),
    (2, 8, 2, 2),
    (2, 4, 8, 4, 2),
    (8, 4, 2, 4, 4),
    (2, 8, 2, 2, 2),
])
@pytest.mark.parametrize('dtype', filtered_dtype)
def test_atomic_add_4d_5d(dtype, shape):
    x0_value = 3
    x0 = torch.full(shape, x0_value, dtype=eval('torch.' + dtype)).npu()
    x1 = torch.full(shape, 2, dtype=eval('torch.' + dtype)).npu()

    x1_ref = x1 + x0_value

    triton_shape = [*shape]
    while len(triton_shape) < 5:
        triton_shape.append(1)
    atomic_add_multi_d[(1, )](x0, x1, *triton_shape)
    test_common.validate_cmp(dtype, x1, x1_ref)


@triton.jit
def atomic_add_5d(x_ptr, y_ptr, out_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, MB: tl.constexpr, NB: tl.constexpr,
                            XB1: tl.constexpr, YB1: tl.constexpr, ZB1: tl.constexpr, MB1: tl.constexpr, NB1: tl.constexpr):
    offsets = tl.arange(0, XB) * (YB * ZB * MB * NB)
    offsets = offsets[:, None] + tl.arange(0, YB)[None, :] * (ZB * MB * NB)
    offsets = offsets[:, :, None] + tl.arange(0, ZB)[None, None, :] * (MB * NB)
    offsets = offsets[:, :, :, None] + tl.arange(0, MB)[None, None, None, :] * NB
    offsets = offsets[:, :, :, :, None] + tl.arange(0, NB)[None, None, None, None, :]

    offsets1 = tl.arange(0, XB1) * (YB1 * ZB1 * MB1 * NB1)
    offsets1 = offsets1[:, None] + tl.arange(0, YB1)[None, :] * (ZB1 * MB1 * NB1)
    offsets1 = offsets1[:, :, None] + tl.arange(0, ZB1)[None, None, :] * (MB1 * NB1)
    offsets1 = offsets1[:, :, :, None] + tl.arange(0, MB1)[None, None, None, :] * NB1
    offsets1 = offsets1[:, :, :, :, None] + tl.arange(0, NB1)[None, None, None, None, :]
    
    tmp0 = tl.load(x_ptr + offsets)
    tmp1 = tl.load(y_ptr + offsets1)
    tl.atomic_add(out_ptr + offsets1, tmp0)
    tl.atomic_add(out_ptr + offsets1, tmp1)


@pytest.mark.parametrize('param_list',
    [
        [(1, 1, 2, 1, 1), (1, 1, 2, 1, 2)],
    ]
    )
@pytest.mark.parametrize('x_dtype_str', filtered_dtype)
@pytest.mark.parametrize('y_dtype_str', filtered_dtype)
def test_atomic_add_5d(x_dtype_str, y_dtype_str, param_list):
    x0_shape, y_shape = param_list

    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)
    y_dtype = eval('torch.' + y_dtype_str)
    if x_dtype == torch.int8 or x_dtype == torch.int16 or x_dtype == torch.int32:
        x0 = torch.randint(low=0, high=100, size=x0_shape, dtype=x_dtype).npu()
    else:
        x0 = torch.randn(x0_shape, dtype=eval('torch.' + x_dtype_str)).npu()

    if y_dtype == torch.int8 or y_dtype == torch.int16 or y_dtype == torch.int32:
        y = torch.randint(low=0, high=100, size=y_shape, dtype=y_dtype).npu()
    else:
        y = torch.randn(y_shape, dtype=eval('torch.' + y_dtype_str)).npu()

    out_dtype = promote_dtype(x_dtype, y_dtype)
    if out_dtype == torch.bfloat16:
        out_dtype = torch.float32
    out = torch.full(y_shape, 0, dtype=out_dtype).npu()

    x0_temp = x0.clone()
    y_temp = y.clone()
    out_temp = out.clone()

    triton_shape = [*x0_shape]
    while len(triton_shape) < 5:
        triton_shape.append(1)
    XB, YB, ZB, MB, NB = triton_shape

    triton_shape1 = [*y_shape]
    while len(triton_shape1) < 5:
        triton_shape1.append(1)
    XB1, YB1, ZB1, MB1, NB1 = triton_shape1

    atomic_add_5d[(1, )](
        x_ptr=x0,
        y_ptr=y, 
        out_ptr=out, 
        XB=XB, YB=YB, ZB=ZB, MB=MB, NB=NB,
        XB1=XB1, YB1=YB1, ZB1=ZB1, MB1=MB1, NB1=NB1,
        )
    
    expected = out_temp + y_temp + x0_temp
    torch.testing.assert_close(out, expected)


@triton.jit
def atomic_add_4d(x_ptr, y_ptr, out_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, MB: tl.constexpr,
                            XB1: tl.constexpr, YB1: tl.constexpr, ZB1: tl.constexpr, MB1: tl.constexpr):
    offsets = tl.arange(0, XB) * (YB * ZB * MB)
    offsets = offsets[:, None] + tl.arange(0, YB)[None, :] * (ZB * MB)
    offsets = offsets[:, :, None] + tl.arange(0, ZB)[None, None, :] * (MB)
    offsets = offsets[:, :, :, None] + tl.arange(0, MB)[None, None, None, :] 

    offsets1 = tl.arange(0, XB1) * (YB1 * ZB1 * MB1)
    offsets1 = offsets1[:, None] + tl.arange(0, YB1)[None, :] * (ZB1 * MB1)
    offsets1 = offsets1[:, :, None] + tl.arange(0, ZB1)[None, None, :] * (MB1)
    offsets1 = offsets1[:, :, :, None] + tl.arange(0, MB1)[None, None, None, :]
    
    tmp0 = tl.load(x_ptr + offsets)
    tmp1 = tl.load(y_ptr + offsets1)
    tl.atomic_add(out_ptr + offsets1, tmp0)
    tl.atomic_add(out_ptr + offsets1, tmp1)


@pytest.mark.parametrize('param_list',
    [
        [(1, 1, 2, 1), (1, 1, 2, 2)],
        [(1, 1, 1, 1), (1, 1, 2, 2)],
    ]
    )
@pytest.mark.parametrize('x_dtype_str', filtered_dtype)
@pytest.mark.parametrize('y_dtype_str', filtered_dtype)
def test_atomic_add_4d(x_dtype_str, y_dtype_str, param_list):
    x0_shape, y_shape = param_list

    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)
    y_dtype = eval('torch.' + y_dtype_str)
    if x_dtype == torch.int8 or x_dtype == torch.int16 or x_dtype == torch.int32:
        x0 = torch.randint(low=0, high=100, size=x0_shape, dtype=x_dtype).npu()
    else:
        x0 = torch.randn(x0_shape, dtype=eval('torch.' + x_dtype_str)).npu()

    if y_dtype == torch.int8 or y_dtype == torch.int16 or y_dtype == torch.int32:
        y = torch.randint(low=0, high=100, size=y_shape, dtype=y_dtype).npu()
    else:
        y = torch.randn(y_shape, dtype=eval('torch.' + y_dtype_str)).npu()

    out_dtype = promote_dtype(x_dtype, y_dtype)
    if out_dtype == torch.bfloat16:
        out_dtype = torch.float32
    out = torch.full(y_shape, 0, dtype=out_dtype).npu()

    x0_temp = x0.clone()
    y_temp = y.clone()
    out_temp = out.clone()

    triton_shape = [*x0_shape]
    while len(triton_shape) < 4:
        triton_shape.append(1)
    XB, YB, ZB, MB = triton_shape

    triton_shape1 = [*y_shape]
    while len(triton_shape1) < 4:
        triton_shape1.append(1)
    XB1, YB1, ZB1, MB1 = triton_shape1

    atomic_add_4d[(1, )](
        x_ptr=x0,
        y_ptr=y, 
        out_ptr=out, 
        XB=XB, YB=YB, ZB=ZB, MB=MB,
        XB1=XB1, YB1=YB1, ZB1=ZB1, MB1=MB1,
        )
    
    expected = out_temp + y_temp + x0_temp
    torch.testing.assert_close(out, expected)


@triton.jit
def atomic_add_3d(x_ptr, y_ptr, out_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr,
                            XB1: tl.constexpr, YB1: tl.constexpr, ZB1: tl.constexpr):
    offsets = tl.arange(0, XB) * (YB * ZB)
    offsets = offsets[:, None] + tl.arange(0, YB)[None, :] * (ZB)
    offsets = offsets[:, :, None] + tl.arange(0, ZB)[None, None, :]

    offsets1 = tl.arange(0, XB1) * (YB1 * ZB1)
    offsets1 = offsets1[:, None] + tl.arange(0, YB1)[None, :] * (ZB1)
    offsets1 = offsets1[:, :, None] + tl.arange(0, ZB1)[None, None, :]
    
    tmp0 = tl.load(x_ptr + offsets)
    tmp1 = tl.load(y_ptr + offsets1)
    tl.atomic_add(out_ptr + offsets1, tmp0)
    tl.atomic_add(out_ptr + offsets1, tmp1)


@pytest.mark.parametrize('param_list',
    [
        [(1, 1, 2), (1, 2, 2)],
    ]
    )
@pytest.mark.parametrize('x_dtype_str', filtered_dtype)
@pytest.mark.parametrize('y_dtype_str', filtered_dtype)
def test_atomic_add_3d_2(x_dtype_str, y_dtype_str, param_list):
    x0_shape, y_shape = param_list

    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)
    y_dtype = eval('torch.' + y_dtype_str)
    if x_dtype == torch.int8 or x_dtype == torch.int16 or x_dtype == torch.int32:
        x0 = torch.randint(low=0, high=100, size=x0_shape, dtype=x_dtype).npu()
    else:
        x0 = torch.randn(x0_shape, dtype=eval('torch.' + x_dtype_str)).npu()

    if y_dtype == torch.int8 or y_dtype == torch.int16 or y_dtype == torch.int32:
        y = torch.randint(low=0, high=100, size=y_shape, dtype=y_dtype).npu()
    else:
        y = torch.randn(y_shape, dtype=eval('torch.' + y_dtype_str)).npu()

    out_dtype = promote_dtype(x_dtype, y_dtype)
    if out_dtype == torch.bfloat16:
        out_dtype = torch.float32
    out = torch.full(y_shape, 0, dtype=out_dtype).npu()

    x0_temp = x0.clone()
    y_temp = y.clone()
    out_temp = out.clone()

    triton_shape = [*x0_shape]
    while len(triton_shape) < 3:
        triton_shape.append(1)
    XB, YB, ZB = triton_shape

    triton_shape1 = [*y_shape]
    while len(triton_shape1) < 3:
        triton_shape1.append(1)
    XB1, YB1, ZB1 = triton_shape1

    atomic_add_3d[(1, )](
        x_ptr=x0,
        y_ptr=y, 
        out_ptr=out, 
        XB=XB, YB=YB, ZB=ZB,
        XB1=XB1, YB1=YB1, ZB1=ZB1,
        )
    
    expected = out_temp + y_temp + x0_temp
    torch.testing.assert_close(out, expected)


@triton.jit
def atomic_add_2d(x_ptr, y_ptr, out_ptr, XB: tl.constexpr, YB: tl.constexpr,
                            XB1: tl.constexpr, YB1: tl.constexpr):
    offsets = tl.arange(0, XB) * (YB)
    offsets = offsets[:, None] + tl.arange(0, YB)[None, :]

    offsets1 = tl.arange(0, XB1) * (YB1)
    offsets1 = offsets1[:, None] + tl.arange(0, YB1)[None, :]
    
    tmp0 = tl.load(x_ptr + offsets)
    tmp1 = tl.load(y_ptr + offsets1)
    tl.atomic_add(out_ptr + offsets1, tmp0)
    tl.atomic_add(out_ptr + offsets1, tmp1)


@pytest.mark.parametrize('param_list',
    [
        [(1, 2), (2, 2)],
         [(1, 1), (2, 2)],
    ]
    )
@pytest.mark.parametrize('x_dtype_str', filtered_dtype)
@pytest.mark.parametrize('y_dtype_str', filtered_dtype)
def test_atomic_add_2d(x_dtype_str, y_dtype_str, param_list):
    x0_shape, y_shape = param_list

    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)
    y_dtype = eval('torch.' + y_dtype_str)
    if x_dtype == torch.int8 or x_dtype == torch.int16 or x_dtype == torch.int32:
        x0 = torch.randint(low=0, high=100, size=x0_shape, dtype=x_dtype).npu()
    else:
        x0 = torch.randn(x0_shape, dtype=eval('torch.' + x_dtype_str)).npu()

    if y_dtype == torch.int8 or y_dtype == torch.int16 or y_dtype == torch.int32:
        y = torch.randint(low=0, high=100, size=y_shape, dtype=y_dtype).npu()
    else:
        y = torch.randn(y_shape, dtype=eval('torch.' + y_dtype_str)).npu()

    out_dtype = promote_dtype(x_dtype, y_dtype)
    if out_dtype == torch.bfloat16:
        out_dtype = torch.float32
    out = torch.full(y_shape, 0, dtype=out_dtype).npu()

    x0_temp = x0.clone()
    y_temp = y.clone()
    out_temp = out.clone()

    triton_shape = [*x0_shape]
    while len(triton_shape) < 2:
        triton_shape.append(1)
    XB, YB = triton_shape

    triton_shape1 = [*y_shape]
    while len(triton_shape1) < 2:
        triton_shape1.append(1)
    XB1, YB1 = triton_shape1

    atomic_add_2d[(1, )](
        x_ptr=x0,
        y_ptr=y, 
        out_ptr=out, 
        XB=XB, YB=YB,
        XB1=XB1, YB1=YB1,
        )
    
    expected = out_temp + y_temp + x0_temp
    torch.testing.assert_close(out, expected)