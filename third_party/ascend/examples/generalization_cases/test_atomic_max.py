import random
import triton
import triton.language as tl
import torch
import pytest
import test_common
from test_common import TestUtils

@triton.jit
def triton_test_fn_atomic_max_dma(in_ptr0, in_ptr1, out_ptr1, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    index = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    mask = index < n_elements
    inp0 = tl.load(in_ptr0 + (index), mask)
    inp1 = tl.load(in_ptr1 + (index), mask)
    tmp1 = tl.atomic_max(out_ptr1 + (index), inp0, mask)
    tmp2 = tl.atomic_max(out_ptr1 + (index), inp1, mask)


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


# torch.max do not support int
@pytest.mark.parametrize('shape', random.sample(TestUtils.test_shape2d + TestUtils.test_shape1d, 5))
@pytest.mark.parametrize('x_dtype_str', ['float32', 'int32', 'int8', 'int16', 'bfloat16', 'float16'])
@pytest.mark.parametrize('y_dtype_str', ['float32', 'int32', 'int8', 'int16', 'bfloat16', 'float16'])
def test_atomic_max(x_dtype_str, y_dtype_str, shape):
    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)
    y_dtype = eval('torch.' + y_dtype_str)

    x0 = test_common.generate_tensor(shape, x_dtype_str)
    x1 = test_common.generate_tensor(shape, y_dtype_str)
    out_dtype = promote_dtype(x_dtype, y_dtype)
    if out_dtype == torch.bfloat16:
        out_dtype = torch.float32
    out = torch.full(x1.shape, 0, dtype=out_dtype)

    out_ref = torch.maximum(out, x0)
    out_ref = torch.maximum(out_ref, x1)
    out_ref = out_ref.npu()
    x0 = x0.npu()
    x1 = x1.npu()
    out = out.npu()

    if len(shape) == 2:
        n_elements = shape[0] * shape[1]
        triton_test_fn_atomic_max_dma[shape[0], 1, 1](x0, x1, out, n_elements, BLOCK_SIZE=shape[1])
    elif len(shape) == 1:
        n_elements = shape[0]
        BLOCK_SIZE = min(1024, shape[0]) # 1024:限制最大线程块大小
        grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE # 向上取整
        triton_test_fn_atomic_max_dma[grid_size, 1, 1](x0, x1, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    torch.testing.assert_close(out, out_ref)

# 3d
testlist = [
    (1,22,39),
    (27,1,39),
    (27,22,1),
    (1,1,23),
    (23,1,1),
    (1,23,1),
    (27,5,3),
    (2,29,4),
    (7,31,7),
    (3,5,8),
    (7,17,15),
    (25,5,16),
    (23,5,31),
    (7,11,32),
    (7,11,33),
    (2,3,255),
    (3,3,256),
    (3,2,257),
]


@pytest.mark.parametrize('shape', random.sample(testlist, 5))
@pytest.mark.parametrize('x_dtype_str', ['float32', 'int32', 'int8', 'int16', 'bfloat16', 'float16'])
@pytest.mark.parametrize('y_dtype_str', ['float32', 'int32', 'int8', 'int16', 'bfloat16', 'float16'])
def test_atomic_max_3d(x_dtype_str, y_dtype_str, shape):
    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)
    y_dtype = eval('torch.' + y_dtype_str)

    ncore = 1
    split_size = shape[0] // ncore
    x0 = test_common.generate_tensor(shape, x_dtype_str)
    x1 = test_common.generate_tensor(shape, y_dtype_str)
    out_dtype = promote_dtype(x_dtype, y_dtype)
    if out_dtype == torch.bfloat16:
        out_dtype = torch.float32
    y = torch.full(shape, 0, dtype=out_dtype)

    out_ref = torch.full_like(x0, 0, dtype=out_dtype)
    out_ref = torch.maximum(out_ref, x0)
    out_ref = torch.maximum(out_ref, x1)
    x0 = x0.npu()
    x1 = x1.npu()
    y = y.npu()

    n_elements = shape[0] * shape[1] * shape[2]
    triton_test_fn_atomic_max_dma[ncore, 1, 1](x0, x1, y, n_elements, BLOCK_SIZE=split_size * shape[1] * shape[2])
    y = y.cpu()
    torch.testing.assert_close(y, out_ref)


@triton.jit
def atomic_max_multi_d(in_ptr0, out_ptr0, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, MB: tl.constexpr, NB: tl.constexpr):
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
    tl.atomic_max(out_ptr0 + offsets, tmp0)


filtered_dtype = [dtype for dtype in TestUtils.full_dtype if dtype not in {'int64', 'bool'}]


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
def test_atomic_max_4d_5d(dtype, shape):
    x0_value = 3
    x0 = torch.full(shape, x0_value, dtype=eval('torch.' + dtype)).npu()
    x1 = torch.full(shape, 2, dtype=eval('torch.' + dtype)).npu()

    x1_ref = torch.maximum(x1, x0)

    triton_shape = [*shape]
    while len(triton_shape) < 5:
        triton_shape.append(1)
    atomic_max_multi_d[(1, )](x0, x1, *triton_shape)
    test_common.validate_cmp(dtype, x1, x1_ref)


@triton.jit
def atomic_max_multi_d_2(in_ptr0, out_ptr0, out_ptr1, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, MB: tl.constexpr, NB: tl.constexpr):
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
    tmp1 = tl.load(out_ptr0 + offsets)
    tl.atomic_max(out_ptr1 + offsets, tmp0)
    tl.atomic_max(out_ptr1 + offsets, tmp1)


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
@pytest.mark.parametrize('x_dtype_str', ['float32', 'int32', 'int8', 'int16', 'bfloat16', 'float16'])
@pytest.mark.parametrize('y_dtype_str', ['float32', 'int32', 'int8', 'int16', 'bfloat16', 'float16'])
def test_atomic_max_4d_5d_2(x_dtype_str, y_dtype_str, shape):
    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)
    y_dtype = eval('torch.' + y_dtype_str)
    
    if x_dtype == torch.int8 or x_dtype == torch.int16 or x_dtype == torch.int32:
        x0 = torch.randint(low=0, high=100, size=shape, dtype=x_dtype).npu()
    else:
        x0 = torch.randn(shape, dtype=eval('torch.' + x_dtype_str)).npu()

    if y_dtype == torch.int8 or y_dtype == torch.int16 or y_dtype == torch.int32:
        x1 = torch.randint(low=0, high=100, size=shape, dtype=y_dtype).npu()
    else:
        x1 = torch.randn(shape, dtype=eval('torch.' + y_dtype_str)).npu()

    out_dtype = promote_dtype(x_dtype, y_dtype)
    if out_dtype == torch.bfloat16:
        out_dtype = torch.float32
    if out_dtype == torch.int8 or out_dtype == torch.int16 or out_dtype == torch.int32:
        y = torch.full(shape, torch.iinfo(out_dtype).min, dtype=out_dtype).npu()
    else:
        y = torch.full(shape, float('-inf'), dtype=out_dtype).npu()

    y_tmp = y
    x1_ref = torch.maximum(y_tmp, x0)
    x1_ref = torch.maximum(x1_ref, x1)

    triton_shape = [*shape]
    while len(triton_shape) < 5:
        triton_shape.append(1)
    atomic_max_multi_d_2[(1, )](x0, x1, y, *triton_shape)
    torch.testing.assert_close(y, x1_ref)
