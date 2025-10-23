import math
import pytest
import torch
import triton

import triton.language as tl

import test_common
from test_common import TestUtils
filtered_dtype = [dtype for dtype in TestUtils.full_dtype if dtype not in {'uint32', 'bfloat16', 'int64', 'bool'}]


@triton.jit
def atomic_xchg(in_ptr0, out_ptr0, n_elements, BLOCK_SIZE: tl.constexpr, BLOCK_NUM: tl.constexpr):
    in_offset = tl.program_id(0) * BLOCK_SIZE
    out_offset = (tl.program_id(0) % BLOCK_NUM) * BLOCK_SIZE
    in_index = in_offset + tl.arange(0, BLOCK_SIZE)
    out_index = out_offset + tl.arange(0, BLOCK_SIZE)
    xmask = in_index < n_elements

    tmp0 = tl.load(in_ptr0 + (in_index), xmask)
    tl.atomic_xchg(out_ptr0 + (out_index), tmp0, xmask)


@triton.jit
def atomic_xchg_ndim(x_ptr, out_ptr, NCORE: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                    DIM0: tl.constexpr, DIM1: tl.constexpr, DIM2: tl.constexpr, DIM3: tl.constexpr, DIM4: tl.constexpr):
    sub_idx = tl.program_id(1)
    base_src = tl.program_id(0) * DIM4 + sub_idx * BLOCK_SIZE
    base_dst = (tl.program_id(0) % (DIM0 * DIM1 * DIM2 * DIM3)) * DIM4 + sub_idx * BLOCK_SIZE
    offsets_src = tl.arange(0, BLOCK_SIZE) + base_src
    offsets_dst = tl.arange(0, BLOCK_SIZE) + base_dst
    mask = tl.arange(0, BLOCK_SIZE) + sub_idx * BLOCK_SIZE < DIM4
    tmp = tl.load(x_ptr + offsets_src, mask)
    tl.atomic_xchg(out_ptr + offsets_dst, tmp, mask)


@triton.jit
def atomic_xchg_broadcast(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    x = tl.load(x_ptr)  # x is scalar or 1D, no mask needed
    
    # Compute y indices
    y_offset = pid * BLOCK_SIZE
    y_indices = y_offset + tl.arange(0, BLOCK_SIZE)
    y_mask = y_indices < n_elements
    
    y_value = tl.load(y_ptr + y_indices, y_mask)
    # Atomic or: y |= x (broadcasted)
    tl.atomic_xchg(out_ptr + y_indices, y_value, mask=y_mask)
    tl.atomic_xchg(out_ptr + y_indices, x, mask=y_mask)


# 定义不同测试场景的参数组合 (x_shape, y_shape, BLOCK_SIZE)
test_cases = [
    ((1, 1, 1, 1), (1, 1, 1, 4), 4),
    ((1, 1, 1, 3), (1, 5, 1, 3), 5),
    ((3,), (2, 3, 3, 3, 3), 81),
    ((3,), (2, 3, 3, 3), 27),
    ((3,), (2, 3, 3), 9),
    ((3,), (2, 3), 3),
]


@pytest.mark.parametrize('shape', TestUtils.test_shape2d + TestUtils.test_shape1d)
@pytest.mark.parametrize('x_dtype_str', filtered_dtype)
def test_atomic_xchg(x_dtype_str, shape):
    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)

    x_shape = list(shape[:])
    x_shape[0] *= 2
    x = test_common.generate_tensor(x_shape, x_dtype_str).npu()
    y = torch.full(shape, 0, dtype=x_dtype).npu()

    # 保存副本用于验证
    x_temp = x.clone()
    y_temp = y.clone()
    
    if len(shape) == 2:
        n_elements = shape[0] * shape[1] * 2
        atomic_xchg[shape[0] * 2, 1, 1](x, y, n_elements, BLOCK_SIZE=shape[1], BLOCK_NUM=shape[0])
    elif len(shape) == 1:
        n_elements = shape[0]
        BLOCK_SIZE = min(1024, shape[0]) # 1024:限制最大线程块大小
        grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE # 向上取整
        aligned_size = grid_size * BLOCK_SIZE
        x_concat = torch.full([aligned_size * 2], 0, dtype=x_dtype).npu()
        x_concat[0:n_elements] = x[0:n_elements]
        x_concat[aligned_size:(aligned_size + n_elements)] = x[n_elements:(n_elements * 2)]
        atomic_xchg[grid_size * 2, 1, 1](x_concat, y, aligned_size * 2, BLOCK_SIZE=BLOCK_SIZE, BLOCK_NUM=grid_size)
    
    expected = x_temp[shape[0]:(shape[0] * 2)].expand(y_temp.shape)
    torch.testing.assert_close(y, expected)


# 3d
@pytest.mark.parametrize('shape', TestUtils.test_shape3d)
@pytest.mark.parametrize('x_dtype_str', filtered_dtype)
def test_atomic_xchg_3d(x_dtype_str, shape):
    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)

    x_shape = list(shape[:])
    x_shape[0] *= 2
    x = test_common.generate_tensor(x_shape, x_dtype_str).npu()
    y = torch.full(shape, 0, dtype=x_dtype).npu()

    # 保存副本用于验证
    x_temp = x.clone()
    y_temp = y.clone()

    n_elements = shape[0] * shape[1] * shape[2]
    atomic_xchg[2, 1, 1](x, y, n_elements * 2, BLOCK_SIZE=shape[0] * shape[1] * shape[2], BLOCK_NUM=1)

    expected = x_temp[shape[0]:(shape[0] * 2)].expand(y_temp.shape)
    torch.testing.assert_close(y, expected)


@triton.jit
def atomic_xchg_multi_d(in_ptr0, out_ptr0, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, MB: tl.constexpr, NB: tl.constexpr):
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
    tl.atomic_xchg(out_ptr0 + offsets, tmp0)


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
def test_atomic_xchg_4d_5d(dtype, shape):
    x0_value = 3
    x0 = torch.full(shape, x0_value, dtype=eval('torch.' + dtype)).npu()
    x1 = torch.full(shape, 2, dtype=eval('torch.' + dtype)).npu()

    x1_ref = x0

    triton_shape = [*shape]
    while len(triton_shape) < 5:
        triton_shape.append(1)
    atomic_xchg_multi_d[(1, )](x0, x1, *triton_shape)
    test_common.validate_cmp(dtype, x1, x1_ref)


@pytest.mark.parametrize('shaape',
    [
        (1, 1, 1, 1, 2),
        (10, 1, 15, 1, 7),
        (1, 1, 1, 1, 257),
    ]
    )
@pytest.mark.parametrize('x_dtype_str', filtered_dtype)
def test_atomic_xchg_5d(x_dtype_str, shaape):
    shape = shaape

    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)
    x_shape = list(shape[:])
    x_shape[0] *= 2
    x = torch.randint(low=0, high=100, size=x_shape, dtype=x_dtype).npu()

    out = torch.full(shape, 0, dtype=x_dtype).npu()

    x_temp = x.clone()
    out_temp = out.clone()

    triton_shape = [*shape]
    while len(triton_shape) < 5:
        triton_shape.append(1)
    XB, YB, ZB, MB, NB = triton_shape
    BLOCK_SIZE = 256
    ncore = (NB + BLOCK_SIZE - 1) // BLOCK_SIZE

    atomic_xchg_ndim[(2 * XB * YB * ZB * MB, ncore)](
        x_ptr=x,
        out_ptr=out,
        NCORE=ncore,
        BLOCK_SIZE=BLOCK_SIZE,
        DIM0=XB, DIM1=YB, DIM2=ZB, DIM3=MB, DIM4=NB,
        )
    
    expected = x_temp[shape[0]:x_shape[0]].expand(out_temp.shape)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize('shaape',
    [
        (1, 1, 1, 1),
        (1, 1, 2, 2),
        (1, 3, 2, 7),
        (1, 3, 2, 651),
    ]
    )
@pytest.mark.parametrize('x_dtype_str', filtered_dtype)
def test_atomic_xchg_4d(x_dtype_str, shaape):
    shape = shaape

    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)
    x_shape = list(shape[:])
    x_shape[0] *= 2
    x = torch.randint(low=0, high=100, size=x_shape, dtype=x_dtype).npu()
    out = torch.full(shape, 0, dtype=x_dtype).npu()

    x_temp = x.clone()
    out_temp = out.clone()

    triton_shape = [*shape]
    while len(triton_shape) < 4:
        triton_shape.append(1)
    XB, YB, ZB, MB = triton_shape

    BLOCK_SIZE = 256
    ncore = (MB + BLOCK_SIZE - 1) // BLOCK_SIZE

    atomic_xchg_ndim[(2 * XB * YB * ZB, ncore)](
        x_ptr=x,
        out_ptr=out, 
        NCORE=ncore,
        BLOCK_SIZE=BLOCK_SIZE,
        DIM0=1, DIM1=XB, DIM2=YB, DIM3=ZB, DIM4=MB,
        )
    
    expected = x_temp[shape[0]:x_shape[0]].expand(out_temp.shape)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize('shaape',
    [
        (1, 1, 1),
        (1, 1, 2),
        (1, 31, 275),
    ]
    )
@pytest.mark.parametrize('x_dtype_str', filtered_dtype)
def test_atomic_xchg_3d_2(x_dtype_str, shaape):
    shape = shaape

    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)
    x_shape = list(shape[:])
    x_shape[0] *= 2
    x = torch.randint(low=0, high=100, size=x_shape, dtype=x_dtype).npu()
    out = torch.full(shape, 0, dtype=x_dtype).npu()

    x_temp = x.clone()
    out_temp = out.clone()

    triton_shape = [*shape]
    while len(triton_shape) < 3:
        triton_shape.append(1)
    XB, YB, ZB = triton_shape
    BLOCK_SIZE = 256
    ncore = (ZB + BLOCK_SIZE - 1) // BLOCK_SIZE

    atomic_xchg_ndim[(2 * XB * YB, ncore)](
        x_ptr=x,
        out_ptr=out, 
        NCORE=ncore,
        BLOCK_SIZE=BLOCK_SIZE,
        DIM0=1, DIM1=1, DIM2=XB, DIM3=YB, DIM4=ZB,
        )
    
    expected = x_temp[shape[0]:x_shape[0]].expand(out_temp.shape)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize('shaape',
    [
        (1, 2),
        (1, 1),
        (257, 1),
        (257, 2),
    ]
    )
@pytest.mark.parametrize('x_dtype_str', filtered_dtype)
def test_atomic_xchg_2d(x_dtype_str, shaape):
    shape = shaape

    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)
    x_shape = list(shape[:])
    x_shape[0] *= 2
    x = torch.randint(low=0, high=100, size=x_shape, dtype=x_dtype).npu()
    out = torch.full(shape, 0, dtype=x_dtype).npu()

    x_temp = x.clone()
    out_temp = out.clone()

    triton_shape = [*shape]
    while len(triton_shape) < 2:
        triton_shape.append(1)
    XB, YB = triton_shape
    BLOCK_SIZE = 256
    ncore = (YB + BLOCK_SIZE - 1) // BLOCK_SIZE

    atomic_xchg_ndim[(2 * XB, ncore)](
        x_ptr=x,
        out_ptr=out, 
        NCORE=ncore,
        BLOCK_SIZE=BLOCK_SIZE,
        DIM0=1, DIM1=1, DIM2=1, DIM3=XB, DIM4=YB,
        )

    expected = x_temp[shape[0]:x_shape[0]].expand(out_temp.shape)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize('shaape', [(1,), (9,), (256,), (257,), (65535,), (65536,)])
@pytest.mark.parametrize('x_dtype_str', filtered_dtype)
def test_atomic_xchg_1d(x_dtype_str, shaape):
    shape = shaape

    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)
    x_shape = list(shape[:])
    x_shape[0] *= 2
    x = torch.randint(low=0, high=100, size=x_shape, dtype=x_dtype).npu()
    out = torch.full(shape, 0, dtype=x_dtype).npu()

    x_temp = x.clone()
    out_temp = out.clone()

    triton_shape = [*shape]
    while len(triton_shape) < 2:
        triton_shape.append(1)
    XB = triton_shape[0]
    BLOCK_SIZE = 256
    ncore = (XB + BLOCK_SIZE - 1) // BLOCK_SIZE

    atomic_xchg_ndim[(2, ncore)](
        x_ptr=x,
        out_ptr=out, 
        NCORE=ncore,
        BLOCK_SIZE=BLOCK_SIZE,
        DIM0=1, DIM1=1, DIM2=1, DIM3=1, DIM4=XB,
        )

    expected = x_temp[shape[0]:x_shape[0]].expand(out_temp.shape)
    torch.testing.assert_close(out, expected)