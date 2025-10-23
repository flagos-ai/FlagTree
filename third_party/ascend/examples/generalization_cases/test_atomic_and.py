import math
import pytest
import torch
import triton

import triton.language as tl

import test_common
from test_common import TestUtils
filtered_dtype = [dtype for dtype in TestUtils.full_dtype if dtype not in {'uint32', 'float16', 'float32', 'bfloat16', 'int64', 'bool'}]


@triton.jit
def atomic_and(in_ptr0, out_ptr0, n_elements, BLOCK_SIZE: tl.constexpr, BLOCK_NUM: tl.constexpr):
    in_offset = tl.program_id(0) * BLOCK_SIZE
    out_offset = (tl.program_id(0) % BLOCK_NUM) * BLOCK_SIZE
    in_index = in_offset + tl.arange(0, BLOCK_SIZE)
    out_index = out_offset + tl.arange(0, BLOCK_SIZE)
    xmask = in_index < n_elements

    tmp0 = tl.load(in_ptr0 + (in_index), xmask)
    tl.atomic_and(out_ptr0 + (out_index), tmp0, xmask)



@triton.jit
def atomic_and_broadcast(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    x = tl.load(x_ptr)  # x is scalar or 1D, no mask needed
    
    # Compute y indices
    y_offset = pid * BLOCK_SIZE
    y_indices = y_offset + tl.arange(0, BLOCK_SIZE)
    y_mask = y_indices < n_elements
    
    y_value = tl.load(y_ptr + y_indices, y_mask)
    # Atomic or: y &= x (broadcasted)
    tl.atomic_and(out_ptr + y_indices, y_value, mask=y_mask)
    tl.atomic_and(out_ptr + y_indices, x, mask=y_mask)


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
def test_atomic_and(x_dtype_str, shape):
    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)
    x_shape = list(shape[:])
    x_shape[0] *= 2
    x = test_common.generate_tensor(x_shape, x_dtype_str).npu()
    # OR的时候任何位和0做OR都不变 任何位和1做AND也都不变，所以为了保持不变 不能用0 只能用1
    y = torch.full(shape, torch.iinfo(x_dtype).max, dtype=x_dtype).npu()

    # 保存副本用于验证
    x_temp = x.clone()
    y_temp = y.clone()
    
    if len(shape) == 2:
        n_elements = shape[0] * shape[1] * 2
        atomic_and[shape[0] * 2, 1, 1](x, y, n_elements, BLOCK_SIZE=shape[1], BLOCK_NUM=shape[0])
    elif len(shape) == 1:
        n_elements = shape[0]
        BLOCK_SIZE = min(1024, shape[0]) # 1024:限制最大线程块大小
        grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE # 向上取整
        aligned_size = grid_size * BLOCK_SIZE
        x_concat = torch.full([aligned_size * 2], 0, dtype=x_dtype).npu()
        x_concat[0:n_elements] = x[0:n_elements]
        x_concat[aligned_size:(aligned_size + n_elements)] = x[n_elements:(n_elements * 2)]
        atomic_and[grid_size * 2, 1, 1](x_concat, y, aligned_size * 2, BLOCK_SIZE=BLOCK_SIZE, BLOCK_NUM=grid_size)
    
    expected = y_temp & x_temp[0:shape[0]] & x_temp[shape[0]:(shape[0] * 2)]
    torch.testing.assert_close(y, expected)


# 3d
@pytest.mark.parametrize('shape', TestUtils.test_shape3d)
@pytest.mark.parametrize('x_dtype_str', filtered_dtype)
def test_atomic_and_3d(x_dtype_str, shape):
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
    atomic_and[2, 1, 1](x, y, n_elements * 2, BLOCK_SIZE=shape[0] * shape[1] * shape[2], BLOCK_NUM=1)

    expected = y_temp & x_temp[0:shape[0]] & x_temp[shape[0]:(shape[0] * 2)]
    torch.testing.assert_close(y, expected)



@pytest.mark.parametrize('shape', TestUtils.test_shape_ub_overflow)
@pytest.mark.parametrize('x_dtype_str', filtered_dtype)
@test_common.raises_with_match(triton.compiler.errors.MLIRCompilationError, "ub overflow")
def test_atomic_and_ub_overflow(x_dtype_str, shape):
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
    atomic_and[2, 1, 1](x, y, n_elements * 2, BLOCK_SIZE=shape[0] * shape[1] * shape[2], BLOCK_NUM=1)



@triton.jit
def atomic_and_multi_d(in_ptr0, out_ptr0, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, MB: tl.constexpr, NB: tl.constexpr):
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
    tl.atomic_and(out_ptr0 + offsets, tmp0)


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
def test_atomic_and_4d_5d(dtype, shape):
    x0_value = 3
    x0 = torch.full(shape, x0_value, dtype=eval('torch.' + dtype)).npu()
    x1 = torch.full(shape, 2, dtype=eval('torch.' + dtype)).npu()

    x1_ref = x1 & x0_value

    triton_shape = [*shape]
    while len(triton_shape) < 5:
        triton_shape.append(1)
    atomic_and_multi_d[(1, )](x0, x1, *triton_shape)
    test_common.validate_cmp(dtype, x1, x1_ref)


@triton.jit
def atomic_and_5d(x_ptr, out_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, MB: tl.constexpr, NB: tl.constexpr,
                            XB1: tl.constexpr, YB1: tl.constexpr, ZB1: tl.constexpr, MB1: tl.constexpr, NB1: tl.constexpr):
    base = tl.program_id(0) * (XB * YB * ZB * MB * NB)
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
    
    based_offsets = offsets + base
    
    tmp0 = tl.load(x_ptr + based_offsets)
    tl.atomic_and(out_ptr + offsets1, tmp0)


@pytest.mark.parametrize('param_list',
    [
        [(1, 1, 2, 1, 1), (1, 1, 2, 1, 2)],
    ]
    )
@pytest.mark.parametrize('x_dtype_str', filtered_dtype)
def test_atomic_and_5d(x_dtype_str, param_list):
    x0_shape, y_shape = param_list

    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)
    x_shape = list(x0_shape[:])
    x_shape[0] *= 2
    x = torch.randint(low=0, high=100, size=x_shape, dtype=x_dtype).npu()

    out = torch.full(y_shape, 0, dtype=x_dtype).npu()

    x_temp = x.clone()
    out_temp = out.clone()

    triton_shape = [*x0_shape]
    while len(triton_shape) < 5:
        triton_shape.append(1)
    XB, YB, ZB, MB, NB = triton_shape

    triton_shape1 = [*y_shape]
    while len(triton_shape1) < 5:
        triton_shape1.append(1)
    XB1, YB1, ZB1, MB1, NB1 = triton_shape1

    atomic_and_5d[(2, )](
        x_ptr=x,
        out_ptr=out, 
        XB=XB, YB=YB, ZB=ZB, MB=MB, NB=NB,
        XB1=XB1, YB1=YB1, ZB1=ZB1, MB1=MB1, NB1=NB1,
        )
    
    expected = out_temp & x_temp[0:x0_shape[0]] & x_temp[x0_shape[0]:x_shape[0]]
    torch.testing.assert_close(out, expected)


@triton.jit
def atomic_and_4d(x_ptr, out_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, MB: tl.constexpr,
                            XB1: tl.constexpr, YB1: tl.constexpr, ZB1: tl.constexpr, MB1: tl.constexpr):
    base = tl.program_id(0) * (XB * YB * ZB * MB)
    offsets = tl.arange(0, XB) * (YB * ZB * MB)
    offsets = offsets[:, None] + tl.arange(0, YB)[None, :] * (ZB * MB)
    offsets = offsets[:, :, None] + tl.arange(0, ZB)[None, None, :] * (MB)
    offsets = offsets[:, :, :, None] + tl.arange(0, MB)[None, None, None, :] 

    offsets1 = tl.arange(0, XB1) * (YB1 * ZB1 * MB1)
    offsets1 = offsets1[:, None] + tl.arange(0, YB1)[None, :] * (ZB1 * MB1)
    offsets1 = offsets1[:, :, None] + tl.arange(0, ZB1)[None, None, :] * (MB1)
    offsets1 = offsets1[:, :, :, None] + tl.arange(0, MB1)[None, None, None, :]
    
    based_offsets = offsets + base

    tmp0 = tl.load(x_ptr + based_offsets)
    tl.atomic_and(out_ptr + offsets1, tmp0)


@pytest.mark.parametrize('param_list',
    [
        [(1, 1, 2, 1), (1, 1, 2, 2)],
        [(1, 1, 1, 1), (1, 1, 2, 2)],
    ]
    )
@pytest.mark.parametrize('x_dtype_str', filtered_dtype)
def test_atomic_and_4d(x_dtype_str, param_list):
    x0_shape, y_shape = param_list

    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)
    x_shape = list(x0_shape[:])
    x_shape[0] *= 2
    x = torch.randint(low=0, high=100, size=x_shape, dtype=x_dtype).npu()
    out = torch.full(y_shape, 0, dtype=x_dtype).npu()

    x_temp = x.clone()
    out_temp = out.clone()

    triton_shape = [*x0_shape]
    while len(triton_shape) < 4:
        triton_shape.append(1)
    XB, YB, ZB, MB = triton_shape

    triton_shape1 = [*y_shape]
    while len(triton_shape1) < 4:
        triton_shape1.append(1)
    XB1, YB1, ZB1, MB1 = triton_shape1

    atomic_and_4d[(2, )](
        x_ptr=x,
        out_ptr=out, 
        XB=XB, YB=YB, ZB=ZB, MB=MB,
        XB1=XB1, YB1=YB1, ZB1=ZB1, MB1=MB1,
        )
    
    expected = out_temp & x_temp[0:x0_shape[0]] & x_temp[x0_shape[0]:x_shape[0]]
    torch.testing.assert_close(out, expected)


@triton.jit
def atomic_and_3d(x_ptr, out_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr,
                            XB1: tl.constexpr, YB1: tl.constexpr, ZB1: tl.constexpr):
    base = tl.program_id(0) * (XB * YB * ZB)
    offsets = tl.arange(0, XB) * (YB * ZB)
    offsets = offsets[:, None] + tl.arange(0, YB)[None, :] * (ZB)
    offsets = offsets[:, :, None] + tl.arange(0, ZB)[None, None, :]

    offsets1 = tl.arange(0, XB1) * (YB1 * ZB1)
    offsets1 = offsets1[:, None] + tl.arange(0, YB1)[None, :] * (ZB1)
    offsets1 = offsets1[:, :, None] + tl.arange(0, ZB1)[None, None, :]
    
    based_offsets = offsets + base

    tmp0 = tl.load(x_ptr + based_offsets)
    tl.atomic_and(out_ptr + offsets1, tmp0)


@pytest.mark.parametrize('param_list',
    [
        [(1, 1, 1), (1, 1, 2)],
        [(1, 1, 2), (1, 2, 2)],
    ]
    )
@pytest.mark.parametrize('x_dtype_str', filtered_dtype)
def test_atomic_and_3d_2(x_dtype_str, param_list):
    x0_shape, y_shape = param_list

    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)
    x_shape = list(x0_shape[:])
    x_shape[0] *= 2
    x = torch.randint(low=0, high=100, size=x_shape, dtype=x_dtype).npu()
    out = torch.full(y_shape, 0, dtype=x_dtype).npu()

    x_temp = x.clone()
    out_temp = out.clone()

    triton_shape = [*x0_shape]
    while len(triton_shape) < 3:
        triton_shape.append(1)
    XB, YB, ZB = triton_shape

    triton_shape1 = [*y_shape]
    while len(triton_shape1) < 3:
        triton_shape1.append(1)
    XB1, YB1, ZB1 = triton_shape1

    atomic_and_3d[(2, )](
        x_ptr=x,
        out_ptr=out, 
        XB=XB, YB=YB, ZB=ZB,
        XB1=XB1, YB1=YB1, ZB1=ZB1,
        )
    
    expected = out_temp & x_temp[0:x0_shape[0]] & x_temp[x0_shape[0]:x_shape[0]]
    torch.testing.assert_close(out, expected)


@triton.jit
def atomic_and_2d(x_ptr, out_ptr, XB: tl.constexpr, YB: tl.constexpr,
                            XB1: tl.constexpr, YB1: tl.constexpr):
    base = tl.program_id(0) * (XB * YB)
    offsets = tl.arange(0, XB) * (YB)
    offsets = offsets[:, None] + tl.arange(0, YB)[None, :]

    offsets1 = tl.arange(0, XB1) * (YB1)
    offsets1 = offsets1[:, None] + tl.arange(0, YB1)[None, :]
    
    based_offsets = offsets + base

    tmp0 = tl.load(x_ptr + based_offsets)
    tl.atomic_and(out_ptr + offsets1, tmp0)


@pytest.mark.parametrize('param_list',
    [
        [(1, 2), (2, 2)],
         [(1, 1), (2, 2)],
    ]
    )
@pytest.mark.parametrize('x_dtype_str', filtered_dtype)
def test_atomic_and_2d(x_dtype_str, param_list):
    x0_shape, y_shape = param_list

    # 获取原始类型
    x_dtype = eval('torch.' + x_dtype_str)
    x_shape = list(x0_shape[:])
    x_shape[0] *= 2
    x = torch.randint(low=0, high=100, size=x_shape, dtype=x_dtype).npu()
    out = torch.full(y_shape, 0, dtype=x_dtype).npu()

    x_temp = x.clone()
    out_temp = out.clone()

    triton_shape = [*x0_shape]
    while len(triton_shape) < 2:
        triton_shape.append(1)
    XB, YB = triton_shape

    triton_shape1 = [*y_shape]
    while len(triton_shape1) < 2:
        triton_shape1.append(1)
    XB1, YB1 = triton_shape1

    atomic_and_2d[(2, )](
        x_ptr=x,
        out_ptr=out, 
        XB=XB, YB=YB,
        XB1=XB1, YB1=YB1,
        )
    
    expected = out_temp & x_temp[0:x0_shape[0]] & x_temp[x0_shape[0]:x_shape[0]]
    torch.testing.assert_close(out, expected)


@triton.jit
def atomic_and(in_ptr0, out_ptr0, n_elements, BLOCK_SIZE: tl.constexpr,
                        BLOCK_NUM: tl.constexpr, mode: tl.constexpr = 0):
    in_offset = tl.program_id(0) * BLOCK_SIZE
    out_offset = (tl.program_id(0) % BLOCK_NUM) * BLOCK_SIZE
    in_index = in_offset + tl.arange(0, BLOCK_SIZE)
    out_index = out_offset + tl.arange(0, BLOCK_SIZE)
    xmask = in_index < n_elements

    tmp0 = tl.load(in_ptr0 + (in_index), xmask)
    if mode ==0:
        tl.atomic_and(out_ptr0 + (out_index), tmp0, xmask, 'acq_rel', 'cta')
    elif mode == 1:
        tl.atomic_and(out_ptr0 + (out_index), tmp0, xmask, "test")
    elif mode == 2:
        tl.atomic_and(out_ptr0 + (out_index), tmp0, xmask, "acq_rel", "test")


invalid_types_int = [
    'int64',
    'bool'
]

@pytest.mark.parametrize("sigtype", invalid_types_int)
@test_common.raises_with_match(triton.compiler.errors.CompilationError, "All support dtypes are int8, int16, int32, float16, float32, bfloat16")
def test_invalid_types_int(sigtype):
    N = 32
    x = test_common.generate_tensor(shape=(N,), dtype=sigtype).npu()
    y = test_common.generate_tensor(shape=(N,), dtype=sigtype).npu()

    atomic_and[1, 1, 1](x, y, 1, 1, 32)


invalid_types_float = [
    'float16',
    'float32',
    'bfloat16'
]

@pytest.mark.parametrize("sigtype", invalid_types_float)
@test_common.raises_with_match(triton.compiler.errors.MLIRCompilationError, "must be signless-integer-like")
def test_invalid_types_float(sigtype):
    N = 32
    x = test_common.generate_tensor(shape=(N,), dtype=sigtype).npu()
    y = test_common.generate_tensor(shape=(N,), dtype=sigtype).npu()

    atomic_and[1, 1, 1](x, y, 1, 1, 32)


default_types = ['int8']

@pytest.mark.parametrize("sigtype", default_types)
@pytest.mark.parametrize("test_type", ["sem", "scope"])
@test_common.raises_with_match(triton.compiler.errors.CompilationError, "Memory semantic test not supported")
def test_invalid_sem_scope(sigtype, test_type):
    N = 32
    x = test_common.generate_tensor(shape=(N,), dtype=sigtype).npu()
    y = test_common.generate_tensor(shape=(N,), dtype=sigtype).npu()

    if test_type == "sem":
        atomic_and[1, 1, 1](x, y, 1, 1, 32, 1)
    elif test_type == "scope":
        atomic_and[1, 1, 1](x, y, 1, 1, 32, 2)


@triton.jit
def _atomic_and_ss(
    in_ptr, out_ptr, n_cols,
    BLOCK_SIZE: tl.constexpr,
    SEM:   tl.constexpr,
    SCOPE: tl.constexpr
):
    pid  = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pid < n_cols
    val  = tl.load(in_ptr + pid, mask)
    tl.atomic_and(out_ptr + pid, val, mask, sem=SEM, scope=SCOPE)

SEMS   = ("relaxed", "acquire", "release", "acq_rel")
SCOPES = ("cta", "gpu", "sys")

@pytest.mark.parametrize("sem", SEMS)
@pytest.mark.parametrize("scope", SCOPES)
def test_atomic_sem_vs_scope(sem: str, scope: str):
    n_cols = 1024
    BLOCK  = 128
    grid   = (triton.cdiv(n_cols, BLOCK),)

    inp = torch.full((n_cols,), 0xFF, dtype=torch.int32, device="npu")

    base = torch.full_like(inp, 0xFF)
    _atomic_and_ss[grid](inp, base, n_cols,
                         BLOCK_SIZE=BLOCK,
                         SEM="acq_rel",
                         SCOPE="gpu")

    cur = torch.full_like(inp, 0xFF)
    _atomic_and_ss[grid](inp, cur, n_cols,
                         BLOCK_SIZE=BLOCK,
                         SEM=sem,
                         SCOPE=scope)
                         
    torch.testing.assert_close(cur, base)
