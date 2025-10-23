import os
import re
import torch
import torch_npu
import math
import logging
from typing import AnyStr
import pytest
import functools

log_level = os.getenv("LOG_LEVEL", "WARN").upper()
level_mapping = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

logging.basicConfig(
    level=level_mapping.get(log_level, logging.WARNING),
    format="[%(asctime)s][%(levelname)s] %(message)s"
)

bisheng_not_support_dtypes = {
    'abs': [],
    'eq':  [],
    'ne':  [],
    'flip':['int64', 'bfloat16'],
    'load_store': ['int64'],
    'permute2d':  ['int64'],
    'permute3d':  ['int64'],
    'trans2d':    ['int64'],
    'trans3d':    ['int64'],
    'matmul':        ['int16', 'int32', 'uint32', 'int64', 'bool']
}

tritonascend_not_support_dtypes = {
    'abs': ['bool'],
    'eq':  ['bool'],
    'ne':  ['bool'],
    'flip':['bool'],
    'load_store': ['bool'],
    'permute2d':  ['bool'],
    'permute3d':  ['bool'],
    'trans2d':    ['bool'],
    'trans3d':    ['bool'],
}

def avoid_not_support(op: AnyStr):
    def decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(shape, dtype, *args, **kwargs):
            if dtype in bisheng_not_support_dtypes.get(op, []):
                logging.warn(f'skiped bisheng not support dtype:{dtype}')
                return
            if dtype in tritonascend_not_support_dtypes.get(op, []):
                logging.warn(f'skiped triton ascend not support dtype:{dtype}')
                return
            return test_func(shape, dtype, *args, **kwargs)
        return wrapper
    return decorator


def get_shape1d(in_shape1d):
    result = []
    for i in in_shape1d:
        v = tuple((i,))
        result.append(v)
    return result

def get_shape2d(in_shape1d, custom_shape):
    result = []
    for a in in_shape1d:
        for b in custom_shape:
            t1 = tuple((a, b))
            t2 = tuple((b, a))
            if t1 not in result:
                result.append(t1)
            if t2 not in result:
                result.append(t2)
    return result

def get_shape3d():
    return [(1,22,39),(27,1,39),(27,22,1),(23,1,1),(1,23,1),(1,1,23),(37,5,3),(2,29,4),(7,31,7),(3,5,8),(7,17,15),(23,5,16),(23,5,31),(7,11,32),(7,11,33),(2,3,255),(3,3,256),(3,2,257)]

def get_shape1_2_3d(in_shape1d, custom_shape):
    return get_shape1d(in_shape1d) + get_shape2d(in_shape1d, custom_shape) + get_shape3d()

class TestUtils:
    in_shape1d = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 37, 741]
    custom_shape = [3, 13, 32, 256]
    batch = [1, 2, 3, 4, 5, 8]
    test_shape1d = get_shape1d(in_shape1d)
    test_shape2d = get_shape2d(in_shape1d, custom_shape)
    test_shape3d = [(1,22,39), (27,1,39), (27,22,1), (1,1,23), (23,1,1), (1,23,1),
                    (37,5,3), (2,29,4), (7,31,7), (3,5,8), (7,17,15), (25,5,16),
                    (23,5,31), (7,11,32), (7,11,33), (2,3,255), (3,3,256), (3,2,257),]
    test_shape4d = [(8, 4, 8, 8), (1, 11, 16, 2)]
    test_shape5d = [(2, 3, 4, 5, 6), (1, 3, 4, 5, 6), (3, 6, 2, 4, 4)]
    test_shape6d = [(2, 3, 5, 6, 3, 2)]
    test_shape7d = [(1, 2, 3, 4, 3, 2, 2)]
    test_shape_ub_overflow = [(10, 50, 1000)]
    test_shape8d = [(1, 2, 3, 2, 5, 3, 7, 2), (1, 3, 2, 5, 6, 7, 2, 1), (2, 3, 7, 3, 2, 3, 2, 3)]
    full_shape_4_8d = test_shape4d + test_shape5d + test_shape6d + test_shape7d + test_shape8d

    full_shape = test_shape1d + test_shape2d + test_shape3d
    test_shape1_2_3d = full_shape
    full_dtype = ['int8', 'int16', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    ub_size = 98304 * 2
    dtype_list = full_dtype

def get_dtype_size(dtype):
    torch_dtype = eval('torch.' + dtype)
    bits = 0
    if torch_dtype == torch.bool:
        bits = 8
    elif torch.is_floating_point(torch.tensor(0, dtype=torch_dtype)):
        bits = torch.finfo(torch_dtype).bits
    else:
        bits = torch.iinfo(torch_dtype).bits
    return bits//8

def check_ub_mem_overflow(dtype, shape):
    bytes = get_dtype_size(dtype)
    if bytes * math.prod(shape) > TestUtils.ub_size:
        logging.warning(f'dtype:{dtype} shape:{shape} mem overflow')
        return True
    return False


def generate_tensor(shape, dtype):
    if dtype == 'float32' or dtype == 'float16' or dtype == 'bfloat16':
        return torch.randn(size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or dtype == 'uint32':
        return torch.randint(low=0, high=2000, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int8':
        return torch.randint(low=0, high=127, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'bool':
        return torch.randint(low=0, high=2, size=shape).bool()
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))


def generate_tensor_int_withSigns(shape, dtype):
    if dtype == 'int32' or dtype == 'int64' or dtype == 'int16':
        return torch.randint(low=-32768, high=32767, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int8':
        return torch.randint(low=-128, high=127, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'bool':
        return torch.randint(low=0, high=2, size=shape).bool()
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))


def get_triton_sig_typename(dtype):
    if dtype == 'float32':
        tyname = "*fp32"
    elif dtype == 'int32':
        tyname = "*i32"
    elif dtype == 'int64':
        tyname = "*i64"
    elif dtype == 'float16':
        tyname = "*fp16"
    elif dtype == 'int16':
        tyname = "*i16"
    elif dtype == 'int8':
        tyname = "*i8"
    elif dtype == 'bool':
        tyname = "*i1"
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))
    return tyname

# Relative error: abs(x_ref - x_cal) / abs(x_ref)
# Absolute error: abs(x_ref - x_cal)

# calculation type operators require different error range
# It is a stricter verification and not satisfied now, save it here
def validate_cal(dtype, y_cal, y_ref):
    if dtype == 'float16':
        if torch.mean(y_ref) < 0.001:
            assert torch.abs(y_cal - y_ref) < 0.001, "|y_cal - y_ref| < 0.001 is required !"
        else:
            diff = torch.div(torch.abs(y_cal - y_ref), torch.abs(y_cal)) < 0.001
            # all true
            assert diff.all(), "Relative error is less than 0.001 !"
    if dtype == 'float32':
        if torch.mean(y_ref) < 0.0001:
            assert torch.abs(y_cal - y_ref) < 0.0001, "|y_cal - y_ref| < 0.0001 is required !"
        else:
            diff = torch.div(torch.abs(y_cal - y_ref), torch.abs(y_cal)) < 0.0001
            assert diff.all(), "Relative error is less than 0.001 !"
    elif dtype == 'bfloat16':
        diff = torch.div(torch.abs(y_cal - y_ref), torch.abs(y_cal)) < 0.001
        assert diff.all(), "Relative error is less than 0.001 !"
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16':
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'bool':
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))


# moving and comparison ops require no precision error
def validate_cmp(dtype, y_cal, y_ref):
    y_cal = y_cal.npu()
    y_ref = y_ref.npu()
    if dtype == 'float16':
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype == 'bfloat16':
        torch.testing.assert_close(y_ref.to(torch.float32), y_cal.to(torch.float32), rtol=1e-03, atol=1e-03,
                                   equal_nan=True)
    elif dtype == 'float32':
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-04, atol=1e-04, equal_nan=True)
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or dtype == 'int8' or dtype == 'uint32':
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'bool':
        assert torch.equal(y_cal.cpu(), y_ref.cpu())
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))


def validate_cmp_with_expection(dtype, y_cal, y_ref, expect):
    if dtype == 'float32' or dtype == 'float16' or dtype == 'bfloat16':
        if expect:
            assert torch.allclose(y_ref, y_cal, rtol=1e-03, atol=1e-03, equal_nan=True)
        else:
            assert not torch.allclose(y_ref, y_cal, rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or dtype == 'int8':
        if expect:
            assert torch.equal(y_cal, y_ref)
        else:
            assert not torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))

def raises_with_match(expected_exception, match_pattern):
    def decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            with pytest.raises(expected_exception, match=match_pattern):
                return test_func(*args, **kwargs)
        return wrapper
    return decorator

def capture_output(expected_output):
    def decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            capsys = kwargs.pop('capsys', None)
            if capsys is None:
                try:
                    capsys = pytest.fixture(capsys)()
                except:
                    raise RuntimeError("This decorator requires pytest's capsys fixture")
            test_func(capsys, *args, **kwargs)
            captured = capsys.readouterr()
            # pybind11::scoped_ostream_redirect captures std::cout with \x00 inserted
            # for now, no idea how to eliminate \x00 from C++ side.
            cleaned = re.sub(r"\x00", "", captured.out)
            assert expected_output in cleaned
        return wrapper
    return decorator
