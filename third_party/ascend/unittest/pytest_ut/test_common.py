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

from typing import Optional
import torch
import torch_npu
import pytest
import functools
import re
import numpy as np

_float_dtypes = ['float32', 'float16', 'bfloat16']
_int_dtypes = ['int32', 'int64', 'int16', 'int8']
_uint_dtypes = ['uint8', 'uint16', 'uint32', 'uint64']
_all_dtypes_no_bool = _float_dtypes + _int_dtypes
_all_dtypes = _all_dtypes_no_bool + ['bool']
_32bit_dtypes = ['float32', 'int32']
_16bit_dtypes = ['float16', 'bfloat16', 'int16']


def generate_numpy(shape, dtype, low=None, high=None):
    if dtype in _int_dtypes + _uint_dtypes:
        iinfo = np.iinfo(getattr(np, dtype))
        low = iinfo.min if low is None else max(low, iinfo.min)
        high = iinfo.max if high is None else min(high, iinfo.max)
        dty = getattr(np, dtype)
        return np.random.randint(low, high, shape, dtype=dty)
    elif dtype == 'float16' or dtype == 'float32':
        return np.random.normal(0, 1, shape).astype(dtype)
    elif dtype == 'bfloat16':
        return (np.random.normal(0, 1, shape).astype('float32').view('uint32') & np.uint32(0xffff0000)).view('float32')
    elif dtype == 'bool':
        return np.random.randint(low=0, high=2, size=shape).astype(bool)
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))


def generate_tensor(shape, dtype):
    if dtype == 'float32' or dtype == 'float16' or dtype == 'bfloat16':
        return torch.randn(size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16':
        return torch.randint(low=0, high=2000, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int8':
        return torch.randint(low=0, high=127, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'bool':
        return torch.randint(low=0, high=2, size=shape).bool()
    elif dtype == 'uint8':
        return torch.randint(low=0, high=255, size=shape, dtype=torch.uint8)
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))


def generate_tensor_fp8(
        shape, dtype, seed=None, data_range=None,
        special_ratio=0.05,  # Ratio of inf/-inf/nan values (only takes effect when dtype is float type)
        precision_ratio=0.35,  # Ratio of precision small floats (only takes effect when dtype is float type)
        extreme_ratio=0.1,  # Ratio of extreme value region (20% for min~min+10, max-10~max, 80% for min and max themselves) (no effect when dtype is bool)
        zero_ratio=0,  # Extra ratio of zeros (no effect when dtype is bool)
        bool_true_ratio=0.5,  # Ratio of True (only takes effect when dtype is bool)
):
    middle_range_dict = {
        'fp8e4m3': [-50, 50],
        'fp8e5m2': [-500, 500],
        'fp8e5b16': [-500, 500],
    }

    float_precision_dict = {
        'fp8e4m3': [-1, 1],
        'fp8e5m2': [-1, 1],
        'fp8e5b16': [-1, 1],
    }

    extreme_range_dict = {
        'fp8e4m3': [-448, 448],
        'fp8e5m2': [-57344, 57344],
        'fp8e5b16': [-57344, 57344],
    }

    ddtype_dict = {
        'fp8e4m3': torch.float8_e4m3fn,
        'fp8e5m2': torch.float8_e5m2,
        'fp8e5b16': None,
    }

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if dtype not in ddtype_dict:
        raise ValueError(f"Unsupported dtype: {dtype}")

    min_val, max_val = extreme_range_dict[dtype]
    total = np.prod(shape)

    left_range = (min_val, min_val + 10)
    right_range = (max_val - 10, max_val)

    if data_range == None:
        mid_range = middle_range_dict[dtype]
    else:
        mid_range = data_range

    # Process float types
    if dtype in ('float16', 'float32', 'bfloat16', 'fp8e4m3', 'fp8e5m2'):

        # Specified distribution probability and handling when exceeding 100%
        if (extreme_ratio + precision_ratio + zero_ratio + special_ratio > 1):
            raise ValueError(f"Total ratio over 100%, unable to allocate sampling:{extreme_ratio + precision_ratio}")

        # Read dictionary to get precision small float data range
        precision_range = float_precision_dict[dtype]

        # Generate initial 1D zero data, float32 to prevent overflow
        data = np.zeros(total, dtype=np.float32)

        # Calculate counts for each interval (extreme region cumulative: near min, exact min, near max, exact max order)
        left_count = int(total * extreme_ratio / 10)
        left_accumulate_count = int(total * extreme_ratio / 2)
        left_right_count = int(total * extreme_ratio * 3 / 5)
        left_right_accmulate_count = int(total * extreme_ratio)
        precision_count = int(total * precision_ratio)
        zero_count = int(total * zero_ratio)
        special_count = int(total * special_ratio)

        # Generate shuffled index mapping array
        indices = np.random.choice(total, size=total, replace=False)

        # Partition interval indices by counts
        left_idx = indices[:left_count]
        left_min_idx = indices[left_count:left_accumulate_count]
        right_idx = indices[left_accumulate_count:left_right_count]
        right_max_idx = indices[left_right_count:left_right_accmulate_count]
        precision_idx = indices[left_right_accmulate_count:left_right_accmulate_count + precision_count]
        zero_idx = indices[left_right_accmulate_count + precision_count:left_right_accmulate_count + precision_count +
                           zero_count]
        special_idx = indices[left_right_accmulate_count + precision_count + zero_count:left_right_accmulate_count +
                              precision_count + zero_count + special_count]
        mid_idx = indices[left_right_accmulate_count + precision_count + zero_count + special_count:]

        # Generate data by interval indices
        data[left_idx] = np.random.uniform(left_range[0], left_range[1], size=len(left_idx))
        data[left_min_idx] = min_val
        data[right_idx] = np.random.uniform(right_range[0], right_range[1], size=len(right_idx))
        data[right_max_idx] = max_val
        data[precision_idx] = np.random.uniform(precision_range[0], precision_range[1], size=len(precision_idx))
        data[zero_idx] = 0
        for idx in special_idx:
            r = np.random.rand()
            if r < 1 / 3:
                data[idx] = float('inf')
            elif r < 2 / 3:
                data[idx] = float('-inf')
            else:
                data[idx] = float('nan')
        data[mid_idx] = np.random.uniform(mid_range[0], mid_range[1], size=len(mid_idx))

        tensor = torch.from_numpy(data.reshape(shape)).to(ddtype_dict[dtype])

    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return tensor


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
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or dtype == 'int8':
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'uint8' or dtype == 'uint16' or dtype == 'uint32' or dtype == 'uint64':
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'bool':
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))


# moving and comparison ops require no precision error
def validate_cmp(dtype, y_cal, y_ref, overflow_mode: Optional[str] = None):
    y_cal = y_cal.npu()
    y_ref = y_ref.npu()
    if overflow_mode == "saturate":
        if dtype in ['float32', 'float16']:
            min_value = -torch.finfo(dtype).min
            max_value = torch.finfo(dtype).max
        elif dtype in ['int32', 'int16', 'int8']:
            min_value = torch.iinfo(dtype).min
            max_value = torch.iinfo(dtype).max
        elif dtype == 'bool':
            min_value = 0
            max_value = 1
        else:
            raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))
        y_ref = torch.clamp(y_ref, min=min_value, max=max_value)
    if dtype == 'float16':
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype == 'bfloat16':
        torch.testing.assert_close(y_ref.to(torch.float32), y_cal.to(torch.float32), rtol=1e-03, atol=1e-03,
                                   equal_nan=True)
    elif dtype == 'float32':
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-04, atol=1e-04, equal_nan=True)
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or dtype == 'int8':
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'uint8' or dtype == 'uint16' or dtype == 'uint32' or dtype == 'uint64':
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'bool':
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))


def validate_cmp_with_expection(dtype, y_cal, y_ref, expect):
    if dtype == 'float32' or dtype == 'float16' or dtype == 'bfloat16':
        if expect:
            assert torch.allclose(y_ref, y_cal, rtol=1e-03, atol=1e-03, equal_nan=True)
        else:
            assert not torch.allclose(y_ref, y_cal, rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or dtype == 'int8' \
        or dtype == 'uint8' or dtype == 'uint16' or dtype == 'uint32' or dtype == 'uint64':
        if expect:
            assert torch.equal(y_cal, y_ref)
        else:
            assert not torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))


# Use the following pytest fixture to run one test case by only single worker.
# Refer to https://pytest-xdist.readthedocs.io/en/stable/how-to.html#making-session-scoped-fixtures-execute-only-once
@pytest.fixture(scope="function")
def pytest_runonce(worker_id, request, cache):
    if (cache.get(request.node.nodeid, "none")) == "none":
        cache.set(request.node.nodeid, worker_id)
    else:
        file_name = f"pytest_{worker_id}.txt"
        with open(file_name, 'a') as file:
            file.write(f"{request.node.nodeid} is already processed by {worker_id}")
        return True
    yield True
    cache.set(request.node.nodeid, "none")


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


def pytest_collection_modifyitems(config, items):
    keep_items = []
    for item in items:
        if not hasattr(item, "callspec"):
            continue
        mark_obj = pytest.mark.case_info(level="P0", type="Functional")
        item.add_marker(mark_obj)
        #if not any(m.name == "skip" for m in item.own_markers):
        #    keep_items.append(item)
    #items[:] = keep_items
