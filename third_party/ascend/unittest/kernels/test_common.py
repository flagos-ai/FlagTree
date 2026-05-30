from typing import Optional
import torch
import pytest

DEVICE_TYPE_NPU = 'npu'


def validate_cmp(dtype, y_cal, y_ref, overflow_mode: Optional[str] = None, device_type: Optional[str] = None):
    if device_type is not None:
        target_device = torch.device(device_type)
        y_cal = y_cal.to(target_device)
        y_ref = y_ref.to(target_device)
    else:
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
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype in ['int64', 'int32', 'int16', 'int8']:
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'bool':
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))


def convert_tensor_with_device_type(indata: dict, device_type: str):
    target_device = torch.device(device_type)
    outdata = {}

    for key, value in indata.items():
        if isinstance(value, torch.Tensor):
            if value.device.type != target_device.type:
                outdata[key] = value.to(target_device)
            else:
                outdata[key] = value
        else:
            outdata[key] = value

    return outdata


def compare_data_precision(dict_ref: dict, dict_cal: dict, device_type: str):
    keys_ref, keys_cal = set(dict_ref.keys()), set(dict_cal.keys())
    if not keys_ref.issubset(keys_cal):
        raise ValueError("The keys of dict_ref is not subset of dict_cal")

    for key in dict_ref.keys():
        val_a, val_b = dict_ref[key], dict_cal[key]
        if not isinstance(val_b, type(val_a)):
            raise ValueError("The data type of two dicts are different")

        if isinstance(val_a, torch.Tensor):
            try:
                promoted_dtype = torch.promote_types(val_a.dtype, val_b.dtype)
            except Exception:
                # Fallback: if promote_types fails for some exotic dtypes, cast both to float32
                promoted_dtype = torch.float32
            val_a_cmp = val_a.to(promoted_dtype) if val_a.dtype != promoted_dtype else val_a
            val_b_cmp = val_b.to(promoted_dtype) if val_b.dtype != promoted_dtype else val_b
            validate_cmp(dtype=str(promoted_dtype).split('.')[-1], y_ref=val_a_cmp, y_cal=val_b_cmp,
                         device_type=device_type)


def run_and_compare_ptfile(ptfile_path: str, kernel_runner, device_type: str = DEVICE_TYPE_NPU):
    try:
        datas = torch.load(ptfile_path, map_location=torch.device('cpu'))
    except Exception as e:
        pytest.fail(f"load file {ptfile_path} failed: {e}")

    def _run_single_case(data):
        if not isinstance(data, dict):
            pytest.fail("Each case loaded from pt file must be a dict")

        input_data = convert_tensor_with_device_type(data.get("input_data", {}), device_type=device_type)
        grid = data.get("grid")
        try:
            kernel_runner(input_data, grid)
        except Exception as e:
            pytest.fail(f"kernel_runner execution failed: {e}")

        output_data_cpu = convert_tensor_with_device_type(input_data, device_type='cpu')
        expected = data.get("gpu_output", {})
        expected_filtered = {k: expected[k] for k in output_data_cpu.keys() if k in expected}
        if not expected_filtered:
            pytest.fail("No matching expected outputs found in pt file for comparison")
        try:
            compare_data_precision(expected_filtered, output_data_cpu, device_type='cpu')
        except Exception as e:
            pytest.fail(f"The testcase failed: {e}")

    # Supports three scenarios:
    # 1) The file stores a single dict (existing behavior)
    # 2) The file stores a list, where each element is a case dict
    # 3) The file stores a dict, but some tensors represent multiple cases in batch on the 0th dimension (no automatic splitting; it is recommended to use a list)
    if isinstance(datas, list):
        for _, data in enumerate(datas):
            _run_single_case(data)
    elif isinstance(datas, dict):
        _run_single_case(datas)
    else:
        pytest.fail("Unsupported pt file format: must be a dict or a list of dicts")
