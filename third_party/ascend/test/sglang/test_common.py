from typing import Optional
import torch
import torch_npu
import pytest

eval_standard = {
    torch.float32: {
        "rtol": 1e-6,
        "small_value": 1e-6,
        "small_value_atol": 1e-9,
        "etol": 1e-4,
    },
    torch.float16: {
        "rtol": 1e-3,
        "small_value": 1e-3,
        "small_value_atol": 1e-5,
        "etol": 1e-3,
    },
    torch.bfloat16: {
        "rtol": 4e-3,
        "small_value": 1e-3,
        "small_value_atol": 1e-5,
        "etol": 1e-3,
    },
}


def validate_cmp(dtype, y_cal, y_ref, overflow_mode: Optional[str] = None, device_type: Optional[str] = None):
    if device_type is not None:
        target_device = torch.device(device_type)
        y_cal = y_cal.to(target_device)
        y_ref = y_ref.to(target_device)
    else:
        y_cal=y_cal.npu()
        y_ref=y_ref.npu()
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
        torch.testing.assert_close(y_ref, y_cal, rtol=5e-03, atol=5e-03, equal_nan=True)
    elif dtype == 'bfloat16':
        torch.testing.assert_close(y_ref.to(torch.float32), y_cal.to(torch.float32), rtol=5e-03, atol=5e-03, equal_nan=True)
    elif dtype == 'float32':
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or dtype == 'int8':
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
        if type(val_a) != type(val_b):
            raise ValueError("The data type of two dicts are different")

        if isinstance(val_a, torch.Tensor):
            validate_cmp(dtype=str(val_a.dtype).split('.')[-1], y_ref=val_a, y_cal=val_b, device_type=device_type)
        else:
            raise ValueError("Non-tensor type is not currently supported")


# 验证浮点型数据的精度时，由于底层实现的差异，难以对比GPU、NPU
# 将NPU结果升级到fp32类型gold，在差异较小的情况下认为NPU结果act和GPU结果std精度通过
# 特别是对于float16和bfloat16类型
def verify_precision_by_gold_standard(gold: torch.Tensor, act: torch.Tensor, std: torch.tensor):
    assert act.dtype == std.dtype, "standard tensor's dtype must equal to actual tensor's dtype!"
    if act.dtype == torch.float16 or act.dtype == torch.float32 or act.dtype == torch.bfloat16:
        assert gold.dtype == torch.float32, "golden should be f32"
        assert not (torch.isnan(act).any() or torch.isinf(act).any()), "actual tensor can not have 'inf' or 'nan'"

    gold = gold.cpu()
    act = act.cpu()
    std = std.cpu()

    eps = eval_standard[act.dtype]['small_value']
    atol = eval_standard[act.dtype]['small_value_atol']

    mask = torch.abs(gold) <= eps
    small_count = mask.sum().item()

    def calculate_relative_errors_except_small(tensor):
        re = torch.abs(gold - tensor) / torch.abs(gold)
        return torch.where(mask, 0, re)

    act_re = calculate_relative_errors_except_small(act)
    std_re = calculate_relative_errors_except_small(std)
    act_ae = torch.abs(gold - std)
    std_ae = torch.abs(gold - std)

    # 小值域的定义为golden小于某个阈值 eps
    act_small_error_count = (mask & (act_ae > atol)).sum().item()
    std_small_error_count = (mask & (std_ae > atol)).sum().item()
    act_total = act.numel()
    std_total = std.numel()

    act_small_error_ratio = act_small_error_count / act_total
    std_small_error_ratio = std_small_error_count / std_total

    def calculate_rmse(tensor):
        dlt2 = (tensor - gold) ** 2
        dlt2_except_small_mean = torch.where(mask, 0, dlt2).sum() / small_count
        return torch.sqrt(dlt2_except_small_mean)

    act_rmse = calculate_rmse(act)
    std_rmse = calculate_rmse(std)

    print(f"act_re.max = {act_re.max()}, std_re.max = {std_re.max()}, limit ratio = 10")
    print(f"act_re.sum = {act_re.sum()}, std_re.sum = {std_re.sum()}, limit_ratio = 2")
    print(
        f"act_small_error_ratio = {act_small_error_ratio}, std_small_error_ratio = {std_small_error_ratio}, limit_ratio = 2")
    print(f"act_rmse = {act_rmse}, std_rmse = {std_rmse}, limit_ratio = 2")

    # 条件 1：actual 与 golden 相对误差最大值超过 10 倍 standard 与 golden 相对误差最大值
    assert act_re.max() <= 10 * std_re.max(), "actual re max > stdandard re max's 10 times"

    # 条件 2：actual 与 golden 相对误差均值超过 2 倍 standard 与 golden 相对误差均值
    assert act_re.sum() <= 2 * std_re.sum(), "actual re sum > stdandard re sum's 2 times"

    # 条件 3：actual 小值域 ERROR 占比超过 standard 的两倍
    assert act_small_error_ratio <= 2 * std_small_error_ratio, "act_small_error_ratio > std_small_error_ratio 's 2 times"

    # 条件 4：actual 均方根误差差于 standard 的两倍
    assert act_rmse <= 2 * std_rmse, "act_rmse > std_rmse 's 2 times"

    return False