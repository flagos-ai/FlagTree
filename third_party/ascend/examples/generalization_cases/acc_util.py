import numpy as np
import torch
import torch_npu

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


def assert_close(gold: torch.Tensor, act: torch.Tensor, eval_type: str = 'DEFAULT'):
    gold = gold.cpu()
    act = act.cpu()
    if act.dtype == torch.float16 or act.dtype == torch.float32 or act.dtype == torch.bfloat16:
        assert gold.dtype == torch.float32, "golden should be f32"
        assert not (torch.isnan(act).any() or torch.isinf(act).any()), "actual tensor can not have 'inf' or 'nan'"
    eps = eval_standard[act.dtype]['small_value']
    rtol = eval_standard[act.dtype]['rtol']
    atol = eval_standard[act.dtype]['small_value_atol']
    if eval_type == 'DEFAULT':
        ae = torch.abs(act - gold)
        re = ae / torch.abs(gold)
        mask = torch.abs(gold) < eps

        print(f"count ae > {atol}: {(ae > atol).sum()}")
        print(f"count re > {rtol}: {(re > rtol).sum()}")

        not_close = torch.where(mask, ae > atol, re > rtol)
        print(f"count not_close = {torch.sum(not_close).item()}")
        print(f"not_close.numel = {not_close.numel()}, gold.numel = {gold.numel()}")
        print(f"not close ratio = {torch.sum(not_close).item() / not_close.numel()}")
        if not torch.any(not_close):
            return False

        assert torch.sum(not_close).item() < not_close.numel() * eps, "actual tensor are not close enough with golden tensor,\
you can use 'benchmark_compare_close' function to compare again!"
    elif eval_type == 'ABS':
        act = act.to(gold.dtype)
        assert torch.equal(gold, act), "actual tensor and golden tensor are not binary equal!"
    else:
        assert 0, "ERROR! invalid eval_type"
    return False


def benchmark_compare_close(gold: torch.Tensor, act: torch.Tensor, std: torch.tensor):
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
