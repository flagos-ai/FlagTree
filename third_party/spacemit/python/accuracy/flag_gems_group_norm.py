import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    N, C, H, W, num_groups = 16, 3, 16, 16, 1
    dtype = torch.float32
    wb_none = False
    res_inp = torch.randn(size=(N, C, H, W), dtype=dtype, device=flag_gems.device)
    if wb_none:
        res_weight = None
        res_bias = None
    else:
        res_weight = torch.randn(size=(C, ), dtype=dtype, device=flag_gems.device)
        res_bias = torch.randn(size=(C, ), dtype=dtype, device=flag_gems.device)
    eps = 1e-5

    ref_out = torch.nn.functional.group_norm(res_inp, num_groups, weight=res_weight, bias=res_bias, eps=eps)

    with flag_gems.use_gems():
        res_out = torch.group_norm(res_inp, num_groups, weight=res_weight, bias=res_bias, eps=eps)

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
