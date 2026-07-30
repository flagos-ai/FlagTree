import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    reduction = "mean"
    weight = True
    shape = (2, 32)
    dtype = torch.float32
    ignore_index = 1
    dim = 1
    target_shape = list(shape)
    del target_shape[dim]

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    target = torch.randint(0, shape[dim], target_shape, device=flag_gems.device)
    if weight:
        weight = torch.randn(shape[dim], dtype=dtype, device=flag_gems.device)
    else:
        weight = None

    ref_out = torch.nn.functional.nll_loss(inp, target, weight, reduction=reduction, ignore_index=ignore_index)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.nll_loss(inp, target, weight, reduction=reduction, ignore_index=ignore_index)
    reduce_dim = 1 if reduction == "none" else target.numel()

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
