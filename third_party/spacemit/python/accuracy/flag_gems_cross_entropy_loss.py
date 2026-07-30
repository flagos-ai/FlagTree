import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    label_smoothing = 0.1
    ignore_index = 1
    shape = (2, 32)
    reduction = "mean"
    weight = True
    dtype = torch.float32
    dim = 1
    up_limit = shape[dim] - 1
    target_shape = list(shape)
    del target_shape[dim]

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    target = torch.randint(0, up_limit, target_shape, device=flag_gems.device)

    if weight:
        wgt = torch.randn(shape[dim], dtype=dtype, device=flag_gems.device)
    else:
        wgt = None
    ref_out = torch.nn.functional.cross_entropy(
        inp,
        target,
        weight=wgt,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
    res_out = flag_gems.cross_entropy_loss(
        inp,
        target,
        weight=wgt,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
