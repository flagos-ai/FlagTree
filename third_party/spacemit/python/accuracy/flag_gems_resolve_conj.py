import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    shape = (2, 19, 7)
    x = torch.randn(size=shape, dtype=torch.cfloat, device="cpu")
    y = x.conj()
    assert y.is_conj()
    with flag_gems.use_gems():
        res_y = y.to(device=flag_gems.device)
        z = res_y.resolve_conj()
    assert not z.is_conj()
    print("PASS")
