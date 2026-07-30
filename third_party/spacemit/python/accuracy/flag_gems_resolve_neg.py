import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    shape = (2, 19, 7)
    x = torch.randn(size=shape, dtype=torch.cfloat, device=flag_gems.device)
    y = x.conj()
    z = y.imag
    assert z.is_neg()
    with flag_gems.use_gems():
        out = z.resolve_neg()
    assert not out.is_neg()
    print("PASS")
