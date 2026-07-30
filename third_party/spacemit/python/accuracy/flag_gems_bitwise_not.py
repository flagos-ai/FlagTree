import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    dtype = torch.int32
    BOOL_TYPES = [torch.bool]
    shape = (2, 19, 7)
    if dtype in BOOL_TYPES:
        inp = torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(flag_gems.device)
    else:
        inp = torch.randint(low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu").to(flag_gems.device)

    ref_out = torch.bitwise_not(inp)
    with flag_gems.use_gems():
        res_out = torch.bitwise_not(inp)

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
