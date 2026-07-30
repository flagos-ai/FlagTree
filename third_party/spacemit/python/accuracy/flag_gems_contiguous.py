import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    shape = (2, 19, 7)
    dtype = torch.float32
    FLOAT_DTYPES = [torch.float16, torch.float32]
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(low=-10000, high=10000, size=shape, dtype=dtype, device=flag_gems.device)
    inp = inp[::2]
    # assert inp.is_contiguous() is False

    ref_out = inp.contiguous()
    with flag_gems.use_gems():
        res_out = inp.contiguous()

    assert res_out.is_contiguous() is True
    assert res_out.is_contiguous() is True
    assert res_out.stride() == ref_out.stride()

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
