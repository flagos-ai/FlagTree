import numpy as np
import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    shape = (2, 19, 7)
    dtype = torch.float32
    p = 0.3
    res_inp = torch.randn(
        shape,
        dtype=dtype,
        device=flag_gems.device,
    )

    # NOTE: ensure that scalars are float32(instead of float64)
    # in some cases, casting up then casting down have different result
    p = np.float32(p)
    one_minus_p = np.float32(1.0) - p

    ref_out = torch.nn.functional.dropout(res_inp, p, True)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.dropout(res_inp, p, True)

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
