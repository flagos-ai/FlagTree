import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    shape = (2, 19, 7)
    dtype = torch.float32
    SPECIAL_VALUES = [float("-inf"), float("inf"), -300]
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if len(shape) == 1:
        special_inputs = torch.tensor(SPECIAL_VALUES, dtype=dtype, device=flag_gems.device)
        inp = torch.cat((inp, special_inputs))

    ref_out = torch.nn.functional.logsigmoid(inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.logsigmoid(inp)

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
