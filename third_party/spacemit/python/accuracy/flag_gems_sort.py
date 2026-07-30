import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    batch_size = 4
    hiddensize = 256
    descending = True
    dtype = torch.float32
    dim = 0
    FLOAT_DTYPES = [torch.float16, torch.float32]
    if dtype in FLOAT_DTYPES:
        x = torch.empty((hiddensize, ), dtype=dtype, device=flag_gems.device)
        tmp = torch.tensor(0, dtype=dtype)
        inf = torch.tensor(float("inf"), dtype=dtype)
        for i in range(0, hiddensize):
            x[i] = tmp.item()
            tmp = torch.nextafter(tmp, inf)
            if tmp.item() == inf.item():
                hiddensize = i
                x = x[:hiddensize]
                break
    else:
        if flag_gems.device == "musa" and dtype == torch.int16:
            # arange short type on torch of mthreads not supported yet.
            x = torch.arange(hiddensize, dtype=torch.int32, device=flag_gems.device).to(dtype)
        else:
            x = torch.arange(hiddensize, dtype=dtype, device=flag_gems.device)
    y = torch.empty((batch_size, hiddensize), dtype=dtype, device=flag_gems.device)

    # Each row use different shuffled index.
    col_indices = torch.randperm(x.size(0))
    for bsz in range(batch_size):
        col_indices = torch.randperm(x.size(0))
        y[bsz, :] = x[col_indices]
    if dim == 0:
        y = torch.movedim(y, dim, -1)
    ref_value, ref_index = torch.sort(y, dim=dim, descending=descending)

    with flag_gems.use_gems():
        res_value, res_index = torch.sort(y, dim=dim, descending=descending)

    torch.testing.assert_close(res_value, ref_value, atol=1e-2, rtol=0)
    torch.testing.assert_close(res_index, ref_index, atol=1e-2, rtol=0)
    print("PASS")
