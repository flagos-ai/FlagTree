import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    hiddensize = 128
    dtype = torch.float32
    batch_size = 4
    topk = 5
    largest = True

    x = torch.arange(hiddensize, dtype=dtype, device=flag_gems.device)
    x = x.repeat(batch_size).reshape(batch_size, hiddensize)

    # Each row use different shuffled index.
    for bsz in range(batch_size):
        col_indices = torch.randperm(x.size(1))
        x[bsz, :] = x[bsz, col_indices]

    ref_value, ref_index = torch.topk(x, topk, largest=largest)

    with flag_gems.use_gems():
        res_value, res_index = torch.topk(x, topk, largest=largest)

    torch.testing.assert_close(res_index, ref_index, atol=0, rtol=0)
    torch.testing.assert_close(res_value, ref_value, atol=1e-2, rtol=0)
    print("PASS")
