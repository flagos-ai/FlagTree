import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    shape = (16, )
    dtype = torch.float32
    n_samples = 1000
    if shape[-1] == 1:
        dist = torch.rand(size=shape, dtype=dtype, device=flag_gems.device)
        res_out = flag_gems.multinomial(dist, n_samples, True)
        assert torch.all(res_out == 0)
    else:
        # Mask p% off of the categories and test the sampling results fall in the rest
        for p in (0.1, 0.5, 0.9):
            dist = torch.rand(size=shape, dtype=dtype, device=flag_gems.device)
            dist[torch.rand(shape) < p] = 0
            # Make sure there's at least one non-zero probability
            dist[..., -1] = 0.5
            res_out = flag_gems.multinomial(dist, n_samples, True)
            res_dist = torch.gather(dist, -1, res_out)
            # assert torch.all(res_dist)
            assert torch.sum(res_dist == 0) / res_dist.numel() < 0.001
    print("PASS")
