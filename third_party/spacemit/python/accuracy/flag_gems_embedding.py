import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    EmbeddingSize = 1024
    Batch = 2
    M = 4
    N = 8
    padding_idx = None
    scale_grad_by_freq = True
    dtype = torch.float32
    res_indices = torch.randint(0, EmbeddingSize, (Batch, M), device=flag_gems.device, requires_grad=False)
    res_embedding = torch.randn((EmbeddingSize, N), device=flag_gems.device, dtype=dtype, requires_grad=True)

    ref_out = torch.nn.functional.embedding(res_indices, res_embedding, padding_idx,
                                            scale_grad_by_freq=scale_grad_by_freq)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.embedding(
            res_indices,
            res_embedding,
            padding_idx,
            scale_grad_by_freq=scale_grad_by_freq,
        )

    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)
    print("PASS")
