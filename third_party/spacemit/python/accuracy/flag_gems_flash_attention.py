import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


def test_flash_attention(Z, H, Q_LEN, KV_LEN, HEAD_DIM, dtype=torch.float32, is_causal=False):
    """Test flash attention with given parameters"""
    DEVICE = "cpu"
    torch.manual_seed(20)

    q = (torch.empty((Z, H, Q_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, KV_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, KV_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5

    torch_result = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        scale=sm_scale,
        is_causal=is_causal,
    )

    import flag_gems.runtime.backend._spacemit.ops as sp
    flaggem_result = sp.flash_attention(q, k, v, scale=sm_scale, is_causal=is_causal)

    print(f"Test case: Z={Z}, H={H}, Q_LEN={Q_LEN}, KV_LEN={KV_LEN}, HEAD_DIM={HEAD_DIM}")
    print(f"  Query shape: {q.shape}, Key shape: {k.shape}, Value shape: {v.shape}")
    torch.testing.assert_close(flaggem_result, torch_result, atol=1e-2, rtol=0)
    print("  ✓ PASS")


if __name__ == "__main__":
    # Test cases extracted from qwen.log - Prefill phase
    print("=" * 60)
    print("Prefill Phase Test Cases")
    print("=" * 60)
    test_flash_attention(Z=1, H=16, Q_LEN=28, KV_LEN=28, HEAD_DIM=64)

    # Test cases extracted from qwen.log - Decode phase
    print("\n" + "=" * 60)
    print("Decode Phase Test Cases")
    print("=" * 60)

    # Decode phase: query length = 1, key/value length increases from 29 to 59
    decode_kv_lengths = [
        29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
        57, 58, 59
    ]

    for kv_len in decode_kv_lengths:
        test_flash_attention(Z=1, H=16, Q_LEN=1, KV_LEN=kv_len, HEAD_DIM=64)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
