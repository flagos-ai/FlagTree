import numpy as np
import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    batch = 8
    num_head = 1
    q_seq_len = 17
    kv_seq_len = 7
    head_size = 64
    add_bias = True
    is_causal = True
    dtype = torch.float16
    device = "cpu"
    np.random.seed(0)
    np_query = np.random.uniform(-0.05, 0.05, (batch, num_head, q_seq_len, head_size)).astype(np.float32)
    np_key = np.random.uniform(-0.05, 0.05, (batch, num_head, kv_seq_len, head_size)).astype(np.float32)
    np_value = np.random.uniform(-0.05, 0.05, (batch, num_head, kv_seq_len, head_size)).astype(np.float32)
    np_attn_bias = np.random.uniform(-0.05, 0.05, (batch, num_head, q_seq_len, kv_seq_len)).astype(np.float32)

    query = torch.tensor(np_query, device=device, dtype=dtype)
    key = torch.tensor(np_key, device=device, dtype=dtype)
    value = torch.tensor(np_value, device=device, dtype=dtype)
    if add_bias:
        attn_bias = torch.tensor(np_attn_bias, device=device, dtype=dtype)
    else:
        attn_bias = None

    scale = float(1.0 / np.sqrt(head_size))

    if is_causal:
        torch_result = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=scale,
            is_causal=is_causal,
        )
    else:
        torch_result = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_bias,
            scale=scale,
            is_causal=is_causal,
        )

    with flag_gems.use_gems():
        if is_causal:
            flaggem_result = torch.nn.functional.scaled_dot_product_attention(query, key, value, scale=scale,
                                                                              is_causal=is_causal)
        else:
            flaggem_result = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attn_bias,
                                                                              scale=scale, is_causal=is_causal)
    torch.testing.assert_close(flaggem_result, torch_result, atol=1e-2, rtol=0)
    print("PASS")
