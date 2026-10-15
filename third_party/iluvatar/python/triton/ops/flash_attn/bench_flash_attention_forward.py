import torch
import triton
import triton.ops

configs = [
    triton.testing.Benchmark(
        x_names=['batch_size', 'nheads_q', 'nheads_k', 'seqlen_q', 'seqlen_k', 'headdim', 'causal',
                 'mode'],  # Argument names to use as an x-axis for the plot
        x_vals=[(1, 64, 64, seqlen, seqlen, 128, causal, mode)
                for mode in ["fwd"]
                for causal in [False, True]
                for seqlen in [1024 * 2**i for i in range(0, 6)]],
        line_arg="provider",
        line_vals=["triton", "torch-aten"],
        line_names=["Triton", "Torch ATen"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="TFLOPS",
        plot_name="flash-attention-mr-performance-bf16",
        args={},
    )
]


def _make_inputs(batch_size, nheads_q, nheads_k, seqlen_q, seqlen_k, headdim, dtype, device):
    q = torch.empty((batch_size, seqlen_q, nheads_q, headdim), dtype=dtype, device=device)
    k = torch.empty((batch_size, seqlen_k, nheads_k, headdim), dtype=dtype, device=device)
    v = torch.empty((batch_size, seqlen_k, nheads_k, headdim), dtype=dtype, device=device)
    q.normal_(mean=0.1, std=0.2)
    k.normal_(mean=0.4, std=0.2)
    v.normal_(mean=0.3, std=0.2)
    return q.requires_grad_(), k.requires_grad_(), v.requires_grad_()


def _triton_flash_fwd(q, k, v, scale, causal):
    return triton.ops._flash_attention_forward(
        q,
        k,
        v,
        None,
        None,
        q.shape[-3],
        k.shape[-3],
        0.0,
        causal,
        False,
        scale=scale,
    )[0]


def _torch_flash_fwd(q, k, v, scale, causal):
    return torch.ops.aten._flash_attention_forward(
        q,
        k,
        v,
        None,
        None,
        q.shape[-3],
        k.shape[-3],
        0.0,
        causal,
        False,
        scale=scale,
    )[0]


@triton.testing.perf_report(configs)
def bench_op_fwd_prefill(
    batch_size,
    nheads_q,
    nheads_k,
    seqlen_q,
    seqlen_k,
    headdim,
    causal,
    mode,
    provider,
    dtype=torch.bfloat16,
    device="cuda",
):
    assert mode in ["fwd", "bwd"]
    warmup = 100
    rep = 100
    sm_scale = 1.3

    torch.manual_seed(42)
    q, k, v = _make_inputs(batch_size, nheads_q, nheads_k, seqlen_q, seqlen_k, headdim, dtype, device)

    if provider == "triton":
        fn = lambda: _triton_flash_fwd(q, k, v, sm_scale, causal)
    elif provider == "torch-aten":
        fn = lambda: _torch_flash_fwd(q, k, v, sm_scale, causal)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # if mode == "bwd":
    #     o = fn()
    #     do = torch.randn_like(o)
    #     fn = lambda: o.backward(do, retain_graph=True)

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    flops_per_matmul = 2.0 * batch_size * nheads_q * seqlen_q * seqlen_k * headdim
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    # if mode == "bwd":
    #     total_flops *= 2.5

    return total_flops / ms * 1e-9


if __name__ == "__main__":
    bench_op_fwd_prefill.run(save_path=".", print_data=True)
