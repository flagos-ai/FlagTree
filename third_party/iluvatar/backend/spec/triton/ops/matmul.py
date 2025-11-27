from triton import language as tl


def only_supports_num_stages_le_2():
    return True


def get_configs_compute_bound():
    import torch
    from triton import Config
    configs = []
    if hasattr(torch, "corex"):
        for block_m in [32, 64, 128, 256]:
            for block_n in [32, 64, 128, 256]:
                for block_k in [32, 64, 128]:
                    for num_stages in [1, 2, 3]:
                        num_warps = 16 if block_m >= 128 or block_n >= 128 or block_k >= 128 else 8
                        configs.append(
                            Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': 1},
                                   num_stages=num_stages, num_warps=num_warps))
    return configs


def get_nv_configs():
    import torch
    from triton import Config
    configs = []
    if hasattr(torch, "corex"):
        return configs
    configs = [
        # basic configs for compute-bound matmuls
        Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=5, num_warps=2),
        # good for int8
        Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=5, num_warps=2),
    ]
    return configs


def matmul_kernel(grid, a, b, c, M, N, K, acc_dtype, input_precision, fp8_fast_accum, ab_dtype):
    from triton import autotune, heuristics, jit
    from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time
    from triton.ops.matmul import get_configs_io_bound

    @autotune(
        configs=get_nv_configs() + get_configs_io_bound() + get_configs_compute_bound(),
        key=['M', 'N', 'K'],
        prune_configs_by={
            'early_config_prune': early_config_prune,
            'perf_model': estimate_matmul_time,
            'top_k': 15,
        },
    )
    @heuristics({
        'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
    })
    @jit
    def _kernel(A, B, C, M, N, K,  #
                stride_am, stride_ak,  #
                stride_bk, stride_bn,  #
                stride_cm, stride_cn,  #
                acc_dtype: tl.constexpr,  #
                input_precision: tl.constexpr,  #
                fp8_fast_accum: tl.constexpr,  #
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
                GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr, AB_DTYPE: tl.constexpr  #
                ):
        # matrix multiplication
        pid = tl.program_id(0)
        pid_z = tl.program_id(1)
        grid_n = tl.cdiv(N, BLOCK_N)
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        # do matrix multiplication
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
        rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
        # pointers
        A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
        B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
        for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
            if EVEN_K:
                a = tl.load(A)
                b = tl.load(B)
            else:
                k_remaining = K - k * (BLOCK_K * SPLIT_K)
                _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
                a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
                b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
            if AB_DTYPE is not None:
                a = a.to(AB_DTYPE)
                b = b.to(AB_DTYPE)
            if fp8_fast_accum:
                acc = tl.dot(a, b, acc, out_dtype=acc_dtype, input_precision=input_precision)
            else:
                acc += tl.dot(a, b, out_dtype=acc_dtype, input_precision=input_precision)
            A += BLOCK_K * SPLIT_K * stride_ak
            B += BLOCK_K * SPLIT_K * stride_bk
        acc = acc.to(C.dtype.element_ty)
        # rematerialize rm and rn to save registers
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        # handles write-back with reduction-splitting
        if SPLIT_K == 1:
            tl.store(C, acc, mask=mask)
        else:
            tl.atomic_add(C, acc, mask=mask)

    return _kernel[grid](
        a, b, c, M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        acc_dtype=acc_dtype,  #
        input_precision=input_precision,  #
        fp8_fast_accum=fp8_fast_accum,  #
        GROUP_M=8, AB_DTYPE=ab_dtype)
