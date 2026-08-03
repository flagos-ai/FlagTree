import imp
import torch
import triton
import triton.language as tl
import triton.experimental.tle as tle
import triton.experimental.tle.language.dsa as dsa
import flag_gems

TILE_NUM = 16
M = 8192
K = 2048
N = 16384
M_SHARD = M // TILE_NUM
AUTOTUNE_BLOCK_M = (64, 256, 512, 1024)
AUTOTUNE_BLOCK_N = (64, 128, 256, 512, 1024)
AUTOTUNE_BLOCK_K = (64, 256, 512, 1024, 2048)
AUTOTUNE_SUB_N = (64, 128, 256, 512, 1024)
AUTOTUNE_SUB_M = (64, 128, 256, 512, 1024)
MAX_DSA_BUFFER_NUMEL = 1024 * 512

ALGORITHM_N_RING = "n_ring"
ALGORITHM_M_RING = "m_ring"
ALGORITHM_FLAG_GEMS_MM = "flaggems_mm"


class Config:
    warm_up = 2
    repetition = 10
    autotune_warm_up = 1
    autotune_repetition = 3


def make_autotune_configs():
    configs = []
    for block_m in AUTOTUNE_BLOCK_M:
        for block_n in AUTOTUNE_BLOCK_N:
            for block_k in AUTOTUNE_BLOCK_K:
                for sub_n in AUTOTUNE_SUB_N:
                    configs.append(
                        triton.Config(kwargs={"BLOCK_M": block_m, "BLOCK_N": block_n,
                                              "BLOCK_K": block_k, "SUB_N": sub_n},
                                      num_stages=1, num_warps=32)
                    )
    return configs


def make_m_autotune_configs():
    configs = []
    for block_m in AUTOTUNE_BLOCK_M:
        for block_n in AUTOTUNE_BLOCK_N:
            for block_k in AUTOTUNE_BLOCK_K:
                for sub_m in AUTOTUNE_SUB_M:
                    configs.append(
                        triton.Config(kwargs={"BLOCK_M": block_m, "BLOCK_N": block_n,
                                              "BLOCK_K": block_k, "SUB_M": sub_m},
                                      num_stages=1, num_warps=32)
                    )
    return configs


def _arg_value(name, named_args, kwargs, default):
    if name in kwargs:
        return kwargs[name]
    if name in named_args:
        return named_args[name]
    return default


def prune_invalid_noc_gemm_configs(configs, named_args, **kwargs):
    m_shard = int(_arg_value("M_SHARD", named_args, kwargs, M_SHARD))
    k = int(_arg_value("K", named_args, kwargs, K))
    n = int(_arg_value("N", named_args, kwargs, N))
    ring_size = int(_arg_value("RING_SIZE", named_args, kwargs, TILE_NUM))

    valid_configs = []
    for config in configs:
        block_m = config.kwargs["BLOCK_M"]
        block_n = config.kwargs["BLOCK_N"]
        block_k = config.kwargs["BLOCK_K"]
        sub_n = config.kwargs["SUB_N"]

        if m_shard % block_m != 0:
            continue
        if k <= max(AUTOTUNE_BLOCK_K) and block_k < k:
            continue
        if sub_n % block_n != 0:
            continue
        if n % (ring_size * sub_n) != 0:
            continue
        if block_k * block_n > MAX_DSA_BUFFER_NUMEL:
            continue

        valid_configs.append(config)

    return valid_configs if valid_configs else [configs[0]]


def prune_invalid_m_noc_gemm_configs(configs, named_args, **kwargs):
    n_shard = int(_arg_value("N_SHARD", named_args, kwargs, N // TILE_NUM))
    k = int(_arg_value("K", named_args, kwargs, K))
    m = int(_arg_value("M", named_args, kwargs, M))
    ring_size = int(_arg_value("RING_SIZE", named_args, kwargs, TILE_NUM))

    valid_configs = []
    for config in configs:
        block_m = config.kwargs["BLOCK_M"]
        block_n = config.kwargs["BLOCK_N"]
        block_k = config.kwargs["BLOCK_K"]
        sub_m = config.kwargs["SUB_M"]

        if n_shard % block_n != 0:
            continue
        if k <= max(AUTOTUNE_BLOCK_K) and block_k < k:
            continue
        if sub_m % block_m != 0:
            continue
        if m % (ring_size * sub_m) != 0:
            continue
        if block_m * block_k > MAX_DSA_BUFFER_NUMEL:
            continue

        valid_configs.append(config)

    return valid_configs if valid_configs else [configs[0]]


def autotune_do_bench(kernel_call, quantiles):
    return triton.testing.do_bench(
        kernel_call,
        warmup=Config.autotune_warm_up,
        rep=Config.autotune_repetition,
        quantiles=quantiles,
    )


def can_run_n_ring(m, n, k, ring_size):
    if m % TILE_NUM != 0:
        return False
    m_shard = m // TILE_NUM
    for block_m in AUTOTUNE_BLOCK_M:
        if m_shard % block_m != 0:
            continue
        for block_n in AUTOTUNE_BLOCK_N:
            for block_k in AUTOTUNE_BLOCK_K:
                if k <= max(AUTOTUNE_BLOCK_K) and block_k < k:
                    continue
                if block_k * block_n > MAX_DSA_BUFFER_NUMEL:
                    continue
                for sub_n in AUTOTUNE_SUB_N:
                    if sub_n % block_n != 0:
                        continue
                    if n % (ring_size * sub_n) == 0:
                        return True
    return False


def can_run_m_ring(m, n, k, ring_size):
    if n % TILE_NUM != 0:
        return False
    n_shard = n // TILE_NUM
    for block_n in AUTOTUNE_BLOCK_N:
        if n_shard % block_n != 0:
            continue
        for block_m in AUTOTUNE_BLOCK_M:
            for block_k in AUTOTUNE_BLOCK_K:
                if k <= max(AUTOTUNE_BLOCK_K) and block_k < k:
                    continue
                if block_m * block_k > MAX_DSA_BUFFER_NUMEL:
                    continue
                for sub_m in AUTOTUNE_SUB_M:
                    if sub_m % block_m != 0:
                        continue
                    if m % (ring_size * sub_m) == 0:
                        return True
    return False


def select_gemm_algorithm(m, n, k, ring_size):
    can_n = can_run_n_ring(m, n, k, ring_size)
    can_m = can_run_m_ring(m, n, k, ring_size)
    if not can_n and not can_m:
        return ALGORITHM_FLAG_GEMS_MM
    if can_m and not can_n:
        return ALGORITHM_M_RING
    if can_n and not can_m:
        return ALGORITHM_N_RING
    if m >= 2 * n:
        return ALGORITHM_M_RING
    return ALGORITHM_N_RING


TILE_PHYSICAL_RELATION = [0, 1, 2, 3, 7, 11, 15, 14, 13, 12, 8, 9, 10, 6, 5, 4]
# Row-group modes keep the same row-snake order and split it by ring_size:
# 2 rings: 8 tiles/ring, 4 rings: 4 tiles/ring, 8 rings: 2 tiles/ring.
ROW_GROUP_PHYSICAL_RELATION = [0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12]

RING_MODE_SPECS = (
    {"mode": "ring", "ring_num": 1, "physical_relation": TILE_PHYSICAL_RELATION},
    {"mode": "two_ring", "ring_num": 2, "physical_relation": ROW_GROUP_PHYSICAL_RELATION},
    {"mode": "four_ring", "ring_num": 4, "physical_relation": ROW_GROUP_PHYSICAL_RELATION},
    {"mode": "eight_ring", "ring_num": 8, "physical_relation": ROW_GROUP_PHYSICAL_RELATION},
)


def make_ring_mesh(physical_relation, shape=None):
    if shape is None:
        shape = (TILE_NUM, )
        dim_names = ("tile", )
        launch_shape = None
        launch_dim_names = None
    else:
        dim_names = ("ring", "ring_tile")
        launch_shape = (TILE_NUM, )
        launch_dim_names = ("tile", )
    return tle.device_mesh(
        None,
        _shape=shape,
        _dim_names=dim_names,
        _physical_ids=tuple(physical_relation),
        _launch_shape=launch_shape,
        _launch_dim_names=launch_dim_names,
    )


RING_MESHES = {
    spec["mode"]: make_ring_mesh(
        spec["physical_relation"],
        None if spec["ring_num"] == 1 else (spec["ring_num"], TILE_NUM // spec["ring_num"]),
    )
    for spec in RING_MODE_SPECS
}


@triton.autotune(
    configs=make_autotune_configs(),
    key=["M", "N", "K", "M_SHARD", "RING_SIZE"],
    prune_configs_by={"early_config_prune": prune_invalid_noc_gemm_configs},
    do_bench=autotune_do_bench,
)
@triton.jit
def dsa_shift_n_gemm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    physical_ids_ptr,
    ring_index_lut_ptr,
    ring_id_lut_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    M_SHARD: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SUB_N: tl.constexpr,
    RING_SIZE: tl.constexpr,
    RING_MESH: tl.constexpr,
):
    pid = tle.shard_id(RING_MESH, axis=0)
    ring_index = tl.load(ring_index_lut_ptr + pid)
    ring_id = tl.load(ring_id_lut_ptr + pid)

    next_ring_pos = tl.where(ring_index == RING_SIZE - 1, 0, ring_index + 1)
    send_next_tile = tl.load(physical_ids_ptr + ring_id * RING_SIZE + next_ring_pos)

    send_buf = dsa.alloc((BLOCK_K, BLOCK_N), tl.float16)
    recv_buf = dsa.alloc((BLOCK_K, BLOCK_N), tl.float16)

    offs_buf_k = tl.arange(0, BLOCK_K)[:, None] + tl.zeros((1, BLOCK_N), dtype=tl.int32)
    offs_buf_n = tl.arange(0, BLOCK_N)[None, :] + tl.zeros((BLOCK_K, 1), dtype=tl.int32)

    send_ptr = dsa.local_ptr(send_buf, [offs_buf_k, offs_buf_n])
    recv_ptr = dsa.local_ptr(recv_buf, [offs_buf_k, offs_buf_n])

    remote_recv_buf = tle.remote(recv_buf, send_next_tile, scope=RING_MESH)
    remote_recv_ptr = dsa.local_ptr(remote_recv_buf, [offs_buf_k, offs_buf_n])
    remote_send_buf = tle.remote(send_buf, send_next_tile, scope=RING_MESH)
    remote_send_ptr = dsa.local_ptr(remote_send_buf, [offs_buf_k, offs_buf_n])

    for m_block in range(0, tl.cdiv(M_SHARD, BLOCK_M)):
        offs_m = pid * M_SHARD + m_block * BLOCK_M + tl.arange(0, BLOCK_M)

        for sub_n_block in range(0, tl.cdiv(N, RING_SIZE * SUB_N)):
            sub_n_base = sub_n_block * RING_SIZE * SUB_N

            for n_block in range(0, tl.cdiv(SUB_N, BLOCK_N)):
                if K <= BLOCK_K:
                    offs_k = tl.arange(0, BLOCK_K)
                    a_ptrs = A_ptr + offs_m[:, None] * K + offs_k[None, :]
                    a = tl.load(a_ptrs, mask=offs_k[None, :] < K, other=0.0)

                    shard_idx = ring_index
                    offs_sub_n = sub_n_base + shard_idx * SUB_N + n_block * BLOCK_N + tl.arange(0, BLOCK_N)
                    b_ptrs = B_ptr + offs_k[:, None] * N + offs_sub_n[None, :]
                    b_init = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0.0)
                    tl.store(send_ptr, b_init)

                    for step in range(0, RING_SIZE, 2):
                        b_cur = tl.load(send_ptr)
                        c_part = tl.dot(a, b_cur)

                        offs_n = sub_n_base + shard_idx * SUB_N + n_block * BLOCK_N + tl.arange(0, BLOCK_N)
                        c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
                        tl.store(c_ptrs, c_part)

                        if step < RING_SIZE - 1:
                            tl.store(remote_recv_ptr, tl.load(send_ptr))
                            # tle.distributed_barrier(RING_MESH)

                            shard_idx = tl.where(shard_idx == 0, RING_SIZE - 1, shard_idx - 1)

                            b_cur = tl.load(recv_ptr)
                            c_part = tl.dot(a, b_cur)

                            offs_n = sub_n_base + shard_idx * SUB_N + n_block * BLOCK_N + tl.arange(0, BLOCK_N)
                            c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
                            tl.store(c_ptrs, c_part)

                            if step < RING_SIZE - 2:
                                tl.store(remote_send_ptr, tl.load(recv_ptr))
                                # tle.distributed_barrier(RING_MESH)

                                shard_idx = tl.where(shard_idx == 0, RING_SIZE - 1, shard_idx - 1)
                else:
                    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
                        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
                        a_ptrs = A_ptr + offs_m[:, None] * K + offs_k[None, :]
                        a = tl.load(a_ptrs, mask=offs_k[None, :] < K, other=0.0)

                        shard_idx = ring_index
                        offs_sub_n = sub_n_base + shard_idx * SUB_N + n_block * BLOCK_N + tl.arange(0, BLOCK_N)
                        b_ptrs = B_ptr + offs_k[:, None] * N + offs_sub_n[None, :]
                        b_init = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0.0)
                        tl.store(send_ptr, b_init)

                        for step in range(0, RING_SIZE, 2):
                            b_cur = tl.load(send_ptr)
                            c_part = tl.dot(a, b_cur, out_dtype=tl.float32)

                            offs_n = sub_n_base + shard_idx * SUB_N + n_block * BLOCK_N + tl.arange(0, BLOCK_N)
                            c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
                            if k_block == 0:
                                c_acc = c_part
                            else:
                                existing = tl.load(c_ptrs).to(tl.float32)
                                c_acc = existing + c_part
                            tl.store(c_ptrs, c_acc)

                            if step < RING_SIZE - 1:
                                tl.store(remote_recv_ptr, tl.load(send_ptr))
                                # tle.distributed_barrier(RING_MESH)

                                shard_idx = tl.where(shard_idx == 0, RING_SIZE - 1, shard_idx - 1)

                                b_cur = tl.load(recv_ptr)
                                c_part = tl.dot(a, b_cur, out_dtype=tl.float32)

                                offs_n = sub_n_base + shard_idx * SUB_N + n_block * BLOCK_N + tl.arange(0, BLOCK_N)
                                c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
                                if k_block == 0:
                                    c_acc = c_part
                                else:
                                    existing = tl.load(c_ptrs).to(tl.float32)
                                    c_acc = existing + c_part
                                tl.store(c_ptrs, c_acc)

                                if step < RING_SIZE - 2:
                                    tl.store(remote_send_ptr, tl.load(recv_ptr))
                                    # tle.distributed_barrier(RING_MESH)

                                    shard_idx = tl.where(shard_idx == 0, RING_SIZE - 1, shard_idx - 1)


@triton.autotune(
    configs=make_m_autotune_configs(),
    key=["M", "N", "K", "N_SHARD", "RING_SIZE"],
    prune_configs_by={"early_config_prune": prune_invalid_m_noc_gemm_configs},
    do_bench=autotune_do_bench,
)
@triton.jit
def dsa_shift_m_gemm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    physical_ids_ptr,
    ring_index_lut_ptr,
    ring_id_lut_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    N_SHARD: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SUB_M: tl.constexpr,
    RING_SIZE: tl.constexpr,
    RING_MESH: tl.constexpr,
):
    pid = tle.shard_id(RING_MESH, axis=0)
    ring_index = tl.load(ring_index_lut_ptr + pid)
    ring_id = tl.load(ring_id_lut_ptr + pid)

    next_ring_pos = tl.where(ring_index == RING_SIZE - 1, 0, ring_index + 1)
    send_next_tile = tl.load(physical_ids_ptr + ring_id * RING_SIZE + next_ring_pos)

    send_buf = dsa.alloc((BLOCK_M, BLOCK_K), tl.float16)
    recv_buf = dsa.alloc((BLOCK_M, BLOCK_K), tl.float16)

    offs_buf_m = tl.arange(0, BLOCK_M)[:, None] + tl.zeros((1, BLOCK_K), dtype=tl.int32)
    offs_buf_k = tl.arange(0, BLOCK_K)[None, :] + tl.zeros((BLOCK_M, 1), dtype=tl.int32)

    send_ptr = dsa.local_ptr(send_buf, [offs_buf_m, offs_buf_k])
    recv_ptr = dsa.local_ptr(recv_buf, [offs_buf_m, offs_buf_k])

    remote_recv_buf = tle.remote(recv_buf, send_next_tile, scope=RING_MESH)
    remote_recv_ptr = dsa.local_ptr(remote_recv_buf, [offs_buf_m, offs_buf_k])
    remote_send_buf = tle.remote(send_buf, send_next_tile, scope=RING_MESH)
    remote_send_ptr = dsa.local_ptr(remote_send_buf, [offs_buf_m, offs_buf_k])

    for n_block in range(0, tl.cdiv(N_SHARD, BLOCK_N)):
        offs_n = pid * N_SHARD + n_block * BLOCK_N + tl.arange(0, BLOCK_N)

        if K <= BLOCK_K:
            offs_k = tl.arange(0, BLOCK_K)
            b_ptrs = B_ptr + offs_k[:, None] * N + offs_n[None, :]
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0.0)

            for sub_m_block in range(0, tl.cdiv(M, RING_SIZE * SUB_M)):
                sub_m_base = sub_m_block * RING_SIZE * SUB_M

                for m_block in range(0, tl.cdiv(SUB_M, BLOCK_M)):
                    shard_idx = ring_index
                    offs_sub_m = sub_m_base + shard_idx * SUB_M + m_block * BLOCK_M + tl.arange(0, BLOCK_M)
                    a_ptrs = A_ptr + offs_sub_m[:, None] * K + offs_k[None, :]
                    a_init = tl.load(a_ptrs, mask=offs_k[None, :] < K, other=0.0)
                    tl.store(send_ptr, a_init)

                    for step in range(0, RING_SIZE, 2):
                        a_cur = tl.load(send_ptr)
                        c_part = tl.dot(a_cur, b)

                        offs_sub_m = sub_m_base + shard_idx * SUB_M + m_block * BLOCK_M + tl.arange(0, BLOCK_M)
                        c_ptrs = C_ptr + offs_sub_m[:, None] * N + offs_n[None, :]
                        tl.store(c_ptrs, c_part)

                        if step < RING_SIZE - 1:
                            tl.store(remote_recv_ptr, tl.load(send_ptr))
                            # tle.distributed_barrier(RING_MESH)

                            shard_idx = tl.where(shard_idx == 0, RING_SIZE - 1, shard_idx - 1)

                            a_cur = tl.load(recv_ptr)
                            c_part = tl.dot(a_cur, b)

                            offs_sub_m = sub_m_base + shard_idx * SUB_M + m_block * BLOCK_M + tl.arange(0, BLOCK_M)
                            c_ptrs = C_ptr + offs_sub_m[:, None] * N + offs_n[None, :]
                            tl.store(c_ptrs, c_part)

                            if step < RING_SIZE - 2:
                                tl.store(remote_send_ptr, tl.load(recv_ptr))
                                # tle.distributed_barrier(RING_MESH)

                                shard_idx = tl.where(shard_idx == 0, RING_SIZE - 1, shard_idx - 1)
        else:
            for k_block in range(0, tl.cdiv(K, BLOCK_K)):
                offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
                b_ptrs = B_ptr + offs_k[:, None] * N + offs_n[None, :]
                b = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0.0)

                for sub_m_block in range(0, tl.cdiv(M, RING_SIZE * SUB_M)):
                    sub_m_base = sub_m_block * RING_SIZE * SUB_M

                    for m_block in range(0, tl.cdiv(SUB_M, BLOCK_M)):
                        shard_idx = ring_index
                        offs_sub_m = sub_m_base + shard_idx * SUB_M + m_block * BLOCK_M + tl.arange(0, BLOCK_M)
                        a_ptrs = A_ptr + offs_sub_m[:, None] * K + offs_k[None, :]
                        a_init = tl.load(a_ptrs, mask=offs_k[None, :] < K, other=0.0)
                        tl.store(send_ptr, a_init)

                        for step in range(0, RING_SIZE, 2):
                            a_cur = tl.load(send_ptr)
                            c_part = tl.dot(a_cur, b, out_dtype=tl.float32)

                            offs_sub_m = sub_m_base + shard_idx * SUB_M + m_block * BLOCK_M + tl.arange(0, BLOCK_M)
                            c_ptrs = C_ptr + offs_sub_m[:, None] * N + offs_n[None, :]
                            if k_block == 0:
                                c_acc = c_part
                            else:
                                existing = tl.load(c_ptrs).to(tl.float32)
                                c_acc = existing + c_part
                            tl.store(c_ptrs, c_acc)

                            if step < RING_SIZE - 1:
                                tl.store(remote_recv_ptr, tl.load(send_ptr))
                                # tle.distributed_barrier(RING_MESH)

                                shard_idx = tl.where(shard_idx == 0, RING_SIZE - 1, shard_idx - 1)

                                a_cur = tl.load(recv_ptr)
                                c_part = tl.dot(a_cur, b, out_dtype=tl.float32)

                                offs_sub_m = sub_m_base + shard_idx * SUB_M + m_block * BLOCK_M + tl.arange(0, BLOCK_M)
                                c_ptrs = C_ptr + offs_sub_m[:, None] * N + offs_n[None, :]
                                if k_block == 0:
                                    c_acc = c_part
                                else:
                                    existing = tl.load(c_ptrs).to(tl.float32)
                                    c_acc = existing + c_part
                                tl.store(c_ptrs, c_acc)

                                if step < RING_SIZE - 2:
                                    tl.store(remote_send_ptr, tl.load(recv_ptr))
                                    # tle.distributed_barrier(RING_MESH)

                                    shard_idx = tl.where(shard_idx == 0, RING_SIZE - 1, shard_idx - 1)


def build_ring_luts(rings, ring_num, ring_size, device):
    """Build LUTs for independent row-group rings.

    ``rings`` is flattened as [ring0 physical ids..., ring1 physical ids...].
    """
    physical_ids = torch.tensor(rings, dtype=torch.int32)
    ring_index = torch.empty(TILE_NUM, dtype=torch.int32)
    ring_id = torch.empty(TILE_NUM, dtype=torch.int32)
    for rid in range(ring_num):
        start = rid * ring_size
        for idx, physical_id in enumerate(rings[start:start + ring_size]):
            ring_index[physical_id] = idx
            ring_id[physical_id] = rid
    return physical_ids.to(device), ring_index.to(device), ring_id.to(device)


def bench(fn):
    latency = triton.testing.do_bench(
        fn,
        warmup=Config.warm_up,
        rep=Config.repetition,
        return_mode="median",
    )
    return latency


def bench_flag_gems_mm(a, b):
    def fn():
        return run_flag_gems_mm(a, b)

    return bench(fn)


def run_flag_gems_mm(a, b):
    with flag_gems.use_gems():
        return torch.mm(a, b)


def get_last_best_config(kernel):
    best_config = getattr(kernel, "best_config", None)
    return "n/a" if best_config is None else str(best_config)


def get_ring_mode_spec(mode):
    for spec in RING_MODE_SPECS:
        if spec["mode"] == mode:
            return spec
    raise ValueError(f"Unknown ring mode: {mode}")


def run_ring_mode(mode):
    spec = get_ring_mode_spec(mode)
    ring_num = spec["ring_num"]
    ring_size = TILE_NUM // ring_num
    m_shard = M // TILE_NUM
    n_shard = N // TILE_NUM
    algorithm = select_gemm_algorithm(M, N, K, ring_size)
    ring_mesh = RING_MESHES[mode]
    physical_relation = spec["physical_relation"]

    assert TILE_NUM % ring_num == 0
    assert ring_size % 2 == 0
    if algorithm == ALGORITHM_N_RING:
        assert M % TILE_NUM == 0
        assert M_SHARD == m_shard
    elif algorithm == ALGORITHM_M_RING:
        assert N % TILE_NUM == 0

    device = triton.runtime.driver.active.get_active_torch_device()
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    b = torch.randn((K, N), device=device, dtype=torch.float16)
    split_k = K > max(AUTOTUNE_BLOCK_K)
    c_dtype = torch.float32 if split_k else torch.float16
    if algorithm == ALGORITHM_FLAG_GEMS_MM:
        c = None
    else:
        c = torch.empty((M, N), device=device, dtype=c_dtype)

    physical_ids, ring_index_lut, ring_id_lut = build_ring_luts(
        physical_relation, ring_num, ring_size, device
    )

    grid = (TILE_NUM, )
    print(
        f"RUN: mode={mode}, ring_num={ring_num}, ring_size={ring_size}, "
        f"m_shard={m_shard}, n_shard={n_shard}, algorithm={algorithm}",
        flush=True,
    )

    if algorithm == ALGORITHM_FLAG_GEMS_MM:
        kernel = None

        def fn():
            return run_flag_gems_mm(a, b)

        c = fn()
    elif algorithm == ALGORITHM_N_RING:
        kernel = dsa_shift_n_gemm_kernel

        def fn():
            dsa_shift_n_gemm_kernel[grid](
                a,
                b,
                c,
                physical_ids,
                ring_index_lut,
                ring_id_lut,
                M=M,
                N=N,
                K=K,
                M_SHARD=m_shard,
                RING_SIZE=ring_size,
                RING_MESH=ring_mesh,
            )

        fn()
    elif algorithm == ALGORITHM_M_RING:
        kernel = dsa_shift_m_gemm_kernel

        def fn():
            dsa_shift_m_gemm_kernel[grid](
                a,
                b,
                c,
                physical_ids,
                ring_index_lut,
                ring_id_lut,
                M=M,
                N=N,
                K=K,
                N_SHARD=n_shard,
                RING_SIZE=ring_size,
                RING_MESH=ring_mesh,
            )

        fn()
    else:
        raise ValueError(f"Unknown GEMM algorithm: {algorithm}")

    with flag_gems.use_gems():
        ref_out = torch.mm(a, b)

    # Compare on CPU to avoid unsupported torch.testing ops on TXDA backend.
    res_out = c.detach().cpu().to(torch.float32)
    golden_cpu = ref_out.detach().cpu().to(torch.float32)
    max_abs = (res_out - golden_cpu).abs().max().item()

    if not torch.allclose(res_out, golden_cpu, atol=1e-3, rtol=1e-2):
        raise AssertionError(f"Mismatch: max_abs_diff={max_abs}")

    latency = bench(fn)
    flag_gems_latency = bench_flag_gems_mm(a, b)
    if algorithm == ALGORITHM_FLAG_GEMS_MM:
        best_config = ALGORITHM_FLAG_GEMS_MM
    else:
        best_config = get_last_best_config(kernel)

    print(
        f"PASS: M={M}, N={N}, K={K}, "
        f"RING_NUM={ring_num}, RING_SIZE={ring_size}, "
        f"TILE_NUM={TILE_NUM}, mode={mode}, "
        f"algorithm={algorithm}, "
        f"best_config={best_config}, "
        f"latency_ms={latency:.4f}, "
        f"flag_gems_latency_ms={flag_gems_latency:.4f}, "
        f"max_abs_diff={max_abs}"
    )
    return latency


def run():
    return run_ring_mode("ring")


def run_two_ring():
    return run_ring_mode("two_ring")


def run_four_ring():
    return run_ring_mode("four_ring")


def run_eight_ring():
    return run_ring_mode("eight_ring")


def run_all():
    for spec in RING_MODE_SPECS:
        run_ring_mode(spec["mode"])


if __name__ == "__main__":
    run()
