"""MoE all-to-all dispatch/combine across chiplet dies.

In MODE 0 each die scatters its tokens' data and per-expert split counts to the
matching expert die; in MODE 1 each die gathers the dispatched data back and
combines it, verifying the round-trip restores the original layout.
"""
import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

N_DIES = 4
NUM_EXPERTS = 16
EXPERTS_PER_RANK = NUM_EXPERTS // N_DIES
TOPK = 1
NUM_TOKENS = 32
MAX_M = NUM_TOKENS * TOPK
HIDDEN = 64
BM = 16
BN = HIDDEN


def splits_to_cumsum(splits):
    out = torch.empty(splits.shape[0] + 1, dtype=splits.dtype, device=splits.device)
    out[0] = 0
    _ = torch.cumsum(splits, 0, out=out[1:])
    return out


def calc_gather_index(exp_indices, row_start, row_end):
    flat = exp_indices.flatten().int()
    topk = exp_indices.shape[1]
    token_idx = torch.arange(flat.shape[0], dtype=torch.int32) // topk
    sorted_order = torch.argsort(flat, stable=True)
    gather_index = token_idx[sorted_order].int()
    return gather_index[row_start:row_end].contiguous()


def generate_random_exp_indices(token_num, total_num_experts, topk):
    import random
    exp_indices = []
    exp_list = list(range(total_num_experts))
    for _ in range(token_num):
        top_selected = random.sample(exp_list, topk)
        exp_indices.append(top_selected)
    return torch.Tensor(exp_indices).int()


@triton.jit
def ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit
def moe_all2all_kernel(
    send_tensor,
    data_src,
    data_dst,
    splits_src,
    splits_dst,
    signal,
    send_splits_cumsum,
    recv_offset,
    combine_splits_cumsum,
    call_count,
    act_pos,
    mesh: tl.constexpr,
    MODE: tl.constexpr,
    HIDDEN: tl.constexpr,
    MAX_M: tl.constexpr,
    NUM_TOT_EXPERTS: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pid = tl.program_id(0)
    pe = tle.shard_id(mesh, "chiplet")
    npes = tle.n_pes(None)
    signal_ptr = signal + act_pos * npes + pe
    wait_ptr = signal + act_pos * npes + pid

    if MODE == 0:
        exp_st = pid * EXPERTS_PER_RANK
        exp_ed = exp_st + EXPERTS_PER_RANK
        m_st = tl.load(send_splits_cumsum + exp_st)
        m_ed = tl.load(send_splits_cumsum + exp_ed)
        num_rows = m_ed - m_st

        split_src = splits_src + pid * (EXPERTS_PER_RANK + 1)
        off0 = exp_st + tl.arange(0, EXPERTS_PER_RANK)
        off1 = exp_st + tl.arange(0, EXPERTS_PER_RANK) + 1
        cumsum_sts = tl.load(send_splits_cumsum + off0)
        cumsum_eds = tl.load(send_splits_cumsum + off1)
        tl.store(split_src + tl.arange(0, EXPERTS_PER_RANK), cumsum_eds - cumsum_sts)
        tl.store(split_src + EXPERTS_PER_RANK, m_st)

        off_m = tl.arange(0, BM)
        off_n = tl.arange(0, BN)
        send_tensor_ptrs = send_tensor + m_st * HIDDEN + off_m[:, None] * HIDDEN + off_n[None, :]
        data_src_ptrs = data_src + m_st * HIDDEN + off_m[:, None] * HIDDEN + off_n[None, :]
        for i in tl.range(ceil_div(num_rows, BM)):
            data_mask = (off_m[:, None] < num_rows - i * BM) & (off_n[None, :] < HIDDEN)
            tl.store(data_src_ptrs, tl.load(send_tensor_ptrs, data_mask), data_mask)
            send_tensor_ptrs += BM * HIDDEN
            data_src_ptrs += BM * HIDDEN

        data_src_ptr = data_src + m_st * HIDDEN
        data_dst_ptr = data_dst + act_pos * npes * MAX_M * HIDDEN + pe * MAX_M * HIDDEN
        split_dst = splits_dst + act_pos * (NUM_TOT_EXPERTS + npes) + pe * (EXPERTS_PER_RANK + 1)
    else:
        src_off = pid * MAX_M
        dst_off = tl.load(recv_offset + pid)
        num_rows = tl.load(combine_splits_cumsum + (pid + 1) * EXPERTS_PER_RANK) - \
                   tl.load(combine_splits_cumsum + pid * EXPERTS_PER_RANK)
        data_src_ptr = data_src + act_pos * npes * MAX_M * HIDDEN + src_off * HIDDEN
        data_dst_ptr = data_dst + dst_off * HIDDEN

    tle.chiplet.putmem(data_dst_ptr, data_src_ptr, num_rows * HIDDEN, pid, blocking=False)
    if MODE == 0:
        tle.chiplet.putmem(split_dst, split_src, EXPERTS_PER_RANK + 1, pid, blocking=False)
    tle.chiplet.fence()
    tle.chiplet.notify(signal_ptr, pid, signal=call_count, sig_op="set")
    tle.chiplet.wait(wait_ptr, wait_value=call_count)


def test_moe_all2all():
    mesh = tle.device_mesh({"chiplet": N_DIES})
    spec_repl = tle.sharding(mesh)
    spec_shard_m = tle.sharding(mesh, split=(("chiplet", ), None))
    spec_shard_0 = tle.sharding(mesh, split=(("chiplet", ), ))

    torch.manual_seed(42)
    scattered_input_host = torch.randn(N_DIES * MAX_M, HIDDEN, dtype=torch.float16)
    split_cumsum_host = torch.zeros(N_DIES * (NUM_EXPERTS + 1), dtype=torch.int32)
    for pe in range(N_DIES):
        import random
        random.seed(42 + pe)
        exp_indices_pe = generate_random_exp_indices(NUM_TOKENS, NUM_EXPERTS, TOPK)
        splits_pe = torch.bincount(exp_indices_pe.view(-1), minlength=NUM_EXPERTS).to(torch.int32)
        cumsum_pe = splits_to_cumsum(splits_pe)
        gather_idx_pe = calc_gather_index(exp_indices_pe, 0, NUM_TOKENS * TOPK)
        scattered_pe = scattered_input_host[pe * MAX_M:(pe + 1) * MAX_M, :][gather_idx_pe]
        scattered_input_host[pe * MAX_M:(pe + 1) * MAX_M, :] = scattered_pe
        split_cumsum_host[pe * (NUM_EXPERTS + 1):(pe + 1) * (NUM_EXPERTS + 1)] = cumsum_pe

    send_buf = tle.make_sharded_tensor(torch.zeros(N_DIES * MAX_M, HIDDEN, dtype=torch.float16), spec_shard_m)
    scattered_input_dev = tle.make_sharded_tensor(scattered_input_host, spec_shard_m)
    splits_cumsum_dev = tle.make_sharded_tensor(split_cumsum_host, spec_shard_0)

    recv_buf = tle.make_sharded_tensor(torch.zeros(N_DIES * MAX_M * 2, HIDDEN, dtype=torch.float16), spec_repl)
    split_send = tle.make_sharded_tensor(torch.zeros(NUM_EXPERTS + N_DIES, dtype=torch.int32), spec_repl)
    split_recv = tle.make_sharded_tensor(torch.zeros((NUM_EXPERTS + N_DIES) * 2 * N_DIES, dtype=torch.int32),
                                         spec_shard_0)
    signal_buf = tle.make_sharded_tensor(torch.zeros(N_DIES * 2, dtype=torch.int64), spec_repl)

    act_pos = 1
    for round_idx in range(1, 4):
        act_pos ^= 1

        moe_all2all_kernel[(N_DIES, )](
            scattered_input_dev,
            send_buf,
            recv_buf,
            split_send,
            split_recv,
            signal_buf,
            splits_cumsum_dev,
            None,
            None,
            call_count=round_idx * 2,
            act_pos=act_pos,
            mesh=mesh,
            MODE=0,
            HIDDEN=HIDDEN,
            MAX_M=MAX_M,
            NUM_TOT_EXPERTS=NUM_EXPERTS,
            EXPERTS_PER_RANK=EXPERTS_PER_RANK,
            BM=BM,
            BN=BN,
        )

        split_recv_host = split_recv.cpu()
        per_die_len = (NUM_EXPERTS + N_DIES) * 2
        combine_offset_per_die = torch.zeros(N_DIES * N_DIES, dtype=torch.int32)
        combine_cumsum_per_die = torch.zeros(N_DIES * (N_DIES * EXPERTS_PER_RANK + 1), dtype=torch.int32)
        for pe in range(N_DIES):
            die_slice = split_recv_host[pe * per_die_len:(pe + 1) * per_die_len]
            act_slice = die_slice[act_pos * (NUM_EXPERTS + N_DIES):act_pos * (NUM_EXPERTS + N_DIES) +
                                  (NUM_EXPERTS + N_DIES)]
            dis_splits_2d = act_slice.reshape(N_DIES, EXPERTS_PER_RANK + 1)
            combine_offset_per_die[pe * N_DIES:(pe + 1) * N_DIES] = dis_splits_2d[:, EXPERTS_PER_RANK]
            combine_send_splits = dis_splits_2d[:, :EXPERTS_PER_RANK].flatten()
            combine_cumsum_per_die[pe * (N_DIES * EXPERTS_PER_RANK + 1):(pe + 1) *
                                   (N_DIES * EXPERTS_PER_RANK + 1)] = splits_to_cumsum(combine_send_splits)

        combine_offset_dev = tle.make_sharded_tensor(combine_offset_per_die, spec_shard_0)
        combine_cumsum_dev = tle.make_sharded_tensor(combine_cumsum_per_die, spec_shard_0)

        moe_all2all_kernel[(N_DIES, )](
            None,
            recv_buf,
            send_buf,
            None,
            None,
            signal_buf,
            None,
            combine_offset_dev,
            combine_cumsum_dev,
            call_count=round_idx * 2 + 1,
            act_pos=act_pos,
            mesh=mesh,
            MODE=1,
            HIDDEN=HIDDEN,
            MAX_M=MAX_M,
            NUM_TOT_EXPERTS=NUM_EXPERTS,
            EXPERTS_PER_RANK=EXPERTS_PER_RANK,
            BM=BM,
            BN=BN,
        )

        combined_host = send_buf.cpu()
        for pe in range(N_DIES):
            actual = combined_host[pe * MAX_M:(pe + 1) * MAX_M, :][:NUM_TOKENS * TOPK, :]
            expected = scattered_input_host[pe * MAX_M:(pe + 1) * MAX_M, :][:NUM_TOKENS * TOPK, :]
            if not torch.allclose(actual, expected, atol=1e-3, rtol=1e-3):
                print(f"[VERIFY FAIL] Round {round_idx} die {pe}:")
                print(f"  actual   (first 8 rows, first elem): {actual[:8, 0].tolist()}")
                print(f"  expected (first 8 rows, first elem): {expected[:8, 0].tolist()}")
                print(f"  actual   (last 8 rows, first elem):  {actual[-8:, 0].tolist()}")
                print(f"  expected (last 8 rows, first elem):  {expected[-8:, 0].tolist()}")
                raise AssertionError(f"Round {round_idx} die {pe}: combine round-trip mismatch")


if __name__ == "__main__":
    test_moe_all2all()
