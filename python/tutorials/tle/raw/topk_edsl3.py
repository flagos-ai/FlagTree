# flake8: noqa: F821, F722, F841
from mlir.dialects import arith, memref, nvvm, scf, llvm
from mlir import ir
import torch
from triton.experimental.tle.raw.mlir.utils import vprintf
import triton
from triton.experimental.tle.raw import dialect, InOut, Input
import triton.experimental.tle.language.raw as tle_raw
import triton.language as tl


@triton.jit
def convert_to_uint16(x):
    hval = x.cast(dtype=tl.float16)
    bits_uint = hval.cast(dtype=tl.uint16, bitcast=True)  # Equivalent to reinterpret
    bits_uint = tl.where(x < 0, ~bits_uint & (0xFFFF), bits_uint | (0x8000))
    return bits_uint >> 8


@triton.jit
def convert_to_uint32(x):
    bits_uint = x.cast(dtype=tl.uint32, bitcast=True)
    bits_uint = tl.where(
        x < 0,
        ~bits_uint & tl.cast((0xFFFFFFFF), tl.uint32, bitcast=True),
        bits_uint | tl.cast((0x80000000), tl.uint32, bitcast=True),
    )
    return bits_uint


#【注意】这个版本的线程数要求==1024，之后还得加上线程资源约束
@dialect(name="mlir")
def edsl1(thre_bin_sum_buf: InOut["memref<?xi32, 3>"], l_new_topk_buf: InOut["memref<?xi32, 3>"],
          indices_base: Input["!llvm.ptr<1>"], s_input_ids_base: Input["!llvm.ptr<1>"], inputs: Input["!llvm.ptr<1>"],
          s_histogram: Input["memref<?xi32, 3>"], l_start_idx: Input["i32"], l_end_idx: Input["i32"], S: Input["i32"],
          BS: Input["i32"], K_tensor: Input["memref<?xi32, 3>"]):
    tidx = nvvm.read_ptx_sreg_tid_x(ir.IntegerType.get_signless(32))
    bidx = nvvm.read_ptx_sreg_ctaid_x(ir.IntegerType.get_signless(32))
    bdimx = nvvm.read_ptx_sreg_ntid_x(ir.IntegerType.get_signless(32))  # blockDim.x
    i32_ty = ir.IntegerType.get_signless(32)
    i16_ty = ir.IntegerType.get_signless(16)
    index_ty = ir.IndexType.get()
    f32_ty = ir.F32Type.get()
    f16_ty = ir.F16Type.get()
    ptr_ty = ir.Type.parse("!llvm.ptr<1>")
    zero_i32 = arith.constant(i32_ty, 0)
    one_i32 = arith.constant(i32_ty, 1)
    eight_i32 = arith.constant(i32_ty, 8)
    zero_f32 = arith.constant(f32_ty, 0.0)
    zero = arith.constant(index_ty, 0)
    one = arith.constant(index_ty, 1)
    mask_0xFFFF = arith.constant(i32_ty, 0xFFFF)
    mask_0x8000 = arith.constant(i32_ty, 0x8000)
    mask_0xFF = arith.constant(i32_ty, 0xFF)
    num_iters = arith.ceildivsi(S, BS)
    num_iters_idx = arith.index_cast(index_ty, num_iters)
    zero_idx = arith.constant(index_ty, 0)

    s_input_ids_base_0 = llvm.getelementptr(ptr_ty, s_input_ids_base, [], [0], i32_ty, 0)
    llvm.store(one_i32, s_input_ids_base_0)

    nvvm.barrier0()
    for s in scf.for_(0, arith.index_cast(index_ty, arith.ceildivsi(S, BS))):
        s_i32 = arith.index_cast(i32_ty, s)
        input_idx_i32 = arith.addi(arith.muli(s_i32, BS), tidx)
        cond = arith.andi(
            arith.andi(arith.cmpi(arith.CmpIPredicate.slt, input_idx_i32, l_end_idx),
                       arith.cmpi(arith.CmpIPredicate.sge, input_idx_i32, l_start_idx)),
            arith.cmpi(arith.CmpIPredicate.slt, input_idx_i32, S))
        if_stmt = scf.if_([], cond)
        cond_i32 = arith.extsi(i32_ty, cond)
        # vprintf("input_idx_i32=%d\n", input_idx_i32)
        thenblock = if_stmt.opview.thenRegion.blocks.append()
        with ir.InsertionPoint(thenblock):

            base_offset = arith.muli(bidx, S)
            full_offset = arith.addi(base_offset, input_idx_i32)

            input_ptr = llvm.getelementptr(ptr_ty, inputs, [full_offset], [-2147483648], f32_ty, 0)
            input_val = llvm.load(f32_ty, input_ptr)

            input_f16 = arith.truncf(f16_ty, input_val)
            input_i16 = arith.bitcast(i16_ty, input_f16)
            input_ui16_i32 = arith.extui(i32_ty, input_i16)

            is_neg = arith.cmpf(arith.CmpFPredicate.OLT, input_val, zero_f32)

            neg_bits = arith.andi(arith.xori(input_ui16_i32, mask_0xFFFF), mask_0xFFFF)
            pos_bits = arith.ori(input_ui16_i32, mask_0x8000)
            processed_bits = arith.select(is_neg, neg_bits, pos_bits)
            bin_id_i16 = arith.shrui(processed_bits, eight_i32)
            val = memref.atomic_rmw(arith.AtomicRMWKind.addi, one_i32, s_histogram,
                                    [arith.index_cast(index_ty, bin_id_i16)])
            # vprintf("tidx: %d, bidx: %d, val: %d\n", tidx, bidx, val)

            scf.yield_([])
        scf.yield_([])
    nvvm.barrier0()

    # Extract K from tensor and use it as initial l_new_topk
    K = memref.load(K_tensor, [zero_idx])
    l_new_topk = K

    # Independent code block: if tx < RADIX
    RADIX = arith.constant(i32_ty, 256)
    tx_lt_radix = arith.cmpi(arith.CmpIPredicate.slt, tidx, RADIX)
    if_tx_lt_radix = scf.if_([], tx_lt_radix)
    then_block = if_tx_lt_radix.opview.thenRegion.blocks.append()
    with ir.InsertionPoint(then_block):
        # for i in T.serial(8):
        eight = arith.constant(ir.IndexType.get(), 8)
        cst3 = arith.constant(i32_ty, 3)
        for i in scf.for_(zero, eight, one):
            # offset = 1 << i
            i_i32 = arith.index_cast(i32_ty, i)
            offset = arith.shli(one_i32, i_i32)
            # Pre-compute tidx_idx for use in both if blocks
            tidx_idx = arith.index_cast(ir.IndexType.get(), tidx)
            # T.sync_threads(3, RADIX)
            nvvm.barrier(barrier_id=cst3, number_of_threads=RADIX)
            # if tx < RADIX - offset:
            radix_minus_offset = arith.subi(RADIX, offset)
            tx_lt_radix_minus_offset = arith.cmpi(arith.CmpIPredicate.slt, tidx, radix_minus_offset)
            if_tx_lt_radix_minus_offset = scf.if_([i32_ty], tx_lt_radix_minus_offset)
            then_block1 = if_tx_lt_radix_minus_offset.owner.opview.thenRegion.blocks.append()
            with ir.InsertionPoint(then_block1):
                # l_val = s_histogram[tx] + s_histogram[tx + offset]
                tidx_plus_offset = arith.addi(tidx, offset)
                tidx_plus_offset_idx = arith.index_cast(ir.IndexType.get(), tidx_plus_offset)

                hist_tx = memref.load(s_histogram, [tidx_idx])
                hist_tx_plus_offset = memref.load(s_histogram, [tidx_plus_offset_idx])
                l_val = arith.addi(hist_tx, hist_tx_plus_offset)
                scf.yield_([l_val])
            else_block1 = if_tx_lt_radix_minus_offset.owner.opview.elseRegion.blocks.append()
            with ir.InsertionPoint(else_block1):
                scf.yield_([arith.constant(i32_ty, 0)])
            l_val = if_tx_lt_radix_minus_offset
            # T.sync_threads(3, RADIX) - second sync before writing back
            nvvm.barrier(barrier_id=cst3, number_of_threads=RADIX)
            # if tx < RADIX - offset:
            if_tx_lt_radix_minus_offset2 = scf.if_([], tx_lt_radix_minus_offset)
            then_block2 = if_tx_lt_radix_minus_offset2.opview.thenRegion.blocks.append()
            with ir.InsertionPoint(then_block2):
                memref.store(l_val, s_histogram, [tidx_idx])
                scf.yield_([])
            scf.yield_([])
        # T.sync_threads(3, RADIX) - after cumsum loop, before finding threshold bin id
        nvvm.barrier(barrier_id=cst3, number_of_threads=RADIX)
        # find threshold bin id
        tidx_idx_for_thre = arith.index_cast(ir.IndexType.get(), tidx)
        tidx_plus_one = arith.addi(tidx, one_i32)
        tidx_plus_one_idx = arith.index_cast(ir.IndexType.get(), tidx_plus_one)

        hist_tx_for_thre = memref.load(s_histogram, [tidx_idx_for_thre])
        hist_tx_plus_one = memref.load(s_histogram, [tidx_plus_one_idx])

        cond1_thre = arith.cmpi(arith.CmpIPredicate.sgt, hist_tx_for_thre, l_new_topk)
        cond2_thre = arith.cmpi(arith.CmpIPredicate.sle, hist_tx_plus_one, l_new_topk)
        cond_thre = arith.andi(cond1_thre, cond2_thre)

        if_find_thre = scf.if_([], cond_thre)
        then_block_thre = if_find_thre.opview.thenRegion.blocks.append()
        with ir.InsertionPoint(then_block_thre):
            memref.store(tidx, thre_bin_sum_buf, [zero_idx])
            scf.yield_([])
        scf.yield_([])
    nvvm.barrier0()
    l_threshold_bin_id_new = memref.load(thre_bin_sum_buf, [zero_idx])
    l_threshold_bin_id_plus_one = arith.addi(l_threshold_bin_id_new, one_i32)
    l_threshold_bin_id_plus_one_idx = arith.index_cast(ir.IndexType.get(), l_threshold_bin_id_plus_one)
    hist_threshold_plus_one = memref.load(s_histogram, [l_threshold_bin_id_plus_one_idx])
    l_new_topk_new = arith.subi(l_new_topk, hist_threshold_plus_one)
    nvvm.barrier0()
    # Store l_new_topk_new to output buffer
    memref.store(l_new_topk_new, l_new_topk_buf, [zero_idx])
    nvvm.barrier0()

    # TileLang: for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
    for s in scf.for_(zero, num_iters_idx, one):

        # Thread-stride loop to match triton's tl.arange(0, BS) behavior
        BS_idx = arith.index_cast(ir.IndexType.get(), BS)
        bdimx_idx = arith.index_cast(ir.IndexType.get(), bdimx)
        num_strides = arith.ceildivsi(BS, bdimx)
        num_strides_idx = arith.index_cast(ir.IndexType.get(), num_strides)

        for stride in scf.for_(zero, num_strides_idx, one):
            s_i32 = arith.index_cast(i32_ty, s)
            stride_i32 = arith.index_cast(i32_ty, stride)
            stride_offset = arith.muli(stride_i32, bdimx)
            input_idx_i32 = arith.addi(arith.addi(arith.muli(s_i32, BS), tidx), stride_offset)
            input_idx_back_i32 = input_idx_i32

            cond1 = arith.cmpi(arith.CmpIPredicate.slt, input_idx_back_i32, l_end_idx)
            cond2 = arith.cmpi(arith.CmpIPredicate.sge, input_idx_back_i32, l_start_idx)
            cond3 = arith.cmpi(arith.CmpIPredicate.slt, input_idx_back_i32, S)
            cond_all = arith.andi(arith.andi(cond1, cond2), cond3)

            ifop = scf.if_([], cond_all)
            thenblock = ifop.opview.thenRegion.blocks.append()
            with ir.InsertionPoint(thenblock):
                base_offset = arith.muli(bidx, S)
                full_offset = arith.addi(base_offset, input_idx_back_i32)

                input_ptr = llvm.getelementptr(ptr_ty, inputs, [full_offset], [-2147483648], f32_ty, 0)
                input_val = llvm.load(f32_ty, input_ptr)

                input_f16 = arith.truncf(f16_ty, input_val)
                input_i16 = arith.bitcast(i16_ty, input_f16)
                input_ui16_i32 = arith.extui(i32_ty, input_i16)

                is_neg = arith.cmpf(arith.CmpFPredicate.OLT, input_val, zero_f32)

                neg_bits = arith.andi(arith.xori(input_ui16_i32, mask_0xFFFF), mask_0xFFFF)
                pos_bits = arith.ori(input_ui16_i32, mask_0x8000)
                processed_bits = arith.select(is_neg, neg_bits, pos_bits)
                bin_id_i32 = arith.shrui(processed_bits, eight_i32)
                bin_id_i32 = arith.andi(bin_id_i32, mask_0xFF)

                l_bin_id32 = bin_id_i32

                over_thre = arith.cmpi(arith.CmpIPredicate.sgt, l_bin_id32, l_threshold_bin_id_new)
                over_thre_if = scf.if_([], over_thre)
                over_thre_then = over_thre_if.opview.thenRegion.blocks.append()
                with ir.InsertionPoint(over_thre_then):
                    bin_id_plus_one = arith.addi(l_bin_id32, one_i32)
                    bin_id_plus_one_idx = arith.index_cast(index_ty, bin_id_plus_one)

                    pos = memref.atomic_rmw(arith.AtomicRMWKind.addi, one_i32, s_histogram, [bin_id_plus_one_idx])
                    indices_ptr = llvm.getelementptr(ptr_ty, indices_base, [pos], [-2147483648], i32_ty, 0)
                    llvm.store(input_idx_back_i32, indices_ptr)
                    scf.yield_([])

                over_thre_else = over_thre_if.opview.elseRegion.blocks.append()
                with ir.InsertionPoint(over_thre_else):
                    eq_thre = arith.cmpi(arith.CmpIPredicate.eq, l_bin_id32, l_threshold_bin_id_new)
                    l_new_topk_gt_zero = arith.cmpi(arith.CmpIPredicate.sgt, l_new_topk_new, zero_i32)
                    eq_thre_and_topk = arith.andi(eq_thre, l_new_topk_gt_zero)
                    eq_thre_if = scf.if_([], eq_thre_and_topk)
                    eq_thre_then = eq_thre_if.opview.thenRegion.blocks.append()
                    with ir.InsertionPoint(eq_thre_then):
                        pos = memref.atomic_rmw(arith.AtomicRMWKind.addi, one_i32, thre_bin_sum_buf, [zero_idx])
                        s_input_ptr = llvm.getelementptr(ptr_ty, s_input_ids_base, [pos], [-2147483648], i32_ty, 0)
                        llvm.store(input_idx_back_i32, s_input_ptr)
                        scf.yield_([])
                    scf.yield_([])

                scf.yield_([])
            scf.yield_([])
        scf.yield_([])

    nvvm.barrier0()


@triton.autotune(
    configs=[
        triton.Config({"BS": 1024, "BSS": 512}, num_stages=3,
                      num_warps=32),  # "BS" should be 1024 and "num_warps" should be 32
    ],
    key=["S", "K"],
)
@triton.jit
def kernel_bucket_sort_topk(  # grid(B,)
        inputs,  # (B, S)
        indices,  # (B, K) topk index array
        s_input_ids,  # (B, SMEM_INPUT_SIZE) Data indices to be filtered in the next round
        starts,  # (B,) for variable length
        ends,  # (B,) for variable length
        S: tl.constexpr,  # sequence length
        K: tl.constexpr,  # k of topk
        HISTOGRAM_SIZE: tl.constexpr, SMEM_INPUT_SIZE: tl.constexpr, BS: tl.constexpr,  # block size of S
        BSS: tl.constexpr,  # block size of SMEM_INPUT
):
    """Kernel: Merged version of kernel1, kernel2, and kernel3 with threshold calculation in edsl1"""
    i_b = tl.program_id(0)

    s_base = inputs + i_b * S
    indices_base = indices + i_b * K
    s_input_ids_base = s_input_ids + i_b * SMEM_INPUT_SIZE

    l_start_idx = tl.load(starts + i_b).to(tl.int32)
    l_end_idx = tl.load(ends + i_b).to(tl.int32)

    # Kernel1: Compute histogram
    s_histogram = tl.zeros([HISTOGRAM_SIZE], dtype=tl.int32)
    # TS = tl.cdiv(S, BS)
    # for s in range(TS):
    #     input_idx = s * BS + tl.arange(0, BS)
    #     input_mask = ((input_idx < l_end_idx) & (input_idx >= l_start_idx) & (input_idx < S))
    #     input = tl.load(s_base + input_idx, input_mask, other=float("-inf")).to(tl.float32)
    #     inval_int16 = convert_to_uint16(input)
    #     s_histogram += inval_int16.to(tl.int32).histogram(HISTOGRAM_SIZE)

    # tl.device_print("s_histogram: ", s_histogram)

    # Kernel2: Call edsl1 for topk selection (threshold calculated in edsl1)
    thre_bin_sum_buf = tl.zeros([1], dtype=tl.int32)
    l_new_topk_buf = tl.zeros([1], dtype=tl.int32)
    s = S
    bs = BS
    k_tensor = tl.full([1], K, dtype=tl.int32)  # Convert constexpr to tensor
    thre_bin_sum_buf, l_new_topk_buf = tle_raw.call(
        edsl1, [thre_bin_sum_buf, l_new_topk_buf],
        [indices_base, s_input_ids_base, inputs, s_histogram, l_start_idx, l_end_idx, s, bs, k_tensor])

    thre_bin_sum = thre_bin_sum_buf.max(0)
    l_new_topk = l_new_topk_buf.max(0)
    sum = K - l_new_topk

    return

    # Kernel3: Continue with while loop
    round = 0
    while round < 4 and l_new_topk > 0:
        ss = tl.cdiv(thre_bin_sum, BSS)
        s_histogram = tl.zeros([HISTOGRAM_SIZE], dtype=tl.int32)
        padding_num = 0.0 if round else float("-inf")
        for s in range(ss):
            s_input_idx = s * BSS + tl.arange(0, BSS)
            s_input_idx_mask = s_input_idx < thre_bin_sum
            input_idx = tl.load(s_input_ids_base + s_input_idx, s_input_idx_mask, other=-1)
            s_input_mask = s_input_idx_mask & (input_idx >= 0) & (input_idx < S)
            s_input = tl.load(s_base + input_idx, s_input_mask, other=padding_num).to(tl.float32)
            inval_int32 = (convert_to_uint32(s_input) >> (24 - round * 8)) & 0xFF
            s_histogram += inval_int32.to(tl.int32).histogram(HISTOGRAM_SIZE)
        s_histogram = s_histogram.cumsum(0, reverse=True)  # Suffix sum
        mv_idx = (tl.arange(1, HISTOGRAM_SIZE + 1) % HISTOGRAM_SIZE)  # Construct offset index matrix
        cond = (s_histogram > l_new_topk) & ((s_histogram.gather(mv_idx, 0) <= l_new_topk) | (mv_idx == 0))
        l_threshold_bin_id = cond.argmax(0)
        l_new_topk -= tl.where(tl.arange(0, HISTOGRAM_SIZE) == l_threshold_bin_id + 1, s_histogram, 0).max(0)
        thre_bin_sum, old_thre_bin_sum = 0, thre_bin_sum

        for s in range(ss):
            s_input_idx = s * BSS + tl.arange(0, BSS)
            s_input_idx_mask = s_input_idx < old_thre_bin_sum
            input_idx = tl.load(s_input_ids_base + s_input_idx, s_input_idx_mask, other=-1)
            s_input_mask = s_input_idx_mask & (input_idx >= 0) & (input_idx < S)
            s_input = tl.load(s_base + input_idx, s_input_mask, other=padding_num).to(tl.float32)
            inval_int32 = (convert_to_uint32(s_input) >> (24 - round * 8)) & 0xFF

            over_thre = inval_int32.to(tl.int32) > l_threshold_bin_id
            cur_sum = over_thre.to(tl.int32).sum(-1)
            eq_thre = inval_int32.to(tl.int32) == l_threshold_bin_id
            thre_bin_cur_sum = eq_thre.to(tl.int32).sum(-1)

            topk_idx = over_thre.to(tl.int32).cumsum(-1)
            thre_bin_idx = eq_thre.to(tl.int32).cumsum(-1)

            concat_mask = tl.cat(over_thre, eq_thre, True)
            concat_input = tl.cat(input_idx, input_idx, True)
            concat_pointer_matrix = tl.cat(
                indices_base + sum + topk_idx - 1,
                s_input_ids_base + thre_bin_sum + thre_bin_idx - 1,
                True,
            )

            tl.store(concat_pointer_matrix, concat_input, mask=concat_mask)

            thre_bin_sum += thre_bin_cur_sum
            sum += cur_sum

        round += 1

    if l_new_topk > 0:
        ss = tl.cdiv(l_new_topk, BSS)
        for s in range(ss):
            s_input_idx = s * BSS + tl.arange(0, BSS)
            s_input_idx_mask = s_input_idx < l_new_topk
            input_idx = tl.load(s_input_ids_base + s_input_idx, s_input_idx_mask, other=-1)
            s_input_mask = s_input_idx_mask
            tl.store(indices_base + sum + tl.arange(0, BSS), input_idx, mask=s_input_mask)
            sum += BSS


def bucket_sort_topk(inputs, starts, ends, topk):
    """Complete topk function using merged kernel"""
    B, S = inputs.shape
    K = topk
    HISTOGRAM_SIZE = 256
    SMEM_INPUT_SIZE = 4096
    indices = torch.full((B, topk), -1, dtype=torch.int32, device=inputs.device)
    s_input_idx = torch.zeros(B, SMEM_INPUT_SIZE, dtype=torch.int32, device=inputs.device)
    grid = (B, )
    kernel_bucket_sort_topk[grid](
        inputs,
        indices,
        s_input_idx,
        starts,
        ends,
        S,
        K,
        HISTOGRAM_SIZE,
        SMEM_INPUT_SIZE,
    )
    return indices


def test_topk_selector(batch=64, seq_len=32 * 1024, topk=2048):

    batch = 64
    seq_len = 32 * 1024
    topk = 2048
    torch.manual_seed(1)
    input = torch.randn(batch, seq_len, dtype=torch.float32).cuda()
    starts = torch.zeros(batch, dtype=torch.int32).cuda()
    ends = torch.ones(batch, dtype=torch.int32).cuda() * seq_len

    indexes = bucket_sort_topk(input, starts, ends, topk)

    indexes_ref = torch.topk(input, topk, dim=-1)[1]

    torch.set_printoptions(threshold=100, edgeitems=100, linewidth=100)
    for i in range(batch):
        ref_np = indexes_ref[i].cpu().to(torch.int32).numpy()
        trt_np = indexes[i].cpu().to(torch.int32).numpy()

        set_ref = set(ref_np)
        set_trt = set(trt_np)
        intersection = set_ref & set_trt
        print("selected/all:", len(intersection), "/", len(set_ref), "=", len(intersection) / len(set_ref))
        if len(intersection) != len(set_ref):
            ref_ordered = input[i][ref_np].sort()
            trt_ordered = input[i][trt_np].sort()
            print(indexes_ref[i][ref_ordered[1]])
            print(indexes[i][trt_ordered[1]])

    # Performance test with CUDA events

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(200):
        _ = bucket_sort_topk(input, starts, ends, topk)
    torch.cuda.synchronize()

    n_iters = 200
    start_event.record()
    for _ in range(n_iters):
        _ = bucket_sort_topk(input, starts, ends, topk)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Average bucket_sort_topk time: {elapsed_time_ms / n_iters:.3f} ms")

    # Torch topk time
    start_event.record()
    for _ in range(n_iters):
        _ = torch.topk(input, topk, dim=-1)[1]
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Average torch.topk time: {elapsed_time_ms / n_iters:.3f} ms")


if __name__ == "__main__":
    test_topk_selector()
