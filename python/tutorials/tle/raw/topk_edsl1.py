# flake8: noqa: F821, F722, F841
from mlir.dialects import arith, memref, nvvm, scf, llvm
from mlir import ir
import torch
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


@dialect(name="mlir")
def edsl0(l_threshold_bin_id_buf: InOut["memref<?xi32, 3>"], l_new_topk_buf: InOut["memref<?xi32, 3>"],  # noqa: F722
          s_histogram: Input["memref<?xi32, 3>"], l_new_topk: Input["i32"]):  # noqa: F722, F821
    tidx = nvvm.read_ptx_sreg_tid_x(ir.IntegerType.get_signless(32))
    bdimx = nvvm.read_ptx_sreg_ntid_x(ir.IntegerType.get_signless(32))
    tidx = arith.index_cast(ir.IndexType.get(), tidx)
    bdimx = arith.index_cast(ir.IndexType.get(), bdimx)
    RADIX = memref.dim(l_threshold_bin_id_buf, arith.constant(ir.IndexType.get(), 0))
    for i in scf.for_(tidx, RADIX, bdimx):
        lh = memref.load(s_histogram, [i])
        rh = memref.load(s_histogram, [arith.addi(i, arith.constant(ir.IndexType.get(), 1))])
        nested_if = scf.if_([],
                            arith.andi(arith.cmpi(arith.CmpIPredicate.sgt, lh, l_new_topk),
                                       arith.cmpi(arith.CmpIPredicate.sle, rh, l_new_topk)))
        nested_thenblock = nested_if.opview.thenRegion.blocks.append()
        with ir.InsertionPoint(nested_thenblock):
            memref.store(arith.index_cast(ir.IntegerType.get_signless(32), i), l_threshold_bin_id_buf,
                         [arith.constant(ir.IndexType.get(), 0)])
            scf.yield_([])
        scf.yield_([])
    nvvm.barrier0()
    ifop = scf.if_([], arith.cmpi(arith.CmpIPredicate.eq, tidx, arith.constant(ir.IndexType.get(), 0)))
    thenblock = ifop.opview.thenRegion.blocks.append()
    with ir.InsertionPoint(thenblock):
        l_threshold_bin_id = memref.load(l_threshold_bin_id_buf, [arith.constant(ir.IndexType.get(), 0)])
        l_new_topk = arith.subi(
            l_new_topk,
            memref.load(s_histogram, [
                arith.addi(arith.index_cast(ir.IndexType.get(), l_threshold_bin_id),
                           arith.constant(ir.IndexType.get(), 1))
            ]))
        memref.store(l_new_topk, l_new_topk_buf, [arith.constant(ir.IndexType.get(), 0)])
        scf.yield_([])


@dialect(name="mlir")
def edsl1(thre_bin_sum_buf: InOut["memref<?xi32, 3>"], indices_base: Input["!llvm.ptr<1>"],
          s_input_ids_base: Input["!llvm.ptr<1>"], inputs: Input["!llvm.ptr<1>"],
          s_histogram: Input["memref<?xi32, 3>"], l_start_idx: Input["i32"], l_end_idx: Input["i32"], S: Input["i32"],
          l_threshold_bin_id: Input["i32"], l_new_topk: Input["i32"], BS: Input["i32"],
          K_tensor: Input["memref<?xi32, 3>"]):
    """
    TileLang equivalent:
    l_end_idx = ~
    l_start_idx = ~
    seq_len = S
    bx = i_b
    l_threshold_bin_id = ~
    s_num_input = thre_bin_sum (thre_bin_sum_buf[0])
    BLOCK_SIZE = BS
    index = indices

    l_bin_id32 = inval_int16
    bin_id = inval_int16

    for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
        T.sync_threads() #我们不用，因为在global mem里
        input_idx = s * BLOCK_SIZE + tx
        if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len:
            bin_id = convert_to_uint16(s_base[input_idx])
            l_bin_id32 = T.Cast(T.int32, bin_id)
            if l_bin_id32 > l_threshold_bin_id:
                pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True)
                indices_base[pos] = input_idx
            #说这个l_new_topk > 0有问题？现在edsl实现删了
            elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                pos = T.atomic_add(s_num_input[0], 1, return_prev=True)
                s_input_ids_base[pos] = input_idx
    """
    tidx = nvvm.read_ptx_sreg_tid_x(ir.IntegerType.get_signless(32))
    bidx = nvvm.read_ptx_sreg_ctaid_x(ir.IntegerType.get_signless(32))
    bdimx = nvvm.read_ptx_sreg_ntid_x(ir.IntegerType.get_signless(32))  # blockDim.x
    # tidx_idx = arith.index_cast(ir.IndexType.get(), tidx)
    zero = arith.constant(ir.IndexType.get(), 0)
    one = arith.constant(ir.IndexType.get(), 1)
    num_iters = arith.ceildivsi(S, BS)
    num_iters_idx = arith.index_cast(ir.IndexType.get(), num_iters)

    i32_ty = ir.IntegerType.get_signless(32)
    f32_ty = ir.F32Type.get()
    ptr_ty = ir.Type.parse("!llvm.ptr<1>")
    zero_i32 = arith.constant(i32_ty, 0)
    one_i32 = arith.constant(i32_ty, 1)
    zero_idx = arith.constant(ir.IndexType.get(), 0)
    # Extract K from tensor
    K = memref.load(K_tensor, [zero_idx])

    # TileLang: for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
    for s in scf.for_(zero, num_iters_idx, one):

        # Thread-stride loop to match triton's tl.arange(0, BS) behavior
        # Each thread processes multiple elements: input_idx = s * BS + tidx + stride * blockDim.x
        BS_idx = arith.index_cast(ir.IndexType.get(), BS)
        bdimx_idx = arith.index_cast(ir.IndexType.get(), bdimx)
        num_strides = arith.ceildivsi(BS, bdimx)
        num_strides_idx = arith.index_cast(ir.IndexType.get(), num_strides)

        for stride in scf.for_(zero, num_strides_idx, one):
            # TileLang: input_idx = s * BLOCK_SIZE + tx + stride * blockDim.x
            s_i32 = arith.index_cast(i32_ty, s)
            stride_i32 = arith.index_cast(i32_ty, stride)
            stride_offset = arith.muli(stride_i32, bdimx)
            input_idx_i32 = arith.addi(arith.addi(arith.muli(s_i32, BS), tidx), stride_offset)
            input_idx_back_i32 = input_idx_i32

            # TileLang: if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len:
            cond1 = arith.cmpi(arith.CmpIPredicate.slt, input_idx_back_i32, l_end_idx)
            cond2 = arith.cmpi(arith.CmpIPredicate.sge, input_idx_back_i32, l_start_idx)
            cond3 = arith.cmpi(arith.CmpIPredicate.slt, input_idx_back_i32, S)
            cond_all = arith.andi(arith.andi(cond1, cond2), cond3)

            ifop = scf.if_([], cond_all)
            thenblock = ifop.opview.thenRegion.blocks.append()
            with ir.InsertionPoint(thenblock):
                # TileLang: bin_id = convert_to_uint16(s_base[input_idx])
                # Load input value: s_base[input_idx] = inputs[i_b * S + input_idx]
                # bidx is already i32 (nvvm.read_ptx_sreg_ctaid_x returns i32)
                base_offset = arith.muli(bidx, S)
                full_offset = arith.addi(base_offset, input_idx_back_i32)

                # llvm.getelementptr expects signless integer indices (not `index`)
                input_ptr = llvm.getelementptr(ptr_ty, inputs, [full_offset], [-2147483648], f32_ty, 0)
                input_val = llvm.load(f32_ty, input_ptr)

                # Convert to uint16: convert_to_uint16 logic
                f16_ty = ir.F16Type.get()
                # arith.bitcast requires signless integer type (not ui16)
                i16_ty = ir.IntegerType.get_signless(16)
                input_f16 = arith.truncf(f16_ty, input_val)
                input_i16 = arith.bitcast(i16_ty, input_f16)
                input_ui16_i32 = arith.extui(i32_ty, input_i16)

                is_neg = arith.cmpf(arith.CmpFPredicate.OLT, input_val, arith.constant(f32_ty, 0.0))
                mask_0xFFFF = arith.constant(i32_ty, 0xFFFF)
                mask_0x8000 = arith.constant(i32_ty, 0x8000)
                mask_0xFF = arith.constant(i32_ty, 0xFF)

                neg_bits = arith.andi(arith.xori(input_ui16_i32, mask_0xFFFF), mask_0xFFFF)
                pos_bits = arith.ori(input_ui16_i32, mask_0x8000)
                processed_bits = arith.select(is_neg, neg_bits, pos_bits)
                bin_id_i32 = arith.shrui(processed_bits, arith.constant(i32_ty, 8))
                bin_id_i32 = arith.andi(bin_id_i32, mask_0xFF)

                # TileLang: l_bin_id32 = T.Cast(T.int32, bin_id)
                l_bin_id32 = bin_id_i32

                # TileLang: if l_bin_id32 > l_threshold_bin_id:
                over_thre = arith.cmpi(arith.CmpIPredicate.sgt, l_bin_id32, l_threshold_bin_id)
                over_thre_if = scf.if_([], over_thre)
                over_thre_then = over_thre_if.opview.thenRegion.blocks.append()
                with ir.InsertionPoint(over_thre_then):
                    # TileLang: pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True)
                    bin_id_plus_one = arith.addi(l_bin_id32, one_i32)
                    bin_id_plus_one_idx = arith.index_cast(ir.IndexType.get(), bin_id_plus_one)

                    # Direct atomic_add on s_histogram, return previous value as pos
                    pos = memref.atomic_rmw(arith.AtomicRMWKind.addi, one_i32, s_histogram, [bin_id_plus_one_idx])
                    # pos is already i32; by suffix-sum invariant, positions for bins > threshold are in-range
                    indices_ptr = llvm.getelementptr(ptr_ty, indices_base, [pos], [-2147483648], i32_ty, 0)
                    llvm.store(input_idx_back_i32, indices_ptr)
                    scf.yield_([])

                # TileLang: elif l_bin_id32 == l_threshold_bin_id:
                # 移除 l_new_topk > 0 检查，与 topk.py 保持一致
                # 嵌套在 else 分支中实现 elif 语义
                over_thre_else = over_thre_if.opview.elseRegion.blocks.append()
                with ir.InsertionPoint(over_thre_else):
                    eq_thre = arith.cmpi(arith.CmpIPredicate.eq, l_bin_id32, l_threshold_bin_id)
                    eq_thre_if = scf.if_([], eq_thre)
                    eq_thre_then = eq_thre_if.opview.thenRegion.blocks.append()
                    with ir.InsertionPoint(eq_thre_then):
                        # TileLang: pos = T.atomic_add(s_num_input[0], 1, return_prev=True)
                        # s_num_input = thre_bin_sum_buf[0]
                        pos = memref.atomic_rmw(arith.AtomicRMWKind.addi, one_i32, thre_bin_sum_buf, [zero_idx])
                        # TileLang: s_input_ids_base[pos] = input_idx
                        # Use llvm.getelementptr and llvm.store to access global memory
                        s_input_ptr = llvm.getelementptr(ptr_ty, s_input_ids_base, [pos], [-2147483648], i32_ty, 0)
                        llvm.store(input_idx_back_i32, s_input_ptr)
                        scf.yield_([])
                    scf.yield_([])

                scf.yield_([])
            scf.yield_([])  # End of stride loop
        scf.yield_([])

    nvvm.barrier0()


@triton.autotune(
    configs=[
        # triton.Config({"BS": 32, "BSS": 32}, num_stages=1, num_warps=1),
        # triton.Config({"BS": 64, "BSS": 32}, num_stages=1, num_warps=1),
        # triton.Config({"BS": 512, "BSS": 64}, num_stages=2, num_warps=2),
        triton.Config({"BS": 1024, "BSS": 256}, num_stages=2, num_warps=32),
        # triton.Config({"BS": 2048, "BSS": 256}, num_stages=2, num_warps=4),
        # triton.Config({"BS": 4096, "BSS": 512}, num_stages=3, num_warps=4),
        # triton.Config({"BS": 8192, "BSS": 512}, num_stages=3, num_warps=8),
        # triton.Config({"BS": 8192, "BSS": 1024}, num_stages=3, num_warps=8),
    ],
    key=["S", "K"],
)
@triton.jit
def kernel_bucket_sort_topk(  # grid(B, BS)
        inputs,  # (B, S) Note: no H because MLA is based on MQA and MHA, not GQA
        indices,  # (B, K) topk index array
        s_input_ids,  # Data indices to be filtered in the next round # s_input_idx --load from gmem--> s_input_ids --load from gmem--> inputs
        starts,  # for variable length
        ends,  # for variable length
        S: tl.constexpr,  # sequence length
        K: tl.constexpr,  # k of topk
        HISTOGRAM_SIZE: tl.constexpr, SMEM_INPUT_SIZE: tl.constexpr,  # to save candidates of next loop
        BS: tl.constexpr,  # block size of S
        BSS: tl.constexpr,  # block size of SMEM_INPUT
):
    # Get thread block id
    i_b = tl.program_id(0)

    # Block base pointer definitions
    s_base = inputs + i_b * S
    indices_base = indices + i_b * K
    s_input_ids_base = s_input_ids + i_b * SMEM_INPUT_SIZE

    # Histogram initialization
    s_histogram = tl.zeros([HISTOGRAM_SIZE], dtype=tl.int32)

    # Support variable length
    l_start_idx = tl.load(starts + i_b).to(tl.int32)
    l_end_idx = tl.load(ends + i_b).to(tl.int32)

    # Record how many positions remain to fill the topk array
    l_new_topk = K

    TS = tl.cdiv(S, BS)
    for s in range(TS):
        input_idx = s * BS + tl.arange(0, BS)
        input_mask = ((input_idx < l_end_idx) & (input_idx >= l_start_idx) & (input_idx < S))
        input = tl.load(s_base + input_idx, input_mask, other=float("-inf")).to(tl.float32)
        inval_int16 = convert_to_uint16(input)
        s_histogram += inval_int16.to(tl.int32).histogram(HISTOGRAM_SIZE)


######################
    s_histogram = s_histogram.cumsum(0, reverse=True)  # Suffix sum
    mv_idx = (tl.arange(1, HISTOGRAM_SIZE + 1) % HISTOGRAM_SIZE)  # Construct offset index matrix

    cond = (s_histogram > l_new_topk) & ((s_histogram.gather(mv_idx, 0) <= l_new_topk) | (mv_idx == 0))
    l_threshold_bin_id = cond.argmax(0)

    l_new_topk -= tl.where(tl.arange(0, HISTOGRAM_SIZE) == l_threshold_bin_id + 1, s_histogram, 0).max(0)

    sum = K - l_new_topk
    ######################

    thre_bin_sum_buf = tl.zeros([1], dtype=tl.int32)
    s = S
    bs = BS
    k_tensor = tl.full([1], K, dtype=tl.int32)  # Convert constexpr to tensor
    thre_bin_sum_buf = tle_raw.call(edsl1, [thre_bin_sum_buf], [
        indices_base, s_input_ids_base, inputs, s_histogram, l_start_idx, l_end_idx, s, l_threshold_bin_id, l_new_topk,
        bs, k_tensor
    ])
    thre_bin_sum = thre_bin_sum_buf.max(0)

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
            inval_int32 = (convert_to_uint32(s_input) >>
                           (24 - round * 8)) & 0xFF  # Ensure all bits except the last eight are zero
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
    seq_len = 4096 * 8
    topk = 2048
    torch.manual_seed(1)
    input = torch.randn(batch, seq_len, dtype=torch.float32).cuda()
    starts = torch.zeros(batch, dtype=torch.int32).cuda()
    ends = torch.ones(batch, dtype=torch.int32).cuda() * seq_len

    indexes = bucket_sort_topk(input, starts, ends, topk)
    # print(indexes)

    indexes_ref = torch.topk(input, topk, dim=-1)[1]
    # print(indexes_ref)

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
