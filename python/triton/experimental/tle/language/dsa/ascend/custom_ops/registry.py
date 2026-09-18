# Copyright 2026- Xcoresigma Technology Co., Ltd
from pathlib import Path

import triton.language as tl
import triton.language.extra.cann.extension as al

CUSTOM_OPS_BITCODE = str(Path(__file__).with_name("custom_ops.bc").resolve())

_GATHER_SUFFIX_BY_DTYPE = {tl.float16: "half", tl.bfloat16: "bf16"}


def _element_dtype(value):
    # Block-pointer tensors (tl.make_block_ptr) carry a nested dtype:
    # pointer<block<[shape], dtype>>. Unwrap element_ty down to the scalar.
    dtype = value.dtype
    while hasattr(dtype, "element_ty"):
        dtype = dtype.element_ty
    return dtype


def _gather_dtype_suffix(value, op_name):
    dtype = _element_dtype(value)
    suffix = _GATHER_SUFFIX_BY_DTYPE.get(dtype)
    assert suffix is not None, (f"{op_name} only supports fp16/bf16, got {dtype}")
    return suffix


@al.register_custom_op
class duplicate_bitwise_mask:
    """
    /*
     * Function:
     *   Copy scalar (variable or immediate) to fill vector in bitwise mask mode, corresponding to the Duplicate in Ascend C API.
     *
     * Inputs:
     *   scalar_value: source
     *   mask: controls the elements involved in computation per iteration
     *   repeat_times: repeat times
     *   dst_block_stride: data block address stride in one iteration
     *   dst_repeat_stride: address stride of dst between adjacent iterations
     *
     * Outputs:
     *   out: required UB destination tensor storing output
     *
     */
    """

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(self, scalar_value, mask, repeat_times, dst_block_stride, dst_repeat_stride, out=None):
        assert out
        assert _element_dtype(out) == _element_dtype(scalar_value), (
            f"duplicate_bitwise_mask requires dst dtype ({_element_dtype(out)}) "
            f"to match scalar_value dtype ({_element_dtype(scalar_value)})")
        if _element_dtype(out) == tl.float16:
            self.symbol = "custom_duplicate_bitwise_mask_half"
        elif _element_dtype(out) == tl.bfloat16:
            self.symbol = "custom_duplicate_bitwise_mask_bf16"
        elif _element_dtype(out) == tl.float32:
            self.symbol = "custom_duplicate_bitwise_mask_float"
        else:
            assert False, (f"duplicate_bitwise_mask does not support dtype {_element_dtype(out)} yet")
        self.bitcode = CUSTOM_OPS_BITCODE
        self.extra_buffers = [(tl.float16, 0)]


@al.register_custom_op
class gather_gm_to_l1:
    """
    /*
     * Function:
     *   Gather fp16/bf16 rows from a row-contiguous 2D GM tensor by discrete
     *   indices, write them into an L1/CBUF tensor and perform the ND2NZ
     *   layout conversion for CUBE use. Output row i comes from source row
     *   index[i]; adjacent indices (index[i + 1] == index[i] + 1) are merged
     *   into a single two-row copy.
     *
     * Inputs:
     *   src: row-contiguous 2D fp16/bf16 source tensor in GM.
     *   index: 2D int32 index tensor in GM (shape (N, 1), stride (1, 1));
     *     output row i takes its data from source row index[i] (0-based row
     *     number); the starting offset of the indices is expressed through
     *     the block ptr offsets, so no extra parameter is needed.
     *   tile_size: number of data rows to gather.
     *   D: number of fp16/bf16 elements in each source row.
     *
     * Outputs:
     *   out: required 4D L1/CBUF fp16/bf16 destination tensor (same dtype as
     *     src), corresponding to dst in the C++ ABI; the gathered result is
     *     written into this tensor directly and it is returned on the Python
     *     side.
     */
    """

    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_MTE2
    mode = al.MODE.SIMD

    def __init__(self, src, index, tile_size, D, out=None):
        assert out is not None, "out buffer is required"
        assert _element_dtype(out) == _element_dtype(src), (
            f"gather_gm_to_l1 requires out dtype ({_element_dtype(out)}) "
            f"to match src dtype ({_element_dtype(src)})")
        assert _element_dtype(index) == tl.int32, (f"gather_gm_to_l1 requires int32 index, "
                                                   f"got {_element_dtype(index)}")
        self.symbol = ("custom_gather_gm_to_l1_" + _gather_dtype_suffix(src, "gather_gm_to_l1"))
        self.bitcode = CUSTOM_OPS_BITCODE


@al.register_custom_op
class gather_gm_to_ub:
    """
    /*
     * Function:
     *   On the VECTOR core, gather fp16/bf16 rows from a row-contiguous 2D
     *   GM tensor by index and write them into a UB tensor. Output row i
     *   comes from source row index[i]; adjacent indices
     *   (index[i + 1] == index[i] + 1) are merged into a single two-row DMA
     *   copy.
     *
     * Inputs:
     *   src: row-contiguous 2D fp16/bf16 source tensor in GM.
     *   index: 2D int32 index tensor in GM (shape (N, 1), stride (1, 1));
     *     output row i takes its data from source row index[i] (0-based row
     *     number); the starting offset of the indices is expressed through
     *     the block ptr offsets, so no extra parameter is needed.
     *   tile_size: number of data rows to gather.
     *   D: number of fp16/bf16 elements copied from each source row.
     *
     * Outputs:
     *   out: required 2D UB fp16/bf16 destination tensor (same dtype as
     *     src), corresponding to dst in the C++ ABI; the stride of the first
     *     dimension must be no smaller than D. The gathered result is
     *     written into this tensor directly and it is returned on the
     *     Python side.
     */
    """

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_MTE2
    mode = al.MODE.SIMD

    def __init__(self, src, index, tile_size, D, out=None):
        assert out is not None, "out buffer is required"
        assert _element_dtype(out) == _element_dtype(src), (
            f"gather_gm_to_ub requires out dtype ({_element_dtype(out)}) "
            f"to match src dtype ({_element_dtype(src)})")
        assert _element_dtype(index) == tl.int32, (f"gather_gm_to_ub requires int32 index, "
                                                   f"got {_element_dtype(index)}")
        self.symbol = ("custom_gather_gm_to_ub_" + _gather_dtype_suffix(src, "gather_gm_to_ub"))
        self.bitcode = CUSTOM_OPS_BITCODE


@al.register_custom_op
class gather_mask_builtin_pattern:
    """
    /*
     * Function:
     *   Select elements from src operand with built-in fixed patterns, corresponding to the GatherMask in Ascend C API.
     *
     * Inputs:
     *   src0: 1D input tensor in UB storing source.
     *   src1_pattern: built-in fixed patterns.
     *   reduce_mode: selects mask parameter mode.
     *   mask: controls elements involved in computation per iteration.
     *   src0_block_stride: the address stride between different DataBlocks of src0 within the same iteration. (struct GatherMaskParams)
     *   repeat_times: number of iterations. (struct GatherMaskParams)
     *   src0_repeat_stride: address stride of src0 between adjacent iterations. (struct GatherMaskParams)
     *   src1_repeat_stride: address stride of src1 between adjacent iterations. (struct GatherMaskParams)
     *
     * Outputs:
     *   out: required two-element output sequence [dst, rsvd_cnt]
     *     - out[0] / dst: UB float tensor for gather result
     *     - out[1] / rsvd_cnt: number of valid elements in dst
     */
    """

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(self, src0, src1_pattern, reduce_mode, mask, src0_block_stride, repeat_times, src0_repeat_stride,
                 src1_repeat_stride, out=None):
        assert out
        assert len(out) == 2, ("gather_mask_builtin_pattern requires out to be [dst, rsvd_cnt]")
        assert _element_dtype(out[0]) == _element_dtype(src0), (
            f"gather_mask_builtin_pattern requires dst dtype ({_element_dtype(out[0])}) "
            f"to match src0 dtype ({_element_dtype(src0)})")
        assert _element_dtype(out[1]) == tl.int64, (
            f"gather_mask_builtin_pattern only supports int64 rsvd_cnt, got {_element_dtype(out[1])}")
        if _element_dtype(src0) == tl.float16:
            self.symbol = "custom_gather_mask_builtin_pattern_half"
        elif _element_dtype(src0) == tl.bfloat16:
            self.symbol = "custom_gather_mask_builtin_pattern_bf16"
        elif _element_dtype(src0) == tl.uint16:
            self.symbol = "custom_gather_mask_builtin_pattern_ushort"
        elif _element_dtype(src0) == tl.int16:
            self.symbol = "custom_gather_mask_builtin_pattern_short"
        elif _element_dtype(src0) == tl.float32:
            self.symbol = "custom_gather_mask_builtin_pattern_float"
        elif _element_dtype(src0) == tl.uint32:
            self.symbol = "custom_gather_mask_builtin_pattern_uint"
        elif _element_dtype(src0) == tl.int32:
            self.symbol = "custom_gather_mask_builtin_pattern_int"
        else:
            assert False, (f"gather_mask_builtin_pattern src0 does not support dtype {_element_dtype(src0)}")
        self.bitcode = CUSTOM_OPS_BITCODE
        self.extra_buffers = [(tl.float16, 0)]


@al.register_custom_op
class gather_mask_custom_pattern:
    """
    /*
     * Function:
     *   Select elements from src operand with user-defined pattern, corresponding to the GatherMask in Ascend C API.
     *
     * Inputs:
     *   src0: 1D input tensor in UB storing source.
     *   src1_pattern: user-defined pattern.
     *   reduce_mode: selects mask parameter mode.
     *   mask: controls elements involved in computation per iteration.
     *   src0_block_stride: the address stride between different DataBlocks of src0 within the same iteration. (struct GatherMaskParams)
     *   repeat_times: number of iterations. (struct GatherMaskParams)
     *   src0_repeat_stride: address stride of src0 between adjacent iterations. (struct GatherMaskParams)
     *   src1_repeat_stride: address stride of src1 between adjacent iterations. (struct GatherMaskParams)
     *
     * Outputs:
     *   out: required two-element output sequence [dst, rsvd_cnt]
     *     - out[0] / dst: UB float tensor for gather result
     *     - out[1] / rsvd_cnt: number of valid elements in dst
     */
    """

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(self, src0, src1_pattern, reduce_mode, mask, src0_block_stride, repeat_times, src0_repeat_stride,
                 src1_repeat_stride, out=None):
        assert out
        assert len(out) == 2, ("gather_mask_custom_pattern requires out to be [dst, rsvd_cnt]")
        assert _element_dtype(out[0]) == _element_dtype(src0), (
            f"gather_mask_custom_pattern requires dst dtype ({_element_dtype(out[0])}) "
            f"to match src0 dtype ({_element_dtype(src0)})")
        assert _element_dtype(out[1]) == tl.int64, (
            f"gather_mask_custom_pattern only supports int64 rsvd_cnt, got {_element_dtype(out[1])}")
        assert isinstance(src1_pattern, tl.tensor), "gather_mask_custom_pattern requires src1_pattern to be a tensor"
        if _element_dtype(src1_pattern) == tl.uint16:
            if _element_dtype(src0) == tl.float16:
                self.symbol = "custom_gather_mask_custom_pattern_half"
            elif _element_dtype(src0) == tl.bfloat16:
                self.symbol = "custom_gather_mask_custom_pattern_bf16"
            elif _element_dtype(src0) == tl.uint16:
                self.symbol = "custom_gather_mask_custom_pattern_ushort"
            elif _element_dtype(src0) == tl.int16:
                self.symbol = "custom_gather_mask_custom_pattern_short"
            else:
                assert False, "when src1_pattern is uint16_t tensor, src0 only supports dtype half / bfloat16 / uint16 / int16"
        elif _element_dtype(src1_pattern) == tl.uint32:
            if _element_dtype(src0) == tl.float32:
                self.symbol = "custom_gather_mask_custom_pattern_float"
            elif _element_dtype(src0) == tl.uint32:
                self.symbol = "custom_gather_mask_custom_pattern_uint"
            elif _element_dtype(src0) == tl.int32:
                self.symbol = "custom_gather_mask_custom_pattern_int"
            else:
                assert False, "when src1_pattern is uint32_t tensor, src0 only supports dtype float32 / uint32/ int32"
        else:
            assert False, (
                f"gather_mask_custom_pattern src1_pattern does not support dtype {_element_dtype(src1_pattern)}")
        self.bitcode = CUSTOM_OPS_BITCODE
        self.extra_buffers = [(tl.float16, 0)]


@al.register_custom_op
class pair_reduce_sum_continuous_mask:
    """
    /*
     * Function:
     *   Sum pairwise for odd and even elements in continuous mask mode, corresponding to the PairReduceSum in Ascend C API.
     *
     * Inputs:
     *   src: 1D input tensor in UB storing source
     *   repeat_times: repeat times
     *   mask: number of leading consecutive elements for computation
     *   dst_rep_stride: address stride of dst between adjacent iterations
     *   src_blk_stride: data block address stride in one iteration
     *   src_rep_stride: Address stride of src between adjacent iterations
     *
     * Outputs:
     *   out: required UB destination tensor storing output
     */
    """

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(self, src, repeat_times, mask, dst_rep_stride, src_blk_stride, src_rep_stride, out=None):
        assert out
        assert _element_dtype(out) == _element_dtype(src), (
            f"pair_reduce_sum_continuous_mask requires dst dtype ({_element_dtype(out)}) "
            f"to match src dtype ({_element_dtype(src)})")
        if _element_dtype(src) == tl.float16:
            self.symbol = "custom_pair_reduce_sum_continuous_mask_half"
        elif _element_dtype(src) == tl.float32:
            self.symbol = "custom_pair_reduce_sum_continuous_mask_float"
        else:
            assert False, (f"pair_reduce_sum_continuous_mask does not support dtype {_element_dtype(src)} yet")
        self.bitcode = CUSTOM_OPS_BITCODE
        self.extra_buffers = [(tl.float16, 0)]


@al.register_custom_op
class sort32:
    """
    /*
     * Function:
     *   Sort 32 elements in descending order in each repeat, corresponding to the Sort32 in Ascend C API.
     *
     * Inputs:
     *   src0: 1D float input tensor in UB storing value
     *   src1: 1D uint32 input tensor in UB storing index
     *   repeat_times: repeat times
     *
     * Outputs:
     *   out: required UB float destination tensor storing sorted pair [value, index]
     */
    """

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(self, src0, src1, repeat_times, out=None):
        assert _element_dtype(src0) == tl.float32, (f"sort32 only supports fp32 value, got {_element_dtype(src0)}")
        assert _element_dtype(src1) == tl.uint32, (f"sort32 only supports uint32 index, got {_element_dtype(src1)}")
        assert out
        assert _element_dtype(out) == tl.float32, (f"sort32 only supports fp32 out, got {_element_dtype(out)}")
        self.symbol = "custom_sort32"
        self.bitcode = CUSTOM_OPS_BITCODE
        self.extra_buffers = [(tl.float16, 0)]


@al.register_custom_op
class sort_1d_pack:
    """
    /*
     * Function:
     *   Sort a 1D float tensor in UB and, according to sort_impl, select the
     *   matching TopK sort implementation, emitting the best TOPK compact
     *   proposals. Each proposal occupies two float slots holding the value
     *   and, in the second slot, the encoded index.
     *
     * Inputs:
     *   src: 1D float input tensor in UB.
     *   tmp_buf: UB scratch tensor holding intermediate sort/merge results.
     *   descending: sort descending when True, ascending when False.
     *   TOPK: number of proposals to keep.
     *   index_offset: offset added to each raw index.
     *   sort_impl: sort implementation number; the device side picks the
     *     matching sort path from it based on the input size.
     *
     * Outputs:
     *   out: required UB float destination tensor, corresponding to
     *     dst_proposals in the C++ ABI; must hold at least 2 * TOPK float
     *     slots storing the TOPK proposals compactly as
     *     [value, encoded_index]. It is returned on the Python side.
     */
    """

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(self, src, tmp_buf, descending, TOPK, index_offset, sort_impl, out=None):
        assert out
        assert _element_dtype(src) == tl.float32, (f"sort_1d_pack only supports fp32 src, got {_element_dtype(src)}")
        assert _element_dtype(tmp_buf) == tl.float32, (f"sort_1d_pack only supports fp32 tmp_buf, "
                                                       f"got {_element_dtype(tmp_buf)}")
        assert _element_dtype(out) == tl.float32, (f"sort_1d_pack only supports fp32 out, got {_element_dtype(out)}")
        self.symbol = "custom_sort_1d_pack_float"
        self.bitcode = CUSTOM_OPS_BITCODE
        self.extra_buffers = [(tl.float16, 0)]


@al.register_custom_op
class merge_exhaust_sort4:
    """
    /*
     * Function:
     *   Perform one exhaustion-mode merge over up to four sorted proposal
     *   ways in UB, emitting the safely determined sorted prefix and the
     *   number of proposals actually consumed from each way. All offsets
     *   and lengths are counted in proposals, not float slots.
     *
     * Inputs:
     *   src_proposals: UB tensor holding multiple sorted compact proposal
     *     ways.
     *   ways: number of active input ways; must match the number of
     *     non-zero values among len0..len3.
     *   off0, off1, off2, off3: starting offset of each of the four input
     *     ways, in proposals.
     *   len0, len1, len2, len3: number of proposals available in each of
     *     the four input ways; 0 marks the way as inactive.
     *
     * Outputs:
     *   out: required two-element output sequence [dst_proposals,
     *     consumed_out]:
     *     - out[0] / dst_proposals: UB float tensor receiving the safely
     *       determined sorted proposal prefix of the merge, corresponding
     *       to dst_proposals in the C++ ABI.
     *     - out[1] / consumed_out: UB tensor holding at least four int32
     *       elements, recording in order the number of proposals actually
     *       consumed this round from original ways 0 through 3,
     *       corresponding to consumed_out in the C++ ABI.
     *   The Python side returns (dst_proposals, consumed_out) in the same
     *   order.
     */
    """

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(self, src_proposals, ways, off0, off1, off2, off3, len0, len1, len2, len3, out=None):
        assert out
        assert len(out) == 2, ("merge_exhaust_sort4 requires out to be "
                               "[dst_proposals, consumed_out]")
        assert _element_dtype(src_proposals) == tl.float32, (f"merge_exhaust_sort4 only supports fp32 src_proposals, "
                                                             f"got {_element_dtype(src_proposals)}")
        assert _element_dtype(out[0]) == tl.float32, (f"merge_exhaust_sort4 only supports fp32 out[0], "
                                                      f"got {_element_dtype(out[0])}")
        assert _element_dtype(out[1]) == tl.int32, (f"merge_exhaust_sort4 only supports int32 out[1], "
                                                    f"got {_element_dtype(out[1])}")
        self.symbol = "custom_merge_exhaust_sort4_float"
        self.bitcode = CUSTOM_OPS_BITCODE
        self.extra_buffers = [(tl.float16, 0)]


@al.register_custom_op
class mrgsort:
    """
    /*
     * Function:
     *   Arrange and merge up to four arranged potential queues into one queue, corresponding to the MrgSort in Ascend C API.
     *
     * Inputs:
     *   src_proposals: UB tensor holding multiple sorted compact proposal
     *     queues.
     *   off0, off1, off2, off3: starting offset of each of the four input
     *     queues, in proposals.
     *   len0, len1, len2, len3: number of proposals available in each of
     *     the four input queues. (struct MrgSort4Info)
     *   if_exhausted_suspension: judge whether to stop after a queue is exhausted. (struct MrgSort4Info)
     *   valid_bit: judge whether a queue is valid or not. (struct MrgSort4Info)
     *   repeat_times: repeat times. (struct MrgSort4Info)
     *
     * Outputs:
     *   out: UB float destination tensor for the merged proposals
     */
    """

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(self, src_proposals, off0, off1, off2, off3, len0, len1, len2, len3, if_exhausted_suspension,
                 valid_bit, repeat_times, out=None):
        assert _element_dtype(src_proposals) == tl.float32, (
            f"mrgsort only supports fp32 src, got {_element_dtype(src_proposals)}")
        assert out
        assert _element_dtype(out) == tl.float32, (f"mrgsort only supports fp32 out, got {_element_dtype(out)}")
        self.symbol = "custom_mrgsort"
        self.bitcode = CUSTOM_OPS_BITCODE
        self.extra_buffers = [(tl.float16, 0)]


@al.register_custom_op
class unpack_sort:
    """
    /*
     * Function:
     *   Split compactly stored TopK proposals in UB into separate float
     *   values and int32 indices. Each input proposal consists of two
     *   float-sized slots.
     *
     * Inputs:
     *   src_proposals: UB tensor holding at least topk compact proposals;
     *     its valid view must contain exactly 2 * topk float slots.
     *   topk: number of proposals to split.
     *
     * Outputs:
     *   out: required two-element output sequence [dst_value, dst_index]:
     *     - out[0] / dst_value: UB float tensor holding at least topk
     *       elements, receiving the value of each proposal, corresponding
     *       to dst_value in the C++ ABI.
     *     - out[1] / dst_index: UB int32 tensor holding at least topk
     *       elements, receiving the decoded index of each proposal,
     *       corresponding to dst_index in the C++ ABI.
     *   The Python side returns (dst_value, dst_index) in the same order.
     */
    """

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(self, src_proposals, topk, out=None):
        assert out
        assert len(out) == 2, ("unpack_sort requires out to be [dst_value, dst_index]")
        assert _element_dtype(src_proposals) == tl.float32, (f"unpack_sort only supports fp32 src_proposals, "
                                                             f"got {_element_dtype(src_proposals)}")
        assert _element_dtype(out[0]) == tl.float32, (f"unpack_sort only supports fp32 out[0], "
                                                      f"got {_element_dtype(out[0])}")
        assert _element_dtype(out[1]) == tl.int32, (f"unpack_sort only supports int32 out[1], "
                                                    f"got {_element_dtype(out[1])}")
        self.symbol = "custom_unpack_sort_float"
        self.bitcode = CUSTOM_OPS_BITCODE
        self.extra_buffers = [(tl.float16, 0)]


# Mask buffers occupy at least one 32-byte UB block.
def _mask_source_size(src, op_name):
    assert src.dtype in (tl.float16, tl.float32), f"{op_name} requires an fp16/fp32 UB tensor"
    assert len(src.shape) == 1, f"{op_name} requires a 1D source"
    size = src.numel.value
    limit = 32768 if src.dtype == tl.float16 else 4096
    assert 256 <= size <= limit and size & (size - 1) == 0, (
        f"{op_name} requires a power-of-two source size in [256, {limit}]")
    return size


def _check_mask_buffer(mask, size, op_name):
    assert mask.dtype in (tl.uint16, tl.uint32), f"{op_name} requires a uint16/uint32 bitmask"
    width = 16 if mask.dtype == tl.uint16 else 32
    assert len(mask.shape) == 1 and mask.numel.value == size // width, (
        f"{op_name} requires a 1D bitmask with {size // width} elements")


@al.register_custom_op
class compare_scalar:
    """Packed scalar comparison of contiguous FP16/FP32 UB values.

    Based on CANN 9.1 `dav_c220/kernel_operator_vec_cmp_impl.h`
    CompareScalarCompute / VcmpvsIntrinsicsImpl (CompareScalar Level 2). All
    parameters of that API are exposed: cmpMode is the CANN CMPMODE value
    0=LT, 1=GT, 2=EQ, 3=LE, 4=GE, 5=NE (utils/kernel_utils_mode.h); count is
    the number of compared elements and must equal the source size. scalar is
    passed as FP32 and rounded to the source dtype before comparison.
    Required out: uint16[N/16], or uint32[N/32] for FP32 sources. The uint32
    form feeds gather_mask_custom_pattern without repacking. FP32 sizes:
    powers of two 256..4096; FP16: 256..32768. Bit i records the predicate
    for element i, least-significant bit first. All bits are defined.
    Inputs/outputs must be 32-byte aligned and disjoint. The op inserts no
    pipeline barrier; ordering is the caller's responsibility.
    """

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(self, src, scalar, cmpMode, count, out=None):
        assert out is not None, "compare_scalar requires an output bitmask"
        size = _mask_source_size(src, "compare_scalar")
        assert isinstance(
            cmpMode, int) and 0 <= cmpMode <= 5, ("compare_scalar cmpMode must be a compile-time CANN CMPMODE value "
                                                  "0=LT, 1=GT, 2=EQ, 3=LE, 4=GE, 5=NE")
        assert isinstance(count, int) and count == size, (
            f"compare_scalar count must be a compile-time element count equal to the source size ({size})")
        assert src.dtype == tl.float32 or out.dtype == tl.uint16, "FP16 comparison requires a uint16 mask"
        _check_mask_buffer(out, count, "compare_scalar")
        self.arg_type["scalar"] = tl.float32
        self.arg_type["cmpMode"] = tl.int32
        self.arg_type["count"] = tl.int32
        suffix = "half" if src.dtype == tl.float16 else "float"
        if out.dtype == tl.uint32:
            suffix += "_mask32"
        self.symbol = "custom_compare_scalar_" + suffix
        self.bitcode = CUSTOM_OPS_BITCODE


@al.register_custom_op
class cast_int4_to_fp16:
    """Unpack signed INT4 values from a 1D uint8 UB tensor into FP16.

    Based on CANN 9.1 `dav_c220/kernel_operator_vec_vconv_impl.h` CastImpl
    (Cast Level 2) and its int4b_t -> half specialization. All parameters of
    that API are exposed: roundMode must be CAST_NONE (0), the only mode
    supported from int4b_t to half on this device; count is the number of
    output FP16 elements and must equal 2 * N. src: uint8[N] in UB, N a
    power of two from 32 through 8192 bytes. Required out: float16[2*N] in
    UB. Each source byte encodes two signed two's-complement values in
    [-8, 7]; the low nibble is emitted first. All output elements are
    defined. No zero-point, scale or layout conversion is applied. Inputs
    and outputs must be contiguous, 32-byte aligned and disjoint. GM
    accesses stay in the caller. The op inserts no pipeline barrier beyond
    the mask set/restore of the reference implementation; ordering is the
    caller's responsibility.
    """

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(self, src, roundMode, count, out=None):
        assert out is not None, "cast_int4_to_fp16 requires an output buffer"
        assert src.dtype == tl.uint8 and len(src.shape) == 1, ("cast_int4_to_fp16 requires a 1D uint8 UB source")
        size = src.numel.value
        assert size in (32, 64, 128, 256, 512, 1024, 2048, 4096,
                        8192), ("cast_int4_to_fp16 requires a power-of-two byte count in [32, 8192]")
        assert isinstance(roundMode, int) and roundMode == 0, (
            "cast_int4_to_fp16 only supports RoundMode::CAST_NONE (0) from int4b_t to half on this device")
        assert isinstance(count, int) and count == 2 * size, (
            f"cast_int4_to_fp16 count must be a compile-time output element count equal to 2 * N ({2 * size})")
        assert out.dtype == tl.float16 and len(out.shape) == 1 and out.numel.value == count, (
            "cast_int4_to_fp16 requires a 1D float16 output with twice the source element count")
        self.arg_type["roundMode"] = tl.int32
        self.arg_type["count"] = tl.int32
        self.symbol = "custom_cast_int4_to_fp16"
        self.bitcode = CUSTOM_OPS_BITCODE


@al.register_custom_op
class cast_int8_to_fp16:
    """Convert a 1D int8 UB tensor into FP16 elementwise.

    Based on CANN 9.1 `dav_c220/kernel_operator_vec_vconv_impl.h` CastImpl
    (Cast Level 2) and its int8_t -> half specialization. All parameters of
    that API are exposed: roundMode must be CAST_NONE (0), the only mode
    supported from int8_t to half on this device; count is the output element
    count and must equal the input size. src: int8[N] in UB, N a power of two
    from 32 through 8192 elements. Required out: float16[N] in UB. All output
    elements are defined. Inputs and outputs must be contiguous, 32-byte
    aligned and disjoint. The op inserts no pipeline barrier beyond the mask
    set/restore of the reference implementation; ordering is the caller's
    responsibility.
    """

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(self, src, roundMode, count, out=None):
        assert out is not None, "cast_int8_to_fp16 requires an output buffer"
        assert src.dtype == tl.int8 and len(src.shape) == 1, ("cast_int8_to_fp16 requires a 1D int8 UB source")
        size = src.numel.value
        assert size in (32, 64, 128, 256, 512, 1024, 2048, 4096,
                        8192), ("cast_int8_to_fp16 requires a power-of-two element count in [32, 8192]")
        assert isinstance(roundMode, int) and roundMode == 0, (
            "cast_int8_to_fp16 only supports RoundMode::CAST_NONE (0) from int8_t to half on this device")
        assert isinstance(count, int) and count == size, (
            f"cast_int8_to_fp16 count must be a compile-time output element count equal to N ({size})")
        assert out.dtype == tl.float16 and len(out.shape) == 1 and out.numel.value == count, (
            "cast_int8_to_fp16 requires a 1D float16 output with the source element count")
        self.arg_type["roundMode"] = tl.int32
        self.arg_type["count"] = tl.int32
        self.symbol = "custom_cast_int8_to_fp16"
        self.bitcode = CUSTOM_OPS_BITCODE
