import triton
import triton.language as tl
import triton.language.extra.cann.extension as extension


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_64x64_inverse_kernel(
    A,
    Ad,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t_val = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        i_t = i_t_val
    else:
        bos, eos = i_b * T, i_b * T + T

    # Base pointers (already offset by batch and head)
    A += (bos * H + i_h) * 64
    Ad += (bos * H + i_h) * 16
    Ai += (bos * H + i_h) * 64

    # load Ai_22 (Ad block at row i_t * 64 + 16, col 0, 16 * 16)
    offs_m = i_t * 64 + 16 + tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    mask_Ad = (offs_m[:, None] < T) & (offs_n[None, :] < 16)
    ptr_Ad = Ad + offs_m[:, None] * (H * 16) + offs_n[None, :]
    Ai_22 = tl.load(ptr_Ad, mask=mask_Ad, other=0.0).to(tl.float32)

    # load A_21 (A block at row i_t * 64 + 16, col 0, 16 * 16)
    mask_A = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_A = A + offs_m[:, None] * (H * 64) + offs_n[None, :]
    A_21 = tl.load(ptr_A, mask=mask_A, other=0.0).to(tl.float32)
    tmp = tl.dot(Ai_22, A_21, input_precision="ieee")

    # load Ai_11 (Ad block at row i_t * 64, col 0, 16 * 16)
    offs_m = i_t * 64 + tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    mask_Ad = (offs_m[:, None] < T) & (offs_n[None, :] < 16)
    ptr_Ad = Ad + offs_m[:, None] * (H * 16) + offs_n[None, :]
    Ai_11 = tl.load(ptr_Ad, mask=mask_Ad, other=0.0).to(tl.float32)

    Ai_21 = -tl.dot(tmp, Ai_11, input_precision="ieee")

    # load Ai_44 (Ad block at row i_t * 64 + 48, col 0, 16 * 16)
    offs_m = i_t * 64 + 48 + tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    mask_Ad = (offs_m[:, None] < T) & (offs_n[None, :] < 16)
    ptr_Ad = Ad + offs_m[:, None] * (H * 16) + offs_n[None, :]
    Ai_44 = tl.load(ptr_Ad, mask=mask_Ad, other=0.0).to(tl.float32)

    # load A_43 (Ad block at row i_t * 64 + 48, col 32, 16 * 16)
    offs_n = 32 + tl.arange(0, 16)
    mask_A = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_A = A + offs_m[:, None] * (H * 64) + offs_n[None, :]
    A_43 = tl.load(ptr_A, mask=mask_A, other=0.0).to(tl.float32)
    tmp = tl.dot(Ai_44, A_43, input_precision="ieee")

    # load Ai_33 (Ad block at row i_t * 64 + 32, col 0, 16 * 16)
    offs_m = i_t * 64 + 32 + tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    mask_Ad = (offs_m[:, None] < T) & (offs_n[None, :] < 16)
    ptr_Ad = Ad + offs_m[:, None] * (H * 16) + offs_n[None, :]
    Ai_33 = tl.load(ptr_Ad, mask=mask_Ad, other=0.0).to(tl.float32)

    Ai_43 = -tl.dot(tmp, Ai_33, input_precision="ieee")

    # build Ai_22_32 (32 * 32)
    Ai_22_32 = tl.zeros((32, 32), tl.float32)
    Ai_22_32 = extension.insert_slice(Ai_22_32, Ai_33, (0, 0), (16, 16), (1, 1))
    Ai_22_32 = extension.insert_slice(Ai_22_32, Ai_44, (16, 16), (16, 16), (1, 1))
    Ai_22_32 = extension.insert_slice(Ai_22_32, Ai_43, (16, 0), (16, 16), (1, 1))

    # load A_21_32 (A block at row i_t * 64 + 32, col 0, 32 * 32)
    offs_m = i_t * 64 + 32 + tl.arange(0, 32)
    offs_n = tl.arange(0, 32)
    mask_A = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_A = A + offs_m[:, None] * (H * 64) + offs_n[None, :]
    A_21_32 = tl.load(ptr_A, mask=mask_A, other=0.0).to(tl.float32)
    tmp = tl.dot(Ai_22_32, A_21_32, input_precision="ieee")

    # build Ai_11_32 (32 * 32)
    Ai_11_32 = tl.zeros((32, 32), tl.float32)
    Ai_11_32 = extension.insert_slice(Ai_11_32, Ai_11, (0, 0), (16, 16), (1, 1))
    Ai_11_32 = extension.insert_slice(Ai_11_32, Ai_22, (16, 16), (16, 16), (1, 1))
    Ai_11_32 = extension.insert_slice(Ai_11_32, Ai_21, (16, 0), (16, 16), (1, 1))

    Ai_21_32 = -tl.dot(tmp, Ai_11_32, input_precision="ieee")

    # store Ai_11_32 to (i_t * 64, 0)
    offs_m = i_t * 64 + tl.arange(0, 32)
    offs_n = tl.arange(0, 32)
    mask_store = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_Ai = Ai + offs_m[:, None] * (H * 64) + offs_n[None, :]
    tl.store(ptr_Ai, Ai_11_32.to(ptr_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), mask=mask_store)

    # store Ai_22_32 to (i_t * 64 + 32, 32)
    offs_m = i_t * 64 + 32 + tl.arange(0, 32)
    offs_n = 32 + tl.arange(0, 32)
    mask_store = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_Ai = Ai + offs_m[:, None] * (H * 64) + offs_n[None, :]
    tl.store(ptr_Ai, Ai_22_32.to(ptr_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), mask=mask_store)

    # store Ai_21_32 to (i_t * 64 + 32, 32)
    offs_n = tl.arange(0, 32)
    mask_store = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_Ai = Ai + offs_m[:, None] * (H * 64) + offs_n[None, :]
    tl.store(ptr_Ai, Ai_21_32.to(ptr_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), mask=mask_store)

    # zero out the upper-right 32 * 32 block (rows 0 ~ 31, cols 32 ~ 63)
    offs_m = i_t * 64 + tl.arange(0, 32)
    offs_n = 32 + tl.arange(0, 32)
    mask_store = (offs_m[:, None] < T) & (offs_n[None, :] < BT)
    ptr_Ai = Ai + offs_m[:, None] * (H * BT) + offs_n[None, :]
    zero_block = tl.zeros((32, 32), dtype=ptr_Ai.dtype.element_ty)
    tl.store(ptr_Ai, zero_block, mask=mask_store)
