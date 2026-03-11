import pytest
import torch

from .flash_mla_triton import (
    torch_mla,
    decode_attention_fwd_grouped,
    decode_attention_fwd_grouped_v1,
    decode_attention_fwd_grouped_v2,
    decode_attention_fwd_grouped_v3,
    decode_attention_fwd_grouped_v5,
    decode_attention_fwd_grouped_v3_2k,
    decode_attention_fwd_grouped_v5_2k,
)

DEFAULT_START_SEQ_LENS = [13, 32, 64, 128, 256, 512, 1024, 2048, 4096, 6 * 1024, 8 * 1024, 16 * 1024]
DEFAULT_END_SEQ_LEN_OFFSET_FULL = 2048
DEFAULT_END_SEQ_LEN_OFFSET_QUICK = 256
DEFAULT_STEP = 128


def _iter_seq_lens():
    end_offset = DEFAULT_END_SEQ_LEN_OFFSET_QUICK
    for start_seq_len in DEFAULT_START_SEQ_LENS:
        end_seq_len = start_seq_len + end_offset
        for seq_len in range(start_seq_len, end_seq_len + 1, DEFAULT_STEP):
            yield seq_len


def _build_fp32_inputs(inputs):
    fp32_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if v.dtype in (torch.float16, torch.bfloat16, torch.float32):
                fp32_inputs[k] = v.float()
            else:
                fp32_inputs[k] = v
        else:
            fp32_inputs[k] = v
    return fp32_inputs


def _build_bf16_inputs(inputs):
    bf16_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if v.dtype in (torch.float16, torch.bfloat16, torch.float32):
                bf16_inputs[k] = v.to(torch.bfloat16)
            else:
                bf16_inputs[k] = v
        else:
            bf16_inputs[k] = v
    return bf16_inputs


def _compute_reference(inputs):
    fp32_inputs = _build_fp32_inputs(inputs)
    bf16_inputs = _build_bf16_inputs(inputs)
    ref_output_fp32 = torch_mla(**fp32_inputs)
    ref_output_bf16 = torch_mla(**bf16_inputs)
    ref_error = torch.max(torch.abs(ref_output_fp32 - ref_output_bf16.float()))
    ref_max = torch.max(torch.abs(ref_output_fp32))
    return ref_output_fp32, ref_error, ref_max


def _allowed_error_for_kernel(kernel_name, ref_error, ref_max, device):
    allowed_multiple = 3.0
    abs_floor = 0.02
    rtol = 0.02
    per_kernel_tolerance = {
        # Split-softmax recombination introduces extra rounding vs single-pass kernels.
        "decode_attention_fwd_grouped": {"multiple": 5.0, "abs_floor": 0.04},
    }
    kernel_tol = per_kernel_tolerance.get(kernel_name, {})
    kernel_multiple = kernel_tol.get("multiple", allowed_multiple)
    kernel_abs_floor = kernel_tol.get("abs_floor", abs_floor)
    kernel_rtol = kernel_tol.get("rtol", rtol)
    return torch.maximum(
        ref_error * kernel_multiple,
        torch.maximum(ref_max * kernel_rtol, torch.tensor(kernel_abs_floor, device=device)),
    )


def _generate_random_inputs(
    b_seq_len,
    bz=1,
    head_num=128,
    head_num_kv=1,
    num_pages=128,
    page_size=64,
    Lq=576,
    Lk=576,
    Lv=512,
):
    num_kv_splits = 4
    sm_scale = 0.1352337788608801
    logit_cap = 0.0
    num_pages_needed = (b_seq_len + page_size - 1) // page_size
    if num_pages < num_pages_needed:
        num_pages = num_pages_needed
    q = torch.randn((bz, head_num, Lq), dtype=torch.bfloat16, device="cuda")
    k_buffer = torch.randn((num_pages, page_size, head_num_kv, Lk), dtype=torch.bfloat16, device="cuda")
    v_buffer = k_buffer[:, :, :, :Lv]
    o = torch.zeros((bz, head_num, 1, Lv), dtype=torch.bfloat16, device="cuda").transpose(1, 2)
    req_to_token = torch.randint(0, num_pages, (bz, num_pages), dtype=torch.int32, device="cuda")
    b_seq_len_tensor = torch.tensor([b_seq_len], dtype=torch.int32, device="cuda")
    attn_logits = torch.randn((bz, head_num, num_kv_splits, Lv + 1), dtype=torch.float32, device="cuda")

    return {
        "q": q,
        "k_buffer": k_buffer,
        "v_buffer": v_buffer,
        "o": o,
        "req_to_token": req_to_token,
        "b_seq_len": b_seq_len_tensor,
        "attn_logits": attn_logits,
        "num_kv_splits": num_kv_splits,
        "sm_scale": sm_scale,
        "page_size": page_size,
        "logit_cap": logit_cap,
    }


@pytest.fixture(scope="module")
def _inputs_cache():
    return {}


@pytest.fixture(scope="module")
def _reference_cache():
    return {}


def _get_inputs(seq_len, cache):
    if seq_len not in cache:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        cache[seq_len] = _generate_random_inputs(seq_len)
    return cache[seq_len]


def _get_reference(seq_len, base_inputs, cache):
    if seq_len not in cache:
        cache[seq_len] = _compute_reference(base_inputs)
    return cache[seq_len]


KERNELS = [
    decode_attention_fwd_grouped,
    decode_attention_fwd_grouped_v1,
    decode_attention_fwd_grouped_v2,
    decode_attention_fwd_grouped_v3,
    decode_attention_fwd_grouped_v5,
    decode_attention_fwd_grouped_v3_2k,
    decode_attention_fwd_grouped_v5_2k,
]


@pytest.mark.parametrize("seq_len", list(_iter_seq_lens()))
@pytest.mark.parametrize("kernel_fn", KERNELS, ids=lambda fn: fn.__name__)
def test_flash_mla_accuracy(seq_len, kernel_fn, _inputs_cache, _reference_cache):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FlashMLA UT")
    inputs = _get_inputs(seq_len, _inputs_cache)
    ref_output_fp32, ref_error, ref_max = _get_reference(seq_len, inputs, _reference_cache)
    bf16_inputs = _build_bf16_inputs(inputs)
    res_out = kernel_fn(**bf16_inputs)
    act_error = torch.max(torch.abs(ref_output_fp32 - res_out.float()))
    allowed_error = _allowed_error_for_kernel(
        kernel_fn.__name__,
        ref_error,
        ref_max,
        ref_output_fp32.device,
    )
    assert act_error <= allowed_error
