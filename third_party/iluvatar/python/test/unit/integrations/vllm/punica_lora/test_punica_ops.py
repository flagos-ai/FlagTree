# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import pytest
import torch

from .punica_test_utils import (PunicaTensors, assert_close, generate_data_for_nslices, parse_case_env, seed_everything)
from .punica_torch_ops import (sgmv_expand, sgmv_expand_slice, sgmv_shrink)
from .punica_triton_ops import (LoRAKernelMeta, _LORA_A_PTR_DICT, _LORA_B_PTR_DICT, lora_expand, lora_shrink)


def sgmv_shrink_for_nslices(nslices: int, inputs_tensor: torch.Tensor, lora_weights_lst: list[torch.Tensor],
                            out_tensor: torch.Tensor, b_seq_start_loc: torch.Tensor, seq_len_tensor: torch.Tensor,
                            prompt_lora_mapping: torch.Tensor, batches: int, max_seq_length: int, num_tokens: int,
                            scaling: float):
    for index in range(nslices):
        sgmv_shrink(
            inputs_tensor,
            lora_weights_lst[index],
            out_tensor[index],
            b_seq_start_loc,
            seq_len_tensor,
            prompt_lora_mapping,
            batches,
            max_seq_length,
            num_tokens,
            scaling,
        )


def sgmv_expand_for_nslices(nslices: int, hidden_size: int, inputs_tensor: torch.Tensor,
                            lora_weights_lst: list[torch.Tensor], out_tensor: torch.Tensor,
                            b_seq_start_loc: torch.Tensor, seq_len_tensor: torch.Tensor,
                            prompt_lora_mapping: torch.Tensor, batches: int, max_seq_length: int, num_tokens: int,
                            add_inputs: bool) -> None:
    if nslices == 1:
        sgmv_expand(
            inputs_tensor[0],
            lora_weights_lst[0],
            out_tensor,
            b_seq_start_loc,
            seq_len_tensor,
            prompt_lora_mapping,
            batches,
            max_seq_length,
            num_tokens,
            add_inputs=add_inputs,
        )
    else:
        slice_offset = 0
        for index in range(nslices):
            lora_weights = lora_weights_lst[index]
            sgmv_expand_slice(
                inputs_tensor[index],
                lora_weights,
                out_tensor,
                b_seq_start_loc,
                seq_len_tensor,
                prompt_lora_mapping,
                batches,
                max_seq_length,
                num_tokens,
                slice_offset,
                hidden_size,
                add_inputs=add_inputs,
            )
            slice_offset += hidden_size


def _full_hidden_sizes() -> list[int]:
    base = [
        128,
        256,
        512,
        896,
        1024,
        1152,
        1216,
        1280,
        1536,
        1664,
        2048,
        2240,
        2304,
        2368,
        2432,
        2560,
        2752,
        3072,
        3328,
        3456,
        3584,
        3712,
        4096,
        4480,
        4608,
        4736,
        4864,
        5120,
        5504,
        5632,
        5888,
        6144,
        6400,
        6848,
        6912,
        7168,
        7424,
        8192,
        8960,
        9216,
        9472,
        10240,
        11008,
        11264,
        13824,
        14336,
        14784,
        14848,
        15360,
        18944,
        22016,
        22528,
        24576,
        27392,
        27648,
        29568,
        29696,
        32000,
        32256,
        32512,
        32768,
        33024,
        36864,
        43264,
        49152,
        49408,
        60544,
        60672,
        64000,
        64256,
        102400,
        102656,
        128000,
        128256,
    ]
    divisibility = [1, 2, 8, 16, 64]
    sizes = set()
    for div in divisibility:
        for hidden_size in base:
            if hidden_size // div > 0:
                sizes.add(hidden_size // div)
    return sorted(sizes)


def _get_param_space():
    level = os.getenv("PUNICA_TEST_LEVEL", "default").lower()
    if level == "full":
        hidden_sizes = _full_hidden_sizes()
        test_params = {
            "hidden_sizes": [2049],
            "batches": [1, 4, 16, 32],
            "num_loras": [1, 8, 32, 128],
            "max_ranks": [1, 4, 8, 16, 32, 64, 128, 256],
        }
        hs_test_params = {
            "hidden_sizes": hidden_sizes,
            "batches": [4],
            "num_loras": [4],
            "max_ranks": [32],
        }
    elif level == "quick":
        test_params = {
            "hidden_sizes": [128, 256],
            "batches": [1, 4],
            "num_loras": [1, 4],
            "max_ranks": [4, 8],
        }
        hs_test_params = {
            "hidden_sizes": [128, 256, 512],
            "batches": [2],
            "num_loras": [2],
            "max_ranks": [8],
        }
    else:
        test_params = {
            "hidden_sizes": [1024, 2049],
            "batches": [1, 4, 16],
            "num_loras": [1, 8, 32],
            "max_ranks": [4, 16, 32, 64],
        }
        hs_test_params = {
            "hidden_sizes": [256, 512, 1024, 2048, 4096],
            "batches": [4],
            "num_loras": [4],
            "max_ranks": [32],
        }

    case = parse_case_env()
    if case:
        test_params = {
            "hidden_sizes": [case.get("hidden_size", test_params["hidden_sizes"][0])],
            "batches": [case.get("batches", test_params["batches"][0])],
            "num_loras": [case.get("num_loras", test_params["num_loras"][0])],
            "max_ranks": [case.get("rank", test_params["max_ranks"][0])],
        }
        hs_test_params = test_params
    return test_params, hs_test_params, case


def _get_op_types(case) -> list[str]:
    op_filter = os.getenv("PUNICA_OP")
    if case and case.get("op"):
        return [case["op"]]
    if op_filter:
        return [op_filter]
    return ["shrink", "expand"]


def _get_dtypes(case) -> list[torch.dtype]:
    if case and case.get("dtype"):
        return [case["dtype"]]
    return [torch.float16, torch.bfloat16]


def _get_device(case) -> str:
    if case and case.get("device"):
        return case["device"]
    return os.getenv("PUNICA_DEVICE", "cuda:0")


def check_lora_shrink_kernel(batches: int, num_loras: int, rank: int, hidden_size: int, nslices: int,
                             dtype: torch.dtype, device: str, seq_length: int, scaling: float):
    data: PunicaTensors = generate_data_for_nslices(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        nslices,
        dtype,
        "shrink",
        device,
    )
    max_seq_length, token_nums = data.meta()

    sgmv_meta_args = (data.b_seq_start_loc, data.seq_len_tensor, data.prompt_lora_mapping, batches, max_seq_length,
                      token_nums)

    lora_meta = LoRAKernelMeta.make(max_loras=num_loras, max_num_tokens=token_nums, device=device)
    lora_meta.prepare_tensors(data.token_lora_mapping)

    ref_out_tensor = data.ref_out_tensor
    out_tensor = data.our_out_tensor.clone()

    _LORA_A_PTR_DICT.clear()
    lora_shrink(
        data.inputs_tensor,
        data.lora_weights,
        out_tensor,
        *lora_meta.meta_args(token_nums=token_nums),
        scaling,
    )

    sgmv_shrink_for_nslices(
        nslices,
        data.inputs_tensor,
        data.lora_weights,
        ref_out_tensor,
        *sgmv_meta_args,
        scaling,
    )

    assert_close(out_tensor, ref_out_tensor)


def check_lora_expand_kernel(batches: int, num_loras: int, rank: int, hidden_size: int, nslices: int,
                             dtype: torch.dtype, device: str, seq_length: int, add_inputs: bool):
    data: PunicaTensors = generate_data_for_nslices(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        nslices,
        dtype,
        "expand",
        device,
    )

    max_seq_length, token_nums = data.meta()

    sgmv_meta_args = (data.b_seq_start_loc, data.seq_len_tensor, data.prompt_lora_mapping, batches, max_seq_length,
                      token_nums)

    lora_meta = LoRAKernelMeta.make(max_loras=num_loras, max_num_tokens=token_nums, device=device)
    lora_meta.prepare_tensors(data.token_lora_mapping)

    ref_out_tensor = data.ref_out_tensor
    out_tensor = data.our_out_tensor.clone()

    _LORA_B_PTR_DICT.clear()
    lora_expand(data.inputs_tensor, data.lora_weights, out_tensor, *lora_meta.meta_args(token_nums=token_nums),
                offset_start=0, add_inputs=add_inputs)

    sgmv_expand_for_nslices(nslices, hidden_size, data.inputs_tensor, data.lora_weights, ref_out_tensor,
                            *sgmv_meta_args, add_inputs=add_inputs)

    assert_close(out_tensor, ref_out_tensor)


def _is_nonconsecutive(mapping: list[int]) -> bool:
    positions = {}
    for idx, val in enumerate(mapping):
        positions.setdefault(val, []).append(idx)
    for pos in positions.values():
        if len(pos) >= 2:
            for i in range(1, len(pos)):
                if pos[i] - pos[i - 1] != 1:
                    return True
    return False


def _generate_nonconsecutive_mappings(num_tokens: int, num_loras: int, num_cases: int, seed: int) -> list[list[int]]:
    import random

    rng = random.Random(seed)
    # Base pattern: each lora has at least one token.
    base = [i % num_loras for i in range(num_tokens)]
    mappings = []
    seen = set()
    attempts = 0
    while len(mappings) < num_cases and attempts < num_cases * 50:
        attempts += 1
        cand = base[:]
        rng.shuffle(cand)
        if not _is_nonconsecutive(cand):
            continue
        key = tuple(cand)
        if key in seen:
            continue
        seen.add(key)
        mappings.append(cand)
    if len(mappings) < num_cases:
        raise RuntimeError("Failed to generate enough nonconsecutive mappings.")
    return mappings


def _build_nonconsec_cases(num_cases: int) -> list[dict[str, int]]:
    import random

    rng = random.Random(11)
    token_sizes = [8, 10, 12, 16, 20, 24, 32]
    lora_sizes = [2, 3, 4]
    ranks = [4, 8, 16]
    hidden_sizes = [32, 64, 128]
    cases = []
    seen = set()
    while len(cases) < num_cases:
        num_tokens = rng.choice(token_sizes)
        num_loras = rng.choice(lora_sizes)
        rank = rng.choice(ranks)
        hidden_size = rng.choice(hidden_sizes)
        key = (num_tokens, num_loras, rank, hidden_size)
        if key in seen:
            continue
        seen.add(key)
        cases.append({
            "num_tokens": num_tokens,
            "num_loras": num_loras,
            "rank": rank,
            "hidden_size": hidden_size,
        })
    return cases


NONCONSEC_PARAMS = _build_nonconsec_cases(num_cases=24)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("params", NONCONSEC_PARAMS)
def test_nonconsecutive_mapping_expand(dtype: torch.dtype, params: dict[str, int]):
    seed_everything(0)
    device = _get_device(None)
    num_tokens = params["num_tokens"]
    num_loras = params["num_loras"]
    rank = params["rank"]
    hidden_size = params["hidden_size"]
    nslices = 1
    mapping = _generate_nonconsecutive_mappings(
        num_tokens=num_tokens,
        num_loras=num_loras,
        num_cases=1,
        seed=123 + num_tokens + num_loras,
    )[0]
    token_lora_mapping = torch.tensor(mapping, dtype=torch.int32, device=device)

    inputs = torch.rand((nslices, num_tokens, rank), dtype=dtype, device=device)
    lora_weights = [torch.rand((num_loras, hidden_size, rank), dtype=dtype, device=device)]
    ref_out = torch.zeros((num_tokens, hidden_size), dtype=dtype, device=device)
    out = ref_out.clone()

    lora_meta = LoRAKernelMeta.make(max_loras=num_loras, max_num_tokens=num_tokens, device=device)
    lora_meta.prepare_tensors(token_lora_mapping)

    _LORA_B_PTR_DICT.clear()
    lora_expand(inputs, lora_weights, out, *lora_meta.meta_args(token_nums=num_tokens), offset_start=0,
                add_inputs=False)

    # Reference: directly use per-token mapping.
    from .punica_torch_ops import bgmv_expand
    bgmv_expand(inputs[0], lora_weights[0], ref_out, token_lora_mapping, add_inputs=False)

    assert_close(out, ref_out)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("params", NONCONSEC_PARAMS)
def test_nonconsecutive_mapping_shrink(dtype: torch.dtype, params: dict[str, int]):
    seed_everything(0)
    device = _get_device(None)
    num_tokens = params["num_tokens"]
    num_loras = params["num_loras"]
    rank = params["rank"]
    hidden_size = params["hidden_size"]
    nslices = 1
    scaling = 0.5
    mapping = _generate_nonconsecutive_mappings(
        num_tokens=num_tokens,
        num_loras=num_loras,
        num_cases=1,
        seed=321 + num_tokens + num_loras,
    )[0]
    token_lora_mapping = torch.tensor(mapping, dtype=torch.int32, device=device)

    inputs = torch.rand((num_tokens, hidden_size), dtype=dtype, device=device)
    lora_weights = [torch.rand((num_loras, rank, hidden_size), dtype=dtype, device=device)]
    ref_out = torch.zeros((num_tokens, rank), dtype=torch.float32, device=device)
    out = torch.zeros((nslices, num_tokens, rank), dtype=torch.float32, device=device)

    lora_meta = LoRAKernelMeta.make(max_loras=num_loras, max_num_tokens=num_tokens, device=device)
    lora_meta.prepare_tensors(token_lora_mapping)

    _LORA_A_PTR_DICT.clear()
    lora_shrink(inputs, lora_weights, out, *lora_meta.meta_args(token_nums=num_tokens), scaling)

    from .punica_torch_ops import bgmv_shrink
    bgmv_shrink(inputs, lora_weights[0], ref_out, token_lora_mapping, scaling)

    assert_close(out[0], ref_out)


test_params, hs_test_params, case = _get_param_space()
op_types = _get_op_types(case)
dtypes = _get_dtypes(case)
device = _get_device(case)
seed = int(case.get("seed", 0) if case else 0)
add_inputs = bool(case.get("add_inputs", True) if case else True)
scaling = float(case.get("scaling", 0.5) if case else 0.5)
seq_length = int(case.get("seq_length", 128) if case else 128)


@pytest.mark.parametrize("batches", test_params['batches'])
@pytest.mark.parametrize("num_loras", test_params['num_loras'])
@pytest.mark.parametrize("rank", test_params['max_ranks'])
@pytest.mark.parametrize("hidden_size", test_params['hidden_sizes'])
@pytest.mark.parametrize("nslices", [1, 2, 3])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("op_type", op_types)
def test_kernels(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
    op_type: str,
):
    seed_everything(seed)
    if op_type == "shrink" and dtype == torch.float32:
        pytest.skip("shrink only supports fp16/bf16")

    if op_type == "shrink":
        check_lora_shrink_kernel(batches=batches, num_loras=num_loras, rank=rank, hidden_size=hidden_size,
                                 nslices=nslices, dtype=dtype, device=device, seq_length=seq_length, scaling=scaling)
    else:
        check_lora_expand_kernel(batches=batches, num_loras=num_loras, rank=rank, hidden_size=hidden_size,
                                 nslices=nslices, dtype=dtype, device=device, seq_length=seq_length,
                                 add_inputs=add_inputs)


@pytest.mark.parametrize("batches", hs_test_params['batches'])
@pytest.mark.parametrize("num_loras", hs_test_params['num_loras'])
@pytest.mark.parametrize("rank", hs_test_params['max_ranks'])
@pytest.mark.parametrize("hidden_size", hs_test_params['hidden_sizes'])
@pytest.mark.parametrize("nslices", [1, 2, 3])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("op_type", op_types)
def test_kernels_hidden_size(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
    op_type: str,
):
    seed_everything(seed)
    if op_type == "shrink" and dtype == torch.float32:
        pytest.skip("shrink only supports fp16/bf16")

    if op_type == "shrink":
        check_lora_shrink_kernel(batches=batches, num_loras=num_loras, rank=rank, hidden_size=hidden_size,
                                 nslices=nslices, dtype=dtype, device=device, seq_length=seq_length, scaling=scaling)
    else:
        check_lora_expand_kernel(batches=batches, num_loras=num_loras, rank=rank, hidden_size=hidden_size,
                                 nslices=nslices, dtype=dtype, device=device, seq_length=seq_length,
                                 add_inputs=add_inputs)
