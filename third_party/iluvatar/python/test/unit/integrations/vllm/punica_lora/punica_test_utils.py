# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional, Union

import torch


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_close(a: torch.Tensor, b: torch.Tensor) -> None:
    rtol, atol = {
        torch.float16: (6e-2, 6e-2),
        torch.bfloat16: (6e-2, 6e-2),
        torch.float32: (1e-2, 1e-2),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


@dataclass
class PunicaTensors:
    inputs_tensor: torch.Tensor
    lora_weights: Union[torch.Tensor, list[torch.Tensor]]
    our_out_tensor: torch.Tensor
    ref_out_tensor: torch.Tensor
    b_seq_start_loc: torch.Tensor
    prompt_lora_mapping: torch.Tensor
    seq_len_tensor: torch.Tensor
    token_lora_mapping: torch.Tensor

    def meta(self) -> tuple[int, int]:
        max_seq_length = self.seq_len_tensor.max()
        token_nums = self.seq_len_tensor.sum().item()
        if isinstance(max_seq_length, tuple):
            max_seq_length = max_seq_length[0].item()
        else:
            max_seq_length = max_seq_length.item()
        return max_seq_length, token_nums


def generate_data_for_nslices(
    batches,
    hidden_size,
    lora_nums,
    max_rank,
    seq_length,
    nslices,
    dtype,
    op_type,
    device,
) -> PunicaTensors:
    seq_len_tensor = torch.randint(seq_length, seq_length + 1, (batches, )).to(device)
    b_seq_start_loc = torch.cumsum(
        torch.tensor([0] + seq_len_tensor[:-1].tolist(), dtype=torch.long),
        dim=0,
    ).to(device)
    total_tokens = seq_len_tensor.sum()

    lora_weights_lst = []
    if op_type == "shrink":
        inputs_tensor = torch.rand((total_tokens, hidden_size), dtype=dtype).to(device)

        for _ in range(nslices):
            lora_weights_lst.append(torch.rand(
                (lora_nums, max_rank, hidden_size),
                dtype=dtype,
            ).to(device))
        our_out_tensor = torch.zeros(
            (nslices, total_tokens, max_rank),
            dtype=torch.float32,
        ).to(device)
    else:
        inputs_tensor = torch.rand(
            (nslices, total_tokens, max_rank),
            dtype=dtype,
        ).to(device)
        for _ in range(nslices):
            lora_weights_lst.append(torch.rand(
                (lora_nums, hidden_size, max_rank),
                dtype=dtype,
            ).to(device))
        our_out_tensor = torch.rand((total_tokens, hidden_size * nslices), dtype=dtype).to(device)

    ref_out_tensor = our_out_tensor.clone()
    lora_indices_tensor = torch.randint(0, lora_nums - 1 if lora_nums > 1 else 1, (batches, ))
    indices = torch.zeros((total_tokens), dtype=torch.long).to(device)
    current_offset = 0
    for b_id in range(batches):
        lora_index = lora_indices_tensor[b_id]
        indices[current_offset:current_offset + seq_len_tensor[b_id]] = (lora_index.item())
        current_offset += seq_len_tensor[b_id].item()

    lora_indices_tensor = lora_indices_tensor.to(device)
    return PunicaTensors(
        inputs_tensor,
        lora_weights_lst,
        our_out_tensor,
        ref_out_tensor,
        b_seq_start_loc,
        lora_indices_tensor,
        seq_len_tensor,
        indices,
    )


def _dtype_from_str(value: str) -> torch.dtype:
    value = value.lower()
    if value in ("fp16", "float16", "f16"):
        return torch.float16
    if value in ("bf16", "bfloat16"):
        return torch.bfloat16
    if value in ("fp32", "float32", "f32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def parse_case_env() -> Optional[dict[str, object]]:
    raw = os.getenv("PUNICA_CASE")
    if not raw:
        return None
    case = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key in ("batches", "num_loras", "rank", "hidden_size", "nslices", "seq_length", "seed"):
            case[key] = int(value)
        elif key in ("dtype", ):
            case[key] = _dtype_from_str(value)
        elif key in ("device", "op"):
            case[key] = value
        elif key in ("add_inputs", ):
            case[key] = value.lower() in ("1", "true", "yes")
        elif key in ("scaling", ):
            case[key] = float(value)
        else:
            raise ValueError(f"Unknown PUNICA_CASE key: {key}")
    return case
