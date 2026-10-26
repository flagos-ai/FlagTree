"""Load the immutable shared Qwen3 FP8 dataset used by CUDA and TLE.

The on-disk tensors deliberately use uint8 views for FP8 payloads.  This keeps
the files readable by both validated Torch environments even if their pickle
support for native float8 tensors differs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

FORMAT_NAME = "megamoe-qwen3-fp8-shared"
FORMAT_VERSION = 2


def _torch_load(path: Path, torch):
    kwargs = {"map_location": "cpu", "weights_only": True}
    try:
        return torch.load(path, mmap=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fp8_from_uint8(bits, torch):
    if bits.dtype != torch.uint8:
        raise TypeError(f"expected uint8 FP8 payload, got {bits.dtype}")
    return bits.contiguous().view(torch.float8_e4m3fn)


def interleave_l1_gate_up_rows(weight, torch, granularity: int = 8):
    """Convert checkpoint [all gate | all up] rows to the SM90 gran-8 layout."""
    if weight.ndim != 3:
        raise ValueError(f"L1 weight must be rank 3, got shape={tuple(weight.shape)}")
    experts, rows, hidden = weight.shape
    if rows % 2:
        raise ValueError(f"L1 gate/up row count must be even, got {rows}")
    half = rows // 2
    if half % granularity:
        raise ValueError(f"L1 rows per gate/up half must be divisible by {granularity}, got {half}")
    gate = weight[:, :half].reshape(experts, half // granularity, granularity, hidden)
    up = weight[:, half:].reshape(experts, half // granularity, granularity, hidden)
    return torch.stack((gate, up), dim=2).reshape(experts, rows, hidden).contiguous()


def _dataset_member(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if path.parent != root:
        raise ValueError(f"dataset file must be directly inside {root}: {name!r}")
    return path


def load_qwen3_rank_data(
    root: str | os.PathLike[str],
    *,
    rank: int,
    num_ranks: int,
    num_tokens: int,
    hidden: int,
    intermediate: int,
    num_experts: int,
    topk: int,
    torch,
) -> dict[str, Any]:
    root_path = Path(root).expanduser().resolve()
    manifest_path = root_path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"shared-data manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    expected_header = {"format": FORMAT_NAME, "format_version": FORMAT_VERSION}
    for key, expected in expected_header.items():
        if manifest.get(key) != expected:
            raise ValueError(f"manifest {key}={manifest.get(key)!r}, expected {expected!r}")

    expected_shape = {
        "num_ranks": num_ranks,
        "hidden_size": hidden,
        "moe_intermediate_size": intermediate,
        "num_experts": num_experts,
        "num_experts_per_tok": topk,
    }
    shape = manifest.get("shape", {})
    mismatches = {
        key: {"dataset": shape.get(key), "runtime": expected}
        for key, expected in expected_shape.items()
        if shape.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"shared-data shape mismatch: {mismatches}")
    dataset_tokens = shape.get("tokens_per_rank")
    if type(dataset_tokens) is not int or dataset_tokens <= 0:
        raise ValueError(f"manifest tokens_per_rank must be a positive integer, got {dataset_tokens!r}")
    if num_tokens <= 0 or num_tokens > dataset_tokens:
        raise ValueError("shared-data token capacity is insufficient: "
                         f"dataset={dataset_tokens}, runtime={num_tokens}")
    if not 0 <= rank < num_ranks:
        raise ValueError(f"rank {rank} is outside [0, {num_ranks})")
    if num_experts % num_ranks:
        raise ValueError(f"num_experts={num_experts} is not divisible by num_ranks={num_ranks}")

    files = manifest.get("files", {})
    shared_entry = files.get("shared", {})
    rank_entry = files.get("ranks", {}).get(str(rank), {})
    shared_path = _dataset_member(root_path, shared_entry.get("name", "shared.pt"))
    rank_path = _dataset_member(root_path, rank_entry.get("name", f"rank{rank:02d}.pt"))
    for path in (shared_path, rank_path):
        if not path.is_file():
            raise FileNotFoundError(f"shared-data tensor file is missing: {path}")

    if os.environ.get("MEGAMOE_DATA_VERIFY_SHA256", "0") == "1":
        for path, entry in ((shared_path, shared_entry), (rank_path, rank_entry)):
            expected = entry.get("sha256")
            actual = _sha256(path)
            if not expected or actual != expected:
                raise ValueError(f"SHA256 mismatch for {path}: {actual} != {expected}")

    shared = _torch_load(shared_path, torch)
    local = _torch_load(rank_path, torch)
    epr = num_experts // num_ranks
    expected_tensors = {
        "hidden_states_bf16": (
            shared,
            torch.bfloat16,
            (num_ranks, dataset_tokens, hidden),
        ),
        "router_logits_fp32": (
            shared,
            torch.float32,
            (num_ranks, dataset_tokens, num_experts),
        ),
        "input_fp8_bits": (shared, torch.uint8, (num_ranks, dataset_tokens, hidden)),
        "input_scales": (
            shared,
            torch.float32,
            (num_ranks, dataset_tokens, hidden // 128),
        ),
        "topk_idx": (shared, torch.int64, (num_ranks, dataset_tokens, topk)),
        "topk_weights": (shared, torch.float32, (num_ranks, dataset_tokens, topk)),
        "l1_weight_fp8_bits": (local, torch.uint8, (epr, 2 * intermediate, hidden)),
        "l1_weight_scale_inv": (
            local,
            torch.float32,
            (epr, 2 * intermediate // 128, hidden // 128),
        ),
        "l2_weight_fp8_bits": (local, torch.uint8, (epr, hidden, intermediate)),
        "l2_weight_scale_inv": (
            local,
            torch.float32,
            (epr, hidden // 128, intermediate // 128),
        ),
    }
    for name, (collection, dtype, tensor_shape) in expected_tensors.items():
        if name not in collection:
            raise KeyError(f"{name} is missing from shared dataset")
        tensor = collection[name]
        if tensor.dtype != dtype or tuple(tensor.shape) != tensor_shape:
            raise ValueError(f"{name}: got dtype={tensor.dtype}, shape={tuple(tensor.shape)}; "
                             f"expected dtype={dtype}, shape={tensor_shape}")

    # One immutable max-token export serves all smaller benchmark shapes.  The
    # same per-rank prefix is selected in every process, so CUDA and TLE still
    # consume byte-identical activations, scales and routes.
    if num_tokens < dataset_tokens:
        shared = {name: tensor[:, :num_tokens] for name, tensor in shared.items()}

    topk_idx = shared["topk_idx"]
    if int(topk_idx.min()) < 0 or int(topk_idx.max()) >= num_experts:
        raise ValueError("real-data routing contains an out-of-range expert id")
    route_sums = shared["topk_weights"].sum(dim=-1)
    # Qwen normalizes the selected probabilities in FP32, then casts the
    # scores back to the BF16 router-logit dtype.  The shared file stores those
    # real, BF16-rounded values as FP32, so their sum includes the accumulated
    # rounding error from all top-k entries.
    route_sum_atol = 4e-3
    if not torch.allclose(
            route_sums,
            torch.ones_like(route_sums),
            atol=route_sum_atol,
            rtol=0.0,
    ):
        max_error = float((route_sums - 1.0).abs().max())
        raise ValueError("Qwen3 top-k routing weights are not normalized within the "
                         f"BF16 rounding tolerance: max_error={max_error}, atol={route_sum_atol}")
    if not torch.isfinite(shared["hidden_states_bf16"].float()).all():
        raise ValueError("real-data hidden states contain non-finite values")
    if not torch.isfinite(shared["router_logits_fp32"]).all():
        raise ValueError("real-data router logits contain non-finite values")

    return {
        "root": root_path,
        "manifest": manifest,
        "shared": shared,
        "local": local,
    }
