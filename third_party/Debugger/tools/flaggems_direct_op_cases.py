#!/usr/bin/env python3
"""Run FlagGems debugger checks through direct operator calls.

This runner keeps the existing copied-worktree instrumentation flow, but
replaces pytest marker execution with one subprocess per direct operator case.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from flaggems_debug_batch import (  # noqa: E402
    DEFAULT_FLAGGEMS_ROOT, DEFAULT_PYTHON, DEFAULT_WORKSPACE_ROOT, InstrumentationStats, build_env, classify_status,
    copy_flaggems_source, create_bootstrap, first_error_from_logs, instrument_flaggems_tree, inventory_by_id,
    load_operator_inventory, now_stamp, op_source_candidates, op_uses_pointwise_dynamic, read_text, select_ops,
    shell_command, write_text,
)
from flaggems_pytest_node_runner import (  # noqa: E402
    build_node_plan, collect_marks, run_node as run_pytest_node, write_node_entry,
)


@dataclass(frozen=True)
class DirectCase:
    op: str
    case_id: str
    include_names: list[str]
    body: str
    description: str = ""


@dataclass
class DirectCaseStatus:
    op: str
    case_id: str
    status: str
    exit_code: int | None
    duration_sec: float
    command: str
    script: str | None
    stdout_log: str
    stderr_log: str
    debug_report_dir: str
    debug_txt_count: int
    debug_json_count: int
    first_error: str
    include_names: list[str]
    description: str
    last_report_time: float | None = None
    report_timeout_sec: int | None = None
    first_report_timeout_sec: int | None = None
    timeout_reason: str = ""


def latest_stage(item: dict[str, Any]) -> str | None:
    stages = item.get("stages") or []
    if not stages:
        return None
    last = stages[-1]
    if isinstance(last, dict) and last:
        return next(iter(last.keys()))
    return None


def sanitize_case_id(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def include_names_for(op: str) -> list[str]:
    names = [op]
    base = op.replace(".", "_")
    if base not in names:
        names.append(base)
    if op.endswith("_out"):
        names.append(op[:-len("_out")])
    if op.endswith(".out"):
        names.append(op[:-len(".out")])
    aliases = {
        "all_dims": ["all"],
        "any_dims": ["any"],
        "arange_start": ["arange"],
        "arange_start_step": ["arange"],
        "prod_dim_int": ["prod"],
        "sum_dim": ["sum"],
        "sum_dim_out": ["sum"],
        "mean_dim": ["mean"],
        "max_dim": ["max"],
        "min_dim": ["min"],
        "cat_out": ["cat"],
        "absolute": ["abs"],
        "negative": ["neg"],
        "clip": ["clamp"],
    }
    names.extend(aliases.get(op, []))
    return list(dict.fromkeys(names))


def indent_body(body: str, spaces: int = 4) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line else "" for line in body.strip().splitlines())


def make_case(op: str, case_id: str, body: str, description: str = "") -> DirectCase:
    return DirectCase(
        op=op,
        case_id=sanitize_case_id(case_id),
        include_names=include_names_for(op),
        body=body.strip(),
        description=description,
    )


def addmm_cases(op: str) -> list[DirectCase]:
    cases: list[DirectCase] = []
    cases.append(
        make_case(
            op,
            "bias_vector_contiguous",
            """
M, N, K = 1, 1, 32
mat1 = torch.randn((M, K), dtype=torch.float32, device=device)
mat2 = torch.randn((K, N), dtype=torch.float32, device=device)
bias = torch.randn((N,), dtype=torch.float32, device=device)
result = torch.addmm(bias, mat1, mat2, alpha=1.0, beta=1.0)
""",
            "torch.addmm with vector bias and contiguous mat2",
        ))
    cases.append(
        make_case(
            op,
            "bias_matrix_contiguous",
            """
M, N, K = 1, 1, 32
mat1 = torch.randn((M, K), dtype=torch.float32, device=device)
mat2 = torch.randn((K, N), dtype=torch.float32, device=device)
bias = torch.randn((M, N), dtype=torch.float32, device=device)
result = torch.addmm(bias, mat1, mat2, alpha=1.0, beta=1.0)
""",
            "torch.addmm with matrix bias and contiguous mat2",
        ))
    cases.append(
        make_case(
            op,
            "bias_vector_column_major",
            """
M, N, K = 1, 1, 32
mat1 = torch.randn((M, K), dtype=torch.float32, device=device)
mat2 = torch.randn((N, K), dtype=torch.float32, device=device).t()
bias = torch.randn((N,), dtype=torch.float32, device=device)
result = torch.addmm(bias, mat1, mat2, alpha=1.0, beta=1.0)
""",
            "torch.addmm with vector bias and transposed mat2",
        ))
    cases.append(
        make_case(
            op,
            "bias_matrix_column_major",
            """
M, N, K = 1, 1, 32
mat1 = torch.randn((M, K), dtype=torch.float32, device=device)
mat2 = torch.randn((N, K), dtype=torch.float32, device=device).t()
bias = torch.randn((M, N), dtype=torch.float32, device=device)
result = torch.addmm(bias, mat1, mat2, alpha=1.0, beta=1.0)
""",
            "torch.addmm with matrix bias and transposed mat2",
        ))
    return cases


def linear_algebra_cases(op: str) -> list[DirectCase]:
    if op == "addmm":
        return addmm_cases(op)
    if op == "addmm_out":
        return [
            make_case(
                op,
                "out_contiguous",
                """
M, N, K = 1, 1, 32
mat1 = torch.randn((M, K), dtype=torch.float32, device=device)
mat2 = torch.randn((K, N), dtype=torch.float32, device=device)
bias = torch.randn((M, N), dtype=torch.float32, device=device)
out = torch.empty((M, N), dtype=torch.float32, device=device)
torch.addmm(bias, mat1, mat2, alpha=1.0, beta=1.0, out=out)
result = out
""",
                "torch.addmm out variant",
            )
        ]
    if op == "mm":
        return [
            make_case(
                op,
                "m8_n8_k16",
                """
a = torch.randn((8, 16), dtype=torch.float32, device=device)
b = torch.randn((16, 8), dtype=torch.float32, device=device)
result = torch.mm(a, b)
""",
                "torch.mm small fp32 matrix multiply",
            )
        ]
    if op == "mm_out":
        return [
            make_case(
                op,
                "m8_n8_k16",
                """
a = torch.randn((8, 16), dtype=torch.float32, device=device)
b = torch.randn((16, 8), dtype=torch.float32, device=device)
out = torch.empty((8, 8), dtype=torch.float32, device=device)
torch.mm(a, b, out=out)
result = out
""",
                "torch.mm out variant",
            )
        ]
    if op == "bmm":
        return [
            make_case(
                op,
                "b2_m8_n8_k16",
                """
a = torch.randn((2, 8, 16), dtype=torch.float32, device=device)
b = torch.randn((2, 16, 8), dtype=torch.float32, device=device)
result = torch.bmm(a, b)
""",
                "torch.bmm small fp32 batched matmul",
            )
        ]
    if op == "bmm_out":
        return [
            make_case(
                op,
                "b2_m8_n8_k16",
                """
a = torch.randn((2, 8, 16), dtype=torch.float32, device=device)
b = torch.randn((2, 16, 8), dtype=torch.float32, device=device)
out = torch.empty((2, 8, 8), dtype=torch.float32, device=device)
torch.bmm(a, b, out=out)
result = out
""",
                "torch.bmm out variant",
            )
        ]
    if op == "addmv":
        return [
            make_case(
                op,
                "m8_n16",
                """
bias = torch.randn((8,), dtype=torch.float32, device=device)
mat = torch.randn((8, 16), dtype=torch.float32, device=device)
vec = torch.randn((16,), dtype=torch.float32, device=device)
result = torch.addmv(bias, mat, vec, alpha=1.0, beta=1.0)
""",
                "torch.addmv small fp32",
            )
        ]
    if op == "addmv_out":
        return [
            make_case(
                op,
                "m8_n16",
                """
bias = torch.randn((8,), dtype=torch.float32, device=device)
mat = torch.randn((8, 16), dtype=torch.float32, device=device)
vec = torch.randn((16,), dtype=torch.float32, device=device)
out = torch.empty((8,), dtype=torch.float32, device=device)
torch.addmv(bias, mat, vec, alpha=1.0, beta=1.0, out=out)
result = out
""",
                "torch.addmv out variant",
            )
        ]
    if op == "mv":
        return [
            make_case(
                op,
                "m8_n16",
                """
mat = torch.randn((8, 16), dtype=torch.float32, device=device)
vec = torch.randn((16,), dtype=torch.float32, device=device)
result = torch.mv(mat, vec)
""",
                "torch.mv small fp32",
            )
        ]
    if op == "dot":
        return [
            make_case(
                op,
                "n32",
                """
a = torch.randn((32,), dtype=torch.float32, device=device)
b = torch.randn((32,), dtype=torch.float32, device=device)
result = torch.dot(a, b)
""",
                "torch.dot fp32 vector dot",
            )
        ]
    if op == "baddbmm":
        return [
            make_case(
                op,
                "b2_m8_n8_k16",
                """
bias = torch.randn((2, 8, 8), dtype=torch.float32, device=device)
a = torch.randn((2, 8, 16), dtype=torch.float32, device=device)
b = torch.randn((2, 16, 8), dtype=torch.float32, device=device)
result = torch.baddbmm(bias, a, b, alpha=1.0, beta=1.0)
""",
                "torch.baddbmm small fp32",
            )
        ]
    if op == "addr":
        return [
            make_case(
                op,
                "m4_n4",
                """
bias = torch.randn((4, 4), dtype=torch.float32, device=device)
x = torch.randn((4,), dtype=torch.float32, device=device)
y = torch.randn((4,), dtype=torch.float32, device=device)
result = torch.addr(bias, x, y, alpha=1.0, beta=1.0)
""",
                "torch.addr small fp32",
            )
        ]
    return []


def reduction_cases(op: str) -> list[DirectCase]:
    x_float = "x = torch.linspace(-8, 7, 16, dtype=torch.float32, device=device).reshape(4, 4)"
    x_bool = "x = (torch.arange(16, device=device).reshape(4, 4) % 3) == 0"
    if op == "all":
        return [make_case(op, "all", f"{x_bool}\nresult = torch.all(x)", "torch.all full reduction")]
    if op == "all_dim":
        return [make_case(op, "dim1", f"{x_bool}\nresult = torch.all(x, dim=1)", "torch.all dim=1")]
    if op == "all_dims":
        return [
            make_case(op, "dims_keepdim", f"{x_bool}\nresult = torch.all(x, dim=1, keepdim=True)",
                      "torch.all dims-compatible direct case")
        ]
    if op == "any":
        return [make_case(op, "any", f"{x_bool}\nresult = torch.any(x)", "torch.any full reduction")]
    if op == "any_dim":
        return [make_case(op, "dim1", f"{x_bool}\nresult = torch.any(x, dim=1)", "torch.any dim=1")]
    if op == "any_dims":
        return [
            make_case(op, "dims_keepdim", f"{x_bool}\nresult = torch.any(x, dim=1, keepdim=True)",
                      "torch.any dims-compatible direct case")
        ]
    if op == "amax":
        return [make_case(op, "dim1", f"{x_float}\nresult = torch.amax(x, dim=1)", "torch.amax dim=1")]
    if op == "aminmax":
        return [make_case(op, "dim1", f"{x_float}\nresult = torch.aminmax(x, dim=1).min", "torch.aminmax dim=1")]
    if op == "max":
        return [make_case(op, "full", f"{x_float}\nresult = torch.max(x)", "torch.max full reduction")]
    if op == "max_dim":
        return [make_case(op, "dim1", f"{x_float}\nresult = torch.max(x, dim=1).values", "torch.max dim=1")]
    if op == "min":
        return [make_case(op, "full", f"{x_float}\nresult = torch.min(x)", "torch.min full reduction")]
    if op == "min_dim":
        return [make_case(op, "dim1", f"{x_float}\nresult = torch.min(x, dim=1).values", "torch.min dim=1")]
    if op == "argmax":
        return [make_case(op, "dim1", f"{x_float}\nresult = torch.argmax(x, dim=1)", "torch.argmax dim=1")]
    if op == "argmin":
        return [make_case(op, "dim1", f"{x_float}\nresult = torch.argmin(x, dim=1)", "torch.argmin dim=1")]
    if op == "sum":
        return [make_case(op, "full", f"{x_float}\nresult = torch.sum(x)", "torch.sum full reduction")]
    if op == "sum_dim":
        return [make_case(op, "dim1", f"{x_float}\nresult = torch.sum(x, dim=1)", "torch.sum dim=1")]
    if op == "sum_out":
        return [
            make_case(
                op,
                "full_out",
                f"{x_float}\nout = torch.empty((), dtype=torch.float32, device=device)\ntorch.sum(x, dim=(0, 1), out=out)\nresult = out",
                "torch.sum out variant",
            )
        ]
    if op == "mean":
        return [make_case(op, "full", f"{x_float}\nresult = torch.mean(x)", "torch.mean full reduction")]
    if op == "mean_dim":
        return [make_case(op, "dim1", f"{x_float}\nresult = torch.mean(x, dim=1)", "torch.mean dim=1")]
    if op == "prod":
        return [make_case(op, "full", f"{x_float}\nresult = torch.prod(x + 9)", "torch.prod full reduction")]
    if op == "prod_dim_int":
        return [make_case(op, "dim1", f"{x_float}\nresult = torch.prod(x + 9, dim=1)", "torch.prod dim=1")]
    if op == "cumsum":
        return [make_case(op, "dim1", f"{x_float}\nresult = torch.cumsum(x, dim=1)", "torch.cumsum dim=1")]
    if op == "cumsum_out":
        return [
            make_case(
                op,
                "dim1_out",
                f"{x_float}\nout = torch.empty_like(x)\ntorch.cumsum(x, dim=1, out=out)\nresult = out",
                "torch.cumsum out variant",
            )
        ]
    return []


def pointwise_cases(op: str) -> list[DirectCase]:
    x = "x = torch.linspace(-3, 3, 16, dtype=torch.float32, device=device)"
    y = "y = torch.linspace(1, 4, 16, dtype=torch.float32, device=device)"
    bool_x = "x = (torch.arange(16, device=device) % 3) == 0"

    unary_torch = {
        "abs": "abs",
        "absolute": "abs",
        "acos": "acos",
        "angle": "angle",
        "arcsinh": "asinh",
        "arctanh": "atanh",
        "asinh": "asinh",
        "atan": "atan",
        "atanh": "atanh",
        "ceil": "ceil",
        "cos": "cos",
        "cosh": "cosh",
        "erf": "erf",
        "exp": "exp",
        "exp2": "exp2",
        "expm1": "expm1",
        "floor": "floor",
        "i0": "i0",
        "log": "log",
        "log10": "log10",
        "log1p": "log1p",
        "neg": "neg",
        "negative": "neg",
        "reciprocal": "reciprocal",
        "round": "round",
        "rsqrt": "rsqrt",
        "sigmoid": "sigmoid",
        "sin": "sin",
        "sqrt": "sqrt",
        "square": "square",
        "tan": "tan",
        "tanh": "tanh",
        "trunc": "trunc",
    }
    if op in unary_torch:
        func = unary_torch[op]
        return [make_case(
            op,
            "shape16",
            f"{x}\nresult = torch.{func}(x)",
            f"torch.{func} pointwise shape=(16,)",
        )]

    inplace_methods = {
        "abs_": "abs_",
        "arcsinh_": "asinh_",
        "arctanh_": "atanh_",
        "asinh_": "asinh_",
        "atan_": "atan_",
        "atanh_": "atanh_",
        "ceil_": "ceil_",
        "cos_": "cos_",
        "cosh_": "cosh_",
        "erf_": "erf_",
        "exp_": "exp_",
        "exp2_": "exp2_",
        "expm1_": "expm1_",
        "floor_": "floor_",
        "neg_": "neg_",
        "reciprocal_": "reciprocal_",
        "relu_": "relu_",
        "round_": "round_",
        "rsqrt_": "rsqrt_",
        "sigmoid_": "sigmoid_",
        "sin_": "sin_",
        "sqrt_": "sqrt_",
        "square_": "square_",
        "tan_": "tan_",
        "tanh_": "tanh_",
        "trunc_": "trunc_",
    }
    if op in inplace_methods:
        method = inplace_methods[op]
        return [
            make_case(
                op,
                "shape16_inplace",
                f"{x}\nx.{method}()\nresult = x",
                f"Tensor.{method} pointwise shape=(16,)",
            )
        ]

    unary_out = {
        "arcsinh_out": "asinh",
        "exp_out": "exp",
        "expm1_out": "expm1",
        "floor_out": "floor",
        "log10_out": "log10",
        "round_out": "round",
        "square_out": "square",
    }
    if op in unary_out:
        func = unary_out[op]
        return [
            make_case(
                op,
                "shape16_out",
                f"{x}\nout = torch.empty_like(x)\ntorch.{func}(x, out=out)\nresult = out",
                f"torch.{func} out pointwise shape=(16,)",
            )
        ]

    if op in {"add", "sub", "mul", "div_tensor", "pow_tensor_tensor"}:
        func = {
            "div_tensor": "div",
            "pow_tensor_tensor": "pow",
        }.get(op, op)
        return [
            make_case(
                op,
                "tensor_tensor",
                f"{x}\n{y}\nresult = torch.{func}(x, y)",
                f"torch.{func} tensor-tensor pointwise",
            )
        ]
    if op in {"addcdiv", "addcmul"}:
        return [
            make_case(
                op,
                "shape16",
                f"{x}\n{y}\nz = torch.linspace(2, 5, 16, dtype=torch.float32, device=device)\nresult = torch.{op}(x, y, z, value=0.5)",
                f"torch.{op} pointwise",
            )
        ]
    if op in {"addcdiv_", "addcmul_"}:
        method = op
        return [
            make_case(
                op,
                "shape16_inplace",
                f"{x}\n{y}\nz = torch.linspace(2, 5, 16, dtype=torch.float32, device=device)\nx.{method}(y, z, value=0.5)\nresult = x",
                f"Tensor.{method} pointwise",
            )
        ]
    if op in {"addcdiv_out", "addcmul_out"}:
        func = op[:-len("_out")]
        return [
            make_case(
                op,
                "shape16_out",
                f"{x}\n{y}\nz = torch.linspace(2, 5, 16, dtype=torch.float32, device=device)\nout = torch.empty_like(x)\ntorch.{func}(x, y, z, value=0.5, out=out)\nresult = out",
                f"torch.{func} out pointwise",
            )
        ]
    if op in {"add_", "sub_", "mul_", "div_tensor_", "pow_tensor_tensor_"}:
        method = {
            "add_": "add_",
            "sub_": "sub_",
            "mul_": "mul_",
            "div_tensor_": "div_",
            "pow_tensor_tensor_": "pow_",
        }[op]
        return [
            make_case(
                op,
                "tensor_tensor_inplace",
                f"{x}\n{y}\nx.{method}(y)\nresult = x",
                f"Tensor.{method} tensor-tensor pointwise",
            )
        ]
    if op in {"eq", "ne", "gt", "ge", "lt", "le"}:
        return [
            make_case(
                op,
                "tensor_tensor",
                f"{x}\n{y}\nresult = torch.{op}(x, y)",
                f"torch.{op} tensor-tensor compare",
            )
        ]
    if op in {"relu", "gelu", "silu"}:
        return [
            make_case(
                op,
                "shape16",
                f"{x}\nresult = torch.nn.functional.{op}(x)",
                f"torch.nn.functional.{op} pointwise",
            )
        ]
    if op in {"clamp", "clamp_tensor", "clip"}:
        return [make_case(
            op,
            "min_max",
            f"{x}\nresult = torch.clamp(x, min=-1.0, max=1.0)",
            "torch.clamp min/max",
        )]
    if op in {"clamp_", "clamp_tensor_", "clip_"}:
        return [
            make_case(
                op,
                "min_max_inplace",
                f"{x}\nx.clamp_(min=-1.0, max=1.0)\nresult = x",
                "Tensor.clamp_ min/max",
            )
        ]
    if op in {"where_self"}:
        return [
            make_case(
                op,
                "mask_tensor_tensor",
                f"{bool_x}\na = torch.ones((16,), dtype=torch.float32, device=device)\nb = torch.zeros((16,), dtype=torch.float32, device=device)\nresult = torch.where(x, a, b)",
                "torch.where mask tensor tensor",
            )
        ]
    if op == "allclose":
        return [
            make_case(
                op,
                "shape16",
                f"{x}\nresult = torch.isclose(x, x + 0.001)",
                "torch.isclose proxy for allclose elementwise path",
            )
        ]
    if op in {"copy", "copy_"}:
        return [make_case(
            op,
            "shape16",
            f"{x}\n{y}\nx.copy_(y)\nresult = x",
            "Tensor.copy_ pointwise copy",
        )]
    if op in {"isfinite", "isinf", "isnan", "logical_not"}:
        func = op
        src = bool_x if op == "logical_not" else x
        return [make_case(
            op,
            "shape16",
            f"{src}\nresult = torch.{func}(x)",
            f"torch.{func} pointwise",
        )]
    return []


def creation_and_shape_cases(op: str) -> list[DirectCase]:
    if op == "arange":
        return [
            make_case(
                op,
                "end16",
                "result = torch.arange(16, dtype=torch.float32, device=device)",
                "torch.arange end-only",
            )
        ]
    if op == "arange_start":
        return [
            make_case(
                op,
                "start_end",
                "result = torch.arange(2, 18, dtype=torch.float32, device=device)",
                "torch.arange start/end",
            )
        ]
    if op == "arange_start_step":
        return [
            make_case(
                op,
                "start_end_step",
                "result = torch.arange(2, 34, 2, dtype=torch.float32, device=device)",
                "torch.arange start/end/step",
            )
        ]
    if op == "zeros":
        return [
            make_case(
                op,
                "shape16",
                "result = torch.zeros((16,), dtype=torch.float32, device=device)",
                "torch.zeros shape=(16,)",
            )
        ]
    if op == "zeros_like":
        return [
            make_case(
                op,
                "shape16",
                "x = torch.randn((16,), dtype=torch.float32, device=device)\nresult = torch.zeros_like(x)",
                "torch.zeros_like",
            )
        ]
    if op == "eye":
        return [make_case(
            op,
            "n16",
            "result = torch.eye(16, dtype=torch.float32, device=device)",
            "torch.eye n=16",
        )]
    if op == "eye_m":
        return [
            make_case(
                op,
                "n16_m8",
                "result = torch.eye(16, 8, dtype=torch.float32, device=device)",
                "torch.eye n=16 m=8",
            )
        ]
    if op == "cat":
        return [
            make_case(
                op,
                "dim0",
                """
a = torch.randn((2, 4), dtype=torch.float32, device=device)
b = torch.randn((2, 4), dtype=torch.float32, device=device)
result = torch.cat([a, b], dim=0)
""",
                "torch.cat dim=0",
            )
        ]
    if op == "cat_out":
        return [
            make_case(
                op,
                "dim0_out",
                """
a = torch.randn((2, 4), dtype=torch.float32, device=device)
b = torch.randn((2, 4), dtype=torch.float32, device=device)
out = torch.empty((4, 4), dtype=torch.float32, device=device)
torch.cat([a, b], dim=0, out=out)
result = out
""",
                "torch.cat out variant",
            )
        ]
    if op in {"concatenate", "vstack", "hstack", "stack"}:
        fn = "torch.concatenate" if op == "concatenate" else f"torch.{op}"
        dim_arg = ", dim=0" if op in {"concatenate", "stack"} else ""
        return [
            make_case(
                op,
                "two_tensors",
                f"""
a = torch.randn((2, 4), dtype=torch.float32, device=device)
b = torch.randn((2, 4), dtype=torch.float32, device=device)
result = {fn}([a, b]{dim_arg})
""",
                f"{fn} two tensors",
            )
        ]
    return []


def normalization_cases(op: str) -> list[DirectCase]:
    if op == "vector_norm":
        return [
            make_case(
                op,
                "dim1_l2",
                """
x = torch.randn((4, 8), dtype=torch.float32, device=device)
result = torch.linalg.vector_norm(x, ord=2, dim=1)
""",
                "torch.linalg.vector_norm dim=1",
            )
        ]
    if op == "rms_norm":
        return [
            make_case(
                op,
                "n2_h16",
                """
x = torch.randn((2, 16), dtype=torch.float32, device=device)
weight = torch.randn((16,), dtype=torch.float32, device=device)
result = torch.nn.functional.rms_norm(x, (16,), weight=weight, eps=1e-5)
""",
                "torch.nn.functional.rms_norm",
            )
        ]
    if op == "layer_norm":
        return [
            make_case(
                op,
                "n2_h16",
                """
x = torch.randn((2, 16), dtype=torch.float32, device=device)
weight = torch.randn((16,), dtype=torch.float32, device=device)
bias = torch.randn((16,), dtype=torch.float32, device=device)
result = torch.nn.functional.layer_norm(x, (16,), weight=weight, bias=bias, eps=1e-5)
""",
                "torch.nn.functional.layer_norm",
            )
        ]
    if op == "group_norm":
        return [
            make_case(
                op,
                "n2_c4_hw2",
                """
x = torch.randn((2, 4, 2, 2), dtype=torch.float32, device=device)
weight = torch.randn((4,), dtype=torch.float32, device=device)
bias = torch.randn((4,), dtype=torch.float32, device=device)
result = torch.nn.functional.group_norm(x, 2, weight=weight, bias=bias, eps=1e-5)
""",
                "torch.nn.functional.group_norm",
            )
        ]
    return []


def pooling_and_conv_cases(op: str) -> list[DirectCase]:
    if op == "avg_pool2d":
        return [
            make_case(
                op,
                "n1_c1_hw4",
                """
x = torch.randn((1, 1, 4, 4), dtype=torch.float32, device=device)
result = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
""",
                "torch.nn.functional.avg_pool2d",
            )
        ]
    if op == "adaptive_avg_pool2d":
        return [
            make_case(
                op,
                "n1_c1_hw4",
                """
x = torch.randn((1, 1, 4, 4), dtype=torch.float32, device=device)
result = torch.nn.functional.adaptive_avg_pool2d(x, (2, 2))
""",
                "torch.nn.functional.adaptive_avg_pool2d",
            )
        ]
    if op == "adaptive_avg_pool3d":
        return [
            make_case(
                op,
                "n1_c1_dhw4",
                """
x = torch.randn((1, 1, 4, 4, 4), dtype=torch.float32, device=device)
result = torch.nn.functional.adaptive_avg_pool3d(x, (2, 2, 2))
""",
                "torch.nn.functional.adaptive_avg_pool3d",
            )
        ]
    if op == "adaptive_avg_pool3d_out":
        return [
            make_case(
                op,
                "n1_c1_dhw4_out",
                """
x = torch.randn((1, 1, 4, 4, 4), dtype=torch.float32, device=device)
out = torch.empty((1, 1, 2, 2, 2), dtype=torch.float32, device=device)
torch.ops.aten.adaptive_avg_pool3d.out(x, [2, 2, 2], out=out)
result = out
""",
                "torch.nn.functional.adaptive_avg_pool3d out",
            )
        ]
    if op == "avg_pool3d":
        return [
            make_case(
                op,
                "n1_c1_dhw4",
                """
x = torch.randn((1, 1, 4, 4, 4), dtype=torch.float32, device=device)
result = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
""",
                "torch.nn.functional.avg_pool3d",
            )
        ]
    if op == "max_pool2d_with_indices":
        return [
            make_case(
                op,
                "n1_c1_hw4",
                """
x = torch.randn((1, 1, 4, 4), dtype=torch.float32, device=device)
result = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)[0]
""",
                "torch.nn.functional.max_pool2d return_indices",
            )
        ]
    if op == "conv1d":
        return [
            make_case(
                op,
                "n1_c1_l8",
                """
x = torch.randn((1, 1, 8), dtype=torch.float32, device=device)
weight = torch.randn((1, 1, 3), dtype=torch.float32, device=device)
result = torch.nn.functional.conv1d(x, weight, padding=1)
""",
                "torch.nn.functional.conv1d",
            )
        ]
    if op == "conv2d":
        return [
            make_case(
                op,
                "n1_c1_hw4",
                """
x = torch.randn((1, 1, 4, 4), dtype=torch.float32, device=device)
weight = torch.randn((1, 1, 3, 3), dtype=torch.float32, device=device)
result = torch.nn.functional.conv2d(x, weight, padding=1)
""",
                "torch.nn.functional.conv2d",
            )
        ]
    return []


def indexing_cases(op: str) -> list[DirectCase]:
    if op == "argsort":
        return [
            make_case(
                op,
                "dim1",
                """
x = torch.randn((4, 8), dtype=torch.float32, device=device)
result = torch.argsort(x, dim=1)
""",
                "torch.argsort dim=1",
            )
        ]
    if op == "topk":
        return [
            make_case(
                op,
                "dim1_k2",
                """
x = torch.randn((4, 8), dtype=torch.float32, device=device)
result = torch.topk(x, k=2, dim=1).values
""",
                "torch.topk dim=1 k=2",
            )
        ]
    if op == "index_select":
        return [
            make_case(
                op,
                "dim0",
                """
x = torch.randn((4, 8), dtype=torch.float32, device=device)
idx = torch.tensor([0, 2], dtype=torch.int64, device=device)
result = torch.index_select(x, 0, idx)
""",
                "torch.index_select dim=0",
            )
        ]
    if op == "gather":
        return [
            make_case(
                op,
                "dim1",
                """
x = torch.randn((4, 8), dtype=torch.float32, device=device)
idx = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.int64, device=device)
result = torch.gather(x, 1, idx)
""",
                "torch.gather dim=1",
            )
        ]
    if op == "embedding":
        return [
            make_case(
                op,
                "small",
                """
weight = torch.randn((8, 4), dtype=torch.float32, device=device)
idx = torch.tensor([0, 2, 4, 6], dtype=torch.int64, device=device)
result = torch.nn.functional.embedding(idx, weight)
""",
                "torch.nn.functional.embedding",
            )
        ]
    if op in {"masked_fill", "masked_fill_scalar"}:
        return [
            make_case(
                op,
                "mask_scalar",
                """
x = torch.randn((4, 4), dtype=torch.float32, device=device)
mask = x > 0
result = torch.masked_fill(x, mask, 0.0)
""",
                "torch.masked_fill scalar",
            )
        ]
    if op == "nonzero":
        return [
            make_case(
                op,
                "small",
                """
x = torch.tensor([0, 1, 0, 2], dtype=torch.float32, device=device)
result = torch.nonzero(x)
""",
                "torch.nonzero small tensor",
            )
        ]
    return []


def softmax_cases(op: str) -> list[DirectCase]:
    if op == "softmax":
        return [
            make_case(
                op,
                "dim1",
                """
x = torch.randn((4, 8), dtype=torch.float32, device=device)
result = torch.nn.functional.softmax(x, dim=1)
""",
                "torch.nn.functional.softmax dim=1",
            )
        ]
    if op == "log_softmax":
        return [
            make_case(
                op,
                "dim1",
                """
x = torch.randn((4, 8), dtype=torch.float32, device=device)
result = torch.nn.functional.log_softmax(x, dim=1)
""",
                "torch.nn.functional.log_softmax dim=1",
            )
        ]
    return []


def generate_cases_for_op(op: str, item: dict[str, Any]) -> list[DirectCase]:
    for generator in (
            linear_algebra_cases,
            reduction_cases,
            pointwise_cases,
            creation_and_shape_cases,
            normalization_cases,
            pooling_and_conv_cases,
            indexing_cases,
            softmax_cases,
    ):
        cases = generator(op)
        if cases:
            return cases
    return []


def filter_cases(cases: list[DirectCase], case_filter: str | None) -> list[DirectCase]:
    if not case_filter:
        return cases
    return [case for case in cases if case_filter in case.case_id or case_filter in case.description]


CASE_TEMPLATE = """\
import contextlib
import json
import os
import platform
import sys

platform.python_implementation = lambda: "CPython"
platform.python_version = lambda: "3.11.15"
platform.python_version_tuple = lambda: ("3", "11", "15")

import torch

try:
    import torch_npu
except Exception:
    torch_npu = None

import triton
from triton.runtime import debugger
import flag_gems


def sync_device():
    if torch_npu is not None and hasattr(torch_npu, "npu"):
        torch_npu.npu.synchronize()
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize(value):
    if isinstance(value, torch.Tensor):
        return {{
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }}
    if isinstance(value, (tuple, list)):
        return [summarize(item) for item in value]
    return {{"kind": type(value).__name__, "repr": repr(value)[:200]}}


torch.manual_seed(0)
if torch_npu is not None and hasattr(torch_npu, "npu"):
    torch_npu.npu.manual_seed_all(0)

output_dir = os.environ.get("FLAGTREE_DEBUGGER_BATCH_OUTPUT_DIR")
if output_dir:
    debugger.configure(
        output_dir=output_dir,
        record_capacity=int(os.environ.get("FLAGTREE_DEBUGGER_BATCH_RECORD_CAPACITY", "4096")),
        export_raw_records=os.environ.get("FLAGTREE_DEBUGGER_BATCH_EXPORT_RAW", "0") == "1",
    )

triton.enable_debug(
    level=int(os.environ.get("FLAGTREE_DEBUGGER_BATCH_LEVEL", "1")),
    addr_level=int(os.environ.get("FLAGTREE_DEBUGGER_BATCH_ADDR_LEVEL", "1")),
)

device = flag_gems.device
include_names = {include_names!r}
manager = (
    flag_gems.use_gems(include=include_names)
    if include_names
    else contextlib.nullcontext()
)

with manager:
{body}

sync_device()
print(json.dumps({{
    "op": {op!r},
    "case_id": {case_id!r},
    "include_names": include_names,
    "result": summarize(result),
}}, sort_keys=True))
"""


def write_case_script(path: Path, case: DirectCase) -> None:
    body = indent_body(case.body, 4)
    write_text(
        path,
        CASE_TEMPLATE.format(
            op=case.op,
            case_id=case.case_id,
            include_names=case.include_names,
            body=body,
        ),
    )


def debug_report_snapshot(debug_dir: Path) -> tuple[int, int, float | None]:
    txt_files = list(debug_dir.glob("*.txt"))
    json_files = list(debug_dir.glob("*.json"))
    mtimes = [path.stat().st_mtime for path in txt_files + json_files if path.exists()]
    return len(txt_files), len(json_files), max(mtimes) if mtimes else None


def complete_report_count(debug_dir: Path) -> int:
    txt_stems = {path.stem for path in debug_dir.glob("*.txt")}
    json_stems = {path.stem for path in debug_dir.glob("*.json")}
    return len(txt_stems & json_stems)


def kill_process_group(proc: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def op_has_debuggable_kernel(worktree: Path, op: str) -> bool:
    search_roots = [
        worktree / "src" / "flag_gems" / "runtime" / "backend" / "_ascend" / "ops",
        worktree / "src" / "flag_gems" / "ops",
        worktree / "src" / "flag_gems" / "experimental_ops",
        worktree / "src" / "flag_gems" / "fused",
    ]
    for root in search_roots:
        for candidate in op_source_candidates(op):
            path = root / f"{candidate}.py"
            if not path.exists():
                continue
            text = read_text(path)
            if "triton.jit" in text or "pointwise_dynamic" in text:
                return True
    return False


def make_pytest_fallback_args(args: argparse.Namespace) -> argparse.Namespace:
    total_timeout = args.case_timeout if args.case_timeout is not None else args.case_total_timeout
    return SimpleNamespace(
        one_node_per_op=True,
        max_nodes_per_op=1,
        max_nodes=None,
        first_report_timeout=args.first_report_timeout,
        report_timeout=args.report_timeout,
        node_total_timeout=total_timeout,
        medium_first_report_timeout=max(args.first_report_timeout, 300),
        medium_report_timeout=max(args.report_timeout, 180),
        medium_node_total_timeout=max(total_timeout, 1200),
        heavy_first_report_timeout=max(args.first_report_timeout, 480),
        heavy_report_timeout=max(args.report_timeout, 240),
        heavy_node_total_timeout=max(total_timeout, 1800),
        level=args.level,
        addr_level=args.addr_level,
        record_capacity=args.record_capacity,
        poll_interval=args.pytest_fallback_poll_interval,
    )


def collect_pytest_fallback_nodes(
    worktree: Path,
    run_dir: Path,
    ops: list[str],
    args: argparse.Namespace,
) -> dict[str, list[Any]]:
    if not args.enable_pytest_fallback:
        return {}

    fallback_root = run_dir / "pytest_fallback"
    fallback_args = make_pytest_fallback_args(args)
    try:
        items = collect_marks(
            worktree,
            args.python,
            fallback_root / "collect",
            [],
        )
        write_text(
            fallback_root / "collect_items.json",
            json.dumps(items, indent=2, sort_keys=True),
        )
        nodes = build_node_plan(items, ops, fallback_args)
        write_text(
            fallback_root / "collected_nodes.json",
            json.dumps(
                [{
                    "nodeid": node.nodeid,
                    "selected_ops": node.selected_ops,
                    "timeout_class": node.timeout_class,
                    "first_report_timeout_sec": node.first_report_timeout_sec,
                    "report_timeout_sec": node.report_timeout_sec,
                    "node_total_timeout_sec": node.node_total_timeout_sec,
                } for node in nodes],
                indent=2,
                sort_keys=True,
            ),
        )
    except Exception as exc:
        write_text(
            fallback_root / "collect_error.txt",
            f"{type(exc).__name__}: {exc}\n",
        )
        return {}

    by_op: dict[str, list[Any]] = {}
    for node in nodes:
        for op in node.selected_ops:
            by_op.setdefault(op, []).append(node)
    return by_op


def run_pytest_fallback_case(
    op: str,
    node: Any,
    worktree: Path,
    run_dir: Path,
    args: argparse.Namespace,
) -> DirectCaseStatus:
    fallback_root = run_dir / "pytest_fallback"
    entry_py = fallback_root / "node_entry.py"
    fallback_root.mkdir(parents=True, exist_ok=True)
    if not entry_py.exists():
        write_node_entry(entry_py)

    fallback_args = make_pytest_fallback_args(args)
    result = run_pytest_node(
        node,
        worktree,
        args.python,
        fallback_root,
        entry_py,
        fallback_args,
    )
    case_id = sanitize_case_id(f"pytest_{result.nodeid}")[:160]
    first_error = result.first_error
    if result.status == "missing_debug_report":
        first_error = ("pytest fallback exited successfully but debugger report is missing "
                       f"or incomplete: txt={result.debug_txt_count}, json={result.debug_json_count}, "
                       f"dir={result.debug_report_dir}")
    return DirectCaseStatus(
        op=op,
        case_id=case_id,
        status=result.status,
        exit_code=result.exit_code,
        duration_sec=result.duration_sec,
        command=f"pytest {result.nodeid}",
        script=str(entry_py),
        stdout_log=result.stdout_log,
        stderr_log=result.stderr_log,
        debug_report_dir=result.debug_report_dir,
        debug_txt_count=result.debug_txt_count,
        debug_json_count=result.debug_json_count,
        first_error=first_error,
        include_names=include_names_for(op),
        description=f"pytest fallback node: {result.nodeid}",
        report_timeout_sec=fallback_args.report_timeout,
        first_report_timeout_sec=fallback_args.first_report_timeout,
        timeout_reason=result.timeout_reason,
    )


def run_direct_case(
    case: DirectCase,
    worktree: Path,
    run_dir: Path,
    bootstrap_dir: Path,
    args: argparse.Namespace,
) -> DirectCaseStatus:
    case_dir = run_dir / case.op / case.case_id
    debug_dir = case_dir / "debug_reports"
    case_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    script = case_dir / "case.py"
    stdout_log = case_dir / "stdout.log"
    stderr_log = case_dir / "stderr.log"
    write_case_script(script, case)

    argv = [str(args.python), str(script)]
    command = shell_command(worktree, argv)
    env = build_env(os.environ, worktree, bootstrap_dir, debug_dir, args)
    start = time.time()
    timed_out = False
    timeout_reason = ""
    exit_code: int | None = None
    last_report_time: float | None = None
    last_complete_reports = 0
    total_timeout = args.case_timeout if args.case_timeout is not None else args.case_total_timeout
    with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
        proc = subprocess.Popen(
            ["/bin/bash", "-lc", command],
            stdout=out,
            stderr=err,
            env=env,
            start_new_session=True,
        )
        while True:
            exit_code = proc.poll()
            now = time.time()
            if exit_code is not None:
                break

            complete_reports = complete_report_count(debug_dir)
            if complete_reports > last_complete_reports:
                last_complete_reports = complete_reports
                last_report_time = now

            elapsed = now - start
            if total_timeout is not None and elapsed > total_timeout:
                timed_out = True
                timeout_reason = "case_total_timeout"
                kill_process_group(proc)
                exit_code = None
                break

            if last_complete_reports == 0:
                if (args.first_report_timeout is not None and elapsed > args.first_report_timeout):
                    timed_out = True
                    timeout_reason = "first_report_timeout"
                    kill_process_group(proc)
                    exit_code = None
                    break
            elif (args.report_timeout is not None and last_report_time is not None
                  and now - last_report_time > args.report_timeout):
                timed_out = True
                timeout_reason = "report_timeout"
                kill_process_group(proc)
                exit_code = None
                break

            time.sleep(1)

    duration = time.time() - start
    debug_txt_count, debug_json_count, report_mtime = debug_report_snapshot(debug_dir)
    status = classify_status(
        exit_code,
        timed_out,
        debug_txt_count,
        debug_json_count,
        stdout_log,
        stderr_log,
    )
    if timed_out and min(debug_txt_count, debug_json_count) > 0:
        status = "partial_timeout"
    if status == "missing_debug_report" and not op_has_debuggable_kernel(worktree, case.op):
        status = "no_triton_kernel"
    first_error = first_error_from_logs(stdout_log, stderr_log)
    if status == "missing_debug_report":
        first_error = ("case exited successfully but debugger report is missing or "
                       f"incomplete: txt={debug_txt_count}, json={debug_json_count}, "
                       f"dir={debug_dir}")
    elif status == "no_triton_kernel":
        first_error = "case exited successfully but selected op has no debuggable Triton kernel"
    elif status in {"timeout", "partial_timeout"}:
        first_error = timeout_reason or first_error
    case_status = DirectCaseStatus(
        op=case.op,
        case_id=case.case_id,
        status=status,
        exit_code=exit_code,
        duration_sec=duration,
        command=command,
        script=str(script),
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        debug_report_dir=str(debug_dir),
        debug_txt_count=debug_txt_count,
        debug_json_count=debug_json_count,
        first_error=first_error,
        include_names=case.include_names,
        description=case.description,
        last_report_time=report_mtime,
        report_timeout_sec=args.report_timeout,
        first_report_timeout_sec=args.first_report_timeout,
        timeout_reason=timeout_reason,
    )
    write_text(case_dir / "status.json", json.dumps(asdict(case_status), indent=2))
    return case_status


def no_direct_case_status(op: str, run_dir: Path, reason: str) -> DirectCaseStatus:
    case_dir = run_dir / op / "no_direct_case"
    debug_dir = case_dir / "debug_reports"
    case_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    write_text(case_dir / "stdout.log", reason + "\n")
    write_text(case_dir / "stderr.log", "")
    status = DirectCaseStatus(
        op=op,
        case_id="no_direct_case",
        status="no_direct_case",
        exit_code=None,
        duration_sec=0.0,
        command="",
        script=None,
        stdout_log=str(case_dir / "stdout.log"),
        stderr_log=str(case_dir / "stderr.log"),
        debug_report_dir=str(debug_dir),
        debug_txt_count=0,
        debug_json_count=0,
        first_error=reason,
        include_names=[],
        description=reason,
    )
    write_text(case_dir / "status.json", json.dumps(asdict(status), indent=2))
    return status


def unsupported_status(op: str, run_dir: Path, reason: str) -> DirectCaseStatus:
    case_dir = run_dir / op / "unsupported_pointwise_dynamic"
    debug_dir = case_dir / "debug_reports"
    case_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    write_text(case_dir / "stdout.log", reason + "\n")
    write_text(case_dir / "stderr.log", "")
    status = DirectCaseStatus(
        op=op,
        case_id="unsupported_pointwise_dynamic",
        status="unsupported_pointwise_dynamic",
        exit_code=None,
        duration_sec=0.0,
        command="",
        script=None,
        stdout_log=str(case_dir / "stdout.log"),
        stderr_log=str(case_dir / "stderr.log"),
        debug_report_dir=str(debug_dir),
        debug_txt_count=0,
        debug_json_count=0,
        first_error=reason,
        include_names=[],
        description=reason,
    )
    write_text(case_dir / "status.json", json.dumps(asdict(status), indent=2))
    return status


def write_summary(run_dir: Path, statuses: list[DirectCaseStatus]) -> None:
    rows = [asdict(status) for status in statuses]
    write_text(run_dir / "summary.json", json.dumps(rows, indent=2, sort_keys=True))
    if not rows:
        write_text(run_dir / "summary.csv", "")
        return
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_status: dict[str, int] = {}
    for status in statuses:
        by_status[status.status] = by_status.get(status.status, 0) + 1
    write_text(run_dir / "status_counts.json", json.dumps(by_status, indent=2, sort_keys=True))
    write_text(
        run_dir / "missing_case_ops.json",
        json.dumps(
            sorted({s.op
                    for s in statuses
                    if s.status == "no_direct_case"}),
            indent=2,
        ),
    )
    write_text(
        run_dir / "failed_cases.json",
        json.dumps(
            [
                asdict(s) for s in statuses if s.status not in {
                    "passed",
                    "no_direct_case",
                    "unsupported_pointwise_dynamic",
                    "no_triton_kernel",
                    "needs_manual_case",
                }
            ],
            indent=2,
            sort_keys=True,
        ),
    )
    write_text(
        run_dir / "passed_cases.json",
        json.dumps(
            [asdict(s) for s in statuses if s.status == "passed"],
            indent=2,
            sort_keys=True,
        ),
    )
    coverage: dict[str, list[str]] = {
        "tested": [],
        "test_case_available": [],
        "needs_manual_case": [],
        "no_triton_kernel": [],
        "unsupported_signature": [],
        "timeout": [],
        "failed": [],
    }
    for status in statuses:
        if status.status == "passed":
            coverage["tested"].append(status.op)
        elif status.status == "dry_run_case":
            coverage["test_case_available"].append(status.op)
        elif status.status == "no_direct_case":
            coverage["needs_manual_case"].append(status.op)
        elif status.status == "no_triton_kernel":
            coverage["no_triton_kernel"].append(status.op)
        elif status.status in {"missing_debug_report", "runtime_error"}:
            coverage["unsupported_signature"].append(status.op)
        elif status.status in {"timeout", "partial_timeout"}:
            coverage["timeout"].append(status.op)
        elif status.status not in {"unsupported_pointwise_dynamic"}:
            coverage["failed"].append(status.op)
    write_text(
        run_dir / "coverage_by_reason.json",
        json.dumps(
            {key: sorted(set(value))
             for key, value in coverage.items()},
            indent=2,
            sort_keys=True,
        ),
    )


def write_manifest(
    run_dir: Path,
    args: argparse.Namespace,
    worktree: Path,
    ops: list[str],
    stats: InstrumentationStats,
) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "direct_op_case",
        "flaggems_root": str(args.flaggems_root),
        "worktree": str(worktree),
        "ops": ops,
        "stages": args.stages,
        "level": args.level,
        "addr_level": args.addr_level,
        "record_capacity": args.record_capacity,
        "case_timeout": args.case_timeout,
        "case_total_timeout": args.case_total_timeout,
        "first_report_timeout": args.first_report_timeout,
        "report_timeout": args.report_timeout,
        "case_filter": args.case_filter,
        "include_status": args.include_status,
        "source_summary": str(args.source_summary) if args.source_summary else None,
        "pointwise_mode": args.pointwise_mode,
        "skip_pointwise_dynamic": args.skip_pointwise_dynamic,
        "normalize_ext_launch_ids": args.normalize_ext_launch_ids,
        "enable_pytest_fallback": args.enable_pytest_fallback,
        "pytest_fallback_poll_interval": args.pytest_fallback_poll_interval,
        "instrumentation": asdict(stats),
    }
    write_text(run_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True))


def add_bool_argument(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
    help_text: str,
) -> None:
    dest = name.replace("-", "_")
    parser.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    parser.add_argument(f"--no-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def latest_summary_path(workspace_root: Path) -> Path | None:
    runs_root = workspace_root / "direct_runs"
    if not runs_root.exists():
        return None
    summaries = sorted(path for path in runs_root.glob("*/summary.json") if path.is_file())
    return summaries[-1] if summaries else None


def apply_status_selection(
    args: argparse.Namespace,
    workspace_root: Path,
    selected_ops: list[str],
) -> list[str]:
    if not args.include_status:
        return selected_ops

    summary_path = args.source_summary or latest_summary_path(workspace_root)
    if summary_path is None:
        raise FileNotFoundError("--include-status requires an existing summary.json")
    rows = json.loads(read_text(summary_path))
    statuses = {status.strip() for status in args.include_status.split(",") if status.strip()}
    selected_set = set(selected_ops)
    result: list[str] = []
    for row in rows:
        op = str(row.get("op", "")).strip()
        status = str(row.get("status", "")).strip()
        if not op or status not in statuses or op in result:
            continue
        if args.ops or args.op_list_file:
            if op not in selected_set:
                continue
        result.append(op)
    if args.max_ops:
        result = result[:args.max_ops]
    args.source_summary = summary_path
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FlagGems debugger checks through direct operator cases.")
    parser.add_argument("--flaggems-root", type=Path, default=DEFAULT_FLAGGEMS_ROOT)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument(
        "--python",
        type=Path,
        default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable),
    )
    parser.add_argument("--ops", help="comma-separated op ids")
    parser.add_argument("--op-list-file")
    parser.add_argument("--stages", default="stable,beta")
    parser.add_argument("--start")
    parser.add_argument("--max-ops", type=int)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--addr-level", type=int, default=1)
    parser.add_argument("--record-capacity", type=int, default=4096)
    parser.add_argument(
        "--case-timeout",
        type=int,
        default=None,
        help="Legacy total timeout override for each direct case.",
    )
    parser.add_argument("--case-total-timeout", type=int, default=1800)
    parser.add_argument("--first-report-timeout", type=int, default=300)
    parser.add_argument("--report-timeout", type=int, default=180)
    parser.add_argument(
        "--include-status",
        help="comma-separated statuses to select from a previous summary.json",
    )
    parser.add_argument(
        "--source-summary",
        type=Path,
        help="summary.json to use with --include-status; defaults to latest direct run",
    )
    parser.add_argument(
        "--pointwise-mode",
        choices=["skip", "wrapper-patch"],
        default="wrapper-patch",
    )
    parser.add_argument(
        "--case-filter",
        help="Run only direct cases whose case_id or description contains this text.",
    )
    add_bool_argument(
        parser,
        "skip-pointwise-dynamic",
        default=False,
        help_text="Skip known pointwise_dynamic ops instead of patching generated wrappers.",
    )
    add_bool_argument(
        parser,
        "normalize-ext-launch-ids",
        default=False,
        help_text="Rewrite ext.program_id/ext.num_programs in the copied worktree.",
    )
    add_bool_argument(
        parser,
        "instrument-pointwise-generated",
        default=True,
        help_text="Patch copied pointwise_dynamic generator wrappers.",
    )
    add_bool_argument(
        parser,
        "enable-pytest-fallback",
        default=True,
        help_text="When a direct case is unavailable, run one collected pytest node for that op.",
    )
    parser.add_argument("--pytest-fallback-poll-interval", type=float, default=2.0)
    parser.add_argument("--export-raw-records", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    add_bool_argument(parser, "keep-worktree", default=True, help_text="Keep copied worktree.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.pointwise_mode == "wrapper-patch":
        args.skip_pointwise_dynamic = False
        args.instrument_pointwise_generated = True
    elif args.pointwise_mode == "skip":
        args.skip_pointwise_dynamic = True
        args.instrument_pointwise_generated = False

    workspace_root = args.workspace_root.resolve()
    runs_root = workspace_root / "direct_runs"
    worktrees_root = workspace_root / "worktrees"
    for directory in (runs_root, worktrees_root):
        directory.mkdir(parents=True, exist_ok=True)

    flaggems_root = args.flaggems_root.resolve()
    if not flaggems_root.exists():
        raise FileNotFoundError(f"FlagGems root not found: {flaggems_root}")

    inventory = load_operator_inventory(flaggems_root)
    inventory_map = inventory_by_id(inventory)
    ops = apply_status_selection(args, workspace_root, select_ops(args, inventory))
    stamp = now_stamp()
    run_dir = runs_root / stamp
    bootstrap_dir = run_dir / "bootstrap"
    worktree = worktrees_root / f"FlagGems_direct_instrumented_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    print(f"[INFO] FlagGems root: {flaggems_root}")
    print(f"[INFO] workspace: {workspace_root}")
    print(f"[INFO] run dir: {run_dir}")
    print(f"[INFO] selected ops: {len(ops)}")

    if args.dry_run:
        stats = InstrumentationStats()
        statuses: list[DirectCaseStatus] = []
        for op in ops:
            cases = filter_cases(
                generate_cases_for_op(op,
                                      inventory_map.get(op) or {}),
                args.case_filter,
            )
            if not cases:
                statuses.append(no_direct_case_status(op, run_dir, "dry-run: no direct case generator for this op"))
            else:
                for case in cases:
                    statuses.append(
                        DirectCaseStatus(
                            op=case.op,
                            case_id=case.case_id,
                            status="dry_run_case",
                            exit_code=None,
                            duration_sec=0.0,
                            command="",
                            script=None,
                            stdout_log="",
                            stderr_log="",
                            debug_report_dir="",
                            debug_txt_count=0,
                            debug_json_count=0,
                            first_error="",
                            include_names=case.include_names,
                            description=case.description,
                        ))
        write_manifest(run_dir, args, worktree, ops, stats)
        write_summary(run_dir, statuses)
        print("[INFO] dry run complete")
        return 0

    create_bootstrap(bootstrap_dir)
    print(f"[INFO] copying FlagGems to {worktree}")
    copy_flaggems_source(flaggems_root, worktree)

    warnings_path = run_dir / "instrumentation_warnings.json"
    classifications_path = run_dir / "jit_function_classifications.json"
    print("[INFO] instrumenting Triton JIT functions")
    stats = instrument_flaggems_tree(
        worktree,
        args.level,
        args.addr_level,
        warnings_path,
        classifications_path,
        args.instrument_pointwise_generated,
        args.normalize_ext_launch_ids,
    )
    write_manifest(run_dir, args, worktree, ops, stats)
    print("[INFO] instrumentation: "
          f"{stats.functions_instrumented} functions in {stats.files_changed} files")
    pytest_fallback_nodes = collect_pytest_fallback_nodes(worktree, run_dir, ops, args)
    if args.enable_pytest_fallback:
        covered_by_fallback = sorted(pytest_fallback_nodes)
        write_text(
            run_dir / "pytest_fallback_ops.json",
            json.dumps(covered_by_fallback, indent=2, sort_keys=True),
        )
        print("[INFO] pytest fallback candidates: "
              f"{len(covered_by_fallback)} op(s)")

    statuses: list[DirectCaseStatus] = []
    for index, op in enumerate(ops, start=1):
        print(f"[INFO] [{index}/{len(ops)}] generating direct cases for {op}")
        if args.skip_pointwise_dynamic and op_uses_pointwise_dynamic(worktree, op, inventory_map):
            reason = ("skipped: current Ascend debugger mode does not support "
                      "FlagGems pointwise_dynamic generated kernels with tt.call")
            status = unsupported_status(op, run_dir, reason)
            statuses.append(status)
            write_summary(run_dir, statuses)
            print(f"[INFO] {op}: {status.status}")
            continue

        cases = filter_cases(
            generate_cases_for_op(op,
                                  inventory_map.get(op) or {}),
            args.case_filter,
        )
        if not cases:
            fallback_node = (pytest_fallback_nodes.get(op) or [None])[0]
            if fallback_node is not None:
                print(f"[INFO] running {op}/pytest_fallback")
                status = run_pytest_fallback_case(
                    op,
                    fallback_node,
                    worktree,
                    run_dir,
                    args,
                )
                statuses.append(status)
                write_summary(run_dir, statuses)
                print(f"[INFO] {op}/{status.case_id}: {status.status} "
                      f"exit={status.exit_code} reports={status.debug_txt_count}")
                continue
            status = no_direct_case_status(op, run_dir, "no direct case generator for this op")
            statuses.append(status)
            write_summary(run_dir, statuses)
            print(f"[INFO] {op}: {status.status}")
            continue

        for case in cases:
            print(f"[INFO] running {op}/{case.case_id}")
            status = run_direct_case(case, worktree, run_dir, bootstrap_dir, args)
            statuses.append(status)
            write_summary(run_dir, statuses)
            print(f"[INFO] {op}/{case.case_id}: {status.status} "
                  f"exit={status.exit_code} reports={status.debug_txt_count}")

    write_summary(run_dir, statuses)
    failing = [
        status for status in statuses if status.status not in {
            "passed",
            "no_direct_case",
            "unsupported_pointwise_dynamic",
            "no_triton_kernel",
            "needs_manual_case",
        }
    ]
    if failing:
        print(f"[WARN] {len(failing)} case(s) did not pass. See {run_dir / 'summary.json'}")
        return 1
    print(f"[INFO] complete. See {run_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
