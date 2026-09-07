# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Execute the production Boxing template over pre-populated owner storage."""

import importlib.util

import pytest

from triton.flagmega.codegen.triton.kernel_call_renderers import prepare_kernel_calls
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from .test_partial_reduce_local_abi import _abi, _raw


@pytest.mark.parametrize("reduce_op", ["sum", "min", "max", "prod"])
@pytest.mark.parametrize("dtype", ["float32", "int32", "int64"])
def test_partial_reduction_obeys_operator_dtype_and_owner_tail(tmp_path, reduce_op, dtype):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    # Fifteen logical owners, three value lanes: both reduction/value tails.
    owners, values = 15, 3
    source_abi = _abi((values,), storage_kind="compact_per_owner", coordinate_space="local",
                      partial_axes=(0, 1), owner_stride=values)
    result_abi = _abi((values,))
    for abi in (source_abi, result_abi):
        abi.update(scalar_dtype=dtype, scalar_itemsize=8 if dtype == "int64" else 4)
        abi["distributed_type"]["placement"]["hierarchy"] = (3, 5)
    source_abi["distributed_type"]["partial"]["reduce_op"] = reduce_op
    call = prepare_kernel_calls((_raw((source_abi,), (result_abi,), ("gather_reduce_scatter",)),), function_name="main")[0]
    source = TritonTemplateRegistry().render(
        "kernels/distributed_boxing/gather_reduce_scatter.py.jinja",
        {"render_calls": (call,), "distributed_entry": False, "mesh_hierarchy": (3, 5)},
    )
    path = tmp_path / "boxing_kernel.py"
    path.write_text("import triton\nimport triton.language as tl\n" + source)
    spec = importlib.util.spec_from_file_location("boxing_test_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = getattr(module, call["symbol"])
    input_value = torch.arange(owners * values, device="cuda").reshape(owners, values) - 20
    if reduce_op == "prod":
        input_value = torch.where(input_value % 3 == 0, -1, 1)
    elif dtype != "float32":
        # FP32 accumulation must not erase the unit in a large exact integer.
        input_value[0] += 2 ** (53 if dtype == "int64" else 24) + 1
    input_value = input_value.to(getattr(torch, dtype))
    reference_input = input_value.cpu()
    expected = {
        "sum": lambda: reference_input.sum(0),
        "min": lambda: reference_input.amin(0),
        "max": lambda: reference_input.amax(0),
        "prod": lambda: reference_input.prod(0),
    }[reduce_op]().to(getattr(torch, dtype))
    output = torch.empty((values,), dtype=getattr(torch, dtype), device="cuda")
    # All owner inputs already exist. One CTA independently materializes this
    # complete partial group, so no cooperative launch/barrier is necessary.
    kernel[(1,)](input_value, output, num_warps=4)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
