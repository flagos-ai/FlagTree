# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Bank aliases accept constant call indices and widen dynamic row products."""
import importlib.util

import pytest

from triton.flagmega.codegen.triton.function_pool_abi import function_pool_aliases


def _kernel(tmp_path, constant, stride=1 << 32):
    plan = {"bindings": [{"formal": "weight", "storage": "rdata",
                         "scalar_dtype": "uint8", "byte_offset": 128,
                         "row_stride_bytes": stride, "alignment_bytes": 128}]}
    alias = function_pool_aliases(plan)["weight"]
    source = f"""
import triton
import triton.language as tl
@triton.jit(noinline=True)
def bank_offset(_flagmega_call_pool_rdata, _flagmega_bank_row_index):
    pointer = {alias}
    return pointer.to(tl.int64) - _flagmega_call_pool_rdata.to(tl.int64)
@triton.jit
def bank_indices(pool, output, row):
    tl.store(output, bank_offset(pool, {constant}))
    tl.store(output + 1, bank_offset(pool, row))
"""
    path = tmp_path / f"bank_index_{constant}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    namespace = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(namespace)
    return namespace.bank_indices


@pytest.mark.parametrize("constant", [0, 3])
@pytest.mark.parametrize("row_type", ["i32", "i64"])
def test_bank_alias_compiles_constant_and_runtime_rows(tmp_path, constant, row_type):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton._C import libtriton
    if not hasattr(libtriton, "iluvatar"):
        pytest.skip("The Iluvatar compiler is required")
    compiled = triton.compile(ASTSource(_kernel(tmp_path, constant),
        signature={"pool": "*u8", "output": "*i64", "row": row_type}, constexprs={}),
        target=GPUTarget("corex", 71, 64), options={"num_warps": 4})
    assert "define" in compiled.asm["llir"]


@pytest.mark.parametrize("constant,row,stride", [(0, 3, 1 << 32),
                                                 (3, (1 << 31) + 1, 1 << 20)])
def test_bank_alias_preserves_64bit_displacements_on_gpu(tmp_path, constant, row, stride):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("An Iluvatar COREX GPU is required")
    pool = torch.empty(1, device="cuda", dtype=torch.uint8)
    output = torch.empty(2, device="cuda", dtype=torch.int64)
    _kernel(tmp_path, constant, stride)[(1,)](pool, output, row, num_warps=4)
    assert output.tolist() == [128 + constant * stride, 128 + row * stride]
