# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton import render_triton_package
from triton.flagmega.codegen.triton.kernel_call_renderers import prepare_kernel_calls
from triton.flagmega.compiler import Compiler
from python.test.flagmega.ir.ops.primitive_helpers import primitive_module
from python.test.flagmega.ir.ops.tensors.test_concat_local_contract import concat_module


@pytest.mark.parametrize("family", ["concat", "broadcast_to"])
def test_canonical_copy_result_has_one_writer_per_broadcast_owner_group(tmp_path, family):
    placement = fm.Placement((2, 2), "xy", "bb")
    if family == "concat":
        module = concat_module(((8, 3), (8, 7)), split=True)
    else:
        value_type = fm.DistributedType(fm.tensor_type("float32", (8, 1)), (fm.SBP.split_block_cyclic(
            (0, ), 2), fm.SBP.broadcast()), placement)
        module = primitive_module(fm.get_definition("tensors.broadcast_to"), (value_type, ), shape=(8, 7))
    module = replace(module, stage="frozen_constants",
                     metadata={"auto_distribution": {"placement": placement.to_data()}})
    package = render_triton_package(Compiler().compile(module).module, tmp_path / "package")
    calls = prepare_kernel_calls(package["runtime_binding"]["call_abi"]["kernel_calls"], function_name="main")
    call = next(call for call in calls if call["family"] == family)
    assert call["writer_active"] != "True"
    source = (tmp_path / "package/generated_kernels.py").read_text()
    assert f"mask=_fm_mask & ({call['writer_active']})" in source
