# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Entry ABI metadata follows the entry outputs it names.

A published ABI contract (``composite_abi``) names entry parameter and output
node ids.  AutoDistributed materialization may satisfy a program-output type by
inserting a reshard node, which replaces the id in ``function.outputs``.  The
contract must be redirected with it, or the runtime adapter resolves a name that
no longer exists in the entry signature.
"""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.entry_abi import redirect_entry_abi
from triton.flagmega.passes.auto_distributed import AutoDistributedPass
from triton.flagmega.targets import NvidiaSm90Target

from .helpers import packed_matmul_module


def _module_with_abi() -> fm.IRModule:
    module = packed_matmul_module()
    return replace(module, metadata={
        **module.metadata,
        "composite_abi": {
            "schema": "flagmega.composite-abi/v1",
            "input_id": "value",
            "logits_output": "output",
            "token_output": "output",
            "tensor_states": ({
                "state_id": 0,
                "input": ("value", "value"),
                "output": ("output", "output"),
            }, ),
            "paged_state": {
                "input": "value",
                "output": "output"
            },
        },
    })


def test_materialization_redirects_entry_abi_onto_the_reshard_output():
    module = _module_with_abi()
    result = AutoDistributedPass.run(module, NvidiaSm90Target())

    function_output = result.function_map["main"].outputs[0]
    assert function_output != "output", "fixture must exercise an inserted program-output reshard"

    abi = result.metadata["composite_abi"]
    outputs = {str(value) for value in result.function_map["main"].outputs}
    assert abi["logits_output"] in outputs
    assert abi["token_output"] in outputs
    assert abi["paged_state"]["output"] in outputs
    assert all(value in outputs for binding in abi["tensor_states"] for value in binding["output"])
    assert abi["logits_output"] == function_output

    # Parameters are untouched by an output redirect.
    assert abi["input_id"] in {str(value) for value in result.function_map["main"].parameters}


def test_redirect_is_a_no_op_without_an_abi():
    module = packed_matmul_module()
    assert redirect_entry_abi(module.metadata, {"output": "output.reshard1"}) is module.metadata


def test_redirect_rejects_an_abi_naming_an_unknown_value():
    metadata = _module_with_abi().metadata
    with pytest.raises(IRVerificationError, match="unknown entry value"):
        redirect_entry_abi(metadata, {"output": "output.reshard1"}, known_values={"output.reshard1"})


def test_redirect_preserves_structural_keys_and_nested_bindings():
    metadata = _module_with_abi().metadata
    updated = redirect_entry_abi(metadata, {"output": "output.reshard1"})
    abi = updated["composite_abi"]

    assert abi["schema"] == "flagmega.composite-abi/v1"
    assert abi["input_id"] == "value", "a parameter is not touched by an output redirect"
    assert abi["logits_output"] == abi["token_output"] == "output.reshard1"
    assert abi["paged_state"] == {"input": "value", "output": "output.reshard1"}
    binding, = abi["tensor_states"]
    assert binding["state_id"] == 0, "a state_id is data, never a node id"
    assert binding["output"] == ("output.reshard1", "output.reshard1")


def test_every_entry_boundary_rewriting_pass_redirects_the_contract():
    """Each site that rebuilds ``function.outputs`` must apply the redirect.

    A new pass that renames an entry output without redirecting the contract
    reintroduces the same runtime-load failure, so pin the known set here.
    """

    import inspect

    from triton.flagmega.codegen.triton import lowering
    from triton.flagmega.passes.auto_distributed import materializer
    from triton.flagmega.passes.functions import distributed_boundary_layout

    for module in (materializer, lowering, distributed_boundary_layout):
        source = inspect.getsource(module)
        assert "outputs=tuple(" in source, f"{module.__name__} no longer rebuilds function outputs"
        assert "redirect_entry_abi(" in source, (
            f"{module.__name__} rewrites entry outputs without redirecting the published entry ABI")
