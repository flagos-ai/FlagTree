# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import json
from pathlib import Path

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.delta_rule_block_update import DeltaRuleBlockUpdate
from triton.flagmega.ir.ops.nn.delta_rule_block_update import delta_rule_block_update
from triton.flagmega.ir.ops.tensors.pack import pack_physical
from triton.flagmega.ir.ops.tensors.unpack import unpack_physical
from python.test.flagmega.ir.ops.primitive_helpers import primitive_module


def test_cuda_block_update_retains_rounded_mma_operand_dtype():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    fixture = json.loads((Path(__file__).resolve().parents[4] / "data" /
                          "native_block_update_mma_rounding.json").read_text())
    operands = tuple(torch.tensor(fixture[name], device="cuda", dtype=dtype) for name, dtype in (
        ("query", torch.bfloat16), ("key", torch.bfloat16), ("value", torch.bfloat16),
        ("coefficients", torch.bfloat16), ("log_prefix", torch.float32)))
    initial = torch.zeros((1, 128, 128), device="cuda")
    actual, _ = delta_rule_block_update(*operands, initial, torch=torch)
    expected = torch.tensor(fixture["output"], device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("packed", [False, True])
def test_single_token_updates_only_named_state_field_and_returns_same_reference(packed):
    torch = pytest.importorskip("torch")
    query = torch.full((1, 1, 4), 0.5, dtype=torch.bfloat16)
    value = torch.tensor([[[1., 2., 3.], [4., 5., 6.]]], dtype=torch.bfloat16)
    coeff = torch.zeros((1, 2, 8, 8), dtype=torch.bfloat16)
    coeff[:, :, 0, 0] = 0.5
    prefix = torch.zeros((1, 2, 8))
    matrix = torch.zeros((2, 3, 4))
    if packed:
        matrix = pack_physical(matrix, 3, (2, ), (2, ))
        state_tensor = fm.tensor_type(fm.VectorType(fm.DType.FLOAT32, (2, )), (2, 3, 2))
    else:
        state_tensor = fm.tensor_type("float32", (2, 3, 4))
    state = {"matrix": matrix, "untouched": torch.tensor([19.])}
    state_type = fm.RefType("state", (("untouched", fm.tensor_type("float32", (1, ))), ("matrix", state_tensor)))
    values = (query, query, value, coeff, prefix)
    types = tuple(fm.tensor_type(str(item.dtype).removeprefix("torch."), item.shape) for item in values)
    module = primitive_module(DeltaRuleBlockUpdate, (*types, state_type), scale=1.,
                              state_vector_axes=("key", ) if packed else ())
    arguments = dict(zip((parameter.name for parameter in DeltaRuleBlockUpdate.input_parameters), (*values, state)))
    output, returned_state = TorchEvaluator(DictWeightResolver({})).run(module, arguments)[0]
    assert returned_state is state
    torch.testing.assert_close(output, value * 0.5, rtol=0, atol=0)
    updated = unpack_physical(matrix, 3, (2, ), (2, )) if packed else matrix
    expected = value[0].float()[:, :, None].expand(2, 3, 4) * 0.25
    torch.testing.assert_close(updated, expected, rtol=0, atol=0)
    assert state["untouched"].item() == 19.
