# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.nn.delta_rule_coefficients import delta_rule_coefficients
from triton.flagmega.ir.ops.nn.delta_rule_block_update import delta_rule_block_update
from triton.flagmega.ir.ops.tensors.pack import pack_physical
from triton.flagmega.ir.ops.tensors.unpack import unpack_physical
from python.test.flagmega.codegen.triton.kernels.delta_rule_block_update.helpers import execute_block_update


@pytest.mark.parametrize("tokens,key_dim,value_dim", [(1, 16, 9), (33, 16, 16), (65, 128, 128)])
@pytest.mark.parametrize("axes,packed,local", [((), False, False), ((0, 1), True, False), ((0, ), True, True)])
def test_block_rounding_state_packing_and_local_operands(tmp_path, tokens, key_dim, value_dim, axes, packed, local):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    generator = torch.Generator().manual_seed(809)
    key = (torch.randint(-2, 3, (tokens, 4, key_dim), generator=generator).float() / 32).bfloat16()
    query = (torch.randint(-2, 3, key.shape, generator=generator).float() / 32).bfloat16()
    value = (torch.randint(-2, 3, (tokens, 8, value_dim), generator=generator).float() / 16).bfloat16()
    initial = torch.randint(-2, 3, (8, value_dim, key_dim), generator=generator).float() / 32
    beta = torch.full((tokens, 8), 0.5)
    coefficients = delta_rule_coefficients(key, beta, 64, torch=torch)
    prefix = torch.zeros(((tokens + 63) // 64, 8, 64))
    expected, expected_state = delta_rule_block_update(query, key, value, coefficients, prefix, initial, scale=1.,
                                                       torch=torch)
    if packed:
        storage = pack_physical(initial, 3, (4, ), (2, ))
        field_type = fm.tensor_type(fm.VectorType(fm.DType.FLOAT32, (4, )), (8, value_dim, key_dim // 4))
    else:
        storage = initial
        field_type = fm.tensor_type("float32", initial.shape)
    state_type = fm.RefType("state", (("untouched", fm.tensor_type("float32", (1, ))), ("matrix", field_type)))
    actual, state, resources = execute_block_update(
        tmp_path, torch, (query, key, value, coefficients, prefix), state_type,
        {"untouched": torch.tensor([23.]), "matrix": storage}, head_axes=axes, local_inputs=local,
        attrs={"scale": 1., "state_vector_axes": ("key", ) if packed else ()})
    actual_state = unpack_physical(state["matrix"], 3, (4, ), (2, )) if packed else state["matrix"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # The CPU arithmetic reference adds a completed matmul to the initial
    # state; device MMA accumulates into that initial FP32 accumulator.
    # Exact native-device comparisons cover that hardware rounding contract.
    torch.testing.assert_close(actual_state, expected_state, rtol=3e-7, atol=1e-8)
    assert state["untouched"].item() == 23.
    assert resources["spill_bytes"] == 0
