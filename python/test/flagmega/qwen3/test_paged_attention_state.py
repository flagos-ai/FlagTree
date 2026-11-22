# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.evaluator import PagedAttentionStateConfig, create_paged_attention_state
from triton.flagmega.ir import DType, VectorType


torch = pytest.importorskip("torch")


def test_state_uses_nncase_axis_order_and_explicit_vector_lane():
    config = PagedAttentionStateConfig(2, 1, 8, block_size=2, num_blocks=2, lanes=8)

    assert config.cache_type.dtype == VectorType(DType.BFLOAT16, (8,))
    assert tuple(dimension.fixed_value for dimension in config.cache_type.shape) == (2, 2, 2, 2, 1, 1)
    assert config.storage_shape == (2, 2, 2, 2, 1, 1, 8)
    assert tuple(name for name, _ in config.ref_type.fields) == (
        "kv_caches", "query_start_loc", "seq_lens", "slot_mapping", "block_table")


def test_append_and_gather_cross_a_physical_block_boundary():
    config = PagedAttentionStateConfig(1, 1, 8, block_size=2, num_blocks=2)
    state = create_paged_attention_state(config)
    keys = [torch.full((1, 8), value, dtype=torch.bfloat16) for value in (1.0, 2.0, 3.0)]
    values = [torch.full((1, 8), value, dtype=torch.bfloat16) for value in (4.0, 5.0, 6.0)]

    for key, value in zip(keys, values):
        state.append(key, value, layer_id=0)

    gathered_key, gathered_value = state.gather(layer_id=0)
    torch.testing.assert_close(gathered_key, torch.stack(keys), rtol=0, atol=0)
    torch.testing.assert_close(gathered_value, torch.stack(values), rtol=0, atol=0)
    assert state.sequence_length == 3
    assert state.slot_mapping.item() == 2
    assert torch.equal(state.kv_caches[1, 0, 0, 0].reshape(1, 8), keys[2])


def test_state_rejects_append_past_capacity():
    state = create_paged_attention_state(PagedAttentionStateConfig(1, 1, 8, block_size=1, num_blocks=1))
    slot = torch.zeros((1, 8), dtype=torch.bfloat16)
    state.append(slot, slot, layer_id=0)

    with pytest.raises(EvaluationError, match="capacity"):
        state.append(slot, slot, layer_id=0)


def test_static_state_validation_is_capture_safe_without_host_value_reads():
    state = create_paged_attention_state(PagedAttentionStateConfig(1, 1, 8, block_size=2, num_blocks=2))
    state.seq_lens.fill_(-1)
    state.block_table.copy_(torch.tensor([[1, 0]], dtype=torch.int32))

    state.validate(dynamic_values=False)
    with pytest.raises(EvaluationError, match="sequence length"):
        state.validate()


def test_f16_state_rounds_f32_writes_and_preserves_layer_and_page_ownership():
    config = PagedAttentionStateConfig(2, 2, 8, block_size=2, num_blocks=3, cache_dtype="float16")
    state = create_paged_attention_state(config)
    state.block_table.copy_(torch.tensor([[2, 0, 1]], dtype=torch.int32))
    empty_key, empty_value = state.gather(layer_id=0)
    assert empty_key.shape == empty_value.shape == (0, 2, 8)
    assert empty_key.dtype == empty_value.dtype == torch.float16
    keys = torch.arange(48, dtype=torch.float32).reshape(3, 2, 8) / 37 + 1.0003
    values = -keys / 3
    for row in range(3):
        state.append(keys[row], values[row], layer_id=0, advance_sequence=False)
        assert state.sequence_length == row
        state.append(keys[row] + 1, values[row] - 1, layer_id=1, advance_sequence=True)
    for layer in range(2):
        key, value = state.gather(layer_id=layer)
        assert torch.equal(key.view(torch.int16), (keys + layer).half().view(torch.int16))
        assert torch.equal(value.view(torch.int16), (values - layer).half().view(torch.int16))
    assert state.sequence_length == 3 and state.slot_mapping.item() == 2
    assert torch.count_nonzero(state.kv_caches[1]) == 0
    assert torch.equal(state.kv_caches[0, 0, 0, 0].reshape(2, 8), keys[2].half())
    other = state.clone()
    other.kv_caches.zero_()
    assert torch.count_nonzero(state.kv_caches) > 0
    assert other.config == config


def test_f16_ref_type_roundtrips_without_importer_state(tmp_path):
    import os
    import subprocess
    import sys
    from triton.flagmega import ir as fm
    from triton.flagmega.ir.ops.nn._paged_attention_state import paged_attention_state_config_from_type

    config = PagedAttentionStateConfig(2, 2, 16, block_size=2, num_blocks=3, cache_dtype=DType.FLOAT16)
    assert config.cache_type.dtype == VectorType(DType.FLOAT16, (8,))
    assert paged_attention_state_config_from_type(config.ref_type) == config
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    state = builder.var("state", config.ref_type, id="state")
    builder.function("main", (state,), (state,))
    module = builder.build(entry="main")
    path = fm.emit_module(module, tmp_path / "f16_state.py")
    assert fm.load_module(path) == module
    code = ("from triton.flagmega import ir as fm; "
            "from triton.flagmega.ir.ops.nn._paged_attention_state import paged_attention_state_config_from_type as parse; "
            "import sys; m=fm.load_module(sys.argv[1]); c=parse(m.node_map['state'].type); "
            "assert c.cache_dtype == fm.DType.FLOAT16; assert m.semantic_hash == sys.argv[2]")
    subprocess.run([sys.executable, "-c", code, str(path), module.semantic_hash], env=os.environ.copy(), check=True)
    default = PagedAttentionStateConfig(2, 2, 16)
    assert default.ref_type == PagedAttentionStateConfig(2, 2, 16, cache_dtype="bfloat16").ref_type
    assert paged_attention_state_config_from_type(default.ref_type) == default


@pytest.mark.parametrize("dtype", ["float32", "int16", "unknown", None])
def test_cache_rejects_unsupported_storage_dtype(dtype):
    with pytest.raises(IRSchemaError, match="cache.*type"):
        PagedAttentionStateConfig(1, 1, 8, cache_dtype=dtype)


def test_f16_state_validation_rejects_mismatched_tensor_dtype():
    state = create_paged_attention_state(PagedAttentionStateConfig(1, 1, 8, cache_dtype="float16"))
    state.kv_caches = state.kv_caches.bfloat16()
    with pytest.raises(EvaluationError, match="float16"):
        state.validate(dynamic_values=False)
