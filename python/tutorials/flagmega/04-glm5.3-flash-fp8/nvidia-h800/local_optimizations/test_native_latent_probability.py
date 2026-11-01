"""Check ordered probability expressions, publication and complete workspace bounds."""
import ast
from pathlib import Path
import sys
import pytest

from jinja2 import Environment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from triton.flagmega.targets import NvidiaSm90Target
from local_optimizations.native_latent_probability_model import (
    IDENTITY, METADATA_CHIP_STRIDE, METADATA_PAYLOAD_FLOATS, declared_model, workspaces,
)
from local_optimizations.native_latent_probability_template import probability_template, KEY
from local_optimizations.native_latent_attention_schedule import _template
from local_optimizations.dsa_column_owners import probability_selected
from triton.flagmega.errors import CodegenError


def sources():
    root = Path(__file__).resolve().parents[6] / 'python/triton/flagmega/codegen/triton'
    raw = (root / KEY).read_text()
    return _template(raw, compact_pipeline=True), probability_template(raw)


def test_typed_workspace_capacity_alignment_and_original_implementation():
    original = NvidiaSm90Target().triton_implementation_model
    model = declared_model(original)
    assert model.implementation('tir.dsa_paged_attention.decode') == original.implementation('tir.dsa_paged_attention.decode')
    selected = model.implementation(IDENTITY)
    assert selected.facts['internal_grid_barriers'] == 3 and 'grid_sync' in selected.requires
    assert selected.parameters['latent_long_column_tile'] == 32
    score, metadata = workspaces()
    assert score['name'] != metadata['name'] and score['alignment'] == metadata['alignment'] == 256
    assert METADATA_CHIP_STRIDE * 4 % 256 == 0
    maxima = {tile * 16 + head for tile in range(64) for head in range(16)}
    coefficients = {1024 + tile * 32 + head * 2 + component
                    for tile in range((2051 + 63) // 64) for head in range(16) for component in range(2)}
    assert not maxima.intersection(coefficients)
    assert max(coefficients) + 1 == METADATA_PAYLOAD_FLOATS <= METADATA_CHIP_STRIDE
    assert 33 * 16 * 64 == 33792


def test_short_path_math_is_unchanged_and_long_pv_covers_all_columns():
    original, changed = sources()
    short = changed.split('    else:\n        if ({{ call.probability_schedule.group.index }}) < 8:\n', 1)[1]
    math = original.split('        _fm_dsa_tokens = tl.arange(0, 64)\n', 1)[1]
    expected = '        _fm_dsa_tokens = tl.arange(0, 64)\n' + math
    restored = '\n'.join(line[4:] if line.startswith('    ') else line for line in short.splitlines()) + '\n'
    assert restored.replace('call.probability_schedule', 'call.native_latent_schedule') == expected
    assert sorted(cta * 32 + column for cta in range(16) for column in range(32)) == list(range(512))


def test_probability_publish_order_and_original_bf16_sum_expressions():
    original, changed = sources()
    long = changed.split('    if _fm_dsa_count >= ', 1)[1].split('    else:\n', 1)[0]
    stages = long.split('tle.distributed_barrier')
    assert len(stages) == 4
    assert 'tl.max(_fm_dsa_score, 1)' in stages[0]
    assert 'tl.associative_scan' in stages[1]
    assert '_fm_dsa_rounded_p.to(tl.float32)' in stages[2] and 'tl.sum(_fm_dsa_p, 1)' in stages[2]
    assert 'long_column_tile' in stages[3] and '_fm_dsa_acc = tl.dot(' in stages[3]
    for name in ('_fm_dsa_previous', '_fm_dsa_p', '_fm_dsa_rounded_p'):
        expression = next(line.strip() for line in original.splitlines() if line.strip().startswith(name + ' = '))
        assert expression in stages[2]
    assert stages[3].count('tl.exp2(') == 0
    assert '_fm_dsa_sum * _fm_dsa_previous + ' in stages[3]


def test_declared_column_owner_rejects_incompatible_geometry():
    model = declared_model(NvidiaSm90Target().triton_implementation_model)
    call = dict(parameters=dict(model.implementation(IDENTITY).parameters), head_dim=512,
                padded_head_dim=512, num_query_heads=16, head_sharded=True,
                index_traversal='compact', compact_topk_capacity=4096, probability_rounding='online_bf16')
    assert probability_selected(call)
    call['parameters']['latent_long_column_tile'] = 64
    with pytest.raises(CodegenError):
        probability_selected(call)
    call['parameters']['latent_long_column_tile'] = 32
    call['probability_rounding'] = 'fp32'
    with pytest.raises(CodegenError):
        probability_selected(call)
