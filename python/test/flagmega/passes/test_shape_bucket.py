# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Shape-bucket specializations bind symbolic dimensions before codegen."""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import IRSchemaError, StageError
from triton.flagmega.options import CompileOptions
from triton.flagmega.passes.shape_bucket import (
    BINDING_SCHEMA,
    DECLARATION_SCHEMA,
    bind_shape_buckets,
    bucket_index_for,
    compute_segment_list,
    shape_bucket_bindings,
    shape_bucket_declaration,
    shape_bucket_segments,
)


def _symbolic_add(tokens=fm.dim('tokens', 1, 2), width=16):
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect='high_level', stage='imported', entry='main')

        def forward(self):
            value_type = fm.tensor_type('bfloat16', (tokens, width))
            lhs = self.input('lhs', value_type, id='lhs')
            rhs = self.input('rhs', value_type, id='rhs')
            out = fm.F.math.add(lhs, rhs, name='out')
            self.function('main', [lhs, rhs], [out])

    return Graph().build()


def _declaration(**overrides):
    declaration = {
        'schema': DECLARATION_SCHEMA,
        'enable': True,
        'range_info': {'tokens': [1, 2]},
        'segments_count': 2,
        'fix_var_map': {},
    }
    declaration.update(overrides)
    return declaration


def _declare(module, **overrides):
    return replace(module, metadata={**module.metadata, 'shape_bucket': _declaration(**overrides)})


def test_segment_list_ports_nncase_ceil_free_split():
    # nncase ShapeBucketHelper.ComputeSegmentList: size = (max - min) // count,
    # the first count-1 upper bounds are min + i*size and the last is max.
    assert compute_segment_list(2, 1, 2) == (1, 2)
    assert compute_segment_list(2, 1, 3) == (1, 3)
    assert compute_segment_list(4, 1, 100) == (1, 25, 49, 100)
    assert compute_segment_list(3, 10, 10) == (10, 10, 10)


def test_segments_report_bounds_bindings_and_dispatch():
    segments = shape_bucket_segments(_declaration(range_info={'tokens': [1, 3]}))
    assert [(segment.index, segment.lower, segment.upper) for segment in segments] == [
        (0, 1, 1),
        (1, 2, 3),
    ]
    assert [dict(segment.bindings) for segment in segments] == [
        {'tokens': 1},
        {'tokens': 3},
    ]
    # Dispatch chooses the first segment whose inclusive upper bound covers
    # the runtime extent; segment 0 also owns the declared minimum.
    assert [bucket_index_for(segments, value=value) for value in (1, 2, 3)] == [0, 1, 1]
    assert [bucket_index_for(shape_bucket_segments(_declaration()), value=value)
            for value in (1, 2)] == [0, 1]
    with pytest.raises(IRSchemaError, match='below the declared minimum'):
        bucket_index_for(segments, value=0)
    with pytest.raises(IRSchemaError, match='above the declared maximum'):
        bucket_index_for(segments, value=4)


def test_bind_selected_segment_fixes_types_and_records_metadata():
    module = _declare(_symbolic_add())
    bound = bind_shape_buckets(module, segment=1)
    assert bound.stage == 'imported'
    assert fm.logical_type(bound.node_map['lhs'].type).shape[0].fixed_value == 2
    assert fm.logical_type(bound.node_map['out'].type).shape[0].fixed_value == 2
    binding = bound.metadata['shape_bucket_binding']
    assert binding['schema'] == BINDING_SCHEMA
    assert binding['segment'] == 1
    assert binding['segments'] == (1, 2)
    assert dict(binding['bindings']) == {'tokens': 2}
    assert dict(binding['range_info']) == {'tokens': (1, 2)}
    # A declared bucket always states its selected value in model metadata.
    assert bound.metadata['shape_bucket_binding']['tokens_per_call'] == 2


def test_default_segment_is_the_declared_maximum():
    module = _declare(_symbolic_add())
    bound = bind_shape_buckets(module)
    assert bound.metadata['shape_bucket_binding']['segment'] == 1
    assert dict(bound.metadata['shape_bucket_binding']['bindings']) == {'tokens': 2}


def test_metadata_can_preselect_a_segment():
    module = _declare(_symbolic_add(), segment=0)
    bound = bind_shape_buckets(module)
    assert bound.metadata['shape_bucket_binding']['segment'] == 0
    assert fm.logical_type(bound.node_map['lhs'].type).shape[0].fixed_value == 1
    # An explicit call wins over the checkpoint's preselection.
    assert bind_shape_buckets(module, segment=1).metadata['shape_bucket_binding']['segment'] == 1


def test_fix_var_map_binds_a_symbolic_dimension_once():
    tokens = fm.dim('tokens', 1, 8)
    module = _declare(_symbolic_add(tokens), fix_var_map={'tokens': 5})
    bound = bind_shape_buckets(module, segment=0)
    assert dict(bound.metadata['shape_bucket_binding']['bindings']) == {'tokens': 5}
    assert fm.logical_type(bound.node_map['lhs'].type).shape[0].fixed_value == 5


def test_bindings_helper_returns_the_selected_segment():
    module = _declare(_symbolic_add(), range_info={'tokens': [1, 3]})
    assert dict(shape_bucket_bindings(module)) == {'tokens': 3}
    assert dict(shape_bucket_bindings(module, segment=0)) == {'tokens': 1}
    assert shape_bucket_bindings(_symbolic_add(fm.dim(2))) is None


def test_unbound_symbolic_dimension_is_an_error():
    module = _declare(_symbolic_add(), range_info={})
    with pytest.raises(IRSchemaError, match="tokens"):
        bind_shape_buckets(module, segment=0)


def test_already_fixed_module_binds_only_metadata():
    tokens = fm.dim('tokens', 1, 4)
    module = _declare(_symbolic_add(tokens), range_info={'tokens': [3, 3]}, segments_count=2)
    fixed = replace(module, nodes=tuple(
        replace(node, type=fm.tensor_type(node.type.dtype, (3, *node.type.shape[1:])))
        if isinstance(node.type, fm.TensorType) else node
        for node in module.nodes))
    bound = bind_shape_buckets(fixed, segment=1)
    assert bound.nodes == fixed.nodes
    assert dict(bound.metadata['shape_bucket_binding']['bindings']) == {'tokens': 3}


@pytest.mark.parametrize('overrides, message', [
    ({'segments_count': 1}, 'segments_count'),
    ({'segments_count': 0}, 'segments_count'),
    ({'range_info': {'tokens': [0, 2]}}, 'minimum'),
    ({'range_info': {'tokens': [3, 2]}}, 'minimum'),
    ({'range_info': {'tokens': [1]}}, 'two bounds'),
    ({'fix_var_map': {'tokens': 0}}, 'positive'),
    ({'schema': 'flagmega.shape-bucket/v0'}, 'schema'),
])
def test_invalid_declarations_are_rejected(overrides, message):
    module = _declare(_symbolic_add(), **overrides)
    with pytest.raises(IRSchemaError, match=message):
        shape_bucket_declaration(module)


def test_re_specializing_a_fixed_module_is_rejected():
    bound = bind_shape_buckets(_declare(_symbolic_add()), segment=1)
    assert bind_shape_buckets(bound, segment=1) is bound
    with pytest.raises(IRSchemaError, match='already specialized'):
        bind_shape_buckets(bound, segment=0)


def test_segment_out_of_range_is_rejected():
    module = _declare(_symbolic_add())
    with pytest.raises(IRSchemaError, match='segment'):
        bind_shape_buckets(module, segment=2)
    with pytest.raises(IRSchemaError, match='segment'):
        bind_shape_buckets(module, segment=-1)


def test_no_declaration_returns_the_identical_module():
    module = _symbolic_add()
    with pytest.raises(IRSchemaError, match='no shape-bucket declaration'):
        bind_shape_buckets(module)
    assert shape_bucket_declaration(module) is None
    assert shape_bucket_bindings(module) is None
    fixed = _symbolic_add(fm.dim(2))
    assert bind_shape_buckets(fixed) is fixed


def test_disabled_declaration_returns_the_identical_module():
    module = _declare(_symbolic_add(), enable=False)
    assert bind_shape_buckets(module) is module


def test_compiler_specializes_one_bucket_before_distribution():
    module = _declare(_symbolic_add())
    result = Compiler(CompileOptions(shape_bucket_segment=1)).compile(module)
    assert result.module.stage == 'bufferized_tir'
    binding = result.module.metadata['shape_bucket_binding']
    assert dict(binding['bindings']) == {'tokens': 2}
    assert binding['segment'] == 1
    arguments = {node.attrs['name']: node for node in result.module.nodes
                 if node.op == 'builtin.var'}
    assert fm.logical_type(arguments['lhs'].type).shape[0].fixed_value == 2


def test_compiler_segment_option_cannot_change_a_bound_checkpoint():
    bound = bind_shape_buckets(_declare(_symbolic_add()), segment=1)
    with pytest.raises(StageError, match='already specialized'):
        Compiler(CompileOptions(shape_bucket_segment=0)).compile(bound)
    result = Compiler(CompileOptions(shape_bucket_segment=1)).compile(bound)
    assert result.module.stage == 'bufferized_tir'
    assert result.module.metadata['shape_bucket_binding']['segment'] == 1


def test_compiler_segment_option_requires_a_declaration():
    with pytest.raises(StageError, match='shape-bucket'):
        Compiler(CompileOptions(shape_bucket_segment=0)).compile(_symbolic_add())


def test_compile_options_reject_negative_segment():
    with pytest.raises(ValueError, match='shape_bucket_segment'):
        CompileOptions(shape_bucket_segment=-1)


def test_stop_after_bind_reaches_the_boundary_for_a_static_module():
    fixed = _symbolic_add(fm.dim(2))
    result = Compiler().compile(fixed, stop_after='bind-shape-buckets')
    assert result.module.stage == 'shape_buckets_bound'
    assert result.module.semantic_hash != fixed.semantic_hash  # stage is hashed


def test_binding_survives_python_checkpoint_roundtrip(tmp_path):
    module = bind_shape_buckets(_declare(_symbolic_add()), segment=1)
    path = fm.emit_module(module, tmp_path / 'bound.py')
    restored = fm.load_module(path)
    assert restored.semantic_hash == module.semantic_hash
    assert restored.metadata['shape_bucket_binding'] == module.metadata['shape_bucket_binding']
