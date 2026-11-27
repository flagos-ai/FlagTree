# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Execution mesh is independent of value distribution and physical storage."""
from collections.abc import Mapping
from dataclasses import replace
import json

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.execution_mesh import effective_execution_mesh


def test_dynamic_metadata_is_canonical_execution_contract():
    node = fm.Node('op', 'math.add', (), fm.tensor_type('float32', (16,)),
                   metadata={'execution_mesh': '*'})
    value = node.metadata['execution_mesh']
    assert isinstance(value, Mapping)
    assert value['kind'] == 'dynamic'
    assert value['scope'] == 'device'
    assert node.type == fm.tensor_type('float32', (16,))


def test_fixed_mesh_identity_includes_axes_order_and_mapping():
    mesh = fm.ExecutionMesh.fixed((8, 16), axes=('row', 'column'))
    assert mesh.fixed_size == 128
    assert mesh != fm.ExecutionMesh.fixed((16, 8), axes=('row', 'column'))
    assert mesh != fm.ExecutionMesh.fixed((8, 16), axes=('column', 'row'))
    assert mesh != fm.ExecutionMesh.fixed((8, 16), axes=('row', 'column'), mapping='backend_x')
    assert fm.ExecutionMesh.from_data(mesh.to_data()) == mesh


def test_dynamic_does_not_create_fixed_block_ownership():
    mesh = fm.execution_mesh('*')
    assert mesh == fm.ExecutionMesh.dynamic()
    assert mesh.fixed_size is None
    assert mesh.shape == mesh.axes == ()
    assert mesh.fusion_key is None
    assert fm.ExecutionMesh.from_data(json.loads(json.dumps(mesh.to_data()))) == mesh


def test_dynamic_persistent_policy_still_has_no_fusion_key():
    node = fm.Node('op', 'math.add', (), fm.tensor_type('float32', (16,)), metadata={
        'execution_mesh': '*', 'kernel_program_kind': fm.KernelProgramKind.PERSISTENT,
    })
    assert node.metadata['kernel_program_kind'] == 'persistent'
    assert fm.node_execution_mesh(node).fusion_key is None


@pytest.mark.parametrize('value', [(), (0,), (-1,), (True,), (1.5,), ('2',)])
def test_invalid_fixed_dimensions_are_rejected(value):
    with pytest.raises(IRSchemaError):
        fm.ExecutionMesh.fixed(value)


@pytest.mark.parametrize('value', [
    {'kind': 'dynamic', 'shape': [8]},
    {'kind': 'dynamic', 'axes': ['x']},
    {'kind': 'dynamic', 'mapping': 'row_major'},
    {'kind': 'dynamic', 'scope': 'device_group'},
    {'kind': 'fixed', 'shape': [8], 'axes': ['x', 'y']},
    {'kind': 'fixed', 'shape': [8, 8], 'axes': ['x', 'x']},
    {'kind': 'fixed', 'shape': [8], 'axes': [[]]},
    {'kind': 'fixed', 'shape': [8], 'unexpected': True},
    {'kind': 'dynamic', 'schema': 'flagmega.execution-mesh/v99'},
])
def test_bad_execution_mesh_metadata_fails_construction(value):
    with pytest.raises(IRSchemaError):
        fm.Node('op', 'math.add', (), fm.tensor_type('float32', (16,)),
                metadata={'execution_mesh': value})


def test_node_round_trip_and_replacement_preserve_policy():
    mesh = fm.ExecutionMesh.fixed((4, 16), axes=('x', 'y'))
    node = fm.Node('op', 'math.add', (), fm.tensor_type('float32', (16,)),
                   metadata={'execution_mesh': mesh, 'other': 'kept'})
    loaded = fm.Node.from_data(json.loads(json.dumps(node.to_data())))
    assert loaded == node
    assert fm.node_execution_mesh(loaded) == mesh
    replaced = replace(node, inputs=('x', 'y'))
    assert replaced.metadata is node.metadata
    assert fm.node_execution_mesh(replaced) == mesh


def test_mesh_annotation_does_not_modify_S_or_E_or_storage():
    placement = fm.Placement((4, 8), 'db', 'db')
    tensor = fm.tensor_type('float32', (32,))
    value_type = fm.DistributedType(tensor, (fm.SBP.split_contiguous((0,)),), placement)
    node = fm.Node('op', 'math.add', (), value_type)
    annotated = fm.with_execution_mesh(node, '*')
    assert annotated.type is value_type
    assert node.metadata == {}
    assert fm.node_execution_mesh(annotated) == fm.ExecutionMesh.dynamic()
    assert annotated.type.placement == placement
    assert annotated.type.exclusive is None


def test_unannotated_node_serialization_remains_unchanged():
    node = fm.Node('op', 'math.add', (), fm.tensor_type('float32', (16,)))
    assert node.to_data()['metadata'] == {}
    assert fm.node_execution_mesh(node) is None


def test_module_default_and_local_override_survive_python_checkpoint(tmp_path):
    mesh = fm.ExecutionMesh.fixed((8, 16))
    builder = fm.IRBuilder(dialect='high_level', stage='imported',
                           metadata={'execution_mesh': mesh})
    x = builder.var('x', fm.tensor_type('float32', (16,)), id='x')
    node = builder.call('math.add', (x, x), x.type, id='op',
                        metadata={'execution_mesh': '*', 'kernel_program_kind': fm.KernelProgramKind.PERSISTENT})
    builder.function('main', (x,), (node,))
    module = builder.build(entry='main')
    checkpoint = fm.emit_module(module, tmp_path / 'meshes.py')
    loaded = fm.load_module(checkpoint)
    assert loaded.semantic_hash == module.semantic_hash
    assert effective_execution_mesh(loaded, loaded.node_map['op']) == fm.ExecutionMesh.dynamic()
    assert effective_execution_mesh(loaded, loaded.node_map['x']) == mesh
    assert loaded.node_map['op'].metadata['kernel_program_kind'] == 'persistent'


def test_program_kind_does_not_replace_existing_participation_kind():
    assert fm.KernelExecutionKind.LOCAL_SHARD.value == 'local_shard'
    assert fm.KernelProgramKind.PERSISTENT.value == 'persistent'
    with pytest.raises(IRSchemaError):
        fm.Node('op', 'math.add', (), fm.tensor_type('float32', (16,)),
                metadata={'execution_mesh': '*', 'kernel_program_kind': 'unknown'})
