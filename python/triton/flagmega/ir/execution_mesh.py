# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Device-local execution mesh, separate from value placement and kernel kind."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum

from triton.flagmega.errors import IRSchemaError

EXECUTION_MESH_SCHEMA = 'flagmega.execution-mesh/v1'


class ExecutionMeshKind(str, Enum):
    FIXED = 'fixed'
    DYNAMIC = 'dynamic'


class KernelProgramKind(str, Enum):
    """Algorithm lifetime; existing KernelExecutionKind describes participation."""
    ORDINARY = 'ordinary'
    PERSISTENT = 'persistent'


@dataclass(frozen=True)
class ExecutionMesh:
    kind: ExecutionMeshKind
    shape: tuple[int, ...] = ()
    axes: tuple[str, ...] = ()
    mapping: str | None = None
    scope: str = 'device'

    def __post_init__(self) -> None:
        try:
            kind = ExecutionMeshKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise IRSchemaError(f'Unknown execution mesh kind {self.kind!r}.') from exc
        if self.scope != 'device':
            raise IRSchemaError('Execution mesh scope must be device; device placement is separate.')
        if isinstance(self.shape, (str, bytes)) or not isinstance(self.shape, Sequence):
            raise IRSchemaError('Execution mesh shape must be a sequence of positive integers.')
        if isinstance(self.axes, (str, bytes)) or not isinstance(self.axes, Sequence):
            raise IRSchemaError('Execution mesh axes must be a sequence of axis names.')
        shape, axes = tuple(self.shape), tuple(self.axes)
        mapping = self.mapping
        if kind is ExecutionMeshKind.DYNAMIC:
            if shape or axes or mapping is not None:
                raise IRSchemaError('Dynamic execution mesh has no fixed shape, axes, or participant mapping.')
        else:
            if not shape or any(type(value) is not int or value <= 0 for value in shape):
                raise IRSchemaError('Fixed execution mesh dimensions must be positive integers.')
            axes = axes or tuple(f'b{i}' for i in range(len(shape)))
            if (len(axes) != len(shape)
                    or any(not isinstance(axis, str) or not axis or axis.strip() != axis for axis in axes)
                    or len(set(axes)) != len(axes)):
                raise IRSchemaError('Fixed execution mesh axes must uniquely name every dimension.')
            mapping = 'row_major' if mapping is None else mapping
            if not isinstance(mapping, str) or not mapping or mapping.strip() != mapping:
                raise IRSchemaError('Fixed execution mesh participant mapping must be named.')
        object.__setattr__(self, 'kind', kind)
        object.__setattr__(self, 'shape', shape)
        object.__setattr__(self, 'axes', axes)
        object.__setattr__(self, 'mapping', mapping)

    @classmethod
    def fixed(cls, shape: int | Sequence[int], *, axes: Sequence[str] = (),
              mapping: str = 'row_major') -> ExecutionMesh:
        shape = (shape,) if type(shape) is int else shape
        return cls(ExecutionMeshKind.FIXED, shape, axes, mapping)

    @classmethod
    def dynamic(cls) -> ExecutionMesh:
        return cls(ExecutionMeshKind.DYNAMIC)

    @property
    def fixed_size(self) -> int | None:
        if self.kind is ExecutionMeshKind.DYNAMIC:
            return None
        result = 1
        for extent in self.shape:
            result *= extent
        return result

    @property
    def fusion_key(self) -> tuple | None:
        """Necessary mesh compatibility; effects/resources still require proof.

        Dynamic never supplies a fusion key, including when its selected
        implementation is persistent or two runtime grids happen to match.
        """
        if self.kind is ExecutionMeshKind.DYNAMIC:
            return None
        return self.scope, self.shape, self.axes, self.mapping

    def to_data(self) -> dict[str, object]:
        data = {'schema': EXECUTION_MESH_SCHEMA, 'kind': self.kind.value, 'scope': self.scope}
        if self.kind is ExecutionMeshKind.FIXED:
            data.update(shape=list(self.shape), axes=list(self.axes), mapping=self.mapping)
        return data

    @classmethod
    def from_data(cls, data: Mapping) -> ExecutionMesh:
        if not isinstance(data, Mapping) or set(data) - {'schema', 'kind', 'scope', 'shape', 'axes', 'mapping'}:
            raise IRSchemaError('Execution mesh has unknown fields or is not a mapping.')
        if data.get('schema', EXECUTION_MESH_SCHEMA) != EXECUTION_MESH_SCHEMA:
            raise IRSchemaError('Unsupported execution mesh schema.')
        return cls(data.get('kind'), data.get('shape', ()), data.get('axes', ()),
                   data.get('mapping'), data.get('scope', 'device'))


def execution_mesh(value) -> ExecutionMesh | None:
    """Normalize an explicit mesh; None means inherit the enclosing policy."""
    if value is None or isinstance(value, ExecutionMesh):
        return value
    if isinstance(value, Mapping):
        return ExecutionMesh.from_data(value)
    if isinstance(value, str):
        if value == '*':
            return ExecutionMesh.dynamic()
        raise IRSchemaError('The dynamic execution mesh spelling is "*".')
    if type(value) is int or isinstance(value, Sequence):
        return ExecutionMesh.fixed(value)
    raise IRSchemaError(f'Unsupported execution mesh value {value!r}.')


def normalize_execution_mesh_metadata(metadata: Mapping) -> Mapping:
    """Canonicalize execution policy without changing unannotated IR identity."""
    if 'execution_mesh' not in metadata and 'kernel_program_kind' not in metadata:
        return metadata
    updates = {}
    if metadata.get('execution_mesh') is not None:
        mesh = execution_mesh(metadata['execution_mesh'])
        data = mesh.to_data()
        if mesh.kind is ExecutionMeshKind.FIXED:
            data.update(shape=mesh.shape, axes=mesh.axes)
        if metadata['execution_mesh'] != data:
            updates['execution_mesh'] = data
    if metadata.get('kernel_program_kind') is not None:
        value = metadata['kernel_program_kind']
        try:
            kind = KernelProgramKind(value)
        except (TypeError, ValueError) as exc:
            raise IRSchemaError(f'Unknown kernel program kind {value!r}.') from exc
        if type(value) is not str or value != kind.value:
            updates['kernel_program_kind'] = kind.value
    return {**metadata, **updates} if updates else metadata


def node_execution_mesh(node) -> ExecutionMesh | None:
    return execution_mesh(node.metadata.get('execution_mesh'))


def effective_execution_mesh(module, node) -> ExecutionMesh | None:
    mesh = node_execution_mesh(node)
    return mesh if mesh is not None else execution_mesh(module.metadata.get('execution_mesh'))


def placement_for_execution_mesh(placement, mesh):
    """Choose an op's data-search domain without using runtime CTA counts.

    Fixed domains use the selected row-major local participants. Dynamic data
    uses unit block axes, hence B within the chip; the implementation's task
    mapping and runtime grid are a separate contract. Keeping those unit axes
    avoids changing existing device-axis indices in the distribution search.
    """
    from triton.flagmega.ir.distributed_type import Placement
    mesh = execution_mesh(mesh)
    if mesh is None:
        return placement
    if mesh.mapping not in (None, 'row_major'):
        raise IRSchemaError(f'No reviewed data mapping for execution mesh {mesh.mapping!r}.')
    block_axes = tuple(i for i, level in enumerate(placement.hierarchy_levels) if level == 'b')
    if mesh.kind is ExecutionMeshKind.DYNAMIC:
        shape = (1,) * len(block_axes)
    else:
        shape = mesh.shape
    if len(shape) == len(block_axes):
        hierarchy = list(placement.hierarchy)
        for axis, extent in zip(block_axes, shape, strict=True):
            hierarchy[axis] = extent
        return Placement(tuple(hierarchy), placement.name, placement.hierarchy_levels)
    # Rank-changing fixed meshes append their local axes after the preserved
    # device/node axes. Names describe data coordinates, independently of the
    # complete execution/fusion identity carried by ExecutionMesh.axes.
    retained = tuple(i for i in range(placement.rank) if i not in block_axes)
    names = ''.join(placement.name[i] for i in retained)
    available = tuple(c for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' if c not in names)
    if len(shape) > len(available):
        raise IRSchemaError('Execution mesh exceeds the named data-placement axis capacity.')
    return Placement(tuple(placement.hierarchy[i] for i in retained) + shape,
                     names + ''.join(available[:len(shape)]),
                     ''.join(placement.hierarchy_levels[i] for i in retained) + 'b' * len(shape))


def with_execution_mesh(value, mesh):
    """Annotate an immutable Node or IRModule without changing value types."""
    from triton.flagmega.ir.model import IRModule, Node
    if not isinstance(value, (Node, IRModule)):
        raise IRSchemaError('Execution mesh annotations require a Node or IRModule.')
    return replace(value, metadata={**value.metadata, 'execution_mesh': execution_mesh(mesh)})
