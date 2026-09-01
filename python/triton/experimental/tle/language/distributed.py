# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# flagtree tle
from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
from itertools import product
from typing import Any, Iterable, Mapping, Sequence, List, Tuple, Union, Optional, Dict, TYPE_CHECKING
from enum import Enum
import triton.language.core as tl

try:
    from triton._C.libtriton.tle import attr, utils
except ImportError:
    pass
if TYPE_CHECKING:
    from . import TLESemantic

Axis = Tuple[str, int]
AxesLike = Union[int, List[Axis]]


def _prod(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _as_positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{label} must be int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{label} must be > 0, got {value}")
    return value


def _parse_src_arg(builder, src, index=0):
    try:
        from triton.runtime import DistributedRtContext
        src = tl._unwrap_if_constexpr(src)
        if isinstance(src, DistributedRtContext):
            return builder.get_int64(src[index])
        elif src:
            return src.handle
        else:
            return None
    except Exception:
        return src.handle


# Get the current device id
@tl.builtin
def _get_local_rank(device_dptr, _semantic: TLESemantic | None = None, ret_dtype=tl.int32):
    builder = _semantic.builder
    ret_ir_ty = ret_dtype.to_ir(builder)
    ptr = _parse_src_arg(builder, device_dptr, 1)
    result = builder.get_device_id(ret_ir_ty, ptr)
    return tl.tensor(result, ret_dtype)


# Get the current world rank
@tl.builtin
def _get_world_rank(device_dptr, _semantic=None, ret_dtype=tl.int32):
    builder = _semantic.builder
    ret_ir_ty = ret_dtype.to_ir(builder)
    ptr = _parse_src_arg(builder, device_dptr, 1)
    result = builder.get_world_rank(ret_ir_ty, ptr)
    return tl.tensor(result, ret_dtype)


# The number of devices on the current node
@tl.builtin
def n_pes(dev_mem_ptr, _semantic: TLESemantic | None = None, ret_dtype=tl.int32):
    builder = _semantic.builder
    ret_ir_ty = ret_dtype.to_ir(builder)
    ptr = _parse_src_arg(builder, dev_mem_ptr, 1)
    result = builder.get_n_pes(ret_ir_ty, ptr)
    return tl.tensor(result, ret_dtype)


class BarrierKind(str, Enum):
    ARRIVE = "arrive"
    WAIT = "wait"
    SYNC = "sync"


class MemoryOrder(str, Enum):
    RELAXED = "relaxed"
    ACQUIRE = "acquire"
    RELEASE = "release"
    ACQ_REL = "acqrel"


class GroupKind(str, Enum):
    THREAD = "thread"
    WARP = "warp"
    BLOCK = "block"
    TILE_SPAN = "tile_span"
    LANES = "lanes"
    GRID = "grid"


_SIGNAL_SPACE_TO_TEAM_KIND = {
    "intra": 0,
    "intra_node": 0,
    "device": 0,
    "inter": 1,
    "inter_node": 1,
    "node": 1,
    "world": 2,
}


def _normalize_signal_scalar(value, name: str, dtype: tl.dtype, _semantic) -> tl.tensor:
    if dtype is None or not dtype.is_int() or dtype.is_bool():
        raise TypeError(f"{name}: target dtype must be a non-bool integer, got {dtype}")

    value = tl._unwrap_if_constexpr(value)
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer scalar, got bool")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
        max_value = dtype.get_int_max_value()
        if value > max_value:
            raise ValueError(f"{name} {value} exceeds {dtype} range [0, {max_value}]")
    value_tensor = value if isinstance(value, tl.tensor) else _semantic.to_tensor(value)
    if not value_tensor.dtype.is_int() or value_tensor.dtype.is_bool():
        raise TypeError(f"{name} must be an integer scalar, got {value_tensor.dtype}")
    if value_tensor.shape != ():
        raise ValueError(f"{name} must be scalar, got shape {value_tensor.shape}")
    if value_tensor.dtype != dtype:
        value_tensor = tl.cast(
            value_tensor,
            dtype,
            _semantic=_semantic,
        )
    return value_tensor


@tl.builtin
def signal(
    device_dptr,
    peer,
    slot_id,
    value: int | None = None,
    op: str | attr.SignalOpKind = "inc",
    space: str | attr.FlagCXTeamKind = "intra_node",
    group_kind: str | GroupKind | attr.FlagCXCoopKind = GroupKind.BLOCK,
    context_idx: int = 0,
    _semantic=None,
):
    """Atomically update a synchronization slot owned by a remote FlagCX peer.

    ``op="inc"`` increments the selected signal slot by one. ``op="add"``
    adds ``value`` to the selected signal slot.
    The primitive only sends a signal; it neither transfers data nor
    waits for completion on the receiving peer.

    ``space`` selects the FlagCX team (``intra_node``, ``inter_node``, or
    ``world``), while ``peer`` is a rank within that team. ``context_idx``
    selects a pre-allocated FlagCX network context. ``slot_id`` selects the
    signal slot to update.

    For ``group_kind="block"`` (the default), every thread in the CTA must
    execute this operation convergently; the group collectively emits one
    remote update. FlagCX S-Path signal supports thread, warp, and block groups.
    """
    builder = _semantic.builder
    if not hasattr(builder, "create_signal"):
        raise NotImplementedError("tle.signal requires rebuilt TLE builder support")

    signal_op = attr.SignalOpKind.from_str(str(tl._unwrap_if_constexpr(op)).lower())
    if signal_op is None:
        raise ValueError(f"op must be 'inc' or 'add', got {signal_op!r}")

    signal_space = str(tl._unwrap_if_constexpr(space)).lower()
    if signal_space not in _SIGNAL_SPACE_TO_TEAM_KIND:
        expected = "intra_node, inter_node, or world"
        raise ValueError(f"space must be {expected}, got {signal_space!r}")
    signal_space = attr.FlagCXTeamKind.from_int(_SIGNAL_SPACE_TO_TEAM_KIND[signal_space])

    group_kind = tl._unwrap_if_constexpr(group_kind)
    group_kind = group_kind.value if isinstance(group_kind, GroupKind) else str(group_kind).lower()
    group_kind = attr.FlagCXCoopKind.from_str(group_kind)
    if group_kind is None:
        expected = "thread, warp, or block"
        raise ValueError(f"group_kind must be {expected}, got {group_kind!r}")

    context_idx = tl._unwrap_if_constexpr(context_idx)
    if not isinstance(context_idx, int):
        raise TypeError(f"context_idx must be a compile-time int, got {type(context_idx).__name__}")
    if context_idx < 0 or context_idx > 0x7FFFFFFF:
        raise ValueError(f"context_idx must be in int32 range, got {context_idx}")

    peer_tensor = _normalize_signal_scalar(peer, "peer", tl.int32, _semantic)
    slot_tensor = _normalize_signal_scalar(slot_id, "slot_id", tl.uint32, _semantic)
    value_value = value.value if isinstance(value, tl.constexpr) else value
    value_tensor = (_normalize_signal_scalar(value_value, "value", tl.uint64, _semantic)
                    if value_value is not None else None)

    utils.verify_signal(signal_op, None if value_tensor is None else value_tensor.handle)

    comm = _parse_src_arg(builder, device_dptr, 1)
    builder.create_signal(
        comm,
        peer_tensor.handle,
        slot_tensor.handle,
        None if value_tensor is None else value_tensor.handle,
        signal_op,
        signal_space,
        group_kind,
        context_idx,
    )
    return None


@dataclass
class MeshConfig:
    """
    Represents a hierarchical mesh topology configuration.

    Fields:
        node:          Inter-node topology (e.g., multi-host layout)
        device:        Intra-node device topology (e.g., GPUs per node)
        block_cluster: Cluster-level partitioning within a device
        block:         Finest-grained block-level partitioning

    Fields set to None are ignored when exporting.
    """
    node: Optional[AxesLike] = None
    device: Optional[AxesLike] = None
    block_cluster: Optional[AxesLike] = None
    block: Optional[AxesLike] = None

    def to_dict(self) -> Dict[str, Any]:
        _dict = asdict(self)
        _dict = {k: v for k, v in _dict.items() if v is not None}
        return _dict

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v}" for k, v in self.to_dict().items())
        return f"MeshConfig({fields})"


class device_mesh:
    """
    Logical view of a physical device topology.
    """

    def __init__(
        self,
        topology: Mapping[str, Any] | MeshConfig | None = None,
        *,
        _shape: Sequence[int] | None = None,
        _dim_names: Sequence[str] | None = None,
        _physical_ids: Sequence[int] | None = None,
        _launch_shape: Sequence[int] | None = None,
        _launch_dim_names: Sequence[str] | None = None,
    ):
        if topology is None:
            if _shape is None or _dim_names is None or _physical_ids is None:
                raise ValueError("internal mesh constructor requires shape/names/physical ids")
            self._shape = tuple(_shape)
            self._dim_names = tuple(_dim_names)
            self._physical_ids = tuple(_physical_ids)
            self._launch_shape = tuple(_launch_shape if _launch_shape is not None else _shape)
            self._launch_dim_names = tuple(_launch_dim_names if _launch_dim_names is not None else _dim_names)
            return

        if not isinstance(topology, Mapping) and not isinstance(topology, MeshConfig):
            raise TypeError(f"topology must be a Mapping or MeshConfig, got {type(topology).__name__}")
        if not topology:
            raise ValueError("topology cannot be empty")
        if isinstance(topology, MeshConfig):
            topology = topology.to_dict()
        shape = []
        dim_names = []
        for level_name, level_desc in topology.items():
            if not isinstance(level_name, str) or not level_name:
                raise ValueError(f"invalid topology level name: {level_name!r}")
            level_shape, level_names = self._parse_level(level_name, level_desc)
            shape.extend(level_shape)
            dim_names.extend(level_names)

        if len(set(dim_names)) != len(dim_names):
            raise ValueError(f"dimension names must be unique, got {dim_names}")

        self._shape = tuple(shape)
        self._dim_names = tuple(dim_names)
        self._physical_ids = tuple(range(_prod(shape)))
        self._launch_shape = self._shape
        self._launch_dim_names = self._dim_names

    @staticmethod
    def _parse_level(level_name: str, level_desc: Any) -> tuple[list[int], list[str]]:
        if isinstance(level_desc, int):
            return [_as_positive_int(level_desc, level_name)], [level_name]
        if not isinstance(level_desc, (tuple, list)):
            raise TypeError(f"topology[{level_name!r}] must be int or list/tuple of (name, size), "
                            f"got {type(level_desc).__name__}")
        if not level_desc:
            raise ValueError(f"topology[{level_name!r}] cannot be empty")

        shape = []
        names = []
        for item in level_desc:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError(f"topology[{level_name!r}] entries must be (name, size), got {item!r}")
            dim_name, dim_size = item
            if not isinstance(dim_name, str) or not dim_name:
                raise ValueError(f"invalid dimension name in {level_name!r}: {dim_name!r}")
            shape.append(_as_positive_int(dim_size, f"{level_name}.{dim_name}"))
            names.append(dim_name)
        return shape, names

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def dim_names(self) -> tuple[str, ...]:
        return self._dim_names

    @property
    def physical_ids(self) -> tuple[int, ...]:
        return self._physical_ids

    @property
    def launch_shape(self) -> tuple[int, ...]:
        return self._launch_shape

    @property
    def launch_dim_names(self) -> tuple[str, ...]:
        return self._launch_dim_names

    @property
    def size(self) -> int:
        return len(self._physical_ids)

    def flatten(self) -> "device_mesh":
        return self.reshape(self.size)

    def reshape(self, *shape: int | Sequence[int]) -> "device_mesh":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            new_shape = tuple(shape[0])
        else:
            new_shape = tuple(shape)
        if not new_shape:
            raise ValueError("new shape cannot be empty")
        new_shape = tuple(_as_positive_int(v, "shape dimension") for v in new_shape)
        if _prod(new_shape) != self.size:
            raise ValueError(f"cannot reshape mesh of size {self.size} into shape {new_shape}")
        if len(new_shape) == self.ndim:
            new_dim_names = self._dim_names
        elif len(new_shape) == 1:
            new_dim_names = ("flat", )
        else:
            new_dim_names = tuple(f"dim{i}" for i in range(len(new_shape)))
        return device_mesh(
            None,
            _shape=new_shape,
            _dim_names=new_dim_names,
            _physical_ids=self._physical_ids,
            _launch_shape=self._launch_shape,
            _launch_dim_names=self._launch_dim_names,
        )

    def _normalize_key(self, key: Any) -> tuple[Any, ...]:
        if not isinstance(key, tuple):
            key = (key, )

        if any(item is Ellipsis for item in key):
            if key.count(Ellipsis) > 1:
                raise IndexError("an index can only have a single ellipsis")
            ellipsis_pos = key.index(Ellipsis)
            missing = self.ndim - (len(key) - 1)
            if missing < 0:
                raise IndexError("too many indices for device_mesh")
            key = key[:ellipsis_pos] + (slice(None), ) * missing + key[ellipsis_pos + 1:]

        if len(key) > self.ndim:
            raise IndexError("too many indices for device_mesh")

        return key + (slice(None), ) * (self.ndim - len(key))

    def _linear_index(self, coords: Sequence[int]) -> int:
        index = 0
        for coord, dim_size in zip(coords, self._shape):
            index = index * dim_size + coord
        return index

    def __getitem__(self, key: Any) -> "device_mesh":
        key = self._normalize_key(key)
        selected_per_dim: list[list[int]] = []
        keep_dim: list[bool] = []

        for dim_size, dim_key in zip(self._shape, key):
            if isinstance(dim_key, int):
                idx = dim_key + dim_size if dim_key < 0 else dim_key
                if idx < 0 or idx >= dim_size:
                    raise IndexError(f"index {dim_key} out of range for dim size {dim_size}")
                selected_per_dim.append([idx])
                keep_dim.append(False)
            elif isinstance(dim_key, slice):
                indices = list(range(*dim_key.indices(dim_size)))
                if not indices:
                    raise ValueError("empty sub-mesh is not supported")
                selected_per_dim.append(indices)
                keep_dim.append(True)
            else:
                raise TypeError(f"device_mesh indices must be int/slice/ellipsis, got {type(dim_key).__name__}")

        new_shape = tuple(len(indices) for indices, keep in zip(selected_per_dim, keep_dim) if keep)
        new_dim_names = tuple(dim_name for dim_name, keep in zip(self._dim_names, keep_dim) if keep)

        new_physical_ids = []
        for coords in product(*selected_per_dim):
            new_physical_ids.append(self._physical_ids[self._linear_index(coords)])

        return device_mesh(
            None,
            _shape=new_shape,
            _dim_names=new_dim_names,
            _physical_ids=tuple(new_physical_ids),
            _launch_shape=self._launch_shape,
            _launch_dim_names=self._launch_dim_names,
        )

    def __repr__(self):
        return f"DeviceMesh(shape={self._shape}, names={self._dim_names})"


class _BroadcastSpec:

    def __repr__(self) -> str:
        return "B"


B = _BroadcastSpec()


@dataclass(frozen=True)
class S:
    axis: str | Sequence[str]


@dataclass(frozen=True)
class P:
    axis: str | Sequence[str]


def _normalize_axis_group(spec: Any, label: str) -> tuple[str, ...]:
    if spec is None or spec is B:
        return tuple()

    if isinstance(spec, S):
        spec = spec.axis
    if isinstance(spec, P):
        spec = spec.axis

    if isinstance(spec, str):
        if not spec:
            raise ValueError(f"{label} axis name cannot be empty")
        return (spec, )

    if isinstance(spec, (tuple, list)):
        if not spec:
            return tuple()
        axes = []
        for axis in spec:
            if not isinstance(axis, str) or not axis:
                raise ValueError(f"{label} axis name must be non-empty str, got {axis!r}")
            axes.append(axis)
        if len(set(axes)) != len(axes):
            raise ValueError(f"{label} axis names must be unique, got {axes}")
        return tuple(axes)

    raise TypeError(f"{label} axis spec must be str/list/tuple/S/P/B, got {type(spec).__name__}")


def _normalize_partial_specs(partial: Any) -> tuple[str, ...]:
    if partial is None:
        return tuple()
    if isinstance(partial, (str, S, P)):
        partial = [partial]
    if not isinstance(partial, (tuple, list)):
        raise TypeError(f"partial must be a list/tuple, got {type(partial).__name__}")

    axes = []
    for item in partial:
        axes.extend(_normalize_axis_group(item, "partial"))
    if len(set(axes)) != len(axes):
        raise ValueError(f"partial axes must be unique, got {axes}")
    return tuple(axes)


@dataclass(frozen=True)
class ShardingSpec:
    mesh: device_mesh
    split: tuple[tuple[str, ...], ...]
    partial: tuple[str, ...]
    broadcast: tuple[str, ...]

    def axis_state(self, axis: str) -> str:
        if axis in self.partial:
            return "P"
        for split_axes in self.split:
            if axis in split_axes:
                return "S"
        return "B"


@dataclass(frozen=True)
class ShardedTensor:
    handle: Any
    sharding: ShardingSpec
    shape: tuple[int, ...] | None = None


def sharding(
    mesh: device_mesh,
    split: Sequence[Any] | None = None,
    partial: Sequence[Any] | None = None,
) -> ShardingSpec:
    """
    Construct a sharding spec bound to a device mesh.

    This is annotation metadata today. Communication lowering is added in later
    phases.
    """
    if not isinstance(mesh, device_mesh):
        raise TypeError(f"mesh must be device_mesh, got {type(mesh).__name__}")

    split_specs: list[tuple[str, ...]] = []
    if split is None:
        split = tuple()
    if not isinstance(split, (tuple, list)):
        raise TypeError(f"split must be a list/tuple, got {type(split).__name__}")
    for split_item in split:
        split_specs.append(_normalize_axis_group(split_item, "split"))

    partial_axes = _normalize_partial_specs(partial)

    split_axes = [axis for split_item in split_specs for axis in split_item]
    if len(set(split_axes)) != len(split_axes):
        raise ValueError(f"split axes must be unique across tensor dims, got {split_axes}")

    split_set = set(split_axes)
    partial_set = set(partial_axes)

    unknown = [axis for axis in split_axes + list(partial_axes) if axis not in mesh.dim_names]
    if unknown:
        raise ValueError(f"unknown mesh axis names: {unknown}; mesh axes are {mesh.dim_names}")

    overlap = split_set.intersection(partial_set)
    if overlap:
        raise ValueError(f"mesh axis cannot be both split and partial: {sorted(overlap)}")

    broadcast = tuple(axis for axis in mesh.dim_names if axis not in split_set and axis not in partial_set)
    return ShardingSpec(
        mesh=mesh,
        split=tuple(split_specs),
        partial=tuple(axis for axis in mesh.dim_names if axis in partial_set),
        broadcast=broadcast,
    )


def make_sharded_tensor(
    handle: Any,
    sharding: ShardingSpec,
    shape: Sequence[int] | None = None,
) -> ShardedTensor:
    if not isinstance(sharding, ShardingSpec):
        raise TypeError(f"sharding must be ShardingSpec, got {type(sharding).__name__}")
    normalized_shape = None
    if shape is not None:
        if not isinstance(shape, (tuple, list)):
            raise TypeError(f"shape must be list/tuple, got {type(shape).__name__}")
        normalized_shape = tuple(_as_positive_int(v, "tensor shape") for v in shape)
        if sharding.split and len(sharding.split) != len(normalized_shape):
            raise ValueError(f"split rank ({len(sharding.split)}) must match tensor rank ({len(normalized_shape)})")
    return ShardedTensor(handle=handle, sharding=sharding, shape=normalized_shape)


def reshard(tensor: ShardedTensor, spec: ShardingSpec) -> ShardedTensor:
    """
    M4 entrypoint. Deferred by roadmap priority.
    """
    raise NotImplementedError("reshard is deferred to M4")


def _shape_to_cluster_dims(shape: Sequence[int]) -> tuple[int, int, int]:
    if not shape:
        return (1, 1, 1)
    dims = tuple(int(v) for v in shape)
    if len(dims) == 1:
        return (dims[0], 1, 1)
    if len(dims) == 2:
        return (dims[0], dims[1], 1)
    if len(dims) == 3:
        return dims
    return (_prod(dims), 1, 1)


def _mesh_to_cluster_dims(mesh: device_mesh) -> tuple[int, int, int]:
    # Prefer explicit cluster axes, then block axes, then fallback to full mesh.
    cluster_axes = [size for name, size in zip(mesh.launch_dim_names, mesh.launch_shape) if "cluster" in name]
    if not cluster_axes:
        cluster_axes = [size for name, size in zip(mesh.launch_dim_names, mesh.launch_shape) if "block" in name]
    if not cluster_axes:
        cluster_axes = list(mesh.launch_shape)
    return _shape_to_cluster_dims(cluster_axes)


def _mesh_has_cluster_axes(mesh: device_mesh) -> bool:
    return any("cluster" in name for name in mesh.dim_names)


def _mesh_has_block_axes(mesh: device_mesh) -> bool:
    return any("block" in name for name in mesh.dim_names)


def _mesh_uses_grid_barrier(mesh: device_mesh) -> bool:
    # Heuristic for auto mode:
    # - explicit cluster axes in the barrier mesh => cluster/submesh barrier
    # - block-only axes in the barrier mesh => grid barrier
    # - empty mesh dims (scalar) fallback to launch mesh naming
    if mesh.dim_names:
        return (not _mesh_has_cluster_axes(mesh)) and _mesh_has_block_axes(mesh)
    return ((not any("cluster" in name for name in mesh.launch_dim_names))
            and any("block" in name for name in mesh.launch_dim_names))


@dataclass(frozen=True)
class _BarrierGroupDescriptor:
    kind: str
    rank: int
    shape: tuple[int, ...]
    axes: tuple[int, ...]
    mask: tuple[int, ...]


def _infer_submesh_barrier_group(
    mesh: device_mesh,
    cluster_dims: Sequence[int],
) -> _BarrierGroupDescriptor | None:
    cluster_size = _prod(cluster_dims)
    if mesh.size == cluster_size:
        return None
    if mesh.size > cluster_size:
        raise ValueError(f"mesh size ({mesh.size}) exceeds inferred cluster size ({cluster_size})")

    launch_size = _prod(mesh.launch_shape)
    if launch_size != cluster_size:
        raise NotImplementedError(
            "sub-mesh distributed_barrier currently requires launch mesh domain "
            f"to match inferred cluster size; launch_size={launch_size}, cluster_size={cluster_size}")

    if not mesh.dim_names:
        raise NotImplementedError("scalar sub-mesh barrier is not implemented yet; provide at least one sliced axis")

    launch_name_to_axis = {name: i for i, name in enumerate(mesh.launch_dim_names)}
    if any(name not in launch_name_to_axis for name in mesh.dim_names):
        raise NotImplementedError("sub-mesh barrier currently supports slicing-derived meshes with "
                                  "axis names inherited from launch mesh")

    axes = tuple(int(launch_name_to_axis[name]) for name in mesh.dim_names)
    if len(set(axes)) != len(axes):
        raise ValueError(f"invalid subgroup axes (duplicate launch axes): {axes}")

    shape = tuple(int(v) for v in mesh.shape)
    if not shape or any(v <= 0 for v in shape):
        raise ValueError(f"invalid subgroup shape inferred from mesh: {shape}")

    mask = tuple(int(v) for v in mesh.physical_ids)
    if not mask:
        raise ValueError("sub-mesh barrier group mask cannot be empty")
    if any(v < 0 or v >= cluster_size for v in mask):
        raise ValueError("sub-mesh barrier group mask contains out-of-range cluster member ids: "
                         f"mask={mask}, cluster_size={cluster_size}")

    return _BarrierGroupDescriptor(
        kind="submesh",
        rank=len(shape),
        shape=shape,
        axes=axes,
        mask=mask,
    )


def _apply_mesh_cluster_launch(mesh: device_mesh, _semantic: TLESemantic | None) -> tuple[int, int, int]:
    cluster_dims = _mesh_to_cluster_dims(mesh)
    options = getattr(_semantic.builder, "options", None)
    if options is None:
        return cluster_dims

    num_ctas = int(getattr(options, "num_ctas", 1))
    if num_ctas != 1:
        raise ValueError("mesh-driven cluster launch requires num_ctas=1; cluster size is inferred from mesh")

    existing = tuple(getattr(options, "cluster_dims", (1, 1, 1)))
    if existing != (1, 1, 1) and existing != cluster_dims:
        raise ValueError(f"conflicting cluster_dims: existing={existing}, inferred_from_mesh={cluster_dims}")
    object.__setattr__(options, "cluster_dims", cluster_dims)
    return cluster_dims


def _apply_mesh_grid_launch(mesh: device_mesh, _semantic: TLESemantic | None) -> None:
    options = getattr(_semantic.builder, "options", None)
    if options is None:
        return

    num_ctas = int(getattr(options, "num_ctas", 1))
    if num_ctas != 1:
        raise ValueError("mesh-driven grid distributed_barrier requires num_ctas=1")

    cluster_dims = tuple(getattr(options, "cluster_dims", (1, 1, 1)))
    if cluster_dims != (1, 1, 1):
        raise ValueError("mesh-driven grid distributed_barrier requires cluster_dims=(1, 1, 1)")
    object.__setattr__(options, "launch_cooperative_grid", True)


def _resolve_launch_axis(mesh: device_mesh, axis: str | int) -> int:
    if isinstance(axis, int):
        ndim = len(mesh.launch_shape)
        axis_idx = axis + ndim if axis < 0 else axis
        if axis_idx < 0 or axis_idx >= ndim:
            raise IndexError(f"axis index {axis} out of range for launch ndim {ndim}")
        return axis_idx

    if isinstance(axis, str):
        if axis not in mesh.launch_dim_names:
            raise ValueError(f"unknown mesh axis {axis!r}; available launch axes: {mesh.launch_dim_names}")
        return mesh.launch_dim_names.index(axis)

    raise TypeError(f"axis must be int or str, got {type(axis).__name__}")


@tl.builtin
def shard_id(
    mesh: device_mesh,
    axis: str | int,
    device_dptr=None,
    _semantic: TLESemantic | None = None,
):
    """
    Return current shard coordinate on the given launch mesh axis.

    `axis` can be axis name (`str`) or axis index (`int`, supports negative).
    `device` returns the intra-node rank; `node` returns the inter-node rank.
    The returned value is a scalar int32 tensor.
    """
    mesh = tl._unwrap_if_constexpr(mesh)
    axis = tl._unwrap_if_constexpr(axis)

    if axis in ("device", "node") and device_dptr is None:
        raise ValueError(f"device_dptr is required for axis {axis!r}")

    if axis == "device":
        return _get_local_rank(device_dptr, _semantic=_semantic, ret_dtype=tl.int32)
    if axis == "node":
        world_rank = _get_world_rank(device_dptr, _semantic=_semantic, ret_dtype=tl.int32)
        local_world_size = n_pes(device_dptr, _semantic=_semantic, ret_dtype=tl.int32)
        return _semantic.floordiv(world_rank, local_world_size)

    if not isinstance(mesh, device_mesh):
        raise TypeError(f"mesh must be device_mesh, got {type(mesh).__name__}")
    axis_idx = _resolve_launch_axis(mesh, axis)
    launch_shape = tuple(int(v) for v in mesh.launch_shape)
    launch_size = _prod(launch_shape)
    if launch_size <= 0:
        raise ValueError(f"invalid launch mesh shape: {launch_shape}")

    _apply_mesh_cluster_launch(mesh, _semantic)
    linear = tl.program_id(0, _semantic=_semantic)
    if launch_size > 1:
        linear = _semantic.mod(linear, launch_size)

    stride = _prod(launch_shape[axis_idx + 1:]) if axis_idx + 1 < len(launch_shape) else 1
    coord = linear
    if stride > 1:
        coord = _semantic.floordiv(coord, stride)
    dim = launch_shape[axis_idx]
    if dim > 1:
        coord = _semantic.mod(coord, dim)
    return coord


def _parse_device_barrier_args(argType) -> str:
    argType = tl._unwrap_if_constexpr(argType)
    argTypes = (BarrierKind, GroupKind, MemoryOrder)
    if isinstance(argType, argTypes):
        return argType.value
    else:
        return str(argType).lower()


def check_and_handle_device_intra_barrier(space: str = None, device_dptr=None,
                                          barrier_kind: BarrierKind | str = BarrierKind.SYNC,
                                          group_kind: str | GroupKind = GroupKind.BLOCK, index: int | None = 0,
                                          order: MemoryOrder | str | int | None = MemoryOrder.ACQ_REL,
                                          _semantic: TLESemantic | None = None):
    if space and space in ("device", "node"):
        builder = _semantic.builder
        ptr = _parse_src_arg(builder, device_dptr, 1)
        builder.create_distributed_barrier(
            src=ptr,
            barrier_index=index or 0,
            space=_parse_device_barrier_args("device"),
            group_kind=_parse_device_barrier_args(group_kind),
            order=_parse_device_barrier_args(order),
            barrier_kind=_parse_device_barrier_args(barrier_kind),
        )
        return True
    return False


@tl.builtin
def distributed_barrier(mesh: device_mesh | None = None, device_dptr=None, space: str = None,
                        group_kind: str | GroupKind = GroupKind.BLOCK,
                        barrier_kind: BarrierKind | str = BarrierKind.SYNC,
                        order: MemoryOrder | str | int | None = MemoryOrder.ACQ_REL,
                        _semantic: TLESemantic | None = None, index: int | None = 0):
    """
    M3 entrypoint: distributed synchronization primitive.

    - cluster mesh: cluster/submesh synchronization
    - block mesh: cooperative grid synchronization
    """
    mesh = tl._unwrap_if_constexpr(mesh)
    if mesh is not None and not isinstance(mesh, device_mesh):
        raise TypeError(f"mesh must be device_mesh or None, got {type(mesh).__name__}")
    subgroup = None
    use_grid = mesh is not None and _mesh_uses_grid_barrier(mesh)

    if check_and_handle_device_intra_barrier(space=space, device_dptr=device_dptr, barrier_kind=barrier_kind,
                                             group_kind=group_kind, index=index, order=order, _semantic=_semantic):
        return None

    if use_grid:
        if mesh is not None:
            _apply_mesh_grid_launch(mesh, _semantic)
        builder = _semantic.builder
        if not hasattr(builder, "create_distributed_barrier"):
            raise NotImplementedError("grid distributed_barrier requires TLE builder support")
        try:
            builder.create_distributed_barrier("grid", [], [], [])
            return None
        except TypeError as exc:
            raise NotImplementedError(
                "grid distributed_barrier requires rebuilt TLE extension with "
                "group-aware create_distributed_barrier(group_kind, group_shape, group_axes, group_mask)") from exc

    if mesh is not None:
        cluster_dims = _apply_mesh_cluster_launch(mesh, _semantic)
        subgroup = _infer_submesh_barrier_group(mesh, cluster_dims)
    builder = _semantic.builder
    if subgroup is not None:
        if not hasattr(builder, "create_distributed_barrier"):
            raise NotImplementedError("sub-mesh distributed_barrier requires TLE builder support; "
                                      f"inferred subgroup descriptor: rank={subgroup.rank}, "
                                      f"shape={subgroup.shape}, axes={subgroup.axes}, size={len(subgroup.mask)}")
        try:
            builder.create_distributed_barrier(
                subgroup.kind,
                list(subgroup.shape),
                list(subgroup.axes),
                list(subgroup.mask),
            )
            return None
        except TypeError as exc:
            raise NotImplementedError(
                "sub-mesh distributed_barrier requires rebuilt TLE extension with "
                "group-aware create_distributed_barrier(group_kind, group_shape, group_axes, group_mask); "
                f"inferred subgroup descriptor: rank={subgroup.rank}, "
                f"shape={subgroup.shape}, axes={subgroup.axes}, size={len(subgroup.mask)}") from exc
    if hasattr(builder, "create_distributed_barrier"):
        builder.create_distributed_barrier()
    else:
        # Compatibility fallback for environments where the C++ extension
        # has not been rebuilt yet.
        builder.create_barrier()
    return None


def _unwrap_remote_shard_id(shard_id: Any):
    shard_id = tl._unwrap_if_constexpr(shard_id)
    # Tuple literals in JIT functions are represented as tl.tuple even when
    # every coordinate is compile-time constant. Convert them back to a Python
    # tuple so the shared compile-time coordinate path can process them.
    if isinstance(shard_id, tl.tuple):
        shard_id = tuple(shard_id)
    return shard_id


def _normalize_remote_shard_id(
    shard_id: Any,
    scope: device_mesh | None,
) -> int:
    shard_id = _unwrap_remote_shard_id(shard_id)
    scope = tl._unwrap_if_constexpr(scope)

    if isinstance(shard_id, int):
        if shard_id < 0:
            raise ValueError(f"shard_id must be >= 0, got {shard_id}")
        return shard_id

    if not isinstance(shard_id, (tuple, list)):
        raise TypeError(f"shard_id must be int or tuple/list of ints, got {type(shard_id).__name__}")
    if not shard_id:
        raise ValueError("shard_id tuple cannot be empty")
    if not all(isinstance(v, int) for v in shard_id):
        raise TypeError(f"shard_id tuple must contain ints, got {shard_id!r}")

    if scope is None:
        raise ValueError("tuple shard_id requires scope=device_mesh to linearize coordinates")
    if not isinstance(scope, device_mesh):
        raise TypeError(f"scope must be device_mesh when shard_id is tuple, got {type(scope).__name__}")
    if len(shard_id) != scope.ndim:
        raise ValueError(f"tuple shard_id rank mismatch: got {len(shard_id)}, expected {scope.ndim}")

    linear = 0
    for idx, dim in zip(shard_id, scope.shape):
        if idx < 0 or idx >= dim:
            raise ValueError(f"shard_id coordinate {idx} out of range for dim size {dim}")
        linear = linear * dim + idx
    return linear


def _is_buffered_tensor_like(value: Any) -> bool:
    return (not isinstance(value, tl.tensor) and value.__class__.__name__ == "buffered_tensor"
            and hasattr(value, "handle") and hasattr(value, "type"))


def _normalize_compile_time_remote_shard_id(
    shard_id: int | tuple[int, ...] | list[int],
    scope: device_mesh | None,
) -> int:
    linear_shard_id = _normalize_remote_shard_id(shard_id, scope)
    if linear_shard_id > 0x7FFFFFFF:
        raise ValueError(f"linearized shard_id {linear_shard_id} exceeds int32 range")
    return linear_shard_id


def _normalize_runtime_remote_shard_id_tensor(shard_id_tensor: tl.tensor) -> tl.tensor:
    if not shard_id_tensor.dtype.is_int() or shard_id_tensor.dtype.primitive_bitwidth != 32:
        raise TypeError("runtime shard_id must be a scalar int32 tensor/value")
    if shard_id_tensor.shape:
        raise ValueError("runtime shard_id must be scalar (shape=())")
    return shard_id_tensor


def _create_remote_pointers_tensor(
    tensor: tl.tensor,
    shard_id_tensor: tl.tensor,
    _semantic: TLESemantic | None,
    dtype: tl.dtype = None,
    space: str = "cluster",
    offset: int | tl.tensor | None = None,
) -> tl.tensor:
    builder = _semantic.builder

    if not hasattr(builder, "create_remote_pointers"):
        raise RuntimeError("remote pointer lowering requires TLE remote_pointers support in the active Triton build")
    if not isinstance(space, str):
        space = tl._unwrap_if_constexpr(space)
    if space == "device":
        dtype = dtype
    else:
        dtype = tensor.dtype.element_ty if dtype is None else dtype

    remote_ptr_dtype = tl.pointer_type(*{
        "cluster": (dtype, 7),
        "device": (dtype, 1),
    }.get(space))
    if space == 'cluster' and tensor and tensor.type.is_block():
        remote_type = tl.block_type(remote_ptr_dtype, list(tensor.shape)).to_ir(builder)
    else:
        remote_type = remote_ptr_dtype.to_ir(builder)

    if space != "device" and offset is not None:
        raise ValueError(f"offset is only supported for device space remote pointers, got space={space!r}")

    if space == "device":
        if offset is None:
            raise ValueError("device space remote pointers require an offset")
        offset_tensor = offset if isinstance(offset, tl.tensor) else _semantic.to_tensor(offset)
        if not offset_tensor.dtype.is_int():
            raise TypeError(f"offset must be an integer scalar, got {offset_tensor.dtype}")
        # flagcxGetIntraPointerC accepts a single int64_t offset, so only
        # scalar offsets (shape == ()) are allowed. Non-scalar tensor offsets
        # are rejected here before reaching the C++ builder.
        if offset_tensor.shape != ():
            raise ValueError(f"offset must be a scalar integer, got shape {offset_tensor.shape}")
        # Normalize to i64 to match the MLIR verifier expectation.
        # Use tl.cast with explicit _semantic because .to() is a builtin and
        # must receive _semantic when called outside the JIT compiler's
        # automatic injection (e.g. inside this helper).
        if offset_tensor.dtype != tl.int64:
            offset_tensor = tl.cast(offset_tensor, tl.int64, _semantic=_semantic)
        # create_dist_tensor exposes registered memory at index 0 and the
        # communicator at index 1.
        ptr = _parse_src_arg(builder, tensor, 0)
        remote_op = builder.create_remote_pointers(remote_type, ptr, shard_id_tensor.handle, space,
                                                   offset_tensor.handle)
    else:
        remote_op = builder.create_remote_pointers(remote_type, tensor.handle, shard_id_tensor.handle, space)
    if space == "cluster" and tensor and tensor.type.is_block():
        return tl.tensor(remote_op.get_result(0), tl.block_type(remote_ptr_dtype, list(tensor.shape)))
    return tl.tensor(remote_op.get_result(0), remote_ptr_dtype)


def _check_cluster_remote_pointer(tensor: tl.tensor, shard_id: int | tuple[int, ...] | list[int],
                                  scope: device_mesh | None) -> None:
    if not isinstance(tensor, tl.tensor):
        raise TypeError(f"tensor must be tl.tensor, got {type(tensor).__name__}")
    if not tensor.dtype.is_ptr():
        raise TypeError(f"{tensor.dtype}, cluster remote pointer internal path requires a pointer tensor")

    if tensor.dtype.address_space == 7:
        # Pointer is already in cluster-shared space. Preserve compatibility
        if isinstance(shard_id, (int, tuple, list)):
            linear_shard_id = _normalize_compile_time_remote_shard_id(shard_id, scope)
            if linear_shard_id == 0:
                return tensor
            raise ValueError("remote(pointer, ...) on cluster-shared pointers only supports shard_id=0")
        raise ValueError("remote(pointer, ...) on cluster-shared pointers requires compile-time shard_id=0")

    if tensor.dtype.address_space != 3:
        raise TypeError(f"{tensor.dtype}, cluster remote pointer internal path requires cluster-shared pointers "
                        "(addrspace=7)")


def _check_device_remote_pointer(tensor: tl.tensor, shard_id: int | tuple[int, ...] | list[int],
                                 scope: device_mesh | None) -> None:
    ...


def _remote_pointer(
    tensor: tl.tensor,
    shard_id,
    space: str = "cluster",
    scope: device_mesh | None = None,
    dtype: tl.dtype = None,
    offset: int | tl.tensor | None = None,
    _semantic: TLESemantic | None = None,
) -> tl.tensor:

    if not isinstance(tensor, tl.tensor) and space != "device":
        raise TypeError(f"tensor must be tl.tensor, got {type(tensor).__name__}")

    space = tl._unwrap_if_constexpr(space)
    res = {
        "cluster": _check_cluster_remote_pointer,
        "device": _check_device_remote_pointer,
    }[space](tensor, shard_id, scope)
    if isinstance(res, tl.tensor):
        return res
    # Compile-time constant shard id path.
    if isinstance(shard_id, (int, tuple, list)):
        linear_shard_id = _normalize_compile_time_remote_shard_id(shard_id, scope)
        shard_id_tensor = _semantic.to_tensor(int(linear_shard_id))
        shard_id_tensor = _normalize_runtime_remote_shard_id_tensor(shard_id_tensor)
        return _create_remote_pointers_tensor(tensor, shard_id_tensor, _semantic, dtype=dtype, space=space,
                                              offset=offset)

    # Runtime shard id path. This materializes a TLE op that carries the
    # runtime i32 shard id through lowering.
    shard_id_tensor = shard_id if isinstance(shard_id, tl.tensor) else _semantic.to_tensor(shard_id)
    shard_id_tensor = _normalize_runtime_remote_shard_id_tensor(shard_id_tensor)

    return _create_remote_pointers_tensor(tensor, shard_id_tensor, _semantic, dtype=dtype, space=space, offset=offset)


def _normalize_node_elem_bytes(dtype) -> int:
    dtype = tl._unwrap_if_constexpr(dtype)
    if not isinstance(dtype, tl.dtype):
        raise TypeError(f"node space dtype must be a scalar Triton dtype, got {type(dtype).__name__}")
    elem_bytes = dtype.itemsize
    if elem_bytes <= 0:
        raise ValueError(f"node space dtype must be byte-addressable, got {dtype}")
    return elem_bytes


def _normalize_node_peer(shard_id, scope, _semantic) -> tl.tensor:
    shard_id = _unwrap_remote_shard_id(shard_id)
    scope = tl._unwrap_if_constexpr(scope)
    if scope is not None and not isinstance(scope, device_mesh):
        raise TypeError(f"node space scope must be device_mesh or None, got {type(scope).__name__}")

    if isinstance(shard_id, (int, tuple, list)):
        is_coordinate = isinstance(shard_id, (tuple, list))
        peer = _normalize_compile_time_remote_shard_id(shard_id, scope)
        if is_coordinate:
            # Coordinates are relative to the selected mesh. Resolve through
            # physical_ids so coordinates on a sliced submesh still produce
            # the corresponding world rank rather than a submesh-local rank.
            peer = scope.physical_ids[peer]
            if peer > 0x7FFFFFFF:
                raise ValueError(f"node space world rank {peer} exceeds int32 range")
        shard_id = _semantic.to_tensor(peer)
    elif not isinstance(shard_id, tl.tensor):
        shard_id = _semantic.to_tensor(shard_id)
    return _normalize_runtime_remote_shard_id_tensor(shard_id)


_NODE_INTER_CONTEXT_COUNT = 4


def _normalize_node_netidx(netidx, _semantic) -> tl.tensor:
    netidx = tl._unwrap_if_constexpr(netidx)
    if isinstance(netidx, bool):
        raise TypeError("node space netidx must be an integer, not bool")
    if isinstance(netidx, int):
        if netidx < 0 or netidx >= _NODE_INTER_CONTEXT_COUNT:
            raise ValueError(f"node space netidx must be in range [0, {_NODE_INTER_CONTEXT_COUNT}), got {netidx}")
        netidx = _semantic.to_tensor(netidx)
    elif not isinstance(netidx, tl.tensor):
        netidx = _semantic.to_tensor(netidx)

    if netidx.shape != ():
        raise ValueError(f"node space netidx must be scalar, got shape {netidx.shape}")
    if netidx.dtype != tl.int32:
        raise TypeError(f"node space runtime netidx must be tl.int32; got {netidx.dtype}")
    return netidx


def _parse_node_context(builder, value, label: str, index: int):
    from triton.runtime import DistributedRtContext
    value = tl._unwrap_if_constexpr(value)
    if not isinstance(value, DistributedRtContext):
        raise TypeError(f"node space {label} must be DistributedRtContext, got {type(value).__name__}")
    return _parse_src_arg(builder, value, index)


def _create_node_remote_pointer(ctx, shard_id, scope, dtype, coopkind, netidx, _semantic) -> tl.tensor:
    if dtype is None:
        raise TypeError('tle.remote(..., space="node") requires dtype')

    builder = _semantic.builder
    if not hasattr(builder, "create_remote_pointers"):
        raise RuntimeError("node remote pointer requires TLE remote_pointers support in the active Triton build")

    peer = _normalize_node_peer(shard_id, scope, _semantic)
    dtype = tl._unwrap_if_constexpr(dtype)
    _normalize_node_elem_bytes(dtype)
    net_idx = _normalize_node_netidx(netidx, _semantic)
    coop_kind = tl._unwrap_if_constexpr(coopkind)
    coop_kind = coop_kind.value if isinstance(coop_kind, GroupKind) else str(coop_kind).lower()
    coop_kind = attr.FlagCXCoopKind.from_str(coop_kind)
    if coop_kind is None:
        expected = "thread, warp, or block"
        raise ValueError(f"coopkind must be {expected}, got {coop_kind!r}")

    src_mem = _parse_node_context(builder, ctx, "ctx", 0)
    comm = _parse_node_context(builder, ctx, "ctx", 1)
    remote_ptr_dtype = tl.pointer_type(dtype, 1)
    remote_op = builder.create_remote_pointers(
        remote_ptr_dtype.to_ir(builder),
        src_mem,
        peer.handle,
        "node",
        None,
        comm,
        net_idx.handle,
        coop_kind,
    )
    return tl.tensor(remote_op.get_result(0), remote_ptr_dtype)


@tl.builtin
def remote(
    tensor=tl.tensor | None,
    shard_id=None,
    scope: device_mesh | None = None,
    space: str = "cluster",
    dtype: tl.dtype = None,
    offset: int | tl.tensor | None = None,
    coopkind: GroupKind | str | None = None,
    netidx: int | tl.tensor = 0,
    _semantic: TLESemantic | None = None,
):
    """
    M3 entrypoint: mark distributed access target.

    Supported input:
    - tle buffered_tensor: returns a remote-marked buffered tensor; caller
      should then use `tle.gpu.local_ptr(...)` to materialize remote pointers.
    - tl.tensor shared-memory pointer (scalar or tensor): returns remote
      pointer directly.
    - DistributedRtContext with `space="node"`: returns a global pointer
      marker consumed by a direct `tl.load`/`tl.store` copy pattern.

    `shard_id` is the target block id inside the current thread block cluster.
    For cluster/device pointer paths, when `scope` is provided, launch cluster
    dimensions are inferred from that mesh and this mode requires `num_ctas=1`
    (one program maps to one block).

    For `space="node"`, `tensor` must be the registered-memory
    `DistributedRtContext` created by `create_dist_tensor`. A load from the
    local buffer followed by a store through the returned pointer performs a
    put; the reverse performs a get. The local buffer root must be the entry
    pointer for that buffer.

    Scalar copies transfer one element and default source/destination offsets
    to zero. Tensor copies require the same contiguous range
    `tl.arange(0, N)`. Unmasked copies or a shared prefix mask
    `offsets < valid_n` are supported; `valid_n` is clamped to `[0, N]`, and
    zero skips the transfer. Sparse, multidimensional, strided, non-zero-start,
    or mismatched ranges are rejected.

    `dtype` is required. `coopkind` defaults to `GroupKind.BLOCK`, and
    `netidx` defaults to zero. Compile-time `netidx` must be in `[0, 4)`;
    runtime values must be scalar `tl.int32`. `shard_id` may be a world rank
    or, with `scope=device_mesh`, a compile-time mesh coordinate.

    For `space="device"`, `offset` is the remote-memory element offset and
    may be a Python `int` or scalar integer `tl.tensor`.
    """
    space = tl._unwrap_if_constexpr(space)
    if not isinstance(space, str):
        raise TypeError(f"space must be str, got {type(space).__name__}")
    if space not in ("cluster", "device", "node"):
        raise ValueError(f"space must be 'cluster', 'device', or 'node', got {space!r}")
    shard_id = _unwrap_remote_shard_id(shard_id)
    scope = tl._unwrap_if_constexpr(scope)
    if space == "node":
        if offset is not None:
            raise TypeError('tle.remote(..., space="node") does not accept offset; '
                            'add element offsets to the returned pointer')
        # Keep None as an internal sentinel so cluster/device can still reject
        # an explicitly supplied node-only coopkind. Node behavior defaults to
        # BLOCK as documented.
        if coopkind is None:
            coopkind = GroupKind.BLOCK
        return _create_node_remote_pointer(tensor, shard_id, scope, dtype, coopkind, netidx, _semantic)
    node_only_args = ["coopkind"] if coopkind is not None else []
    unwrapped_netidx = tl._unwrap_if_constexpr(netidx)
    if not isinstance(unwrapped_netidx, int) or unwrapped_netidx != 0:
        node_only_args.append("netidx")
    if node_only_args:
        raise TypeError(f'{space} space does not accept node-only argument(s): '
                        f'{", ".join(node_only_args)}')
    if scope is not None and not isinstance(scope, device_mesh):
        raise TypeError(f"scope must be device_mesh or None, got {type(scope).__name__}")
    if scope is not None:
        _apply_mesh_cluster_launch(scope, _semantic)

    # Direct pointer path: support local_ptr scalar/tensor values and return
    # remote pointer with preserved shape.
    if isinstance(tensor, tl.tensor) or space == "device":
        return _remote_pointer(tensor, shard_id, scope=scope, space=space, _semantic=_semantic, dtype=dtype,
                               offset=offset)

    # Buffered tensor path: carry remote metadata and let `local_ptr` materialize
    # remote pointers later.
    if _is_buffered_tensor_like(tensor):
        if offset is not None:
            raise NotImplementedError("offset is not supported for buffered_tensor in tle.remote; "
                                      "use tle.remote(in_ptr, ..., offset=...) with a pointer tensor instead")
        if (hasattr(tensor, "_tle_remote_shard_id") or hasattr(tensor, "_tle_remote_scope")
                or hasattr(tensor.type, "_tle_remote_shard_id") or hasattr(tensor.type, "_tle_remote_scope")):
            raise ValueError("remote(buffered_tensor, ...) cannot be applied twice; "
                             "materialize pointer views with tle.gpu.local_ptr(remote_buffer, indices)")
        if isinstance(shard_id, (int, tuple, list)):
            shard_id = _normalize_compile_time_remote_shard_id(shard_id, scope)
        else:
            shard_id_tensor = shard_id if isinstance(shard_id, tl.tensor) else None
            if shard_id_tensor is None:
                if _semantic is None:
                    raise TypeError("runtime shard_id for remote(buffered_tensor, ...) must be scalar int32 "
                                    "and requires JIT semantic context for materialization")
                shard_id_tensor = _semantic.to_tensor(shard_id)
            shard_id = _normalize_runtime_remote_shard_id_tensor(shard_id_tensor)
        # Keep remote metadata on buffered_tensor.type so it survives value
        # reconstruction in JIT interpreter paths (value-level attrs can drop).
        remote_buffer = copy.copy(tensor)
        remote_type = copy.copy(tensor.type)
        try:
            setattr(remote_type, "_tle_remote_shard_id", shard_id)
            setattr(remote_type, "_tle_remote_scope", scope)
            remote_buffer.type = remote_type
        except AttributeError:
            # Type object may be immutable for unit-test stubs.
            pass
        # Keep value-level metadata as a secondary carrier to maximize
        # compatibility with existing JIT object reconstruction paths.
        setattr(remote_buffer, "_tle_remote_shard_id", shard_id)
        setattr(remote_buffer, "_tle_remote_scope", scope)
        return remote_buffer

    raise TypeError(f"tensor must be tle.buffered_tensor, got {type(tensor).__name__}")


@tl.builtin
def signal_wait(
    device_dptr,
    slot_id,
    wait_kind: str | attr.SignalWaitKind,
    target: int | None = None,
    group_kind: str | GroupKind = GroupKind.BLOCK,
    context_idx: int = 0,
    _semantic=None,
):
    """Wait until a local FlagCX synchronization slot reaches its target.

    ``target`` is required for ``wait_kind="signal"`` and
    ``wait_kind="counter"``.  ``wait_kind="shadow"`` instead reads the target
    from FlagCX's locally maintained shadow buffer, so ``target`` must be
    omitted. ``slot_id`` is interpreted in the signal slot namespace.
    """
    builder = _semantic.builder

    wait_kind = tl._unwrap_if_constexpr(wait_kind)
    wait_kind_val = (wait_kind if isinstance(wait_kind, attr.SignalWaitKind) else attr.SignalWaitKind.from_str(
        str(wait_kind).lower()))
    if wait_kind_val is None:
        expected = "signal, counter, or shadow"
        raise ValueError(f"wait kind must be {expected}, got {wait_kind!r}")

    group_kind = tl._unwrap_if_constexpr(group_kind)
    group_kind = group_kind.value if isinstance(group_kind, GroupKind) else str(group_kind).lower()
    group_kind = attr.FlagCXCoopKind.from_str(group_kind)
    if group_kind is None:
        expected = "thread, warp, or block"
        raise ValueError(f"group kind must be {expected}, got {group_kind!r}")

    context_idx = tl._unwrap_if_constexpr(context_idx)
    if not isinstance(context_idx, int):
        raise TypeError(f"context_idx must be a compile-time int, got {type(context_idx).__name__}")
    if context_idx < 0 or context_idx > 0x7FFFFFFF:
        raise ValueError(f"context_idx must be in int32 range, got {context_idx}")

    comm = _parse_src_arg(builder, device_dptr, 1)
    slot_tensor = _normalize_signal_scalar(slot_id, "slot_id", tl.int32, _semantic)
    target_value = target.value if isinstance(target, tl.constexpr) else target
    target_tensor = (_normalize_signal_scalar(target_value, "target", tl.int64, _semantic)
                     if target_value is not None else None)

    utils.verify_signal_wait(wait_kind_val, None if target_tensor is None else target_tensor.handle)

    builder.create_signal_wait(
        comm,
        slot_tensor.handle,
        wait_kind_val,
        None if target_tensor is None else target_tensor.handle,
        group_kind,
        context_idx,
    )


def distributed_dot(a: ShardedTensor, b: ShardedTensor, c: ShardedTensor | None = None):
    raise NotImplementedError("distributed_dot is deferred to M5")
