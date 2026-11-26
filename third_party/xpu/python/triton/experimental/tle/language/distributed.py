"""XPU TLE mesh metadata and explicit-handle FlagCX device operations.

Pass the context returned by create_dist_tensor as an ordinary kernel argument
to supply both FlagCX device handles at launch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Mapping, Sequence

import triton.language as tl
from triton.language.core import _unwrap_if_constexpr, builtin
from triton.runtime import DistributedRtContext

from ._gluon import gluon_builder


def _prod(values: Sequence[int]) -> int:
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


@dataclass
class MeshConfig:
    node: int | list[tuple[str, int]] | None = None
    device: int | list[tuple[str, int]] | None = None
    chiplet: int | list[tuple[str, int]] | None = None
    block_cluster: int | list[tuple[str, int]] | None = None
    block: int | list[tuple[str, int]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}

    def __repr__(self) -> str:
        fields = ", ".join(f"{key}={value}" for key, value in self.to_dict().items())
        return f"MeshConfig({fields})"


class device_mesh:
    """Immutable logical topology descriptor; does not launch or synchronize devices."""

    def __init__(self, topology: Mapping[str, Any] | MeshConfig | None = None, *, _shape: Sequence[int] | None = None,
                 _dim_names: Sequence[str] | None = None, _physical_ids: Sequence[int] | None = None,
                 _launch_shape: Sequence[int] | None = None, _launch_dim_names: Sequence[str] | None = None):
        if topology is None:
            if _shape is None or _dim_names is None or _physical_ids is None:
                raise ValueError("internal mesh constructor requires shape/names/physical ids")
            self._shape = tuple(_shape)
            self._dim_names = tuple(_dim_names)
            self._physical_ids = tuple(_physical_ids)
            self._launch_shape = tuple(_launch_shape if _launch_shape is not None else _shape)
            self._launch_dim_names = tuple(_launch_dim_names if _launch_dim_names is not None else _dim_names)
            return

        if not isinstance(topology, (Mapping, MeshConfig)):
            raise TypeError(f"topology must be a Mapping or MeshConfig, got {type(topology).__name__}")
        if isinstance(topology, MeshConfig):
            topology = topology.to_dict()
        if not topology:
            raise ValueError("topology cannot be empty")

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
            name, size = item
            if not isinstance(name, str) or not name:
                raise ValueError(f"invalid dimension name in {level_name!r}: {name!r}")
            shape.append(_as_positive_int(size, f"{level_name}.{name}"))
            names.append(name)
        return shape, names

    def __deepcopy__(self, memo):
        return self

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

    def flatten(self) -> device_mesh:
        return self.reshape(self.size)

    def reshape(self, *shape: int | Sequence[int]) -> device_mesh:
        new_shape = tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else tuple(shape)
        if not new_shape:
            raise ValueError("new shape cannot be empty")
        new_shape = tuple(_as_positive_int(dim, "shape dimension") for dim in new_shape)
        if _prod(new_shape) != self.size:
            raise ValueError(f"cannot reshape mesh of size {self.size} into shape {new_shape}")
        if len(new_shape) == self.ndim:
            names = self._dim_names
        elif len(new_shape) == 1:
            names = ("flat", )
        else:
            names = tuple(f"dim{i}" for i in range(len(new_shape)))
        return device_mesh(None, _shape=new_shape, _dim_names=names, _physical_ids=self._physical_ids,
                           _launch_shape=self._launch_shape, _launch_dim_names=self._launch_dim_names)

    def _normalize_key(self, key: Any) -> tuple[Any, ...]:
        if not isinstance(key, tuple):
            key = (key, )
        if any(item is Ellipsis for item in key):
            if key.count(Ellipsis) > 1:
                raise IndexError("an index can only have a single ellipsis")
            position = key.index(Ellipsis)
            missing = self.ndim - (len(key) - 1)
            if missing < 0:
                raise IndexError("too many indices for device_mesh")
            key = key[:position] + (slice(None), ) * missing + key[position + 1:]
        if len(key) > self.ndim:
            raise IndexError("too many indices for device_mesh")
        return key + (slice(None), ) * (self.ndim - len(key))

    def _linear_index(self, coords: Sequence[int]) -> int:
        index = 0
        for coord, dim_size in zip(coords, self._shape):
            index = index * dim_size + coord
        return index

    def __getitem__(self, key: Any) -> device_mesh:
        key = self._normalize_key(key)
        selected = []
        keep_dim = []
        for dim_size, dim_key in zip(self._shape, key):
            if isinstance(dim_key, int):
                index = dim_key + dim_size if dim_key < 0 else dim_key
                if index < 0 or index >= dim_size:
                    raise IndexError(f"index {dim_key} out of range for dim size {dim_size}")
                selected.append([index])
                keep_dim.append(False)
            elif isinstance(dim_key, slice):
                indices = list(range(*dim_key.indices(dim_size)))
                if not indices:
                    raise ValueError("empty sub-mesh is not supported")
                selected.append(indices)
                keep_dim.append(True)
            else:
                raise TypeError(f"device_mesh indices must be int/slice/ellipsis, got {type(dim_key).__name__}")
        shape = tuple(len(indices) for indices, keep in zip(selected, keep_dim) if keep)
        names = tuple(name for name, keep in zip(self._dim_names, keep_dim) if keep)
        ids = tuple(self._physical_ids[self._linear_index(coords)] for coords in product(*selected))
        return device_mesh(None, _shape=shape, _dim_names=names, _physical_ids=ids, _launch_shape=self._launch_shape,
                           _launch_dim_names=self._launch_dim_names)

    def __repr__(self) -> str:
        return f"DeviceMesh(shape={self._shape}, names={self._dim_names})"


def _parse_src_arg(builder, src, index, name):
    src = _unwrap_if_constexpr(src)
    if isinstance(src, tl.tuple):
        src = src[index]
    if isinstance(src, tl.tensor) and src.dtype == tl.int64:
        return src.handle
    if isinstance(src, DistributedRtContext) and src[index] is not None:
        return builder.get_int64(int(src[index]))
    raise TypeError(f"{name} must be a runtime i64 or a FlagCX DistributedRtContext")


@builtin
def _get_local_rank(comm=None, _semantic=None, _generator=None):
    """Read the intra-node rank from an explicit FlagCX device communicator."""
    builder = gluon_builder(_semantic, _generator, "get_device_id")
    comm = _unwrap_if_constexpr(comm)
    handle = None if comm is None else _parse_src_arg(builder, comm, 1, "comm")
    return tl.tensor(builder.get_device_id(tl.int32.to_ir(builder), handle), tl.int32)


@builtin
def n_pes(comm, _semantic=None, _generator=None):
    """Read the intra-node device count from an explicit FlagCX communicator."""
    builder = gluon_builder(_semantic, _generator, "get_n_pes")
    handle = _parse_src_arg(builder, comm, 1, "comm")
    return tl.tensor(builder.get_n_pes(tl.int32.to_ir(builder), handle), tl.int32)


@builtin
def shard_id(mesh, axis, device_dptr=None, _semantic=None, _generator=None):
    """Return the device coordinate from an explicit FlagCX communicator."""
    mesh = _unwrap_if_constexpr(mesh)
    axis = _unwrap_if_constexpr(axis)
    if not isinstance(mesh, device_mesh):
        raise TypeError("mesh must be a device_mesh")
    if axis != "device":
        raise NotImplementedError(f"XPU TLE shard_id axis {axis!r} is not implemented")
    if _unwrap_if_constexpr(device_dptr) is None:
        raise ValueError("device_dptr is required for the device axis")
    return _get_local_rank(device_dptr, _semantic=_semantic, _generator=_generator)


_COOP_KINDS = {"thread": 0, "warp": 1, "block": 2}
_TEAM_KINDS = {"intra_node": 0, "inter_node": 1, "world": 2}
_MEMORY_ORDERS = {"relaxed": 0, "acquire": 1, "release": 2, "acq_rel": 3}
_MEMORY_SCOPES = {"system": 0, "device": 1, "block": 2, "thread": 3}


def _enum_value(options, value, name):
    value = _unwrap_if_constexpr(value)
    if value not in options:
        raise ValueError(f"{name} must be one of {', '.join(options)}, got {value!r}")
    return options[value]


@builtin
def distributed_barrier(mesh=None, device_dptr=None, *, space=None, barrier_kind="sync", group_kind="block",
                        barrier_index=0, order="acq_rel", _semantic=None, _generator=None):
    """Synchronize an intra-node FlagCX device group using an explicit comm."""
    mesh = _unwrap_if_constexpr(mesh)
    space = _unwrap_if_constexpr(space)
    if space != "device" or not isinstance(mesh, device_mesh) or "device" not in mesh.dim_names:
        raise NotImplementedError("XPU TLE supports only an explicit device-space barrier on a device mesh")
    builder = gluon_builder(_semantic, _generator, "create_distributed_barrier")
    builder.create_distributed_barrier(
        _parse_src_arg(builder, device_dptr, 1, "comm"), _unwrap_if_constexpr(barrier_kind),
        _enum_value(_COOP_KINDS, group_kind, "group_kind"), _unwrap_if_constexpr(barrier_index),
        _enum_value(_MEMORY_ORDERS, order, "order"))


def _scalar_int_handle(value, name, dtype, _semantic):
    value = _semantic.to_tensor(_unwrap_if_constexpr(value))
    if value.shape != () or not value.dtype.is_int():
        raise TypeError(f"{name} must be a scalar integer")
    if value.dtype != dtype:
        value = tl.cast(value, dtype, _semantic=_semantic)
    return value.handle


def _signal_context_id(context_id):
    context_id = _unwrap_if_constexpr(context_id)
    if not isinstance(context_id, int) or isinstance(context_id, bool):
        raise TypeError(f"context_id must be a compile-time int, got {type(context_id).__name__}")
    return context_id


@builtin
def init(device_dptr, _semantic=None, _generator=None):
    """Select the communicator for XPU kernel initialization explicitly."""
    builder = gluon_builder(_semantic, _generator, "create_device_init")
    builder.create_device_init(_parse_src_arg(builder, device_dptr, 1, "comm"))


@builtin
def signal(device_dptr, peer, slot_id, value=None, op="inc", space="intra_node", group_kind="block", context_id=0,
           scope="system", _semantic=None, _generator=None):
    """Publish an update to a slot owned by a remote FlagCX peer.

    ``op="add"`` adds ``value``, ``op="inc"`` adds one. ``space`` names the team
    ``peer`` indexes; ``context_id`` picks the arena ``slot_id`` lives in.
    """
    op = _unwrap_if_constexpr(op)
    value = _unwrap_if_constexpr(value)
    if op == "add" and value is None:
        raise ValueError("op='add' requires value")
    if op == "inc" and value is not None:
        raise ValueError("op='inc' takes no value")
    builder = gluon_builder(_semantic, _generator, "create_signal")
    builder.create_signal(
        _parse_src_arg(builder, device_dptr, 1, "comm"),
        _scalar_int_handle(peer, "peer", tl.int32, _semantic),
        _scalar_int_handle(slot_id, "slot_id", tl.int32, _semantic),
        None if value is None else _scalar_int_handle(value, "value", tl.int64, _semantic),
        op, _enum_value(_TEAM_KINDS, space, "space"),
        _enum_value(_COOP_KINDS, group_kind, "group_kind"), _signal_context_id(context_id),
        _enum_value(_MEMORY_SCOPES, scope, "scope"))


@builtin
def signal_wait(device_dptr, slot_id, wait_kind, target=None, group_kind="block", context_id=0, order="acquire",
                _semantic=None, _generator=None):
    """Spin until a local FlagCX slot reaches its target.

    ``"signal"`` and ``"counter"`` wait for ``target`` in their own slot
    namespace; ``"shadow"`` reads the target from FlagCX and takes no ``target``.
    """
    wait_kind = _unwrap_if_constexpr(wait_kind)
    target = _unwrap_if_constexpr(target)
    if wait_kind == "shadow" and target is not None:
        raise ValueError("wait_kind='shadow' takes no target")
    if wait_kind in ("signal", "counter") and target is None:
        raise ValueError(f"wait_kind={wait_kind!r} requires target")
    builder = gluon_builder(_semantic, _generator, "create_signal_wait")
    builder.create_signal_wait(
        _parse_src_arg(builder, device_dptr, 1, "comm"),
        _scalar_int_handle(slot_id, "slot_id", tl.int32, _semantic),
        None if target is None else _scalar_int_handle(target, "target", tl.int64, _semantic),
        wait_kind, _enum_value(_COOP_KINDS, group_kind, "group_kind"),
        _signal_context_id(context_id), _enum_value(_MEMORY_ORDERS, order, "order"))


@builtin
def remote(tensor, shard_id=None, scope=None, space="cluster", dtype=None, offset=None, _semantic=None,
           _generator=None):
    """Map a registered device buffer to a peer for stores, returning a write-only global pointer."""
    space = _unwrap_if_constexpr(space)
    scope = _unwrap_if_constexpr(scope)
    dtype = _unwrap_if_constexpr(dtype)
    if space != "device":
        raise NotImplementedError(f"XPU TLE remote space {space!r} is not implemented")
    if scope is not None and (not isinstance(scope, device_mesh) or "device" not in scope.dim_names):
        raise ValueError("device remote scope must be a device mesh")
    if offset is None:
        raise ValueError("device space remote pointers require an offset")
    if shard_id is None:
        raise ValueError("device space remote pointers require a shard_id")
    if not isinstance(dtype, tl.dtype):
        raise TypeError("device space remote pointers require a scalar dtype")
    bits = int(dtype.primitive_bitwidth)
    if bits < 8 or bits % 8:
        raise ValueError("remote pointee type must be byte addressable")
    builder = gluon_builder(_semantic, _generator, "create_remote_pointers")
    ptr_type = tl.pointer_type(dtype, 1)
    memory = _unwrap_if_constexpr(tensor)
    comm = None
    if isinstance(memory, tl.tuple) or (isinstance(memory, DistributedRtContext) and memory[1] is not None):
        comm = _parse_src_arg(builder, memory, 1, "comm")
    result = builder.create_remote_pointers(
        ptr_type.to_ir(builder), _parse_src_arg(builder, memory, 0, "memory"),
        _scalar_int_handle(shard_id, "shard_id", tl.int32, _semantic),
        _scalar_int_handle(offset, "offset", tl.int64, _semantic), bits // 8, comm)
    return tl.tensor(result, ptr_type)


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
        return ()
    if isinstance(spec, (S, P)):
        spec = spec.axis
    if isinstance(spec, str):
        if not spec:
            raise ValueError(f"{label} axis name cannot be empty")
        return (spec, )
    if isinstance(spec, (tuple, list)):
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
        return ()
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
        if any(axis in group for group in self.split):
            return "S"
        return "B"


@dataclass(frozen=True)
class ShardedTensor:
    handle: Any
    sharding: ShardingSpec
    shape: tuple[int, ...] | None = None


def sharding(mesh: device_mesh, split: Sequence[Any] | None = None,
             partial: Sequence[Any] | None = None) -> ShardingSpec:
    """Describe tensor placement on a mesh; no communication is emitted."""
    if not isinstance(mesh, device_mesh):
        raise TypeError(f"mesh must be device_mesh, got {type(mesh).__name__}")
    if split is None:
        split = ()
    if not isinstance(split, (tuple, list)):
        raise TypeError(f"split must be a list/tuple, got {type(split).__name__}")
    groups = tuple(_normalize_axis_group(item, "split") for item in split)
    partial_axes = _normalize_partial_specs(partial)
    split_axes = [axis for group in groups for axis in group]
    if len(set(split_axes)) != len(split_axes):
        raise ValueError(f"split axes must be unique across tensor dims, got {split_axes}")
    unknown = [axis for axis in split_axes + list(partial_axes) if axis not in mesh.dim_names]
    if unknown:
        raise ValueError(f"unknown mesh axis names: {unknown}; mesh axes are {mesh.dim_names}")
    overlap = set(split_axes) & set(partial_axes)
    if overlap:
        raise ValueError(f"mesh axis cannot be both split and partial: {sorted(overlap)}")
    broadcast = tuple(axis for axis in mesh.dim_names if axis not in split_axes and axis not in partial_axes)
    return ShardingSpec(mesh, groups, tuple(axis for axis in mesh.dim_names if axis in partial_axes), broadcast)


def make_sharded_tensor(handle: Any, sharding: ShardingSpec, shape: Sequence[int] | None = None) -> ShardedTensor:
    if not isinstance(sharding, ShardingSpec):
        raise TypeError(f"sharding must be ShardingSpec, got {type(sharding).__name__}")
    normalized_shape = None
    if shape is not None:
        if not isinstance(shape, (tuple, list)):
            raise TypeError(f"shape must be list/tuple, got {type(shape).__name__}")
        normalized_shape = tuple(_as_positive_int(dim, "tensor shape") for dim in shape)
        if sharding.split and len(sharding.split) != len(normalized_shape):
            raise ValueError(f"split rank ({len(sharding.split)}) must match tensor rank ({len(normalized_shape)})")
    return ShardedTensor(handle=handle, sharding=sharding, shape=normalized_shape)
