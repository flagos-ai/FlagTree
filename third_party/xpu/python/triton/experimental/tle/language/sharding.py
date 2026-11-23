# Triton 3.6 XPU TLE - sharding annotations (`tle.sharding` and friends)
#
# Port of flagtree's sharding surface (its
# `python/triton/experimental/tle/language/distributed.py:548-707`): the `S`/`P`/
# `B` axis specs, `ShardingSpec`, `ShardedTensor`, `sharding()`,
# `make_sharded_tensor()` and `reshard()`.
#
# It is METADATA, on flagtree too -- its own `sharding()` docstring says
# "annotation metadata today. Communication lowering is added in later phases",
# and `reshard` is a roadmap stub there ("deferred to M4"). So is
# `distributed_dot` ("deferred to M5"). Nothing here emits IR, and nothing here
# is xpu5-specific: it exists so a kernel written against flagtree can be
# *annotated* identically on this backend, and so the two stubs fail with the
# same message instead of an AttributeError.
#
# One deliberate omission: flagtree's `make_sharded_tensor` forwards to its
# `dsa.thrive.distributed.make_sharded_tensor_impl` when FLAGTREE_BACKEND is
# "thrive", which rewrites the handle into a torch_thrive ParallelInfo. That
# runtime belongs to another backend and does not exist here, and this fork does
# not provide the `tle.dsa.thrive` namespace at all, so the forward is left out
# rather than faked.
#
# It lives in its own module because the types are host-side dataclasses with no
# builder involvement, unlike everything in distributed.py; flagtree keeps them
# in one file, and `tle.<name>` resolves the same either way.
from dataclasses import dataclass

from .distributed import _as_positive_int, device_mesh

__all__ = [
    "B", "S", "P", "ShardingSpec", "ShardedTensor", "sharding", "make_sharded_tensor", "reshard", "distributed_dot"
]


class _BroadcastSpec:
    """`B`: this axis holds a full copy, so it is neither split nor partial."""

    def __repr__(self):
        return "B"


B = _BroadcastSpec()


@dataclass(frozen=True)
class S:
    """Split: the tensor dimension is divided across the named mesh axis/axes."""
    axis: object


@dataclass(frozen=True)
class P:
    """Partial: every shard holds a partial sum over the named mesh axis/axes."""
    axis: object


def _normalize_axis_group(spec, label):
    """One tensor dimension's axis spec -> a tuple of mesh axis names."""
    if spec is None or spec is B:
        return tuple()
    if isinstance(spec, (S, P)):
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
    raise TypeError(f"{label} axis spec must be str/list/tuple/S/P/B, got "
                    f"{type(spec).__name__}")


def _normalize_partial_specs(partial):
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
    """How one tensor is laid out over a mesh.

    ``split[d]`` names the mesh axes tensor dimension ``d`` is divided over,
    ``partial`` the axes a reduction is still outstanding on, and ``broadcast``
    everything left -- every mesh axis is in exactly one of the three, which is
    what :meth:`axis_state` reads back.
    """
    mesh: object
    split: tuple
    partial: tuple
    broadcast: tuple

    def axis_state(self, axis):
        if axis in self.partial:
            return "P"
        for split_axes in self.split:
            if axis in split_axes:
                return "S"
        return "B"


@dataclass(frozen=True)
class ShardedTensor:
    handle: object
    sharding: ShardingSpec
    shape: object = None


def sharding(mesh, split=None, partial=None):
    """Build a :class:`ShardingSpec` bound to ``mesh``.

    Annotation metadata, as in flagtree: it validates the layout and records
    it. No communication is derived from it on this backend -- a resharding
    collective would be :func:`reshard`, which flagtree has not landed either.
    """
    if not isinstance(mesh, device_mesh):
        raise TypeError(f"mesh must be device_mesh, got {type(mesh).__name__}")

    if split is None:
        split = tuple()
    if not isinstance(split, (tuple, list)):
        raise TypeError(f"split must be a list/tuple, got {type(split).__name__}")
    split_specs = [_normalize_axis_group(item, "split") for item in split]
    partial_axes = _normalize_partial_specs(partial)

    split_axes = [axis for item in split_specs for axis in item]
    if len(set(split_axes)) != len(split_axes):
        raise ValueError(f"split axes must be unique across tensor dims, got {split_axes}")

    split_set, partial_set = set(split_axes), set(partial_axes)
    unknown = [a for a in split_axes + list(partial_axes) if a not in mesh.dim_names]
    if unknown:
        raise ValueError(f"unknown mesh axis names: {unknown}; mesh axes are {mesh.dim_names}")
    overlap = split_set & partial_set
    if overlap:
        raise ValueError(f"mesh axis cannot be both split and partial: {sorted(overlap)}")

    return ShardingSpec(
        mesh=mesh,
        split=tuple(split_specs),
        partial=tuple(a for a in mesh.dim_names if a in partial_set),
        broadcast=tuple(a for a in mesh.dim_names if a not in split_set and a not in partial_set),
    )


def make_sharded_tensor(handle, sharding, shape=None):
    """Attach a :class:`ShardingSpec` to a tensor handle."""
    if not isinstance(sharding, ShardingSpec):
        raise TypeError(f"sharding must be ShardingSpec, got {type(sharding).__name__}")
    normalized_shape = None
    if shape is not None:
        if not isinstance(shape, (tuple, list)):
            raise TypeError(f"shape must be list/tuple, got {type(shape).__name__}")
        normalized_shape = tuple(_as_positive_int(v, "tensor shape") for v in shape)
        if sharding.split and len(sharding.split) != len(normalized_shape):
            raise ValueError(f"split rank ({len(sharding.split)}) must match tensor "
                             f"rank ({len(normalized_shape)})")
    return ShardedTensor(handle=handle, sharding=sharding, shape=normalized_shape)


def reshard(tensor, spec):
    """Re-lay a sharded tensor out over a new spec.

    flagtree's M4 entry point, and a stub there too: the collective it would
    emit does not exist on either backend yet. Kept so the failure is this
    message rather than an AttributeError.
    """
    raise NotImplementedError("reshard is deferred to M4")


def distributed_dot(a, b, c=None):
    """Matmul over sharded operands.

    flagtree's M5 entry point, and a stub there too (its distributed.py:1664).
    A real implementation needs the resharding collective :func:`reshard` would
    provide, so it cannot be ahead of it. Build the matmul out of
    :func:`~.distributed.remote` plus ``tl.dot`` meanwhile -- that is what
    ``test/tle/distributed/07-intra-node-reduce-scatter.py`` does.
    """
    raise NotImplementedError("distributed_dot is deferred to M5")
