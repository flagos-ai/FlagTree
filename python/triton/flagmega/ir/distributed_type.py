# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""nncase-compatible placement and SBP value objects.

These objects deliberately live beside, rather than inside, ``model.py`` so
distributed layout semantics have one reviewable home.  ``DistributedType``
itself remains an ``IRType`` in :mod:`model` to avoid a base-type import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from itertools import combinations, product
from math import prod
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.dim_expr import Dimension, ceil_div, dim, try_div_exactly

if TYPE_CHECKING:
    from triton.flagmega.ir.model import DistributedType, IRType, TensorType


class ReduceOp(str, Enum):
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    PROD = "prod"


class SplitDistribution:
    """How owners of a placement axis receive a tensor-axis partition."""

    def to_data(self) -> dict[str, object]:
        raise NotImplementedError


@dataclass(frozen=True)
class ContiguousSplit(SplitDistribution):
    granularity: Dimension | None = None

    def __post_init__(self) -> None:
        if self.granularity is not None:
            value = dim(self.granularity)
            if value.is_fixed and value.fixed_value <= 0:
                raise IRSchemaError("Contiguous split granularity must be positive.")
            object.__setattr__(self, "granularity", value)

    def to_data(self) -> dict[str, object]:
        return {
            "kind": "contiguous",
            "granularity": None if self.granularity is None else self.granularity.to_data(),
        }

    def __str__(self) -> str:
        return "C" if self.granularity is None else f"C({self.granularity})"


@dataclass(frozen=True)
class BlockCyclicSplit(SplitDistribution):
    block_size: int

    def __post_init__(self) -> None:
        if isinstance(self.block_size, bool) or self.block_size <= 0:
            raise IRSchemaError("Block-cyclic split block_size must be positive.")

    def to_data(self) -> dict[str, object]:
        return {"kind": "block_cyclic", "block_size": self.block_size}

    def __str__(self) -> str:
        return f"BC({self.block_size})"


@dataclass(frozen=True)
class SplitStage:
    hierarchy_axes: tuple[int, ...]
    distribution: SplitDistribution

    def __post_init__(self) -> None:
        axes = tuple(int(value) for value in self.hierarchy_axes)
        if not axes or any(value < 0 for value in axes) or len(set(axes)) != len(axes):
            raise IRSchemaError("SplitStage axes must be a non-empty unique set of non-negative axes.")
        if not isinstance(self.distribution, SplitDistribution):
            raise IRSchemaError("SplitStage requires a SplitDistribution.")
        object.__setattr__(self, "hierarchy_axes", axes)

    @classmethod
    def contiguous(
        cls,
        hierarchy_axes: Sequence[int],
        granularity: int | str | Dimension | None = None,
    ) -> SplitStage:
        return cls(tuple(hierarchy_axes), ContiguousSplit(None if granularity is None else dim(granularity)))

    @classmethod
    def block_cyclic(cls, hierarchy_axes: Sequence[int], block_size: int) -> SplitStage:
        return cls(tuple(hierarchy_axes), BlockCyclicSplit(block_size))

    def to_data(self) -> dict[str, object]:
        return {
            "hierarchy_axes": list(self.hierarchy_axes),
            "distribution": self.distribution.to_data(),
        }

    def __str__(self) -> str:
        axes = ",".join(str(value) for value in self.hierarchy_axes)
        return f"{self.distribution}@[{axes}]"


class SBP:
    """Split/Broadcast/Partial policy for one logical tensor axis."""

    @staticmethod
    def broadcast() -> SBPBroadCast:
        return SBPBroadCast()

    @staticmethod
    def partial(axes: Sequence[int], reduce_op: ReduceOp | str = ReduceOp.SUM) -> SBPPartial:
        return SBPPartial(tuple(axes), ReduceOp(reduce_op))

    @staticmethod
    def split(*stages: SplitStage) -> SBPSplit:
        return SBPSplit(tuple(stages))

    @staticmethod
    def split_contiguous(
        hierarchy_axes: Sequence[int],
        granularity: int | str | Dimension | None = None,
    ) -> SBPSplit:
        return SBP.split(SplitStage.contiguous(hierarchy_axes, granularity))

    @staticmethod
    def split_block_cyclic(hierarchy_axes: Sequence[int], block_size: int) -> SBPSplit:
        return SBP.split(SplitStage.block_cyclic(hierarchy_axes, block_size))

    def to_data(self) -> dict[str, object]:
        raise NotImplementedError


@dataclass(frozen=True)
class SBPBroadCast(SBP):
    def to_data(self) -> dict[str, object]:
        return {"kind": "broadcast"}

    def __str__(self) -> str:
        return "B"


@dataclass(frozen=True)
class SBPPartial(SBP):
    axes: tuple[int, ...]
    reduce_op: ReduceOp = ReduceOp.SUM

    def __post_init__(self) -> None:
        axes = tuple(int(value) for value in self.axes)
        if not axes or any(value < 0 for value in axes) or len(set(axes)) != len(axes):
            raise IRSchemaError("Partial axes must be a non-empty unique set of non-negative axes.")
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "reduce_op", ReduceOp(self.reduce_op))

    def to_data(self) -> dict[str, object]:
        return {"kind": "partial", "axes": list(self.axes), "reduce_op": self.reduce_op.value}

    def __str__(self) -> str:
        return f"P([{','.join(str(value) for value in self.axes)}], {self.reduce_op.value})"


@dataclass(frozen=True)
class SBPSplit(SBP):
    stages: tuple[SplitStage, ...]

    def __post_init__(self) -> None:
        stages = tuple(self.stages)
        if not stages:
            raise IRSchemaError("A split policy requires at least one SplitStage.")
        axes = tuple(axis for stage in stages for axis in stage.hierarchy_axes)
        if len(set(axes)) != len(axes):
            raise IRSchemaError("A hierarchy axis can occur in only one SplitStage.")
        object.__setattr__(self, "stages", stages)

    @property
    def hierarchy_axes(self) -> tuple[int, ...]:
        return tuple(axis for stage in self.stages for axis in stage.hierarchy_axes)

    @property
    def is_contiguous(self) -> bool:
        return all(isinstance(stage.distribution, ContiguousSplit) for stage in self.stages)

    def to_data(self) -> dict[str, object]:
        return {"kind": "split", "stages": [stage.to_data() for stage in self.stages]}

    def __str__(self) -> str:
        return "S(" + ", ".join(str(stage) for stage in self.stages) + ")"


@dataclass(frozen=True)
class Placement:
    hierarchy: tuple[int, ...]
    name: str
    hierarchy_levels: str

    def __post_init__(self) -> None:
        hierarchy = tuple(int(value) for value in self.hierarchy)
        if not hierarchy or any(value <= 0 for value in hierarchy):
            raise IRSchemaError("Placement hierarchy must contain positive dimensions.")
        names = self.normalize_axis_string(self.name)
        if len(names) != len(hierarchy):
            raise IRSchemaError(f"Placement name {self.name!r} must contain {len(hierarchy)} axes.")
        levels = self.normalize_hierarchy_levels(self.hierarchy_levels, len(hierarchy))
        object.__setattr__(self, "hierarchy", hierarchy)
        object.__setattr__(self, "name", names)
        object.__setattr__(self, "hierarchy_levels", levels)

    @property
    def rank(self) -> int:
        return len(self.hierarchy)

    @property
    def size(self) -> int:
        result = 1
        for value in self.hierarchy:
            result *= value
        return result

    @property
    def normalized_hierarchy_names(self) -> str:
        return self.name

    @property
    def normalized_hierarchy_levels(self) -> str:
        return self.hierarchy_levels

    def is_physical_block_axis(self, axis: int) -> bool:
        return self.hierarchy_levels[axis] == "b"

    def physical_level_size(self, level: str) -> int:
        wanted = level.lower()
        result = 1
        for axis, axis_level in enumerate(self.hierarchy_levels):
            if axis_level == wanted:
                result *= self.hierarchy[axis]
        return result

    def to_data(self) -> dict[str, object]:
        return {
            "hierarchy": list(self.hierarchy),
            "name": self.name,
            "hierarchy_levels": self.hierarchy_levels,
        }

    @staticmethod
    def normalize_axis_string(value: str | None) -> str:
        return "".join(character for character in (value or "") if character.isalnum())

    @staticmethod
    def normalize_hierarchy_levels(value: str | None, rank: int) -> str:
        levels = Placement.normalize_axis_string(value).lower()
        if len(levels) != rank or any(level not in {"c", "d", "b"} for level in levels):
            raise IRSchemaError(
                f"Placement hierarchy_levels must have {rank} entries drawn from c/d/b, got {value!r}.")
        return levels

    def __str__(self) -> str:
        return "[" + ",".join(f"{name}:{size}" for name, size in zip(self.name, self.hierarchy)) + "]"


def sbp_from_data(data: Mapping[str, object]) -> SBP:
    kind = str(data.get("kind"))
    if kind == "broadcast":
        return SBP.broadcast()
    if kind == "partial":
        return SBP.partial(tuple(int(value) for value in data.get("axes", ())), str(data.get("reduce_op", "sum")))
    if kind == "split":
        return SBP.split(*(split_stage_from_data(value) for value in data.get("stages", ())))  # type: ignore[arg-type]
    raise IRSchemaError(f"Unknown SBP kind {kind!r}.")


def split_stage_from_data(data: Mapping[str, object]) -> SplitStage:
    distribution = data.get("distribution")
    if not isinstance(distribution, Mapping):
        raise IRSchemaError("SplitStage distribution must be an object.")
    kind = str(distribution.get("kind"))
    axes = tuple(int(value) for value in data.get("hierarchy_axes", ()))
    if kind == "contiguous":
        granularity = distribution.get("granularity")
        return SplitStage.contiguous(
            axes,
            None if granularity is None else Dimension.from_data(granularity),  # type: ignore[arg-type]
        )
    if kind == "block_cyclic":
        return SplitStage.block_cyclic(axes, int(distribution["block_size"]))
    raise IRSchemaError(f"Unknown split distribution kind {kind!r}.")


def placement_from_data(data: Mapping[str, object]) -> Placement:
    return Placement(
        tuple(int(value) for value in data.get("hierarchy", ())),
        str(data.get("name", "")),
        str(data.get("hierarchy_levels", "")),
    )


def is_distributable(tensor: TensorType, policies: Sequence[SBP], placement: Placement) -> bool:
    if len(policies) != tensor.rank:
        return False
    split_axes: list[int] = []
    for tensor_axis, policy in enumerate(policies):
        if isinstance(policy, SBPSplit):
            split_axes.extend(policy.hierarchy_axes)
            divisor = 1
            for stage in policy.stages:
                for hierarchy_axis in stage.hierarchy_axes:
                    if hierarchy_axis >= placement.rank:
                        return False
                    if isinstance(stage.distribution, ContiguousSplit):
                        divisor *= placement.hierarchy[hierarchy_axis]
            dimension = tensor.shape[tensor_axis]
            if dimension.is_fixed and dimension.fixed_value % divisor:
                return False
        elif isinstance(policy, SBPPartial) and any(axis >= placement.rank for axis in policy.axes):
            return False
    return len(split_axes) == len(set(split_axes))


def sharded_view_error(source_type: IRType, target_type: DistributedType) -> str | None:
    """Validate the semantic, zero-copy ``Distributed.ShardedView`` contract.

    This mirrors nncase ``DistributedUtility.TryValidateShardedView``.  The
    target always names the same logical tensor; a distributed source must use
    the same placement and neither side may carry a partial value.  Whether a
    legal semantic view is physically realizable is deliberately left to the
    target realization policy.
    """

    from triton.flagmega.ir.model import DistributedType, TensorType

    if not isinstance(target_type, DistributedType):
        return "ShardedView target must be a DistributedType."
    if isinstance(source_type, DistributedType):
        source_tensor = source_type.tensor
        if source_type.placement != target_type.placement:
            return "ShardedView source and target placements must match."
        if _has_partial(source_type):
            return "ShardedView source cannot contain a partial value."
        if not _has_only_split_or_broadcast(source_type):
            return "ShardedView source policies must contain only Split or Broadcast."
        if source_type == target_type:
            return "ShardedView source and target distributed types must differ."
    elif isinstance(source_type, TensorType):
        source_tensor = source_type
    else:
        return "ShardedView source must be a TensorType or DistributedType."
    if source_tensor != target_type.tensor:
        return "ShardedView cannot change the logical tensor type."
    if _has_partial(target_type):
        return "ShardedView target cannot contain a partial value."
    if not _has_only_split_or_broadcast(target_type):
        return "ShardedView target policies must contain only Split or Broadcast."
    return None


@lru_cache(maxsize=4096)
def is_local_shard_subview(
    source_type: DistributedType,
    target_type: DistributedType,
) -> bool:
    """Return whether each target owner shard is contained in its source shard.

    The structural nncase cases are handled without descriptor construction.
    FlagMega additionally proves contiguous split refinements using bounded
    symbolic owner coordinates.  The latter admits, for example, a ``y``
    shard as the block-local backing of the corresponding ``yx`` subshard,
    while sibling ownership and coarsening remain rejected.
    """

    if (
        source_type.tensor != target_type.tensor
        or source_type.placement != target_type.placement
        or _has_partial(source_type)
        or _has_partial(target_type)
        or len(source_type.axis_policies) != len(target_type.axis_policies)
    ):
        return False
    structural = True
    for source_policy, target_policy in zip(
        source_type.axis_policies,
        target_type.axis_policies,
    ):
        if isinstance(source_policy, SBPBroadCast) and isinstance(
            target_policy, (SBPBroadCast, SBPSplit)
        ):
            continue
        if isinstance(source_policy, SBPSplit) and source_policy == target_policy:
            continue
        # A broadcast target owns the complete logical tensor on every
        # placement coordinate.  A genuine source split cannot contain that
        # region on every owner, so this is a proven coarsening rather than a
        # case that needs the considerably more expensive symbolic interval
        # construction below.  Returning False is also conservative for an
        # empty tensor or a split over unit mesh axes, where block-local
        # promotion has no useful storage benefit.
        if isinstance(source_policy, SBPSplit) and isinstance(
            target_policy, SBPBroadCast
        ):
            return False
        if (
            isinstance(source_policy, SBPSplit)
            and isinstance(target_policy, SBPSplit)
            and _is_contiguous_split_refinement(
                source_policy,
                target_policy,
                source_type.placement,
            )
        ):
            continue
        structural = False
        break
    if structural:
        return True

    # Keep non-contiguous/block-cyclic mappings conservative.  A rectangular
    # local alias needs one dense parent-shard origin for every tensor axis.
    # Bounded coordinates make the interval proof valid for every placement
    # owner rather than for one sampled owner.
    from triton.flagmega.ir.local_shard import local_shard_descriptor

    coordinates = tuple(
        dim(f"_owner_coord_{axis}", 0, extent - 1)
        for axis, extent in enumerate(source_type.placement.hierarchy)
    )
    source_region = local_shard_descriptor(
        source_type, coordinates
    ).contiguous_region
    target_region = local_shard_descriptor(
        target_type, coordinates
    ).contiguous_region
    if source_region is None or target_region is None:
        return False
    for source_offset, source_shape, target_offset, target_shape in zip(
        source_region.offset,
        source_region.shape,
        target_region.offset,
        target_region.shape,
    ):
        relative_start = (target_offset - source_offset).simplify()
        remaining = (
            source_offset
            + source_shape
            - target_offset
            - target_shape
        ).simplify()
        if (
            relative_start.minimum is None
            or relative_start.minimum < 0
            or remaining.minimum is None
            or remaining.minimum < 0
        ):
            return False
    return True


def _is_contiguous_split_refinement(
    source: SBPSplit,
    target: SBPSplit,
    placement: Placement,
) -> bool:
    """Prove common parent/child contiguous shards without symbolic algebra."""

    if (
        len(target.stages) > len(source.stages)
        and target.stages[:len(source.stages)] == source.stages
    ):
        return all(
            isinstance(stage.distribution, ContiguousSplit)
            for stage in target.stages[len(source.stages):]
        )
    if len(source.stages) != 1 or len(target.stages) != 1:
        return False
    source_stage = source.stages[0]
    target_stage = target.stages[0]
    if not (
        isinstance(source_stage.distribution, ContiguousSplit)
        and isinstance(target_stage.distribution, ContiguousSplit)
    ):
        return False
    source_axes = source_stage.hierarchy_axes
    target_axes = target_stage.hierarchy_axes
    if (
        len(target_axes) <= len(source_axes)
        or target_axes[:len(source_axes)] != source_axes
    ):
        return False
    source_granularity = source_stage.distribution.granularity
    target_granularity = target_stage.distribution.granularity
    if source_granularity is None or target_granularity is None:
        return False
    refinement_factor = prod(
        placement.hierarchy[axis]
        for axis in target_axes[len(source_axes):]
    )
    return source_granularity == target_granularity * refinement_factor


def is_fully_replicated(value: DistributedType) -> bool:
    """Return whether every placement owner holds the complete tensor."""

    return value.partial is None and all(
        isinstance(policy, SBPBroadCast) for policy in value.axis_policies
    )


def _has_partial(value: DistributedType) -> bool:
    return value.partial is not None or any(
        isinstance(policy, SBPPartial) for policy in value.axis_policies
    )


def _has_only_split_or_broadcast(value: DistributedType) -> bool:
    return all(
        isinstance(policy, (SBPSplit, SBPBroadCast))
        for policy in value.axis_policies
    )


def scale_split_units(
    split: SBPSplit,
    numerator: int,
    denominator: int,
) -> SBPSplit | None:
    """Scale explicit split units across pack/unpack element boundaries."""

    if numerator <= 0 or denominator <= 0:
        return None
    common = _greatest_common_divisor(numerator, denominator)
    numerator //= common
    denominator //= common
    stages: list[SplitStage] = []
    for stage in split.stages:
        distribution = stage.distribution
        if isinstance(distribution, ContiguousSplit):
            if distribution.granularity is None:
                scaled = distribution
            else:
                granularity = try_div_exactly(
                    distribution.granularity * numerator,
                    denominator,
                )
                if granularity is None:
                    return None
                scaled = ContiguousSplit(granularity)
        elif isinstance(distribution, BlockCyclicSplit):
            block = distribution.block_size * numerator
            if block % denominator:
                return None
            scaled = BlockCyclicSplit(block // denominator)
        else:  # pragma: no cover - sealed by SplitDistribution today.
            return None
        stages.append(SplitStage(stage.hierarchy_axes, scaled))
    return SBP.split(*stages)


def _greatest_common_divisor(lhs: int, rhs: int) -> int:
    while rhs:
        lhs, rhs = rhs, lhs % rhs
    return lhs


def local_shape(distributed_type: DistributedType) -> tuple[Dimension, ...]:
    """Return the maximum dense component shape held by one placement owner.

    Contiguous stages divide the active extent. Block-cyclic stages reserve
    enough whole blocks for the owner with the largest component. Partial and
    broadcast axes do not divide a tensor dimension.
    """

    result: list[Dimension] = []
    for dimension, policy in zip(
        distributed_type.tensor.shape,
        distributed_type.axis_policies,
    ):
        extent = dimension
        if isinstance(policy, SBPSplit):
            for stage in policy.stages:
                owners = 1
                for hierarchy_axis in stage.hierarchy_axes:
                    owners *= distributed_type.placement.hierarchy[hierarchy_axis]
                if isinstance(stage.distribution, ContiguousSplit):
                    extent = (
                        stage.distribution.granularity
                        if stage.distribution.granularity is not None
                        else ceil_div(extent, owners)
                    )
                elif isinstance(stage.distribution, BlockCyclicSplit):
                    block = stage.distribution.block_size
                    extent = ceil_div(ceil_div(extent, block), owners) * block
                else:  # pragma: no cover - sealed by SplitDistribution today.
                    raise IRSchemaError(
                        f"Unsupported split distribution {type(stage.distribution).__name__}."
                    )
        result.append(extent.simplify())
    return tuple(result)


def local_tensor_type(distributed_type: DistributedType) -> TensorType:
    """Return the dense per-owner tensor type used for physical sizing."""

    from triton.flagmega.ir.model import TensorType

    return TensorType(
        distributed_type.tensor.dtype,
        local_shape(distributed_type),
        distributed_type.tensor.layout,
    )


def placement_owner_count(distributed_type: DistributedType) -> int:
    result = 1
    for extent in distributed_type.placement.hierarchy:
        result *= extent
    return result


def is_fully_sharded_across_placement(distributed_type: DistributedType) -> bool:
    split_axes = {
        hierarchy_axis
        for policy in distributed_type.axis_policies
        if isinstance(policy, SBPSplit)
        for hierarchy_axis in policy.hierarchy_axes
    }
    return bool(distributed_type.placement.rank) and len(split_axes) == distributed_type.placement.rank


def leaf_candidate_policies(
    tensor: TensorType,
    placement: Placement,
    *,
    split_candidates: Callable[[TensorType, int, tuple[int, ...]], Sequence[SBPSplit]] | None = None,
) -> tuple[tuple[SBP, ...], ...]:
    """Port of nncase ``DistributedUtility.GetLeafCandidatePolicies``.

    Every tensor axis may be broadcast or split by any non-empty combination
    of placement axes when a fixed extent divides exactly.  Cartesian products
    are filtered by the same one-owner-axis-per-tensor-axis invariant.
    A target may supply the split distributions without introducing a compiler
    policy dependency into IR; the default remains portable contiguous SBP.
    """

    placement_combinations = tuple(
        axes
        for count in range(1, placement.rank + 1)
        for axes in combinations(range(placement.rank), count)
    )
    by_tensor_axis: list[tuple[SBP, ...]] = []
    for tensor_axis, dimension in enumerate(tensor.shape):
        values: list[SBP] = []
        for axes in placement_combinations:
            divisor = 1
            for axis in axes:
                divisor *= placement.hierarchy[axis]
            if divisor > 1 and (not dimension.is_fixed or dimension.fixed_value % divisor == 0):
                granularity = None if not dimension.is_fixed else dimension.fixed_value // divisor
                candidates = (
                    (SBP.split_contiguous(axes, granularity),)
                    if split_candidates is None
                    else split_candidates(tensor, tensor_axis, axes)
                )
                for candidate in candidates:
                    if candidate not in values:
                        values.append(candidate)
        values.append(SBP.broadcast())
        by_tensor_axis.append(tuple(values))
    return tuple(
        tuple(candidate)
        for candidate in product(*by_tensor_axis)
        if is_distributable(tensor, candidate, placement)
    )


def distributed_type_signature(value: DistributedType) -> str:
    policies = ",".join(str(policy) for policy in value.axis_policies)
    partial = "" if value.partial is None else f";partial={value.partial}"
    return f"{value.placement}:{policies}{partial}"


__all__ = [
    "BlockCyclicSplit",
    "ContiguousSplit",
    "Placement",
    "ReduceOp",
    "SBP",
    "SBPBroadCast",
    "SBPPartial",
    "SBPSplit",
    "SplitDistribution",
    "SplitStage",
    "distributed_type_signature",
    "is_distributable",
    "is_fully_sharded_across_placement",
    "is_fully_replicated",
    "is_local_shard_subview",
    "leaf_candidate_policies",
    "local_shape",
    "local_tensor_type",
    "placement_owner_count",
    "scale_split_units",
    "sharded_view_error",
    "placement_from_data",
    "sbp_from_data",
]
