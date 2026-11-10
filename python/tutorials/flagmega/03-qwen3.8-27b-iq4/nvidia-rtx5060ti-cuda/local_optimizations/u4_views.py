# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""U4-L tensor coordinates and concrete physical binding verification.

The records consume core PhysicalBuffer/MemSpan objects; allocation, generic
serialization, program lifetimes and formal substitution remain core-owned.
Lifetimes retain inclusive ordinals from the owning core plan. Mapping these
to actual program completion remains a separate integration obligation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod
from typing import Mapping

from triton.flagmega.ir.bufferization import MemSpan, PhysicalBuffer
from .u4_q8_abi import Q8Layout, exact, identity, integer, require

_ITEMSIZE = {"uint8": 1, "bfloat16": 2, "float32": 4}


@dataclass(frozen=True)
class TensorLayout:
    scalar_dtype: str
    shape: tuple[int, ...]
    lanes: tuple[int, ...] = ()

    def __post_init__(self):
        require(type(self.scalar_dtype) is str and self.scalar_dtype in _ITEMSIZE and type(self.shape) is tuple and bool(self.shape)
                and all(integer(n, 1) for n in self.shape) and type(self.lanes) is tuple
                and all(integer(n, 1) for n in self.lanes), "invalid scalar/lane TensorLayout")
        require(self.nbytes < 2**63, "tensor extent overflows 64-bit index")

    @property
    def itemsize(self):
        return _ITEMSIZE[self.scalar_dtype]

    @property
    def lane_count(self):
        return prod(self.lanes)

    @property
    def strides(self):
        return tuple(prod(self.shape[i + 1:]) * self.lane_count for i in range(len(self.shape)))

    @property
    def nbytes(self):
        return prod(self.shape) * self.lane_count * self.itemsize

    @property
    def dtype(self):
        return self.scalar_dtype + ("<" + ",".join(map(str, self.lanes)) + ">" if self.lanes else "")

    def scalar_offset(self, outer, lane=()):
        require(type(outer) is tuple and len(outer) == len(self.shape)
                and all(integer(i) and i < n for i, n in zip(outer, self.shape)), "outer coordinate outside tensor")
        require(type(lane) is tuple and len(lane) == len(self.lanes)
                and all(integer(i) and i < n for i, n in zip(lane, self.lanes)), "lane outside vector packet")
        return sum(i * s for i, s in zip(outer, self.strides)) + sum(
            i * prod(self.lanes[j + 1:]) for j, i in enumerate(lane))

    def to_data(self):
        return {"schema": "flagmega.u4-tensor-layout/v1", "dtype": self.dtype,
                "scalar_dtype": self.scalar_dtype, "shape": list(self.shape),
                "scalar_lane_shape": list(self.lanes), "scalar_lane_count": self.lane_count,
                "scalar_storage_strides": list(self.strides), "nbytes": self.nbytes}

    @classmethod
    def from_data(cls, data):
        require(isinstance(data, Mapping) and type(data.get("shape")) in (list, tuple)
                and type(data.get("scalar_lane_shape")) in (list, tuple), "missing tensor layout record")
        layout = cls(data.get("scalar_dtype"), tuple(data["shape"]), tuple(data["scalar_lane_shape"]))
        require(exact(data, layout.to_data()), "tensor lane/dtype/stride metadata differs")
        return layout

    def validate_tensor_type(self, tensor):
        from triton.flagmega.ir import TensorType
        require(isinstance(tensor, TensorType) and tensor.dtype.value == self.dtype
                and all(d.is_fixed for d in tensor.shape)
                and tuple(d.fixed_value for d in tensor.shape) == self.shape,
                "actual TensorType dtype/lanes/shape differs")


_SCALAR5120 = TensorLayout("bfloat16", (1, 5120))
_PACKED5120 = TensorLayout("bfloat16", (1, 640), (8,))
_Q_GATE = TensorLayout("bfloat16", (1, 24, 32), (8,))
SITE_LAYOUTS = {
    "linear_glu": (TensorLayout("bfloat16", (1, 17408)),),
    "attention_glu": (TensorLayout("bfloat16", (1, 17408)),),
    "linear_down": (_SCALAR5120,), "attention_down": (_SCALAR5120,),
    "linear_out": (_SCALAR5120,), "attention_out": (_SCALAR5120,),
    "linear_qkv": (TensorLayout("bfloat16", (1, 10240)),),
    "linear_z": (TensorLayout("bfloat16", (1, 6144)),),
    "attention_k": (TensorLayout("bfloat16", (1, 128), (8,)),),
    "attention_v": (TensorLayout("bfloat16", (1, 128), (8,)),),
    "attention_q_gate": (_Q_GATE, _Q_GATE),
    "logits": (TensorLayout("float32", (1, 248320)),),
}


def output_layouts(site):
    require(site in SITE_LAYOUTS, "unknown full12 site")
    return SITE_LAYOUTS[site]


def cyclic32_coordinate(cta, local):
    require(integer(cta) and cta < 96 and integer(local)
            and local < (64 if cta < 64 else 32), "inactive cyclic32 lane must be masked before addressing")
    return 32 * cta + 3072 * (local // 32) + local % 32


def cyclic32_packed_coordinate(cta, local):
    """The same scalar owner viewed through BF16<8>[1,640]."""
    h = cyclic32_coordinate(cta, local)
    return (0, h // 8), (h % 8,)


def q_gate_coordinate(native_row):
    require(integer(native_row) and native_row < 12288, "interleaved row outside tuple")
    head, field, dim = native_row // 512, native_row % 512 // 256, native_row % 256
    return field, (0, head, dim // 8), (dim % 8,)


def q_read_coordinate(x, y, dim):
    require(integer(x) and x < 24 and integer(y) and y < 4 and integer(dim) and dim < 256,
            "Q reader outside fixed96 owner map")
    return 256 * x + dim


def gate_read_coordinate(x, y, dim):
    require(integer(x) and x < 24 and integer(y) and y < 4 and integer(dim) and dim < 64,
            "gate reader outside fixed96 owner map")
    return 256 * x + 64 * y + dim


def gdn_z_coordinate(cta, local):
    require(integer(cta) and cta < 96 and integer(local) and local < 64, "GDN Z owner out of bounds")
    return 64 * cta + local


def output_coordinate(site, native_row):
    """Native row -> (tuple field, outer coordinate, complete lane tuple)."""
    layouts = output_layouts(site)
    if site == "attention_q_gate":
        return q_gate_coordinate(native_row)
    layout = layouts[0]
    require(integer(native_row) and native_row < prod(layout.shape) * layout.lane_count,
            "native output row out of bounds")
    if layout.lanes:
        return 0, (0, native_row // 8), (native_row % 8,)
    return 0, (0, native_row), ()


@dataclass(frozen=True)
class PhysicalView:
    """One actual logical view rooted in a core MemSpan and scoped pool binding.

    frame_offset binds function-relative PhysicalBuffer.start to the arena.
    scope_stride_bytes describes replicated block-private pools, not tensor strides.
    plan ordinals must cover the last read; program completion is verified at
    integration. Shared is forbidden
    here; intra-entry GDN Shared storage is checked by its unchanged validator.
    """
    name: str
    span: MemSpan
    layout: TensorLayout
    arena: str
    frame: str
    epoch: str
    live_start: int
    live_end: int
    frame_offset: int = 0
    sharing_scope: str = "chip"
    coordinate_space: str = "canonical_global"
    scope_stride_bytes: int = 0
    scope_count: int = 1
    writable: bool = False
    alias_of: str | None = None
    coordinate_map: str = "identity"
    block_pool: GDNZBlockPool | None = None

    def __post_init__(self):
        require(isinstance(self.span, MemSpan) and isinstance(self.span.buffer, PhysicalBuffer)
                and isinstance(self.layout, TensorLayout), "view needs actual core MemSpan/PhysicalBuffer")
        require(all(identity(s) for s in (self.name, self.arena, self.frame, self.epoch)), "missing view identity")
        require(all(integer(n) for n in (self.live_start, self.live_end, self.frame_offset, self.scope_stride_bytes))
                and self.live_start <= self.live_end and integer(self.scope_count, 1)
                and type(self.writable) is bool, "invalid view lifetime/frame/scope")
        root = self.span.buffer
        if self.block_pool is None:
            require(root.memory_space == "workspace", "Shared/private memory space cannot escape a launched program")
        else:
            require(isinstance(self.block_pool, GDNZBlockPool), "untyped private-pool proof")
            self.block_pool.validate_view(self)
        require(root.function is not None, "workspace root needs a function identity")
        require(integer(root.live_start) and integer(root.live_end) and root.live_start <= self.live_start
                and self.live_end <= root.live_end, "view outlives physical root completion lifetime")
        require(self.span.nbytes == self.layout.nbytes and self.span.byte_offset >= 0
                and self.span.byte_offset + self.span.nbytes <= root.nbytes, "view extent differs from actual root span")
        require(self.frame_offset % root.alignment == 0 and root.offset % root.alignment == 0
                and self.offset % self.layout.itemsize == 0, "misaligned frame/root/typed view")
        require(self.offset + self.scope_stride_bytes * (self.scope_count - 1) + self.span.nbytes < 2**63,
                "view address overflows signed 64-bit")
        require(self.alias_of is None or identity(self.alias_of), "invalid alias identity")
        if self.coordinate_space == "canonical_global":
            require(self.sharing_scope == "chip" and self.scope_stride_bytes == 0 and self.scope_count == 1,
                    "canonical global view retains a stale private scope stride")
        else:
            require((self.coordinate_space == "parent_shard_local" or
                     self.coordinate_space == "local" and self.block_pool is not None) and self.sharing_scope == "block"
                    and self.scope_count == (self.block_pool.ctas if self.block_pool is not None else 96)
                    and self.scope_stride_bytes >= root.offset + root.nbytes
                    and self.scope_stride_bytes % root.alignment == 0,
                    "private view needs all96 disjoint aligned parent pool scopes")

    @property
    def offset(self):
        return self.frame_offset + self.span.offset

    @property
    def root_id(self):
        return self.span.buffer.id

    def intervals(self):
        return tuple((self.offset + i * self.scope_stride_bytes,
                      self.offset + i * self.scope_stride_bytes + self.span.nbytes) for i in range(self.scope_count))

    def to_data(self):
        return {"schema": "flagmega.u4-physical-view/v1", "name": self.name, "physical_buffer": self.root_id,
                "mem_span": self.span.to_data(), "layout": self.layout.to_data(), "arena": self.arena,
                "frame": self.frame, "epoch": self.epoch, "live_start": self.live_start, "live_end": self.live_end,
                "frame_offset": self.frame_offset, "sharing_scope": self.sharing_scope,
                "coordinate_space": self.coordinate_space, "scope_stride_bytes": self.scope_stride_bytes,
                "scope_count": self.scope_count, "writable": self.writable, "alias_of": self.alias_of,
                "coordinate_map": self.coordinate_map,
                **({"block_pool": self.block_pool.to_data()} if self.block_pool is not None else {})}

    def validate_buffer_descriptor(self, descriptor):
        """Bind layout and alias facts to actual normal bufferization output."""
        from triton.flagmega.ir.bufferization import BufferDescriptor
        require(isinstance(descriptor, BufferDescriptor), "expected core BufferDescriptor")
        require(descriptor.mem_span == self.span and descriptor.physical_id == self.root_id,
                "descriptor root/span identity differs")
        if self.block_pool is not None:
            require(descriptor == self.block_pool.descriptor, "descriptor differs from proven private view")
        shape = descriptor.component_shape if self.block_pool is not None else descriptor.shape
        require(descriptor.dtype.value == self.layout.dtype and shape == self.layout.shape,
                "descriptor type/lane geometry differs")
        # Core BufferDescriptor strides are vector-element strides; the local
        # primitive ABI records scalar strides. Do not compare unequal units.
        require(tuple(s * self.layout.lane_count for s in descriptor.strides) == self.layout.strides,
                "descriptor scalar storage strides differ")
        require(descriptor.alias_of == self.alias_of and descriptor.function == self.span.buffer.function,
                "descriptor alias/function identity differs")
        life = ((descriptor.mem_span.buffer.live_start, descriptor.mem_span.buffer.live_end)
                if self.block_pool is not None and descriptor == self.block_pool.allocation
                else (descriptor.live_start, descriptor.live_end))
        require(life == (self.live_start, self.live_end), "descriptor/root lifetime differs")
        if descriptor.distributed_type is not None:
            require(descriptor.distributed_storage_kind.exposes_logical_coordinates ==
                    (self.coordinate_space == "canonical_global"), "descriptor storage kind differs")


def canonical_read_view(source, *, name, coordinate_map, live_start=None, live_end=None):
    require(source.coordinate_space == "canonical_global", "read view must retain canonical root")
    require(coordinate_map in ("identity", "cyclic32_5120", "q_replicated_y4", "gate_partition_y4"),
            "unsupported read coordinate map")
    if coordinate_map == "cyclic32_5120":
        require(source.layout in (_SCALAR5120, _PACKED5120),
                "cyclic32 requires complete5120 scalar or packedBF16x8 backing")
    if coordinate_map in ("q_replicated_y4", "gate_partition_y4"):
        require(source.layout == _Q_GATE, "Q/gate reader requires full packed tuple-field strides")
    return replace(source, name=name, alias_of=source.name, writable=False, coordinate_map=coordinate_map,
                   live_start=source.live_start if live_start is None else live_start,
                   live_end=source.live_end if live_end is None else live_end)


def validate_q8_view(view, schema):
    from .u4_q8_abi import validate_q8_schema
    layout = validate_q8_schema(schema)
    require(view.layout == TensorLayout("uint8", layout.shape) and view.coordinate_space == "canonical_global"
            and view.sharing_scope == "chip" and view.span.buffer.alignment >= 16 and view.offset % 16 == 0
            and view.span.byte_offset == 0 and view.span.buffer.nbytes == layout.nbytes
            and view.alias_of is None and view.coordinate_map == "identity",
            "Q8 must be a distinct complete canonical aligned allocation")


def validate_gdn_z(source, destination):
    require(source.layout == SITE_LAYOUTS["linear_z"][0] and source.coordinate_space == "canonical_global",
            "GDN Z gather source must be complete canonical BF16[1,6144]")
    values, ctas = ((128, 48) if destination.layout == TensorLayout("bfloat16", (1, 128)) else (64, 96))
    require(destination.layout == TensorLayout("bfloat16", (1, values)) and destination.sharing_scope == "block"
            and (destination.coordinate_space == "parent_shard_local" or
                 destination.coordinate_space == "local" and destination.block_pool is not None)
            and destination.scope_count == ctas
            and destination.writable and destination.alias_of is None
            and destination.root_id != source.root_id, "GDN Z destination must retain private64-value ABI")
    require(source.frame == destination.frame and source.epoch == destination.epoch
            and source.live_start <= destination.live_end and destination.live_start <= source.live_end,
            "GDN Z gather source and destination must share a live frame/epoch")
    _require_disjoint(source, destination)


def validate_q_gate_roots(q, gate, derived_q):
    require(q.layout == gate.layout == _Q_GATE and q.coordinate_space == gate.coordinate_space == "canonical_global"
            and q.root_id != gate.root_id, "Q and gate require distinct canonical packed roots")
    _require_disjoint(q, gate)
    require(bool(derived_q), "derived Q roots must be enumerated")
    names, roots = set(), {q.root_id, gate.root_id}
    for view in derived_q:
        require(view.name not in names and view.root_id not in roots and view.alias_of is None
                and view.coordinate_space == "parent_shard_local" and view.sharing_scope == "block",
                "statistics/q_norm/RoPE must retain distinct private derived roots")
        names.add(view.name)
        roots.add(view.root_id)
        for other in (q, gate):
            _require_disjoint(view, other)
    validate_live_views(tuple(derived_q))


def _overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def _require_disjoint(a, b):
    require(a.root_id != b.root_id, "simultaneous independent values alias the same physical root")
    if a.arena == b.arena:
        require(not any(_overlap(x, y) for x in a.intervals() for y in b.intervals()), "live arena spans overlap")


def validate_live_views(views):
    """Prove live physical reuse and explicit read aliases, across nested frames/epochs."""
    by_name, roots = {}, {}
    for view in views:
        require(isinstance(view, PhysicalView) and view.name not in by_name, "duplicate/untyped view")
        by_name[view.name] = view
        root = view.span.buffer
        require(root.id not in roots or roots[root.id] == root, "physical ID refers to inconsistent root metadata")
        roots[root.id] = root
    for view in views:
        if view.alias_of is not None:
            require(view.alias_of in by_name, "view aliases missing root binding")
            owner = by_name[view.alias_of]
            require(not view.writable and view.root_id == owner.root_id and view.span == owner.span
                    and view.arena == owner.arena and view.frame_offset == owner.frame_offset
                    and view.epoch == owner.epoch and owner.live_start <= view.live_start <= view.live_end <= owner.live_end
                    and view.scope_stride_bytes == owner.scope_stride_bytes and view.scope_count == owner.scope_count,
                    "read alias escapes root/span/epoch/lifetime/scope")
    def ultimate(view):
        seen = set()
        while view.alias_of is not None:
            require(view.name not in seen, "cyclic alias chain")
            seen.add(view.name)
            view = by_name[view.alias_of]
        return view.name
    for i, a in enumerate(views):
        for b in views[i + 1:]:
            if a.live_start <= b.live_end and b.live_start <= a.live_end:
                if a.root_id == b.root_id and ultimate(a) == ultimate(b):
                    continue
                _require_disjoint(a, b)


@dataclass(frozen=True)
class FrameBinding:
    """One function invocation's concrete pool footprint and completion epoch."""
    call_path: str
    function: str
    epoch: str
    arena: str
    offset: int
    nbytes: int
    live_start: int
    live_end: int
    views: tuple[PhysicalView, ...]
    parent: str | None = None

    def __post_init__(self):
        require(all(identity(x) for x in (self.call_path, self.function, self.epoch, self.arena))
                and all(integer(x) for x in (self.offset, self.nbytes, self.live_start, self.live_end))
                and self.nbytes > 0 and self.live_start <= self.live_end, "invalid concrete function frame")
        for view in self.views:
            require(view.frame == self.call_path and view.span.buffer.function == self.function
                    and view.epoch == self.epoch and view.arena == self.arena
                    and self.live_start <= view.live_start <= view.live_end <= self.live_end,
                    "view escapes its function frame/epoch/completion")
            require(all(self.offset <= start < end <= self.offset + self.nbytes for start, end in view.intervals()),
                    "view/scope lies outside frame allocation")


def validate_frames(frames, *, expected_call_paths=None):
    by_path = {f.call_path: f for f in frames}
    require(len(by_path) == len(frames), "duplicate call/frame identity")
    if expected_call_paths is not None:
        require(len(set(expected_call_paths)) == len(expected_call_paths) and set(by_path) == set(expected_call_paths),
                "incomplete or stale all-caller frame binding")
    for frame in frames:
        if frame.parent is not None:
            require(frame.parent in by_path and frame.parent != frame.call_path, "unknown/recursive frame parent")
            parent = by_path[frame.parent]
            require(parent.arena == frame.arena and parent.epoch == frame.epoch
                    and parent.offset <= frame.offset and frame.offset + frame.nbytes <= parent.offset + parent.nbytes
                    and parent.live_start <= frame.live_start <= frame.live_end <= parent.live_end,
                    "nested frame escapes parent lifetime/span/epoch")
    def ancestors(frame):
        result = set()
        while frame.parent is not None:
            require(frame.parent not in result, "recursive frame chain")
            result.add(frame.parent)
            frame = by_path[frame.parent]
        return result
    for i, a in enumerate(frames):
        aa = ancestors(a)
        for b in frames[i + 1:]:
            if a.call_path in ancestors(b) or b.call_path in aa:
                continue
            if a.arena == b.arena and a.live_start <= b.live_end and b.live_start <= a.live_end:
                require(not _overlap((a.offset, a.offset + a.nbytes), (b.offset, b.offset + b.nbytes)),
                        "overlapping live invocation frames/epochs")
    validate_live_views(tuple(v for f in frames for v in f.views))


def validate_call_view(actual, bound, formal):
    """Check one actual/formal binding after core has substituted the caller root.

    This does not perform substitution. ``formal`` is the normal callee
    BufferDescriptor; ``bound`` is its materialized caller PhysicalView.
    """
    from triton.flagmega.ir.bufferization import BufferDescriptor
    require(isinstance(formal, BufferDescriptor) and isinstance(actual, PhysicalView)
            and isinstance(bound, PhysicalView), "call binding needs typed formal and actual views")
    coordinate_match = bound.coordinate_space == actual.coordinate_space
    if bound.block_pool is not None and actual.block_pool is not None:
        validate_gdn_z_reader(actual, bound)
        coordinate_match = True
    require(bound.span == actual.span and bound.arena == actual.arena
            and bound.frame_offset == actual.frame_offset and bound.epoch == actual.epoch
            and bound.scope_stride_bytes == actual.scope_stride_bytes and bound.scope_count == actual.scope_count
            and coordinate_match
            and actual.live_start <= bound.live_start <= bound.live_end <= actual.live_end,
            "callee binding changed caller root/offset/scope/epoch/completion lifetime")
    if bound.block_pool is not None:
        if bound == actual:
            # A normal identity reshard can disappear when producer and reader
            # have exactly the same typed shard. This is a use of the same
            # writable allocation, not a fabricated read-only alias.
            require(formal == bound.block_pool.allocation == bound.block_pool.descriptor
                    and formal.distributed_backing_type is None
                    and bound.coordinate_space == "local", "GDN direct formal lost its actual own-root")
        else:
            require(formal.distributed_type in (_gdn_z_type(values=bound.layout.shape[1]), _gdn_z_type(explicit=False, values=bound.layout.shape[1]))
                    and formal.distributed_backing_type == _gdn_z_type(values=bound.layout.shape[1])
                    and bound.coordinate_space == "parent_shard_local", "GDN formal lost exact parent-shard backing")
    shape = formal.component_shape if bound.block_pool is not None else formal.shape
    require(formal.dtype.value == bound.layout.dtype and shape == bound.layout.shape
            and tuple(s * bound.layout.lane_count for s in formal.strides) == bound.layout.strides,
            "callee formal dtype/lanes/full scalar strides differ from actual backing")


def _gdn_z_type(*, explicit=True, values=64):
    from triton.flagmega.ir import DistributedType, Placement, SBP, tensor_type
    require(values in (64, 128), "unsupported GDN private shard size")
    return DistributedType(tensor_type("bfloat16", (1, 6144)),
        (SBP.broadcast(), SBP.split_contiguous((0, 1), values if explicit else None)),
        Placement((4, 12) if values == 128 else (4, 24), "yx", "bb"))


@dataclass(frozen=True)
class GDNZBlockPool:
    """Proof for the existing global-backed GDN Z block pool, not a new allocator.

    ``allocation`` is the own-root descriptor in the actual BufferPlan;
    ``descriptor`` is either that descriptor or a checked read alias. Normal
    local ABI and runtime-pool records must match those actual core objects.
    Scope payload, root size and whole-pool allocation are different extents.
    """
    allocation: object
    descriptor: object
    plan: object
    buffer_abi: Mapping
    runtime_pool: Mapping
    call_scope: GDNZCallScope | None = None

    def __post_init__(self):
        from triton.flagmega.ir.model import canonical_attributes
        from triton.flagmega.ir.bufferization import BufferDescriptor, BufferPlan, AliasKind
        from triton.flagmega.ir.distributed_storage import DistributedBufferStorageKind
        from triton.flagmega.codegen.triton.call_abi import describe_local_buffer_abi
        a, d, plan = self.allocation, self.descriptor, self.plan
        require(isinstance(a, BufferDescriptor) and isinstance(d, BufferDescriptor)
                and isinstance(plan, BufferPlan) and isinstance(self.buffer_abi, Mapping)
                and isinstance(self.runtime_pool, Mapping), "private pool needs actual core plan/descriptors/ABI")
        require(plan.buffer_map.get(a.id) == a and plan.physical_buffer_map.get(a.physical_id) == a.mem_span.buffer,
                "private root is not the actual allocation in the buffer plan")
        root = a.mem_span.buffer
        values = a.component_shape[1]
        require(values in (64, 128), "GDN Z private shape must select an explicit64/128 contract")
        ctas = 6144 // values
        require(a.function == root.function and a.alias is None and root.role == "workspace"
                and a.dtype.value == "bfloat16" and a.shape == (1, 6144) and a.component_shape == (1, values)
                and a.strides == (values, 1) and a.distributed_type == _gdn_z_type(values=values)
                and a.distributed_backing_type is None
                and a.distributed_storage_kind is DistributedBufferStorageKind.COMPACT_LOCAL
                and a.mem_span.byte_offset == 0 and a.nbytes == root.nbytes == values * 2,
                "GDN Z requires its own exact compact split64 BF16 root")
        space = plan.memory_space_map.get(root.memory_space)
        function = plan.function_map.get(root.function)
        require(space is not None and function is not None and space.name == "block_local_data"
                and a.storage == space.name and space.kind == "device" and space.sharing_scope.value == "block"
                and space.allocation_scope.value == "function" and space.supports_lifetime_reuse,
                "private Z must use a global-backed block-scoped function pool")
        pool = function.memory_pool_map.get(space.name)
        require(pool is not None and root.id in pool.allocations and pool.scope_bytes >= root.offset + root.nbytes
                and pool.scope_bytes == space.allocation_bytes(pool.scope_bytes)
                and root.offset % root.alignment == 0 and pool.alignment >= root.alignment
                and pool.scope_bytes % root.alignment == 0 and a.alignment == root.alignment,
                "private root is absent, misaligned or outside its actual function pool")
        require(integer(root.live_start) and integer(root.live_end)
                and integer(a.live_start) and integer(a.live_end)
                and root.live_start <= a.live_start <= a.live_end <= root.live_end,
                "private allocation lacks bounded physical lifetime")
        # Audit all allocations in the same actual pool, including unrelated
        # values that might overlap Z while live. Scope replication preserves
        # the per-scope proof only if the full high-water mark fits the pitch.
        for other_id in pool.allocations:
            other = plan.physical_buffer_map.get(other_id)
            require(other is not None and other.function == root.function and other.memory_space == space.name
                    and 0 <= other.offset and other.offset + other.nbytes <= pool.scope_bytes
                    and integer(other.live_start) and integer(other.live_end), "invalid peer allocation in private pool")
            if other.id != root.id and root.live_start <= other.live_end and other.live_start <= root.live_end:
                require(not _overlap((root.offset, root.offset + root.nbytes),
                                     (other.offset, other.offset + other.nbytes)), "live private pool allocations overlap")
        if self.call_scope is not None:
            require(isinstance(self.call_scope, GDNZCallScope) and self.call_scope.plan == plan
                    and self.call_scope.call.callee == root.function, "private call scope differs from actual callee")
            self.call_scope.resolve(a)
        scoped = self.call_scope is not None
        allocation_scopes = 1 if scoped else self.runtime_pool.get("scope_count")
        require(allocation_scopes == 1 if scoped else allocation_scopes in (ctas, 96),
                "runtime pool must cover exact GDN owners or enclosing96-scope program")
        expected_pool = {"name": space.name, "storage": space.name, "scope": "block",
                         "scope_count": allocation_scopes, "scope_nbytes": pool.scope_bytes,
                         "nbytes": pool.scope_bytes * allocation_scopes, "scope_index": "program_id_x", "scope_local": scoped}
        require(all(k in self.runtime_pool and exact(self.runtime_pool[k], v) for k, v in expected_pool.items()),
                "runtime pool is not the full96-scope planned allocation")
        require(d.mem_span == a.mem_span and d.dtype == a.dtype and d.shape == a.shape
                and d.strides == a.strides and d.storage == a.storage and d.function == a.function
                and d.alignment == a.alignment and d.distributed_type in (_gdn_z_type(values=values), _gdn_z_type(explicit=False, values=values))
                and d.distributed_storage_kind == a.distributed_storage_kind
                and integer(d.live_start) and integer(d.live_end)
                and root.live_start <= d.live_start <= d.live_end <= root.live_end,
                "GDN Z reader changed allocation/type/scope/lifetime")
        if d != a:
            require(d.id != a.id and d.alias is not None and d.alias.source == a.id
                    and d.alias.kind is AliasKind.VIEW and d.distributed_backing_type == a.distributed_type,
                    "retained GDN reader needs typed alias and exact split64 backing")
        expected_abi = describe_local_buffer_abi(d, plan, pool_scope_count=(
            self.call_scope.parent_runtime_pool["scope_count"] if scoped else allocation_scopes))
        if scoped:
            from triton.flagmega.codegen.triton.runtime_binding import _make_pool_abi_scope_local
            _make_pool_abi_scope_local(plan, expected_abi)
        require(exact(self.buffer_abi, expected_abi), "private ABI differs from actual core root/type/stride/scope")
        require(expected_abi["coordinate_space"] == ("local" if d == a else "parent_shard_local"),
                "core did not derive the expected GDN own-root/parent-shard coordinates")
        object.__setattr__(self, "buffer_abi", canonical_attributes(self.buffer_abi))
        object.__setattr__(self, "runtime_pool", canonical_attributes(expected_pool))

    @property
    def values_per_cta(self):
        return self.allocation.component_shape[1]

    @property
    def ctas(self):
        return 6144 // self.values_per_cta

    def validate_view(self, view):
        d = self.descriptor
        values, ctas = self.values_per_cta, self.ctas
        parent = self.call_scope.parent_runtime_pool if self.call_scope is not None else self.runtime_pool
        offset = self.call_scope.pool_offset("block_local_data") if self.call_scope is not None else 0
        life = ((d.mem_span.buffer.live_start, d.mem_span.buffer.live_end) if d == self.allocation
                else (d.live_start, d.live_end))
        if self.call_scope is not None:
            require(view.frame == self.call_scope.frame and view.epoch == self.call_scope.epoch,
                    "private view escaped actual caller frame/epoch")
        require(view.name == d.id and view.span == d.mem_span
                and view.layout == TensorLayout("bfloat16", (1, values))
                and view.arena == parent["name"] and view.frame_offset == offset
                and view.sharing_scope == "block" and view.scope_count == ctas
                and view.scope_stride_bytes == parent["scope_nbytes"]
                and view.coordinate_space == self.buffer_abi["coordinate_space"]
                and view.alias_of == d.alias_of
                and view.writable == (d == self.allocation)
                and (view.live_start, view.live_end) == life,
                "physical view does not retain the proven private root/pool/lifetime/alias")
        require(view.offset + (ctas - 1) * view.scope_stride_bytes + view.span.nbytes <= parent["nbytes"],
                "private view's last scope escapes runtime allocation")

    def to_data(self):
        from triton.flagmega.codegen.triton.call_abi import describe_local_buffer_abi
        return {"schema": "flagmega.u4-gdn-z-pool/v1", "allocation": self.allocation.id,
                "descriptor": self.descriptor.to_data(),
                "buffer_abi": _plain_data(self.buffer_abi), "runtime_pool": dict(self.runtime_pool),
                "call_scope": None if self.call_scope is None else self.call_scope.to_data()}


def bind_gdn_z_gather(source, destination, plan, *, source_abi, destination_abi, runtime_pool, frame, epoch, call_scope=None):
    """Adapt normal gather descriptors without changing allocation or lifetime.

    The supplied pool is the actual runtime_binding.pools entry. A scope-local
    callee additionally requires GDNZCallScope with the actual enclosing96-scope
    runtime pool and core call-frame binding. Core liveness ordinals are
    preserved; I0 still binds them to the program completion schedule.
    """
    from triton.flagmega.ir.bufferization import BufferDescriptor, BufferPlan
    from triton.flagmega.codegen.triton.call_abi import describe_local_buffer_abi
    require(isinstance(source, BufferDescriptor) and isinstance(plan, BufferPlan), "missing canonical source core descriptor")
    require(plan.buffer_map.get(source.id) == source
            and plan.physical_buffer_map.get(source.physical_id) == source.mem_span.buffer
            and source.alias is None and source.dtype.value == "bfloat16" and source.shape == (1, 6144)
            and source.strides == (6144, 1) and source.nbytes == 12288
            and source.distributed_storage_kind.value == "canonical_global"
            and source.mem_span.buffer.memory_space == "workspace"
            and source.mem_span.buffer.role == "workspace" and source.function == destination.function,
            "gather source lost canonical workspace root/type/span")
    source_space = plan.memory_space_map.get("workspace")
    require(source_space is not None and source_space.kind == "device"
            and source_space.sharing_scope.value == "chip"
            and exact(source_abi, describe_local_buffer_abi(source, plan)), "canonical source ABI/pool differs")
    proof = GDNZBlockPool(destination, destination, plan, destination_abi, runtime_pool, call_scope)
    src = PhysicalView(source.id, source.mem_span, TensorLayout("bfloat16", (1, 6144)), "workspace",
                       frame, epoch, source.live_start, source.live_end,
                       frame_offset=call_scope.pool_offset("workspace") if call_scope is not None else 0)
    if call_scope is not None:
        call_scope.resolve(source)
    dst = PhysicalView(destination.id, destination.mem_span, TensorLayout("bfloat16", (1, proof.values_per_cta)),
                       runtime_pool["name"], frame, epoch, destination.mem_span.buffer.live_start, destination.mem_span.buffer.live_end,
                       frame_offset=call_scope.pool_offset("block_local_data") if call_scope is not None else 0,
                       sharing_scope="block", coordinate_space="local", scope_stride_bytes=(
                           call_scope.parent_runtime_pool["scope_nbytes"] if call_scope is not None else runtime_pool["scope_nbytes"]),
                       scope_count=proof.ctas, writable=True, block_pool=proof)
    validate_gdn_z(src, dst)
    src.validate_buffer_descriptor(source)
    dst.validate_buffer_descriptor(destination)
    return src, dst


def make_gdn_z_reader(gathered, *, name, live_start=None, live_end=None):
    """Construct a real typed read alias; no IR mutation or pointer rewrite.

    The caller must integrate this descriptor through normal core planning and
    validate the eventual GDN operand. A successful construction is not proof
    that the actual generated GDN call consumes this alias.
    """
    from triton.flagmega.ir.bufferization import AliasInfo, AliasKind
    from triton.flagmega.codegen.triton.call_abi import describe_local_buffer_abi
    require(isinstance(gathered, PhysicalView) and gathered.block_pool is not None
            and gathered.writable and gathered.alias_of is None, "GDN reader needs a proven gathered own-root")
    proof = gathered.block_pool
    require(identity(name) and name != gathered.name and name not in proof.plan.buffer_map, "reader requires fresh alias identity")
    descriptor = replace(proof.allocation, id=name, alias=AliasInfo(gathered.name, AliasKind.VIEW), role="view",
                         distributed_backing_type=proof.allocation.distributed_type,
                         live_start=gathered.live_start if live_start is None else live_start,
                         live_end=gathered.live_end if live_end is None else live_end)
    abi = describe_local_buffer_abi(descriptor, proof.plan, pool_scope_count=(
        proof.call_scope.parent_runtime_pool["scope_count"] if proof.call_scope is not None
        else proof.runtime_pool["scope_count"]))
    if proof.call_scope is not None:
        from triton.flagmega.codegen.triton.runtime_binding import _make_pool_abi_scope_local
        _make_pool_abi_scope_local(proof.plan, abi)
    return descriptor, bind_gdn_z_reader(gathered, descriptor, abi)


def validate_gdn_z_reader(gathered, reader):
    require(isinstance(gathered, PhysicalView) and isinstance(reader, PhysicalView)
            and gathered.block_pool is not None and reader.block_pool is not None
            and gathered.block_pool.allocation == reader.block_pool.allocation
            and gathered.block_pool.plan == reader.block_pool.plan
            and gathered.block_pool.runtime_pool == reader.block_pool.runtime_pool
            and gathered.block_pool.call_scope == reader.block_pool.call_scope
            and reader.frame == gathered.frame and reader.epoch == gathered.epoch,
            "GDN reader escaped its proven gather root/frame/epoch")
    if reader == gathered:
        require(reader.block_pool.descriptor == reader.block_pool.allocation
                and reader.coordinate_space == "local" and reader.writable
                and reader.alias_of is None, "GDN direct reader is not the actual gathered own-root")
        validate_live_views((gathered,))
    else:
        require(reader.coordinate_space == "parent_shard_local" and not reader.writable
                and reader.alias_of == gathered.name,
                "GDN read alias lost its exact parent-shard backing or read permission")
        validate_live_views((gathered, reader))
    reader.validate_buffer_descriptor(reader.block_pool.descriptor)
    return reader.block_pool.buffer_abi


def _plain_data(value):
    if isinstance(value, Mapping):
        return {k: _plain_data(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_data(v) for v in value]
    return value


@dataclass(frozen=True)
class GDNZCallScope:
    """One real callee frame inside the caller's full96-scope block pool.

    Callee liveness and caller-frame liveness are separate ordinal domains;
    neither is silently translated into the other. The core call binding is
    the authority for their nesting and physical frame identity.
    """
    plan: object
    call: object
    parent_runtime_pool: Mapping
    epoch: str

    def __post_init__(self):
        from triton.flagmega.ir.bufferization import BufferPlan, CallBufferBinding
        from triton.flagmega.ir.model import canonical_attributes
        require(isinstance(self.plan, BufferPlan) and isinstance(self.call, CallBufferBinding)
                and self.plan.call_map.get(self.call.call) == self.call and identity(self.epoch),
                "scope-local ABI requires an actual core call binding and epoch")
        plan, call = self.plan, self.call
        require(not call.loop_carries and call.caller != call.callee, "unsupported recursive/loop-carried private frame")
        for memory in ("workspace", "block_local_data"):
            binding = call.memory_pool_map.get(memory)
            caller = plan.function_map[call.caller].memory_pool_map.get(memory)
            callee = plan.function_map[call.callee].memory_pool_map.get(memory)
            require(binding is not None and caller is not None and callee is not None,
                    "call lacks a complete source/destination function pool binding")
            root = plan.physical_buffer_map.get(binding.allocation)
            require(root is not None and root.function == call.caller and root.memory_space == memory
                    and root.role == "call_memory_pool" and root.id in caller.allocations
                    and root.offset == binding.offset and root.nbytes == binding.scope_bytes == callee.scope_bytes
                    and 0 <= root.offset and root.offset + root.nbytes <= caller.scope_bytes
                    and root.offset % callee.alignment == 0 and root.alignment >= callee.alignment
                    and integer(root.live_start) and integer(root.live_end),
                    "scope-local pointer frame does not match core allocation/span/lifetime")
        parent = plan.function_map[call.caller].memory_pool_map["block_local_data"]
        expected = {"name": "block_local_data", "storage": "block_local_data", "scope": "block",
                    "scope_nbytes": parent.scope_bytes, "scope_count": 96, "scope_local": False,
                    "scope_index": "program_id_x", "nbytes": 96 * parent.scope_bytes}
        require(isinstance(self.parent_runtime_pool, Mapping)
                and all(k in self.parent_runtime_pool and exact(self.parent_runtime_pool[k], v) for k, v in expected.items()),
                "scope-local pointer lacks the actual enclosing96-scope runtime pool")
        object.__setattr__(self, "parent_runtime_pool", canonical_attributes(expected))

    @property
    def frame(self):
        return self.call.caller + "/" + self.call.call

    def pool_offset(self, memory):
        return self.call.memory_pool_map[memory].offset

    def resolve(self, descriptor):
        require(descriptor.function == self.call.callee, "descriptor belongs to a different callee frame")
        actual = self.plan.resolve_call_buffer(self.call.call, descriptor.id)
        binding = self.call.memory_pool_map[descriptor.mem_span.buffer.memory_space]
        root = self.plan.physical_buffer_map[binding.allocation]
        require(actual.mem_span.buffer == root and actual.mem_span.byte_offset == descriptor.mem_span.offset
                and actual.nbytes == descriptor.nbytes and actual.dtype == descriptor.dtype
                and actual.strides == descriptor.strides and actual.shape == descriptor.shape,
                "core call substitution changed GDN scalar/root/span coordinates")
        return actual

    def to_data(self):
        return {"schema": "flagmega.u4-gdn-z-call-scope/v1", "call": self.call.to_data(),
                "epoch": self.epoch, "parent_runtime_pool": dict(self.parent_runtime_pool),
                "frames": {memory: self.plan.physical_buffer_map[binding.allocation].to_data()
                           for memory, binding in self.call.memory_pool_map.items()}}


def bind_gdn_z_reader(gathered, descriptor, reader_abi):
    """Validate the actual normal-plan GDN operand, including refined type.

    The gathered root's physical lifetime includes every alias consumer; the
    producer's logical descriptor may end before the retained reader finishes.
    """
    require(isinstance(gathered, PhysicalView) and gathered.block_pool is not None
            and gathered.writable and gathered.alias_of is None, "reader needs proven gathered allocation")
    proof = gathered.block_pool
    registered = proof.plan.buffer_map.get(descriptor.id)
    require(registered is None or registered == descriptor, "reader differs from its actual plan descriptor")
    reader_proof = replace(proof, descriptor=descriptor, buffer_abi=reader_abi)
    if proof.call_scope is not None and registered is not None:
        proof.call_scope.resolve(descriptor)
    if descriptor == proof.allocation:
        require(registered == descriptor and reader_proof == proof,
                "GDN direct read must retain the actual registered allocation and identical ABI")
        validate_gdn_z_reader(gathered, gathered)
        return gathered
    reader = replace(gathered, name=descriptor.id, writable=False, alias_of=descriptor.alias_of,
                     coordinate_space="parent_shard_local", block_pool=reader_proof,
                     live_start=descriptor.live_start, live_end=descriptor.live_end)
    validate_gdn_z_reader(gathered, reader)
    return reader


def fullhead48_gdn_z_coordinate(cta, local):
    require(integer(cta) and cta < 48 and integer(local) and local < 128,
            "fullhead48 GDN Z owner out of bounds")
    return 128 * cta + local
