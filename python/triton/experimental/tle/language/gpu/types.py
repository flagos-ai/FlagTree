# flagtree tle

import triton.language.core as tl
from typing import Optional, List, Tuple
from abc import abstractmethod
from triton._C.libtriton import ir
from triton.language.semantic import TritonSemantic


class scope():
    """Storage type enum, defines storage location for TLE buffers"""
    NVIDIA = ['share_memory', 'tensor_memory']

    def __init__(self, name: str):
        self.name = name
        assert name in scope.NVIDIA, name

    def __repr__(self):
        return self.name

    def to_ir(self, builder: ir.builder) -> None:
        raise NotImplementedError(f"{self.__class__.__name__}.to_ir() must be overridden in subclasses")


smem = scope('share_memory')
tmem = scope('tensor_memory')


def _storage_to_memdesc_space(storage: scope) -> str:
    if storage is smem:
        return "smem"
    if storage is tmem:
        return "tmem"
    raise ValueError(f"Unsupported TLE buffered_tensor storage: {storage}")


class layout:

    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__

    def to_ir(self, builder: ir.builder) -> None:
        raise NotImplementedError(f"{self.__class__.__name__}.to_ir() must be overridden in subclasses")


class shared_layout(layout):

    def __init__(self):
        super().__init__()
        pass

    """
    Create a new layout object that is a permutation of the current layout.
    """

    @abstractmethod
    def make_permute(self, dims):
        raise NotImplementedError(f"{self.__class__.__name__}.make_permute() must be overridden in subclasses")

    def to_ir(self, builder: ir.builder) -> None:
        raise NotImplementedError(f"{self.__class__.__name__}.to_ir() must be overridden in subclasses")


class swizzled_shared_layout(shared_layout):

    def __init__(self, vectorSize, perPhase, maxPhase, order, numCTAs, numCTAsPerCGA, numCTASplit, numCTAOrder):
        super().__init__()
        self.vectorSize = vectorSize
        self.perPhase = perPhase
        self.maxPhase = maxPhase
        self.order = order
        self.numCTAs = numCTAs
        self.numCTAsPerCGA = numCTAsPerCGA
        self.numCTASplit = numCTASplit
        self.numCTAOrder = numCTAOrder

    """
    Make a default non-swizzled shared layout encoding.
    """

    @classmethod
    def make_default(cls, rank):
        return cls(
            vectorSize=1,
            perPhase=1,
            maxPhase=1,
            order=list(reversed(range(rank))),  # e.g, [1, 0] as a row-major order
            numCTAs=[1] * rank,
            numCTAsPerCGA=[1] * rank,
            numCTASplit=[1] * rank,
            numCTAOrder=list(reversed(range(rank))),
        )

    """
    Create a new layout that is a permutation of the given layout.
    """

    def make_permute(self, dims):
        permuted_order = tuple(self.order[d] for d in dims)
        return swizzled_shared_layout(self.vectorSize, self.perPhase, self.maxPhase, permuted_order, self.numCTAs,
                                      self.numCTAsPerCGA, self.numCTASplit, self.numCTAOrder)

    def to_ir(self, builder: ir.builder) -> None:
        return builder.make_swizzled_shared_encoding_attr(
            self.vectorSize,
            self.perPhase,
            self.maxPhase,
            self.order,
            self.numCTAsPerCGA,
            self.numCTASplit,
            self.numCTAOrder,
        )


class tensor_memory_layout(shared_layout):

    def __init__(self, blockM, blockN, colStride, CTASplitM, CTASplitN, twoCTAs=False):
        super().__init__()
        self.blockM = blockM
        self.blockN = blockN
        self.colStride = colStride
        self.CTASplitM = CTASplitM
        self.CTASplitN = CTASplitN
        self.twoCTAs = twoCTAs

    """
    Make a default tensor memory layout encoding.
    """

    @classmethod
    def make_default(cls, shape):
        return cls(
            blockM=shape[0],
            blockN=shape[1],
            colStride=2,
            CTASplitM=1,
            CTASplitN=1,
            twoCTAs=False,
        )

    def to_ir(self, builder: ir.builder) -> None:
        return builder.make_tensor_memory_encoding_attr(
            self.blockM,
            self.blockN,
            self.colStride,
            self.CTASplitM,
            self.CTASplitN,
            self.twoCTAs,
        )


class nv_mma_shared_layout(shared_layout):

    def __init__(self, shape, order, elemType, numCTAsPerCGA, numCTASplit, numCTAOrder, fp4Padded, swizzled):
        super().__init__()
        self.shape = shape
        self.order = order
        self.elemType = elemType
        self.numCTAsPerCGA = numCTAsPerCGA
        self.numCTASplit = numCTASplit
        self.numCTAOrder = numCTAOrder
        self.fp4Padded = fp4Padded
        self.swizzled = swizzled

    """
    Make a default NVMMA shared layout encoding.
    """

    @classmethod
    def make_default(cls, shape, elemType):
        rank = len(shape)
        return cls(
            shape=shape,
            order=list(reversed(range(rank))),  # e.g, [1, 0] as a row-major order
            elemType=elemType,
            numCTAsPerCGA=[1] * rank,
            numCTASplit=[1] * rank,
            numCTAOrder=list(reversed(range(rank))),
            fp4Padded=False,
            swizzled=True,
        )

    """
    Create a new layout that is a permutation of the given layout.
    """

    def make_permute(self, dims):
        permuted_order = tuple(self.order[d] for d in dims)
        return nv_mma_shared_layout(
            self.shape,
            permuted_order,
            self.elemType,
            self.numCTAsPerCGA,
            self.numCTASplit,
            self.numCTAOrder,
            self.fp4Padded,
            self.swizzled,
        )

    def to_ir(self, builder: ir.builder) -> None:
        return builder.make_nv_mma_shared_encoding_attr(
            [int(x) for x in self.shape],
            self.order,
            self.elemType.to_ir(builder),
            self.numCTAsPerCGA,
            self.numCTASplit,
            self.numCTAOrder,
            self.fp4Padded,
            self.swizzled,
        )

    def __str__(self) -> str:
        return f"nv_mma_shared_layout<{self.shape}, {self.order}, {self.elemType}, {self.numCTAsPerCGA}, {self.numCTASplit}, {self.numCTAOrder}, {self.fp4Padded}, {self.swizzled}>"

    def __eq__(self, other) -> bool:
        return (type(self) is type(other) and self.shape == other.shape and self.order == other.order
                and self.elemType == other.elemType and self.numCTAsPerCGA == other.numCTAsPerCGA
                and self.numCTASplit == other.numCTASplit and self.numCTAOrder == other.numCTAOrder
                and self.fp4Padded == other.fp4Padded and self.swizzled == other.swizzled)


def _drop_leading_dim_order(order):
    return [dim - 1 for dim in order if dim != 0]


def _drop_leading_dim_values(values):
    return list(values[1:])


def _make_slot_layout(src_layout: shared_layout, slot_shape: List[int]) -> shared_layout:
    if isinstance(src_layout, swizzled_shared_layout):
        return swizzled_shared_layout(
            src_layout.vectorSize,
            src_layout.perPhase,
            src_layout.maxPhase,
            _drop_leading_dim_order(src_layout.order),
            _drop_leading_dim_values(src_layout.numCTAs),
            _drop_leading_dim_values(src_layout.numCTAsPerCGA),
            _drop_leading_dim_values(src_layout.numCTASplit),
            _drop_leading_dim_order(src_layout.numCTAOrder),
        )
    if isinstance(src_layout, nv_mma_shared_layout):
        return nv_mma_shared_layout(
            list(slot_shape),
            _drop_leading_dim_order(src_layout.order),
            src_layout.elemType,
            _drop_leading_dim_values(src_layout.numCTAsPerCGA),
            _drop_leading_dim_values(src_layout.numCTASplit),
            _drop_leading_dim_order(src_layout.numCTAOrder),
            src_layout.fp4Padded,
            src_layout.swizzled,
        )
    raise ValueError(f"buffered_tensor.slot does not support layout {type(src_layout).__name__}")


class buffered_tensor(tl.base_value):
    """
    A symbolic type representing a tensor allocated in a manually managed buffer
    such as shared memory (SMEM).

    This type is to model data that is not stored in global memory or registers
    but instead resides in hardware-close memory spaces with specialized
    allocation, access, or swizzling patterns.

    Unlike regular `tl.tensor`, which models values computed by operations,
    `buffered_tensor` reflects a memory-backed buffer that may be explicitly
    allocated and reused across program regions. It is primarily used with
    low-level intrinsics such as `tlx.local_alloc()`.

    Examples:
        a = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float16, num=4)

    Attributes:
        handle: The backing IR value representing the buffer allocation.
    """

    def __init__(self, handle, element_ty: tl.dtype, shape: List, storage: scope,
                 layout: Optional[shared_layout] = None, semantic: TritonSemantic = None, alloc_shape: List = None):
        """Not called by user code."""
        super().__init__()
        # IR handle
        self.handle = handle
        # Block shape
        self.shape = shape
        self.type = buffered_tensor_type(element_ty, shape, storage, layout, semantic, alloc_shape=alloc_shape)
        # Following the practice in pytorch, dtype is scalar type
        self.dtype = element_ty

    def _flatten_ir(self, handles) -> None:
        handles.append(self.handle)

    @tl.builtin
    def slot(self, stage, _semantic=None):
        if len(self.shape) < 2:
            raise ValueError("buffered_tensor.slot requires a rank >= 2 buffer")
        if self.type.storage is not smem:
            raise ValueError(f"buffered_tensor.slot currently supports only smem storage, got {self.type.storage}")

        stage_tensor = _semantic.to_tensor(stage)
        stage_ty = stage_tensor.type
        if getattr(stage_ty, "is_block", lambda: False)():
            raise ValueError("buffered_tensor.slot stage must be a scalar int32 value")
        if not getattr(stage_ty, "is_int", lambda: False)():
            raise ValueError(f"buffered_tensor.slot stage must be integer, got {stage_ty}")
        if stage_ty != tl.int32:
            raise ValueError(f"buffered_tensor.slot stage must be int32, got {stage_ty}")

        slot_shape = list(self.shape[1:])
        slot_layout = _make_slot_layout(self.type.layout, slot_shape)
        slot_ty = buffered_tensor_type(self.dtype, slot_shape, self.type.storage, slot_layout, _semantic,
                                       alloc_shape=slot_shape)
        slot_handle = _semantic.builder.create_memdesc_index(slot_ty.to_ir(_semantic.builder), self.handle,
                                                             stage_tensor.handle)
        return buffered_tensor(slot_handle, self.dtype, slot_shape, self.type.storage, slot_layout, _semantic,
                               alloc_shape=slot_ty.alloc_shape)

    def make_permute(self, handle, dims):
        permuted_layout = self.type.layout.make_permute(dims)
        return buffered_tensor(
            handle,
            self.dtype,
            [self.shape[d] for d in dims],
            self.type.num,
            self.type.storage,
            permuted_layout,
        )


class buffered_tensor_type(tl.block_type):

    def __init__(self, element_ty: tl.dtype, shape: List, storage: scope, layout: Optional[shared_layout] = None,
                 semantic: TritonSemantic = None, alloc_shape: List = None):
        super().__init__(element_ty, shape)
        # Storage
        self.storage = storage
        # layout encoding
        self.layout = layout
        self.alloc_shape = list(shape if alloc_shape is None else alloc_shape)
        # Buffer number. 0 means a single buffer, 1+ means a buffer array.
        assert semantic, "buffered_tensor array must be created with a builder"
        self.semantic = semantic

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[buffered_tensor, int]:
        value = buffered_tensor(handles[cursor], self.scalar, self.shape, self.storage, self.layout, self.semantic,
                                alloc_shape=self.alloc_shape)
        if hasattr(self, "_tle_remote_shard_id"):
            shard_id = getattr(self, "_tle_remote_shard_id")
            scope = getattr(self, "_tle_remote_scope", None)
            setattr(value, "_tle_remote_shard_id", shard_id)
            setattr(value, "_tle_remote_scope", scope)
            setattr(value.type, "_tle_remote_shard_id", shard_id)
            setattr(value.type, "_tle_remote_scope", scope)
        return value, cursor + 1

    def mangle(self) -> str:
        elt = self.scalar.mangle()
        shape = '_'.join(map(str, self.shape))
        alloc_suffix = ""
        if self.alloc_shape != self.shape:
            alloc_shape = '_'.join(map(str, self.alloc_shape))
            alloc_suffix = f"A{alloc_shape}"
        remote_suffix = ""
        shard_id = getattr(self, "_tle_remote_shard_id", None)
        if shard_id is not None:
            if isinstance(shard_id, int):
                remote_suffix = f"_R{shard_id}"
            else:
                remote_suffix = "_Rdyn"
        return f'buffered_{elt}S{shape}{alloc_suffix}{remote_suffix}'

    def __str__(self) -> str:
        return f"buffered_tensor_<{self.element_ty}, {self.shape}, {self.layout}, {self.alloc_shape}, >"

    def __eq__(self, other) -> bool:
        if not (type(self) is type(other) and self.shape == other.shape and self.layout == other.layout
                and self.alloc_shape == other.alloc_shape):
            return False
        self_shard = getattr(self, "_tle_remote_shard_id", None)
        other_shard = getattr(other, "_tle_remote_shard_id", None)
        self_scope = getattr(self, "_tle_remote_scope", None)
        other_scope = getattr(other, "_tle_remote_scope", None)
        if self_shard is None and other_shard is None and self_scope is None and other_scope is None:
            return True
        # If either side carries remote metadata, require equivalent remote marker.
        if (self_shard is None) != (other_shard is None):
            return False
        if isinstance(self_shard, int) and isinstance(other_shard, int):
            if self_shard != other_shard:
                return False
        elif self_shard is not other_shard:
            return False
        self_scope_key = None if self_scope is None else (
            tuple(getattr(self_scope, "shape", tuple())),
            tuple(getattr(self_scope, "dim_names", tuple())),
        )
        other_scope_key = None if other_scope is None else (
            tuple(getattr(other_scope, "shape", tuple())),
            tuple(getattr(other_scope, "dim_names", tuple())),
        )
        return self_scope_key == other_scope_key

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        out.append(self.to_ir(builder))

    def to_ir(self, builder: ir.builder) -> None:
        shape = self.shape
        builder = self.semantic.builder
        return builder.get_memdesc_type(
            shape,
            self.element_ty.to_ir(builder),
            self.layout.to_ir(builder),
            _storage_to_memdesc_space(self.storage),
            self.alloc_shape,
        )

    def _flatten_ir(self, handles) -> None:
        handles.append(self.handle)


class pipe_slot_type(tl.base_type):

    def __init__(self, fields):
        self.fields = list(fields)

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple["pipe_slot", int]:
        values = {}
        for name, ty in self.fields:
            value, cursor = ty._unflatten_ir(handles, cursor)
            values[name] = value
        return pipe_slot(values), cursor

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        for _, ty in self.fields:
            ty._flatten_ir_types(builder, out)

    def mangle(self) -> str:
        fields = "_".join(f"{name}_{ty.mangle()}" for name, ty in self.fields)
        return f"pipe_slot_{fields}"

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and self.fields == other.fields

    def __str__(self) -> str:
        fields = ", ".join(f"{name}: {ty}" for name, ty in self.fields)
        return f"pipe_slot<{fields}>"


class pipe_slot(tl.base_value):

    def __init__(self, fields):
        super().__init__()
        self.fields = dict(fields)
        for name, value in self.fields.items():
            setattr(self, name, value)

    def _flatten_ir(self, handles) -> None:
        for field in self.fields.values():
            field._flatten_ir(handles)

    @property
    def type(self):
        return pipe_slot_type([(name, value.type) for name, value in self.fields.items()])


class pipe_wait_result_type(tl.base_type):

    def __init__(self, slot_type: pipe_slot_type):
        self.slot_type = slot_type

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple["pipe_wait_result", int]:
        slot, cursor = self.slot_type._unflatten_ir(handles, cursor)
        is_closed, cursor = tl.int1._unflatten_ir(handles, cursor)
        return pipe_wait_result(slot, is_closed), cursor

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        self.slot_type._flatten_ir_types(builder, out)
        tl.int1._flatten_ir_types(builder, out)

    def mangle(self) -> str:
        return f"pipe_wait_result_{self.slot_type.mangle()}"

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and self.slot_type == other.slot_type

    def __str__(self) -> str:
        return f"pipe_wait_result<slot: {self.slot_type}, is_closed: {tl.int1}>"


class pipe_wait_result(tl.base_value):

    def __init__(self, slot: pipe_slot, is_closed: tl.tensor):
        super().__init__()
        self.slot = slot
        self.is_closed = is_closed

    def _flatten_ir(self, handles) -> None:
        self.slot._flatten_ir(handles)
        self.is_closed._flatten_ir(handles)

    @property
    def type(self):
        return pipe_wait_result_type(self.slot.type)


def _unwrap_pipe_constexpr(value):
    if isinstance(value, tl.constexpr):
        value = value.value
    if isinstance(value, tl.tuple):
        return tuple(_unwrap_pipe_constexpr(item) for item in value.values)
    if isinstance(value, (tuple, list)):
        return type(value)(_unwrap_pipe_constexpr(item) for item in value)
    if value is None or isinstance(value, str):
        return value
    value = tl._unwrap_if_constexpr(value)
    if isinstance(value, (tuple, list)):
        return type(value)(_unwrap_pipe_constexpr(item) for item in value)
    return value


def _validate_pipe_reader_name(pipe, name):
    name = _unwrap_pipe_constexpr(name)
    if pipe.readers is None:
        if name is not None:
            raise ValueError("tle.pipe.reader name requires pipe readers=...")
        return None
    if name is None:
        raise ValueError("tle.pipe.reader requires a reader name when pipe readers are declared")
    if not isinstance(name, str):
        raise ValueError(f"tle.pipe.reader name must be a string, got {type(name).__name__}")
    if name not in pipe.readers:
        raise ValueError(f"tle.pipe.reader name {name!r} is not declared in pipe readers={pipe.readers!r}")
    return name


def _validate_pipe_reader_fields(pipe, fields):
    fields = _unwrap_pipe_constexpr(fields)
    if fields is None:
        return None
    if isinstance(fields, str) or not isinstance(fields, (tuple, list)):
        raise ValueError("tle.pipe.reader fields must be a compile-time tuple/list of field names")
    if not fields:
        raise ValueError("tle.pipe.reader fields must not be empty")

    names = []
    seen = set()
    for field in fields:
        field = _unwrap_pipe_constexpr(field)
        if not isinstance(field, str):
            raise ValueError(f"tle.pipe.reader field name must be a string, got {type(field).__name__}")
        if field not in pipe.fields:
            raise ValueError(f"tle.pipe.reader field {field!r} is not a pipe field")
        if field in seen:
            raise ValueError(f"tle.pipe.reader fields must be unique, got duplicate {field!r}")
        seen.add(field)
        names.append(field)
    return tuple(names)


class pipe_value_type(tl.base_type):

    def __init__(self, capacity: int, scope: str, name: Optional[str], fields, readers=None, one_shot=False):
        self.capacity = capacity
        self.scope = scope
        self.name = name
        self.fields = list(fields)
        self.readers = None if readers is None else tuple(readers)
        self.one_shot = one_shot

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple["pipe_value", int]:
        values = {}
        for field_name, ty in self.fields:
            value, cursor = ty._unflatten_ir(handles, cursor)
            values[field_name] = value
        return pipe_value(self.capacity, self.scope, self.name, values, self.readers, one_shot=self.one_shot), cursor

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        for _, ty in self.fields:
            ty._flatten_ir_types(builder, out)

    def mangle(self) -> str:
        name = "anon" if self.name is None else self.name
        fields = "_".join(f"{field_name}_{ty.mangle()}" for field_name, ty in self.fields)
        readers = "spsc" if self.readers is None else "readers_" + "_".join(self.readers)
        mode = "oneshot" if self.one_shot else "cyclic"
        return f"pipe_{self.scope}_{self.capacity}_{name}_{mode}_{readers}_{fields}"

    def __eq__(self, other) -> bool:
        return (type(self) is type(other) and self.capacity == other.capacity and self.scope == other.scope
                and self.name == other.name and self.fields == other.fields and self.readers == other.readers
                and self.one_shot == other.one_shot)

    def __str__(self) -> str:
        fields = ", ".join(f"{name}: {ty}" for name, ty in self.fields)
        readers = "default" if self.readers is None else ", ".join(self.readers)
        mode = "one_shot" if self.one_shot else "cyclic"
        return f"pipe<{self.scope}, {self.capacity}, {mode}, readers=[{readers}], {fields}>"


class pipe_value(tl.base_value):

    def __init__(self, capacity: int, scope: str, name: Optional[str], fields, readers=None, one_shot=False):
        super().__init__()
        self.capacity = capacity
        self.scope = scope
        self.name = name
        self.fields = dict(fields)
        self.readers = None if readers is None else tuple(readers)
        self.one_shot = one_shot

    def _flatten_ir(self, handles) -> None:
        for field in self.fields.values():
            field._flatten_ir(handles)

    @property
    def type(self):
        return pipe_value_type(self.capacity, self.scope, self.name,
                               [(name, value.type) for name, value in self.fields.items()], self.readers,
                               one_shot=self.one_shot)

    def _field_handles(self):
        return [field.handle for field in self.fields.values()]

    def _field_names(self):
        return list(self.fields.keys())

    def _ir_name(self):
        return "" if self.name is None else self.name

    def _make_slot(self, stage, _semantic=None, field_names=None):
        if field_names is None:
            fields = self.fields.items()
        else:
            fields = ((name, self.fields[name]) for name in field_names)
        return pipe_slot({name: field.slot(stage, _semantic=_semantic) for name, field in fields})

    @tl.builtin
    def _stage_phase(self, iter, _semantic=None):
        iter = tl._unwrap_if_constexpr(iter)
        if isinstance(iter, int):
            stage = iter % self.capacity
            phase = ((iter // self.capacity) % 2) != 0
            return _semantic.to_tensor(stage), _semantic.to_tensor(phase)
        iter = _semantic.to_tensor(iter)
        stage = iter.__mod__(self.capacity, _semantic=_semantic)
        phase = iter.__floordiv__(self.capacity, _semantic=_semantic)
        phase = phase.__mod__(2, _semantic=_semantic)
        phase = phase.__ne__(0, _semantic=_semantic)
        return _semantic.to_tensor(stage), _semantic.to_tensor(phase)

    @tl.builtin
    def writer(self, _semantic=None):
        return pipe_writer(self)

    @tl.builtin
    def reader(self, name=None, fields=None, _semantic=None):
        reader_name = _validate_pipe_reader_name(self, name)
        field_names = _validate_pipe_reader_fields(self, fields)
        return pipe_reader(self, reader_name=reader_name, field_names=field_names)


class _pipe_endpoint_type(tl.base_type):

    endpoint = "endpoint"
    value_cls = None

    def __init__(self, pipe_type: pipe_value_type, reader_name=None, field_names=None):
        self.pipe_type = pipe_type
        self.reader_name = reader_name
        self.field_names = None if field_names is None else tuple(field_names)

    def _unflatten_ir(self, handles: List[ir.value], cursor: int):
        pipe, cursor = self.pipe_type._unflatten_ir(handles, cursor)
        return self.value_cls(pipe, self.reader_name, self.field_names), cursor

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        self.pipe_type._flatten_ir_types(builder, out)

    def mangle(self) -> str:
        reader = "default" if self.reader_name is None else self.reader_name
        fields = "all" if self.field_names is None else "_".join(self.field_names)
        return f"pipe_{self.endpoint}_{reader}_{fields}_{self.pipe_type.mangle()}"

    def __eq__(self, other) -> bool:
        return (type(self) is type(other) and self.pipe_type == other.pipe_type
                and self.reader_name == other.reader_name and self.field_names == other.field_names)

    def __str__(self) -> str:
        return f"pipe_{self.endpoint}<{self.pipe_type}>"


class pipe_writer_type(_pipe_endpoint_type):
    endpoint = "writer"


class pipe_reader_type(_pipe_endpoint_type):
    endpoint = "reader"


class _pipe_endpoint(tl.base_value):

    type_cls = _pipe_endpoint_type

    def __init__(self, pipe: pipe_value, reader_name=None, field_names=None):
        super().__init__()
        self.pipe = pipe
        self.reader_name = reader_name
        self.field_names = None if field_names is None else tuple(field_names)

    @property
    def capacity(self):
        return self.pipe.capacity

    @property
    def scope(self):
        return self.pipe.scope

    @property
    def name(self):
        return self.pipe.name

    @property
    def fields(self):
        if self.field_names is None:
            return self.pipe.fields
        return {name: self.pipe.fields[name] for name in self.field_names}

    def _flatten_ir(self, handles) -> None:
        self.pipe._flatten_ir(handles)

    @property
    def type(self):
        return self.type_cls(self.pipe.type, self.reader_name, self.field_names)


class pipe_writer(_pipe_endpoint):

    type_cls = pipe_writer_type

    def __init__(self, pipe: pipe_value, reader_name=None, field_names=None):
        if reader_name is not None or field_names is not None:
            raise ValueError("tle.pipe.writer does not accept reader endpoint metadata")
        super().__init__(pipe)

    @tl.builtin
    def acquire(self, iter, _semantic=None):
        stage, phase = self.pipe._stage_phase(iter, _semantic=_semantic)
        _semantic.builder.create_pipe_writer_acquire(self.pipe._field_handles(), stage.handle, phase.handle,
                                                     self.pipe.capacity, self.pipe.scope, self.pipe._ir_name(),
                                                     self.pipe._field_names())
        return self.pipe._make_slot(stage, _semantic=_semantic)

    @tl.builtin
    def commit(self, iter, _semantic=None):
        stage, _ = self.pipe._stage_phase(iter, _semantic=_semantic)
        _semantic.builder.create_pipe_writer_commit(self.pipe._field_handles(), stage.handle, self.pipe.capacity,
                                                    self.pipe.scope, self.pipe._ir_name(), self.pipe._field_names())

    @tl.builtin
    def close(self, iter, _semantic=None):
        if self.pipe.one_shot:
            raise ValueError("tle.pipe one_shot pipes do not support close")
        stage, phase = self.pipe._stage_phase(iter, _semantic=_semantic)
        _semantic.builder.create_pipe_writer_close(self.pipe._field_handles(), stage.handle, phase.handle,
                                                   self.pipe.capacity, self.pipe.scope, self.pipe._ir_name(),
                                                   self.pipe._field_names())


class pipe_reader(_pipe_endpoint):

    type_cls = pipe_reader_type

    def _reader_field_names(self):
        if self.field_names is None:
            return self.pipe._field_names()
        return list(self.field_names)

    @tl.builtin
    def wait(self, iter, _semantic=None):
        stage, phase = self.pipe._stage_phase(iter, _semantic=_semantic)
        is_closed = _semantic.builder.create_pipe_reader_wait(self.pipe._field_handles(), stage.handle, phase.handle,
                                                              self.pipe.capacity, self.pipe.scope, self.pipe._ir_name(),
                                                              self.pipe._field_names(), self.reader_name or "",
                                                              self._reader_field_names())
        slot = self.pipe._make_slot(stage, _semantic=_semantic, field_names=self.field_names)
        return pipe_wait_result(slot, tl.tensor(is_closed, tl.int1))

    @tl.builtin
    def release(self, iter, _semantic=None):
        stage, _ = self.pipe._stage_phase(iter, _semantic=_semantic)
        _semantic.builder.create_pipe_reader_release(self.pipe._field_handles(), stage.handle,
                                                     self.pipe.capacity, self.pipe.scope, self.pipe._ir_name(),
                                                     self.pipe._field_names(), self.reader_name or "",
                                                     self._reader_field_names())


pipe_writer_type.value_cls = pipe_writer
pipe_reader_type.value_cls = pipe_reader
