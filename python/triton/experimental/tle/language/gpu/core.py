# flagtree tle
import builtins
import triton.language.core as tl
from typing import Optional, Sequence
from enum import Enum
from . import types as tle
from .mthreads import copy as mthreads_copy
from .iluvatar import copy as iluvatar_copy
from triton.compiler.code_generator import flatten_values_to_ir, unflatten_ir_values

from triton.language.core import (
    constexpr,
    tensor,
    range,
)

# Address space 3 matches the shared-memory space used in TritonGPU lowering.
SHARED_MEMORY_ADDRESS_SPACE = 3

_WGMMA_PIPELINE_MODE_ATTR = "tle.wgmma_pipeline_mode"
_WGMMA_PIPELINE_MODE_USER_PROMISE = "user_promise"


def _mark_wgmma_user_promise(_semantic, _generator):
    if _generator is None or _semantic is None:
        return
    _generator.module.set_attr(
        _WGMMA_PIPELINE_MODE_ATTR,
        _semantic.builder.get_string_attr(_WGMMA_PIPELINE_MODE_USER_PROMISE),
    )


def _is_wgmma_user_promise_marked(_generator):
    if _generator is None:
        return False
    mode = _generator.module.get_operation().get_str_attr(_WGMMA_PIPELINE_MODE_ATTR)
    return mode == _WGMMA_PIPELINE_MODE_USER_PROMISE


class pipeline(range):
    """
    Iterator that counts upward forever, with parallel execution semantics.

    This is a special iterator used to implement similar semantics to Python's :code:`range` in the context of
    :code:`triton.jit` functions. In addition, it allows user to pass extra attributes to the compiler.
    :param bind_sub_block: Tells the compiler if multiple vector cores participate in the loop.
        This is used in the mixed cube-vector kernel on 910B. The number of vector cores is determined by the number of
        iteration in this loop. Currently on 910B, max 2 vector cores could be used.
    """

    def __init__(self, arg1, arg2=None, step=None, num_stages=None, loop_unroll_factor=None):
        super().__init__(arg1, arg2, step, num_stages, loop_unroll_factor)


class WarpSpecializeCallerContext:

    def __init__(self, num_warps: int):
        self.num_warps = num_warps

    def mangle(self):
        return f"_TLEWSNW{self.num_warps}"

    def initialize_callee(self, fn, builder):
        fn.set_attr("ttg.num-warps", builder.get_int32_attr(self.num_warps))


def _as_call_args(args):
    args = tl._unwrap_if_constexpr(args)
    if isinstance(args, tl.tuple):
        args = tuple(args.values)
    elif isinstance(args, tuple):
        args = args
    else:
        raise ValueError(f"warp_specialize function arguments must be a tuple, got {type(args).__name__}")

    def normalize(arg):
        if isinstance(arg, (tl.dtype, float, int, bool)):
            return tl.constexpr(arg)
        return arg

    return tuple(normalize(arg) for arg in args)


def _as_warp_specialize_entry(entry, index: int):
    entry = tl._unwrap_if_constexpr(entry)
    if isinstance(entry, tl.tuple):
        entry = tuple(entry.values)
    if not isinstance(entry, tuple):
        raise ValueError(f"warp_specialize entry {index} must be a tuple, got {type(entry).__name__}")
    if len(entry) != 2:
        raise ValueError(f"warp_specialize entry {index} must be (fn, args); pass cid as a normal constexpr arg")
    if not isinstance(tl._unwrap_if_constexpr(entry[1]), (tuple, tl.tuple)):
        raise ValueError(f"warp_specialize entry {index} args must be a tuple")
    return entry


def _as_result_values(results):
    if results is None:
        return tuple()
    if isinstance(results, tl.tuple):
        return tuple(results.values)
    if isinstance(results, tuple):
        return results
    return (results, )


def _warp_specialize_capture_key(handle):
    try:
        hash(handle)
    except TypeError:
        return id(handle)
    return handle


def _deduplicate_warp_specialize_captures(worker_items):
    capture_handles = []
    capture_indices = {}
    deduplicated_items = []
    for worker_fn, worker_args, flattened in worker_items:
        remapped = []
        for handle in flattened:
            key = _warp_specialize_capture_key(handle)
            index = capture_indices.get(key)
            if index is None:
                index = len(capture_handles)
                capture_indices[key] = index
                capture_handles.append(handle)
            remapped.append(index)
        deduplicated_items.append((worker_fn, worker_args, flattened, remapped))
    return capture_handles, deduplicated_items


@tl.builtin
def warp_specialize(functions_and_args, worker_num_warps, worker_num_regs, _semantic=None, _generator=None):
    """
    Create an explicit GPU warp-specialized region.

    ``functions_and_args[0]`` is emitted into the default partition. Later
    entries are emitted into worker partitions, with their warp counts and
    requested register counts provided by ``worker_num_warps`` and
    ``worker_num_regs``.
    """
    if _generator is None:
        raise ValueError("warp_specialize requires a Triton code generator")
    functions_and_args = tl._unwrap_if_constexpr(functions_and_args)
    worker_num_warps = [tl._unwrap_if_constexpr(w) for w in tl._unwrap_if_constexpr(worker_num_warps)]
    worker_num_regs = [tl._unwrap_if_constexpr(r) for r in tl._unwrap_if_constexpr(worker_num_regs)]
    if len(functions_and_args) < 1:
        raise ValueError("warp_specialize requires at least a default partition function")
    num_partitions = len(functions_and_args) - 1
    if num_partitions != len(worker_num_warps):
        raise ValueError(
            f"warp_specialize got {num_partitions} worker functions but {len(worker_num_warps)} warp counts")
    if num_partitions != len(worker_num_regs):
        raise ValueError(
            f"warp_specialize got {num_partitions} worker functions but {len(worker_num_regs)} register counts")

    builder = _semantic.builder
    insert_pt = builder.get_insertion_point()
    inline_user_promise = _is_wgmma_user_promise_marked(_generator)
    if inline_user_promise:
        call_jit_function = _generator.inline_JitFunction
    else:
        call_jit_function = _generator.call_JitFunction

    default_fn, default_args = _as_warp_specialize_entry(functions_and_args[0], 0)
    default_args = _as_call_args(default_args)
    default_block = builder.create_block()
    builder.set_insertion_point_to_start(default_block)
    default_results = call_jit_function(default_fn, default_args, kwargs={})
    default_result_values = _as_result_values(default_results)
    default_result_handles = flatten_values_to_ir(default_result_values)
    builder.set_insertion_point_to_end(default_block)
    result_types = [result.get_type() for result in default_result_handles]

    worker_items = []
    for idx, entry in enumerate(functions_and_args[1:], start=1):
        worker_fn, worker_args = _as_warp_specialize_entry(entry, idx)
        worker_args = _as_call_args(worker_args)
        flattened = flatten_values_to_ir(worker_args)
        worker_items.append((worker_fn, worker_args, flattened))
    worker_arg_handles, worker_items = _deduplicate_warp_specialize_captures(worker_items)

    builder.restore_insertion_point(insert_pt)
    ws_op = builder.create_warp_specialize(result_types, worker_arg_handles, worker_num_warps)
    real_default_block = builder.create_block_with_parent(ws_op.get_default_region(), [])
    default_block.merge_block_before(real_default_block)
    default_block = real_default_block
    builder.set_insertion_point_to_end(default_block)
    builder.create_warp_yield(default_result_handles)
    ws_op.set_requested_registers(worker_num_regs)

    builder.create_block_with_parent(ws_op.get_partition_op_holder(), [])
    partitions_op = builder.create_warp_specialize_partitions(num_partitions)
    partition_arg_types = [arg.get_type() for arg in worker_arg_handles]
    for idx, (worker_fn, worker_args, flattened, remapped) in enumerate(worker_items):
        block = builder.create_block_with_parent(partitions_op.get_region(idx), partition_arg_types)
        block_args = [block.get_argument(remapped[j]) for j in builtins.range(len(flattened))]
        block_values = tuple(unflatten_ir_values(block_args, [arg.type for arg in worker_args]))
        caller_context = WarpSpecializeCallerContext(worker_num_warps[idx])
        call_jit_function(worker_fn, block_values, kwargs={}, caller_context=caller_context)
        builder.set_insertion_point_to_end(block)
        builder.create_warp_return()

    builder.set_insertion_point_after(ws_op.get_operation())
    if not default_result_values:
        return None
    result_handles = [ws_op.get_result(i) for i in builtins.range(len(result_types))]
    result_values = tuple(unflatten_ir_values(result_handles, [value.type for value in default_result_values]))
    if len(result_values) == 1:
        return result_values[0]
    return result_values


@tl.builtin
def memory_space(input, space, _builder=None, _semantic=None):
    '''
    Assign a memory space to the tensor :code:`input`.

    :param input: the input tensor
    :type input: Tensor
    :param space: the memory space to assign. Can be one of "shared_memory", "tensor_memory", "register" or any other target-specific memory space.
    :type space: str
    '''
    space = tl._unwrap_if_constexpr(space)
    input.handle.set_attr("tt.memory_space", _semantic.builder.get_string_attr(space))
    return input


@tl.builtin
def alloc(
    shape: tuple,
    dtype: tl.dtype,
    layout: Optional[tle.shared_layout] = None,
    scope: tle.scope = tle.smem,
    init_value: Optional[tl.tensor] = None,
    nv_mma_shared_layout=True,
    _semantic=None,
) -> tle.buffered_tensor:
    """
    Allocate local memory buffer

    Args:
        shape: Buffer shape
        dtype: Data type
        layout: Memory layout encoding (optional)
        scope: Storage type (default to shared memory)
        _semantic: Semantic analyzer (internal use)

    Returns:
        Allocated buffer tensor

    Raises:
        ValueError: When parameters are invalid
        RuntimeError: When allocation fails
    """
    # Parameter validation
    if not isinstance(shape, (tuple, list)):
        # Try to handle Triton tuple-like objects
        if hasattr(shape, '__iter__'):
            shape = tuple(shape)
        else:
            raise ValueError(f"Shape parameter must be tuple or list, but got {type(shape)}")

    if not isinstance(dtype, tl.dtype):
        raise ValueError(f"Data type must be tl.dtype, but got {type(dtype)}")

    if not isinstance(scope, tle.scope):
        raise ValueError(f"Storage type must be tle.scope, but got {type(scope)}")

    layout = tl._unwrap_if_constexpr(layout)
    if layout is not None and not isinstance(layout, tle.shared_layout):
        # Handle constexpr None
        if hasattr(layout, 'value') and layout.value is None:
            layout = None
        else:
            raise ValueError(f"Layout must be tle.shared_layout or None, but got {type(layout)}")

    # Semantic analysis
    try:
        from .semantic import TLESemantic
        if isinstance(_semantic, TLESemantic):
            shape, dtype = _semantic.analyze_alloc_operation(shape, dtype, layout, scope)
    except ImportError:
        # If semantic analysis module is not available, continue with warning
        import warnings
        warnings.warn("TLE semantic analysis module not available, skipping validation", UserWarning)

    # Map scope to storage (backward compatibility)
    storage = scope

    try:
        unwrapped_shape = [tl._unwrap_if_constexpr(dim) for dim in shape]
        full_shape = unwrapped_shape
        dtype = tl._unwrap_if_constexpr(dtype)
        elem_type = dtype.to_ir(_semantic.builder)

        if layout is None:
            if storage == tle.smem:
                if not nv_mma_shared_layout:
                    layout = tle.swizzled_shared_layout.make_default(rank=len(shape))
                    layout_handle = _semantic.builder.make_swizzled_shared_encoding_attr(
                        layout.vectorSize,
                        layout.perPhase,
                        layout.maxPhase,
                        layout.order,
                        layout.numCTAsPerCGA,
                        layout.numCTASplit,
                        layout.numCTAOrder,
                    )
                else:
                    layout = tle.nv_mma_shared_layout.make_default(shape, dtype)
                    layout_handle = _semantic.builder.make_nv_mma_shared_encoding_attr(
                        [int(x) for x in layout.shape],
                        layout.order,
                        layout.elemType.to_ir(_semantic.builder),
                        layout.numCTAsPerCGA,
                        layout.numCTASplit,
                        layout.numCTAOrder,
                        layout.fp4Padded,
                        layout.swizzled,
                    )
            else:
                layout = tle.tensor_memory_layout.make_default(shape)
                layout_handle = _semantic.builder.make_tensor_memory_encoding_attr(
                    layout.blockM,
                    layout.blockN,
                    layout.colStride,
                    layout.CTASplitM,
                    layout.CTASplitN,
                    layout.twoCTAs,
                )
        else:
            # Use provided layout
            layout_handle = layout.to_ir(_semantic.builder)

        if storage == tle.smem:
            if init_value is not None:
                mutable_ty = _semantic.builder.get_memdesc_type(full_shape, elem_type, layout_handle, "smem")
                tensor_handle = _semantic.builder.create_local_alloc(mutable_ty, init_value.handle)
            else:
                tensor_handle = _semantic.builder.create_local_alloc(full_shape, elem_type, layout_handle)
        else:
            raise ValueError(f"Storage type {storage} not yet supported")

        return tle.buffered_tensor(tensor_handle, dtype, unwrapped_shape, storage, layout, _semantic)

    except Exception as e:
        raise RuntimeError(f"Memory allocation failed: {str(e)}") from e


def _unwrap_barrier_constexpr(value):
    value = tl._unwrap_if_constexpr(value)
    if isinstance(value, tl.constexpr):
        return value.value
    return value


def _require_barrier_int(value, name: str) -> int:
    value = _unwrap_barrier_constexpr(value)
    if not isinstance(value, int):
        raise ValueError(f"{name} must be a compile-time integer")
    return value


def _normalize_barrier_init(init) -> str:
    init = _unwrap_barrier_constexpr(init)
    if isinstance(init, str):
        normalized = init.lower()
        if normalized in (tle.PENDING, "pending"):
            return tle.PENDING
        if normalized in (tle.READY, "ready"):
            return tle.READY
    raise ValueError("barrier init must be tle.gpu.PENDING or tle.gpu.READY")


# NVIDIA hardware named barriers are ids 0..15. TLE frontend assigns virtual
# ids from 16 upward so each source barrier is distinguishable until the late
# compiler pass remaps them to conflict-free physical ids.
_FIRST_VIRTUAL_NAMED_BARRIER_ID = 16


def _reserve_named_barrier_ids(_semantic, count: int) -> int:
    next_id = getattr(_semantic, "_tle_next_named_barrier_id", _FIRST_VIRTUAL_NAMED_BARRIER_ID)
    last_id = next_id + count - 1
    setattr(_semantic, "_tle_next_named_barrier_id", last_id + 1)
    return next_id


def _barrier_handle_key(handle):
    try:
        hash(handle)
        return handle
    except TypeError:
        return id(handle)


def _ensure_named_barrier_ids(slot: tle.barrier, _semantic) -> None:
    if slot.named_base_id > 0:
        return
    key = slot.allocation_key
    if key is None:
        key = _barrier_handle_key(slot.handle)
    named_bases = getattr(_semantic, "_tle_barrier_named_bases", None)
    if named_bases is None:
        named_bases = {}
        setattr(_semantic, "_tle_barrier_named_bases", named_bases)
    base_id = named_bases.get(key)
    if base_id is None:
        base_id = _reserve_named_barrier_ids(_semantic, slot.num_barriers)
        named_bases[key] = base_id
    slot.named_base_id = base_id
    slot.type.named_base_id = base_id


def _barrier_phase_tensor(phaseIdx, init: str, _semantic) -> tl.tensor:
    init_polarity = 1 if init == tle.READY else 0
    raw_phase = _unwrap_barrier_constexpr(phaseIdx)
    if isinstance(raw_phase, int):
        return _semantic.to_tensor((raw_phase & 1) ^ init_polarity).to(tl.int32, _semantic=_semantic)

    phase = _semantic.to_tensor(phaseIdx)
    if getattr(phase.type, "is_block", lambda: False)():
        raise ValueError("barrier phaseIdx must be a scalar integer")
    if not getattr(phase.type, "is_int", lambda: False)():
        raise ValueError(f"barrier phaseIdx must be integer, got {phase.type}")
    if phase.dtype != tl.int32:
        phase = phase.to(tl.int32, _semantic=_semantic)
    phase = phase.__and__(1, _semantic=_semantic)
    if init_polarity:
        phase = phase.__xor__(1, _semantic=_semantic)
    return phase


def _barrier_slot(value: tle.barrier, _semantic) -> tle.barrier:
    if not isinstance(value, tle.barrier):
        raise ValueError(f"barrier operation expects tle.gpu barrier, got {type(value).__name__}")
    if value.is_slot:
        return value
    return value.__getitem__(0, _semantic=_semantic)


def _record_barrier_backend(slot: tle.barrier, backend: str, _semantic) -> None:
    if slot.allocation_key is not None and slot.static_index is not None:
        key = (slot.allocation_key, slot.static_index)
    elif slot.named_base_id > 0 and slot.static_index is not None:
        key = (slot.named_base_id, slot.static_index)
    else:
        key = _barrier_handle_key(slot.handle)
    uses = getattr(_semantic, "_tle_barrier_backend_uses", None)
    if uses is None:
        uses = {}
        setattr(_semantic, "_tle_barrier_backend_uses", uses)
    previous = uses.get(key)
    if previous is not None and previous != backend:
        raise ValueError("cannot mix named and mbarrier backends for the same barrier slot")
    uses[key] = backend


def _tma_completion_barrier_slot(value, _semantic) -> tle.barrier:
    if not isinstance(value, tle.barrier):
        raise ValueError(f"TMA copy barrier expects tle.gpu barrier, got {type(value).__name__}")
    if not value.is_slot and value.num_barriers != 1:
        raise ValueError("TMA copy barrier arrays must be indexed, e.g. bars[i]")

    slot = _barrier_slot(value, _semantic)
    if slot.expect_bytes is None:
        raise ValueError("TMA copy barrier must be allocated with expect_bytes")
    if not isinstance(slot.expect_bytes, int) or slot.expect_bytes <= 0:
        raise ValueError("TMA copy barrier expect_bytes must be a positive compile-time integer")

    _record_barrier_backend(slot, "mbarrier", _semantic)
    return slot


@tl.builtin
def alloc_barriers(
    num_barriers,
    arrive_count=1,
    init=tle.PENDING,
    expect_bytes=None,
    _semantic=None,
    _generator=None,
) -> tle.barrier:
    """Allocate a TLE GPU barrier array."""
    _mark_wgmma_user_promise(_semantic, _generator)

    num_barriers = _require_barrier_int(num_barriers, "num_barriers")
    arrive_count = _require_barrier_int(arrive_count, "arrive_count")
    init = _normalize_barrier_init(init)
    if num_barriers <= 0:
        raise ValueError("num_barriers must be positive")
    if arrive_count <= 0:
        raise ValueError("arrive_count must be positive")

    expect_bytes = _unwrap_barrier_constexpr(expect_bytes)
    if expect_bytes is not None:
        if not isinstance(expect_bytes, int):
            raise ValueError("expect_bytes must be a compile-time integer or None")
        if expect_bytes <= 0:
            raise ValueError("expect_bytes must be positive when provided")

    layout = tle.swizzled_shared_layout.make_default(rank=2)
    named_base_id = 0
    barrier_ty = tle.barrier_type(num_barriers, arrive_count, init, expect_bytes, layout, _semantic,
                                  shape=[num_barriers, 1], named_base_id=named_base_id)
    handle = _semantic.builder.create_barrier_alloc(
        barrier_ty.to_ir(_semantic.builder),
        num_barriers,
        arrive_count,
        1 if init == tle.READY else 0,
        -1 if expect_bytes is None else expect_bytes,
    )
    allocation_key = _barrier_handle_key(handle)
    barrier_ty.allocation_key = allocation_key
    return tle.barrier(handle, num_barriers, arrive_count, init, expect_bytes, layout, _semantic,
                       shape=[num_barriers, 1], named_base_id=named_base_id, allocation_key=allocation_key)


@tl.builtin
def alloc_barrier(
    arrive_count=1,
    init=tle.PENDING,
    expect_bytes=None,
    _semantic=None,
    _generator=None,
) -> tle.barrier:
    """Allocate a single TLE GPU barrier."""
    _mark_wgmma_user_promise(_semantic, _generator)
    return alloc_barriers(
        1,
        arrive_count,
        init,
        expect_bytes,
        _semantic=_semantic,
    )


@tl.builtin
def barrier_wait(barr, phaseIdx=None, _semantic=None) -> None:
    """Wait on a TLE GPU barrier slot."""
    slot = _barrier_slot(barr, _semantic)
    if phaseIdx is None:
        if slot.expect_bytes is not None:
            raise ValueError("barrier_wait on a barrier with expect_bytes requires phaseIdx")
        if slot.init == tle.READY:
            raise ValueError("barrier_wait without phaseIdx selects named barrier, which does not support READY")
        if slot.static_index is None:
            raise ValueError("named barrier backend requires a static barrier slot index")
        _ensure_named_barrier_ids(slot, _semantic)
        _record_barrier_backend(slot, "named", _semantic)
        _semantic.builder.create_barrier_wait_named(slot.handle, slot.named_base_id + slot.static_index,
                                                    slot.arrive_count)
        return

    phase = _barrier_phase_tensor(phaseIdx, slot.init, _semantic)
    _record_barrier_backend(slot, "mbarrier", _semantic)
    _semantic.builder.create_barrier_wait_mbarrier(slot.handle, phase.handle)


@tl.builtin
def barrier_arrive(barr, arrive_count=1, phaseIdx=None, _semantic=None) -> None:
    """Arrive on a TLE GPU barrier slot."""
    slot = _barrier_slot(barr, _semantic)
    arrive_count = _require_barrier_int(arrive_count, "arrive_count")
    if arrive_count <= 0:
        raise ValueError("arrive_count must be positive")

    if phaseIdx is None:
        if slot.expect_bytes is not None:
            raise ValueError("barrier_arrive on a barrier with expect_bytes requires phaseIdx")
        if slot.init == tle.READY:
            raise ValueError("barrier_arrive without phaseIdx selects named barrier, which does not support READY")
        if arrive_count != 1:
            raise ValueError("named barrier backend requires barrier_arrive arrive_count = 1")
        if slot.static_index is None:
            raise ValueError("named barrier backend requires a static barrier slot index")
        _ensure_named_barrier_ids(slot, _semantic)
        _record_barrier_backend(slot, "named", _semantic)
        _semantic.builder.create_barrier_arrive_named(slot.handle, slot.named_base_id + slot.static_index,
                                                      slot.arrive_count)
        return

    phase = _barrier_phase_tensor(phaseIdx, slot.init, _semantic)
    _record_barrier_backend(slot, "mbarrier", _semantic)
    _semantic.builder.create_barrier_arrive_mbarrier(slot.handle, arrive_count, phase.handle)


def _require_wgmma_int(value, name: str) -> int:
    value = tl._unwrap_if_constexpr(value)
    if isinstance(value, tl.constexpr):
        value = value.value
    if not isinstance(value, int):
        raise ValueError(f"{name} must be a compile-time integer")
    return value


def _require_wgmma_smem_operand(value, name: str) -> tle.buffered_tensor:
    if not isinstance(value, tle.buffered_tensor):
        raise ValueError(f"{name} must be a tle.gpu buffered_tensor in shared memory")
    if value.type.storage is not tle.smem:
        raise ValueError(f"{name} must live in shared memory")
    if not isinstance(value.type.layout, tle.nv_mma_shared_layout):
        raise ValueError(f"{name} must use nv_mma_shared_layout; allocate it with the default tle.gpu.alloc layout")
    if len(value.type.shape) != 2:
        raise ValueError(f"{name} must be a rank-2 shared tile")
    return value


def _require_wgmma_bool(value, name: str) -> bool:
    value = tl._unwrap_if_constexpr(value)
    if isinstance(value, tl.constexpr):
        value = value.value
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a compile-time bool")
    return value


def _require_rank2_wgmma_operand(value, name: str):
    if len(value.type.shape) != 2:
        raise ValueError(f"{name} transpose currently supports only rank-2 operands")


def _require_transpose_order(order, rank: int, name: str):
    if len(order) != rank or sorted(order) != list(builtins.range(rank)):
        raise ValueError(f"{name} transpose order must be a permutation of rank {rank}")


def _transpose_wgmma_smem_operand(value: tle.buffered_tensor, name: str, _semantic) -> tle.buffered_tensor:
    _require_rank2_wgmma_operand(value, name)
    order = [1, 0]
    _require_transpose_order(order, len(value.type.shape), name)
    handle = _semantic.builder.create_memdesc_trans(value.handle, order)
    shape = [value.type.shape[i] for i in order]

    alloc_shape = value.type.alloc_shape
    leading_rank = len(alloc_shape) - len(value.type.shape)
    alloc_tail = alloc_shape[leading_rank:]
    transposed_alloc_shape = alloc_shape[:leading_rank] + [alloc_tail[i] for i in order]

    layout = value.type.layout.make_permute(order)
    return tle.buffered_tensor(
        handle,
        value.dtype,
        shape,
        value.type.storage,
        layout,
        _semantic,
        alloc_shape=transposed_alloc_shape,
    )


_WGMMA_ALLOWED_OPERAND_TYPE_PAIRS = (
    (tle.buffered_tensor, tle.buffered_tensor),
    (tl.tensor, tle.buffered_tensor),
)


def _canonicalize_wgmma_operands(a, b, trans_a: bool, trans_b: bool, _semantic):
    a = tl._unwrap_if_constexpr(a)
    b = tl._unwrap_if_constexpr(b)

    if not any(isinstance(a, a_ty) and isinstance(b, b_ty) for a_ty, b_ty in _WGMMA_ALLOWED_OPERAND_TYPE_PAIRS):
        if isinstance(b, tl.tensor):
            raise ValueError(
                "wgmma b currently supports only shared-memory buffered_tensor operands; tensor B is unsupported")
        raise ValueError("wgmma operands must be one of: "
                         "(shared-memory buffered_tensor, shared-memory buffered_tensor) or "
                         "(tl.tensor, shared-memory buffered_tensor)")

    if isinstance(a, tle.buffered_tensor):
        a = _require_wgmma_smem_operand(a, "wgmma a")
        if trans_a:
            a = _transpose_wgmma_smem_operand(a, "wgmma a", _semantic)
    elif trans_a:
        _require_rank2_wgmma_operand(a, "wgmma a")
        a = tl.trans(a, _semantic=_semantic)

    b = _require_wgmma_smem_operand(b, "wgmma b")
    if trans_b:
        b = _transpose_wgmma_smem_operand(b, "wgmma b", _semantic)
    return a, b


def _wgmma_ret_scalar_ty(lhs_dtype, out_dtype):
    if lhs_dtype.is_int():
        if lhs_dtype != tl.int8:
            raise ValueError("wgmma integer operands currently support only tl.int8")
        return tl.int32
    if out_dtype.is_bf16():
        raise ValueError("wgmma out_dtype=bfloat16 is unsupported; use float32/float16 and cast afterward")
    if lhs_dtype.is_fp32() or lhs_dtype.is_bf16():
        return tl.float32
    return out_dtype


def _wgmma_zero_value(builder, scalar_ty):
    if scalar_ty.is_int():
        return builder.get_int32(0)
    if scalar_ty.is_fp64():
        return builder.get_fp64(0)
    if scalar_ty.is_fp16():
        return builder.get_fp16(0)
    return builder.get_fp32(0)


@tl.builtin
def wgmma(
    a,
    b,
    acc=None,
    input_precision=None,
    max_num_imprecise_acc=None,
    out_dtype=tl.float32,
    trans_a: tl.constexpr = False,
    trans_b: tl.constexpr = False,
    _semantic=None,
) -> tl.tensor:
    """
    Issue an asynchronous Hopper WGMMA.

    A may be a shared-memory TLE buffer or a register tensor. B must currently
    be a shared-memory TLE buffer. ``trans_a`` and ``trans_b`` are rank-2
    descriptor/tensor transpose requests; shared-memory transposes are emitted
    as descriptor-only views.

    The returned accumulator is an async WGMMA dependency value. Use
    ``tle.gpu.wgmma_wait(pendings, acc)`` before consuming it with ordinary
    tensor operations or storing it.
    """
    trans_a = _require_wgmma_bool(trans_a, "trans_a")
    trans_b = _require_wgmma_bool(trans_b, "trans_b")
    a, b = _canonicalize_wgmma_operands(a, b, trans_a, trans_b, _semantic)

    m, k = [int(tl._unwrap_if_constexpr(dim)) for dim in a.type.shape]
    k_b, n = [int(tl._unwrap_if_constexpr(dim)) for dim in b.type.shape]
    if k != k_b:
        raise ValueError(f"wgmma shape mismatch: a is {a.type.shape}, b is {b.type.shape}")
    if m < 64 or m % 64 != 0:
        raise ValueError("wgmma result M dimension must be divisible by 64")
    if n < 8 or n % 8 != 0:
        raise ValueError("wgmma result N dimension must be divisible by 8")
    if k < 16:
        raise ValueError("wgmma K dimension must be at least 16")

    if not (a.dtype.is_fp8() and b.dtype.is_fp8()):
        if a.dtype != b.dtype:
            raise ValueError(f"wgmma operands must have the same dtype, got {a.dtype} and {b.dtype}")
        if a.dtype not in (tl.int8, tl.float16, tl.bfloat16, tl.float32):
            raise ValueError(f"unsupported wgmma operand dtype {a.dtype}")

    out_dtype = tl._unwrap_if_constexpr(out_dtype)
    if not isinstance(out_dtype, tl.dtype):
        raise ValueError(f"wgmma out_dtype must be a Triton dtype, got {type(out_dtype).__name__}")

    if input_precision is None:
        input_precision = _semantic.builder.options.default_dot_input_precision
    input_precision = _semantic._str_to_dot_input_precision(tl._unwrap_if_constexpr(input_precision))

    max_num_imprecise_acc = tl._unwrap_if_constexpr(max_num_imprecise_acc)
    if isinstance(max_num_imprecise_acc, tl.constexpr):
        max_num_imprecise_acc = max_num_imprecise_acc.value
    if max_num_imprecise_acc is None:
        if a.dtype.is_fp8() and b.dtype.is_fp8():
            max_num_imprecise_acc = _semantic.builder.options.max_num_imprecise_acc_default
        else:
            max_num_imprecise_acc = 0
    else:
        max_num_imprecise_acc = _require_wgmma_int(max_num_imprecise_acc, "max_num_imprecise_acc")
        if max_num_imprecise_acc < 0:
            raise ValueError("max_num_imprecise_acc must be non-negative")

    ret_scalar_ty = _wgmma_ret_scalar_ty(a.dtype, out_dtype)
    ret_ty = tl.block_type(ret_scalar_ty, [m, n])
    builder = _semantic.builder

    acc = tl._unwrap_if_constexpr(acc)
    if acc is None:
        zero = _wgmma_zero_value(builder, ret_scalar_ty)
        acc_handle = builder.create_splat(ret_ty.to_ir(builder), zero)
    else:
        if not isinstance(acc, tl.tensor):
            raise ValueError(f"wgmma acc must be a tl.tensor or None, got {type(acc).__name__}")
        if tuple(int(tl._unwrap_if_constexpr(dim)) for dim in acc.type.shape) != (m, n):
            raise ValueError(f"wgmma acc shape must be {(m, n)}, got {acc.type.shape}")
        if acc.dtype != ret_scalar_ty:
            raise ValueError(f"wgmma acc dtype must be {ret_scalar_ty}, got {acc.dtype}")
        acc_handle = acc.handle

    result = builder.create_tle_wgmma(
        a.handle,
        b.handle,
        acc_handle,
        input_precision,
        max_num_imprecise_acc,
        True,
    )
    return tensor(result, ret_ty)


@tl.builtin
def wgmma_wait(pendings, acc=None, _semantic=None, _generator=None) -> tl.tensor:
    """Wait until ``pendings`` or fewer async WGMMA groups remain outstanding."""
    _mark_wgmma_user_promise(_semantic, _generator)
    if acc is None and isinstance(pendings, tl.tensor):
        acc = pendings
        pendings = 0
    pendings = _require_wgmma_int(pendings, "pendings")
    if pendings < 0:
        raise ValueError("wgmma_wait pendings must be non-negative")
    if not isinstance(acc, tl.tensor):
        raise ValueError(f"wgmma_wait acc must be a tl.tensor, got {type(acc).__name__}")
    result = _semantic.builder.create_tle_wgmma_wait(acc.handle, pendings)
    return tensor(result, acc.type)


class CopyDirection(Enum):
    """Copy direction enum for data transfer operations"""
    GM_TO_LOCAL = "GMTOLOCAL"  # Global memory to local memory
    LOCAL_TO_GM = "LOCALTOGM"  # Local memory to global memory


@tl.builtin
def copy(
    src,
    dst,
    shape,
    offsets: Sequence[constexpr | tensor] = None,
    barrier=None,
    _semantic=None,
) -> None:
    """
    High-performance data copy operation supporting TMA (Tensor Memory Accelerator) transfers.

    This function provides efficient data movement between different memory spaces, with support for:
    - TMA descriptor-based transfers (for NVIDIA Hopper architecture and later)
    - Standard global ↔ local memory transfers
    - Bidirectional data movement with automatic direction detection

    Transfer Direction Modes:
    - GM_TO_LOCAL: Global memory → Local memory (shared memory)
    - LOCAL_TO_GM: Local memory → Global memory

    Direction is automatically determined based on operand types:
    - If src is tl.tensor/tensor_descriptor and dst is tle.buffered_tensor: GM_TO_LOCAL
    - If src is tle.buffered_tensor and dst is tl.tensor/tensor_descriptor: LOCAL_TO_GM

    Args:
        src: Source data - can be:
            - tl.tensor (global memory tensor)
            - tl.tensor_descriptor (TMA descriptor for global memory)
            - tle.buffered_tensor (local memory buffer)
        dst: Destination - can be:
            - tle.buffered_tensor (local memory buffer)
            - tl.tensor (global memory tensor)
            - tl.tensor_descriptor (TMA descriptor for global memory)
        shape: Tuple specifying the dimensions of the data to copy
        offsets: Sequence of offsets for multi-dimensional addressing. Used with TMA operations
            to specify the starting coordinates within the tensor. Required for TMA copy.
        barrier: Optional TLE GPU mbarrier completion barrier for global-to-shared TMA copy.
            The barrier must come from ``tle.gpu.alloc_barrier(s)(expect_bytes=...)``.
        _semantic: Internal semantic analyzer for validation and compilation (user-provided)

    Raises:
        ValueError: When parameter types are incompatible or offsets missing for TMA operations
        RuntimeError: When copy operation fails during execution or compilation

    Examples:
        Standard global → local copy:
            local_buf = tle.alloc([256, 256], dtype=tl.float32, scope=tle.smem)
            tle.copy(global_tensor, local_buf, [256, 256])

        TMA copy with offsets:
            tle.copy(tma_desc, local_buf, [64, 64], [x_offset, y_offset])

        TMA copy with explicit completion barrier:
            bar = tle.gpu.alloc_barrier(expect_bytes=64 * 64 * 2)
            tle.copy(tma_desc, local_buf, [64, 64], [x_offset, y_offset], barrier=bar)
            tle.gpu.barrier_wait(bar, phaseIdx=0)
    """
    mthreads_enabled = mthreads_copy.enabled()
    iluvatar_enabled = iluvatar_copy.enabled()

    def normcopy(
        src: tl.tensor,
        dst: tle.buffered_tensor,
        shape: tuple,
        direction,
        _semantic=None,
    ) -> None:
        if mthreads_enabled:
            mthreads_copy.validate_normal_copy(src, dst, shape, direction)

        # Semantic analysis
        try:
            from .semantic import TLESemantic
            if isinstance(_semantic, TLESemantic):
                _semantic.analyze_copy_operation(src, dst, shape)
        except ImportError:
            # If semantic analysis module is not available, continue with warning
            import warnings
            warnings.warn("TLE semantic analysis module not available, skipping validation", UserWarning)

        mask = None
        other = None
        boundary_check = ()
        padding_option = ""
        cache_modifier = ""
        eviction_policy = ""
        volatile = False

        try:
            if direction == CopyDirection.GM_TO_LOCAL:
                # None fills the FlagTree hints slot; TLE copy has no hints to pass.
                load_extra_args = () if (mthreads_enabled or iluvatar_enabled) else (None, )
                tt_load = _semantic.load(src, mask, other, boundary_check, padding_option, cache_modifier,
                                         eviction_policy, volatile, *load_extra_args)
                local_ptrs = local_ptr(dst, _make_full_indices(dst, _semantic), _semantic=_semantic)
                _semantic.store(local_ptrs, tt_load, mask, boundary_check, cache_modifier, eviction_policy)
            else:
                local_ptrs = local_ptr(src, _make_full_indices(src, _semantic), _semantic=_semantic)
                load = tl.load(local_ptrs, _semantic=_semantic)
                _semantic.store(dst, load, mask, boundary_check, cache_modifier, eviction_policy)
        except Exception as e:
            raise RuntimeError(f"copy operation failed: {str(e)}") from e

    # this api is use for tma copy
    def tmacopy(
        src: tle.buffered_tensor | tl.tensor_descriptor,
        dst: tle.buffered_tensor | tl.tensor_descriptor,
        direction,
        shape: tuple,
        offsets: Sequence[constexpr | tensor],
        barrier=None,
        _semantic=None,
    ) -> None:
        # Parameter validation
        valid_types = (tle.buffered_tensor, tl.tensor_descriptor)

        if not isinstance(src, valid_types):
            raise ValueError(
                f"Source parameter must be tle.buffered_tensor or tl.tensor_descriptor, but got {type(src).__name__}")

        if not isinstance(dst, valid_types):
            raise ValueError(
                f"Destination parameter must be tle.buffered_tensor or tl.tensor_descriptor, but got {type(dst).__name__}"
            )

        # Auto-determine copy direction based on operand types
        if isinstance(src, tle.buffered_tensor) and isinstance(dst, tl.tensor_descriptor):
            desc = dst
        elif isinstance(src, tl.tensor_descriptor) and isinstance(dst, tle.buffered_tensor):
            desc = src
        else:
            raise ValueError(
                f"Invalid copy combination: src={type(src).__name__}, dst={type(dst).__name__}. "
                "One operand must be tl.tensor_descriptor (global memory) and the other must be tle.buffered_tensor (local memory)"
            )

        if not isinstance(shape, (tuple, list)):
            # Try to handle Triton tuple-like objects
            if hasattr(shape, '__iter__'):
                shape = tuple(shape)
            else:
                raise ValueError(f"Shape parameter must be tuple or list, but got {type(shape)}")

        if not isinstance(offsets, (tuple, list)):
            # Try to handle Triton tuple-like objects
            if hasattr(offsets, '__iter__'):
                offsets = tuple(offsets)
            else:
                raise ValueError(f"Shape parameter must be tuple or list, but got {type(shape)}")

        barrier_slot = None
        expect_bytes = -1
        if barrier is not None:
            if direction != CopyDirection.GM_TO_LOCAL:
                raise ValueError("TMA copy barrier is only supported for global-to-shared TMA copy")
            barrier_slot = _tma_completion_barrier_slot(barrier, _semantic)
            expect_bytes = barrier_slot.expect_bytes

        # Note: Skip shape assertion at this level since it requires _semantic context
        # assert desc.shape == shape, "Shape mismatch between descriptor and provided shape"
        assert len(offsets) == len(desc.shape), "Offsets and shape must have the same length"
        offsets = _semantic._convert_to_ir_values(offsets, require_i64=False)
        _semantic.builder.create_tma_copy(src.handle, dst.handle, offsets,
                                          None if barrier_slot is None else barrier_slot.handle, expect_bytes)
        return

    # Parameter validation
    valid_types = (tl.tensor, tle.buffered_tensor, tl.tensor_descriptor)

    if not isinstance(src, valid_types):
        raise ValueError(
            f"Source parameter must be tl.tensor or tle.buffered_tensor  tl.tensor_descriptor , but got {type(src).__name__}"
        )

    if not isinstance(dst, valid_types):
        raise ValueError(
            f"Destination parameter must be tl.tensor or tle.buffered_tensor  tl.tensor_descriptor, but got {type(dst).__name__}"
        )

    # Auto-determine copy direction based on operand types
    if isinstance(src, tle.buffered_tensor) and isinstance(dst, tl.tensor):
        direction = CopyDirection.LOCAL_TO_GM
        is_normcopy = True
    elif isinstance(src, tl.tensor) and isinstance(dst, tle.buffered_tensor):
        direction = CopyDirection.GM_TO_LOCAL
        is_normcopy = True
    elif isinstance(src, tle.buffered_tensor) and isinstance(dst, tl.tensor_descriptor):
        direction = CopyDirection.LOCAL_TO_GM
        is_normcopy = False
    elif isinstance(src, tl.tensor_descriptor) and isinstance(dst, tle.buffered_tensor):
        direction = CopyDirection.GM_TO_LOCAL
        is_normcopy = False
    else:
        raise ValueError(
            f"Invalid copy combination: src={type(src).__name__}, dst={type(dst).__name__}. "
            "One operand must be tl.tensor (global memory) and the other must be tle.buffered_tensor (local memory)")

    if not isinstance(shape, (tuple, list)):
        # Try to handle Triton tuple-like objects
        if hasattr(shape, '__iter__'):
            shape = tuple(shape)
        else:
            raise ValueError(f"Shape parameter must be tuple or list, but got {type(shape)}")
    if is_normcopy:
        if barrier is not None:
            raise ValueError("copy barrier is only supported for TMA global-to-shared copy")
        return normcopy(src, dst, shape, direction, _semantic)
    if mthreads_enabled:
        if barrier is not None:
            raise ValueError("TMA copy barrier is only supported on NVIDIA backend")
        return mthreads_copy.tmacopy(src, dst, direction, shape, offsets, _semantic)
    else:
        return tmacopy(src, dst, direction, shape, offsets, barrier, _semantic)


def _expand_index_to_shape(index: tl.tensor, shape: Sequence[int], axis: int, _semantic) -> tl.tensor:
    idx = index
    for _ in builtins.range(axis):
        idx = tl.expand_dims(idx, 0, _semantic=_semantic)
    for _ in builtins.range(len(shape) - axis - 1):
        idx = tl.expand_dims(idx, len(idx.shape), _semantic=_semantic)
    return tl.broadcast_to(idx, *shape, _semantic=_semantic)


def _make_full_indices(buffer: tle.buffered_tensor, _semantic) -> tuple[tl.tensor, ...]:
    shape = tuple(int(tl._unwrap_if_constexpr(dim)) for dim in buffer.type.shape)
    indices = []
    for axis, dim in enumerate(shape):
        idx = tl.arange(0, dim, _semantic=_semantic)
        idx = _expand_index_to_shape(idx, shape, axis, _semantic)
        indices.append(idx)
    return tuple(indices)


@tl.builtin
def local_ptr(
    buffer: tle.buffered_tensor,
    indices: Optional[Sequence] = None,
    _semantic=None,
    _generator=None,
) -> tl.tensor:
    """
    Materialize shared-memory pointers that cover the given buffered tensor.

    Args:
        buffer: Local memory buffer tensor returned by ``tle.alloc``.
        indices: Optional tuple of integer index tensors. If provided, tuple
            length must equal ``rank(buffer)`` and every tensor must have the
            same shape. If ``None``, emit a full-view pointer tensor over
            ``buffer`` shape (or scalar pointer for rank-0 buffer).
        _semantic: Semantic analyzer (internal use).
        _generator: Triton code generator (internal use).

    Returns:
        Tensor of pointers suitable for ``tl.load``/``tl.store``.
    """
    if not isinstance(buffer, tle.buffered_tensor):
        raise ValueError(f"Buffer parameter must be tle.buffered_tensor, but got {type(buffer)}")

    # Preferred metadata source: buffered_tensor.type (survives JIT value
    # reconstruction). Keep value attrs as backward-compatibility fallback.
    remote_shard_id = getattr(buffer.type, "_tle_remote_shard_id", None)
    remote_scope = getattr(buffer.type, "_tle_remote_scope", None)
    if remote_shard_id is None:
        remote_shard_id = getattr(buffer, "_tle_remote_shard_id", None)
        remote_scope = getattr(buffer, "_tle_remote_scope", None)
    remote_buffer_marker = remote_shard_id is not None

    buffer_shape = tuple(int(tl._unwrap_if_constexpr(dim)) for dim in buffer.type.shape)
    indices = tl._unwrap_if_constexpr(indices)
    no_indices = indices is None
    if no_indices:
        indices_tuple = tuple()
    elif isinstance(indices, tl.tuple):
        indices_tuple = tuple(indices.values)
    elif isinstance(indices, (tuple, list)):
        indices_tuple = tuple(indices)
    else:
        raise ValueError("local_ptr indices must be a tuple/list of tensors or None")
    if not no_indices and len(indices_tuple) != len(buffer_shape):
        raise ValueError(f"local_ptr indices must provide {len(buffer_shape)} tensors, got {len(indices_tuple)}")

    idx_tensors: list[tensor] = []
    view_shape: Optional[tuple[int, ...]] = None
    scalar_index_flags: list[bool] = []
    if not no_indices:
        for idx in indices_tuple:
            idx_tensor = idx if isinstance(idx, tensor) else _semantic.to_tensor(idx)
            if not idx_tensor.dtype.is_int():
                raise ValueError("local_ptr indices must use integer dtypes")
            is_scalar_index = not idx_tensor.type.is_block()
            scalar_index_flags.append(is_scalar_index)
            if is_scalar_index:
                idx_tensors.append(idx_tensor)
                continue
            if view_shape is None:
                view_shape = tuple(idx_tensor.shape)
            elif tuple(idx_tensor.shape) != view_shape:
                raise ValueError("local_ptr indices must have identical shapes")
            idx_tensors.append(idx_tensor)

        if not idx_tensors:
            raise ValueError("local_ptr indices cannot be empty")
        all_scalar_indices = all(scalar_index_flags)
        any_scalar_indices = any(scalar_index_flags)
        if any_scalar_indices and not all_scalar_indices:
            raise ValueError("local_ptr indices must be either all scalar or all tensors with identical shapes")
        if not all_scalar_indices and view_shape is None:
            view_shape = tuple()
    else:
        all_scalar_indices = len(buffer_shape) == 0
        if not all_scalar_indices:
            view_shape = buffer_shape

    try:
        from .semantic import TLESemantic
        if isinstance(_semantic, TLESemantic):
            _semantic.analyze_local_pointer_operation(buffer, idx_tensors)
    except ImportError:
        import warnings
        warnings.warn("TLE semantic analysis module not available, skipping validation", UserWarning)

    ptr_dtype = tl.pointer_type(buffer.type.element_ty, SHARED_MEMORY_ADDRESS_SPACE)
    insert_block = _semantic.builder.get_insertion_block()
    if insert_block is None:
        raise RuntimeError("TLE local_ptr called without an insertion block")
    if no_indices:
        if len(buffer_shape) == 0:
            result_ty = ptr_dtype
            result_ir = ptr_dtype.to_ir(_semantic.builder)
        else:
            result_ty = tl.block_type(ptr_dtype, list(buffer_shape))
            result_ir = result_ty.to_ir(_semantic.builder)
    elif all_scalar_indices:
        result_ty = ptr_dtype
        result_ir = ptr_dtype.to_ir(_semantic.builder)
    else:
        result_ty = tl.block_type(ptr_dtype, list(view_shape))
        result_ir = result_ty.to_ir(_semantic.builder)
    handles = [idx.handle for idx in idx_tensors]
    local_ptr_op = _semantic.builder.create_local_pointers(result_ir, buffer.handle, *handles)

    result_tensor = tl.tensor(local_ptr_op.get_result(0), result_ty)

    if remote_buffer_marker:
        # Keep remote semantics attached to the source buffered tensor and
        # materialize them only when pointer view is requested. This applies
        # to both block and scalar pointer views so remote/local_ptr semantics
        # stay aligned with local shared-memory local_ptr.
        from triton.experimental.tle.language import distributed as _tle_distributed
        result_tensor = _tle_distributed._remote_pointer(
            result_tensor,
            remote_shard_id,
            scope=remote_scope,
            _semantic=_semantic,
        )

    return result_tensor
