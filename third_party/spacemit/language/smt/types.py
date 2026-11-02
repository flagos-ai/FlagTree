import triton.language.core as tl
from typing import List, Tuple
from triton._C.libtriton import ir
from triton.language.semantic import TritonSemantic


class buffered_tensor(tl.tensor):

    def __init__(self, handle, element_ty: tl.dtype, shape: List, copies: int, scope: str,
                 semantic: TritonSemantic = None):
        buf_type = buffered_tensor_type(element_ty, shape, copies, scope, semantic)
        super().__init__(handle, buf_type)

        self.type = buf_type
        self.shape = shape
        self.element_ty = element_ty
        self.copies = copies
        self.scope = scope
        self.semantic = semantic

    def __getitem__(self, buffer_idx):
        return buffer_view(self, buffer_idx, _semantic=self.semantic)

    def _flatten_ir(self, handles) -> None:
        handles.append(self.handle)

    def __str__(self) -> str:
        return f"buffered_tensor<{self.element_ty}, {self.shape}, copies={self.copies}>"

    def make_permute(self, handle, dims):
        return buffered_tensor(
            handle,
            self.element_ty,
            [self.shape[d] for d in dims],
            self.type.copies,
            self.type.scope,
        )


class buffered_tensor_type(tl.pointer_type):

    def __init__(self, element_ty: tl.dtype, shape: List, copies: int, scope: str, semantic: TritonSemantic = None):
        super().__init__(element_ty, shape)
        self.scope = scope
        self.copies = copies
        self.semantic = semantic
        self.element_ty = element_ty
        self.shape = shape

        assert semantic or copies == 0, "buffered_tensor array must be created with a builder"

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[buffered_tensor, int]:
        value = buffered_tensor(handles[cursor], self.scalar, self.shape, self.copies, self.scope, self.semantic)
        return value, cursor + 1

    def mangle(self) -> str:
        elt = self.scalar.mangle()
        shape = '_'.join(map(str, self.shape))
        if self.copies > 0:
            shape += f'_{self.copies}'
        return f'buffered_{elt}S{shape}'

    def __str__(self) -> str:
        return f"buffered_tensor_ptr_<{self.element_ty}, {self.shape}, {self.copies}>"

    def __eq__(self, other) -> bool:
        return (type(self) is type(other) and self.shape == other.shape and self.copies == other.copies)

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        out.append(self.to_ir(builder))

    def to_ir(self, builder: ir.builder) -> None:
        shape = self.shape
        builder = self.semantic.builder
        if self.copies >= 1:
            shape = [self.copies] + list(shape)

        return builder.create_smt_buffer_type(
            shape,
            self.element_ty.to_ir(builder),
            self.copies,
            self.scope.value,
        )

    @property
    def scalar(self):
        return self.element_ty.scalar if hasattr(self.element_ty, 'scalar') else self.element_ty

    def _flatten_ir(self, handles) -> None:
        handles.append(self.handle)


@tl.builtin
def buffer_view(
    local_allocated_buffers: buffered_tensor,
    buffer_idx: int,
    _semantic=None,
) -> tl.tensor:
    """
    Returns a subview of the buffer.
    """
    buffer_idx = _semantic._convert_elem_to_ir_value(buffer_idx, require_i64=False)
    view_handle = _semantic.builder.create_buffer_tensor_subview(local_allocated_buffers.handle, buffer_idx)

    original_shape = local_allocated_buffers.shape
    if local_allocated_buffers.type.copies == 0:
        if len(original_shape) == 1:
            new_shape = [1]
        else:
            new_shape = original_shape[1:]
    else:
        new_shape = original_shape[1:]

    element_ty = local_allocated_buffers.element_ty

    return tl.tensor(view_handle, tl.pointer_type(tl.block_type(element_ty, new_shape)))


class mbarrier_type(tl.dtype):
    """Type for multi-copy mbarrier."""

    def __init__(self, num: int, semantics=None):
        self.num = num
        self.semantics = semantics

    def to_ir(self, builder):
        return builder.get_mbarrier_type(self.num)

    def __repr__(self):
        return f"mbarrier_type(num={self.num})"

    def __eq__(self, other):
        if not isinstance(other, mbarrier_type):
            return False
        return self.num == other.num

    def __hash__(self):
        return hash(('mbarrier_type', self.num))


class barrier_view:
    """
    A view into a single barrier from mbarrier copies.

    The handle is an I64 value representing the barrier slot/offset.
    """

    def __init__(self, handle, parent_mbarrier, index: int, semantics=None):
        """
        Args:
            handle: mlir::Value of I64 type (barrier slot)
            parent_mbarrier: the parent mbarrier object
            index: the index used to create this view
            semantics: triton semantic context
        """
        self.handle = handle
        self.parent = parent_mbarrier
        self.index = index
        self.semantics = semantics

    def __repr__(self):
        return f"barrier_view(index={self.index}, parent_num={self.parent.num})"


class mbarrier(tl.tensor):
    """
    Handle for multi-copy mbarrier.
    """

    def __init__(self, handle, num: int, flag: int = 0, arrive_count: int = 0, transaction_count: int = 0,
                 expect_count: int = 1, semantics=None):
        self._type = mbarrier_type(num, semantics)
        self.handle = handle
        self.num = num
        self.flag = flag
        self.arrive_count = arrive_count
        self.transaction_count = transaction_count
        self.expect_count = expect_count
        self.semantics = semantics

    @property
    def type(self):
        return self._type

    def __getitem__(self, index):
        idx = tl._unwrap_if_constexpr(index)
        if isinstance(idx, tl.constexpr):
            idx = idx.value
        if isinstance(idx, int):
            if idx < 0 or idx >= self.num:
                raise IndexError(f"barrier index {idx} out of range [0, {self.num})")
            idx_handle = self.semantics.builder.create_i64_constant(idx)
        elif isinstance(idx, tl.tensor):
            idx_handle = idx.handle
        elif hasattr(idx, 'handle'):
            idx_handle = idx.handle
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

        handle = self.semantics.builder.create_mbarrier_subview(self.handle, idx_handle)
        return barrier_view(handle, self, idx, self.semantics)

    def __repr__(self):
        return f"mbarrier(num={self.num}, flag={self.flag}, arrive={self.arrive_count}, txn={self.transaction_count}, expect={self.expect_count})"
