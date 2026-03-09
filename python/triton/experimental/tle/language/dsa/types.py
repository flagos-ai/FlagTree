from triton._C.libtriton import ir

from typing import List
import triton.language.core as tl


class address_space:
    """Represents a buffer's address space.

    The :code:`address_space` of a buffer is a target-specific attribute.
    """

    def to_ir(self, builder: ir.builder) -> ir.type:
        raise NotImplementedError("Abstract address_space cannot be converted to ir")


class buffer_type(tl.dtype):

    def __init__(self, element_ty: tl.dtype, shape: List, space: address_space = None, strides: List = None):
        self.element_ty = element_ty
        self.shape = shape if isinstance(shape, list) else list(shape)
        self.space = space
        self.strides = strides if strides is not None else []
        self.name = self._make_name()

    def _make_name(self):
        res = '<buffer ' + 'x'.join(str(s) for s in self.shape) + 'x' + str(self.element_ty)
        if self.strides:
            res += ', strides=[' + ', '.join(str(s) for s in self.strides) + ']'
        if self.space:
            res += ', ' + str(self.space)
        return res + '>'

    def to_ir(self, builder: ir.builder) -> ir.type:
        element_ty_ir = self.element_ty.to_ir(builder)
        addr_space_attr = self.space.to_ir(builder) if self.space else builder.get_null_attr()

        # use the method with strides if strides is not empty
        if self.strides:
            return builder.dsa_get_buffer_type_with_strides(self.shape, element_ty_ir, self.strides, addr_space_attr)
        else:
            return builder.dsa_get_buffer_ty(self.shape, element_ty_ir, addr_space_attr)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other) -> bool:
        if not isinstance(other, buffer_type):
            return False
        return (self.element_ty == other.element_ty and self.shape == other.shape and self.space == other.space
                and self.strides == other.strides)

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    @property
    def scalar(self):
        return self.element_ty


# -----------------------
# buffer
# -----------------------


class buffer(tl._value):
    """Represents a region of memory.

    :code:`buffer` is the fundamental data structure for Triton programs using
    the buffer language extension. Most functions in
    :py:mod:`triton.extension.buffer.language` operate on and return buffers.

    Most of the named member functions here are duplicates of the free functions
    in :code:`triton.language`.  For example, :code:`triton.language.sqrt(x)` is
    equivalent to :code:`x.sqrt()`.

    .. rubric:: Constructors
    ..
       For some reason Sphinx includes __init__ before printing the full table
       of methods.  Not what I want, but I can't figure out how to fix it.  Give
       it its own section so it looks intentional. :)
    """

    def __init__(self, handle, buffer_ty: buffer_type):
        """Not called by user code."""
        super().__init__(handle)
        self.type = buffer_ty
        self.dtype = buffer_ty.element_ty.scalar
        self.shape = buffer_ty.shape
        self.space = buffer_ty.space
        self.strides = buffer_ty.strides

    def __str__(self) -> str:
        # ex. "<16x32xfloat32, address_space>"
        res = '<' + 'x'.join(str(s) for s in self.shape) + 'x' + str(self.dtype)
        if self.space:
            res += ', ' + str(self.space)
        return res + '>'
