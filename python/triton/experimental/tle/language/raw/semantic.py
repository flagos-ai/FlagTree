from typing import TypeVar

from triton._C.libtriton import ir
from triton.language.semantic import TritonSemantic

TensorTy = TypeVar("TensorTy")


class TLERawSemantic(TritonSemantic[TensorTy]):

    def __init__(self, builder: ir.builder, *args, **kwargs) -> None:
        super().__init__(builder, *args, **kwargs)
