from typing import TypeVar

from triton._C.libtriton.flagtree_ir import FlagTreeOpBuilder
from triton.language.semantic import TritonSemantic

TensorTy = TypeVar("TensorTy")


class FlagTreeSemantic(TritonSemantic[TensorTy]):

    def __init__(self, builder: FlagTreeOpBuilder, *args, **kwargs) -> None:
        super().__init__(builder, *args, **kwargs)

    def call(self, func, operands):
        operands = [operand.handle for operand in operands]
        dsl_region_op = self.builder.create_edsl_region_by_llvm_func(f"{func.make_llir()}", func.fnname, operands)
        return dsl_region_op
