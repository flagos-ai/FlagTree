from typing import TypeVar

from triton._C.libtriton.flagtree_ir import FlagTreeOpBuilder
import triton.language as tl
from triton.language.core import tensor
from triton.language.semantic import TritonSemantic

TensorTy = TypeVar("TensorTy")


class FlagTreeSemantic(TritonSemantic[TensorTy]):

    def __init__(self, builder: FlagTreeOpBuilder, *args, **kwargs) -> None:
        super().__init__(builder, *args, **kwargs)

    def call(self, func, outputs, inputs):
        dsl_region_op = self.builder.create_edsl_region_by_llvm_func(f"{func.llvm}", func.fnname,
                                                                     [output.handle for output in outputs],
                                                                     [input.handle for input in inputs])
        tensors = [tensor(result, output.type) for result, output in zip(dsl_region_op.get_results(), outputs)]
        if len(tensors) == 1:
            return tensors[0]
        else:
            return tl.tuple(tensors)
