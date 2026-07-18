# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from triton.language.semantic import TritonSemantic
from triton.language import core as tl
from typing import TypeVar, Type, List

TensorTy = TypeVar('TensorTy')


class ThriveSemantic(TritonSemantic):
    tensor: Type[TensorTy] = tl.tensor

    def __init__(self, builder):
        """
        Args:
            builder: ThriveBuilder instance
        """
        super().__init__(builder)

    # ==================== Custom API ====================

    def asin(self, input: TensorTy) -> TensorTy:
        result = self.builder.create_asin(input.handle)
        return self.tensor(result, input.type)

    def asinh(self, input: TensorTy) -> TensorTy:
        result = self.builder.create_asinh(input.handle)
        return self.tensor(result, input.type)

    def acos(self, input: TensorTy) -> TensorTy:
        result = self.builder.create_acos(input.handle)
        return self.tensor(result, input.type)

    def acosh(self, input: TensorTy) -> TensorTy:
        result = self.builder.create_acosh(input.handle)
        return self.tensor(result, input.type)

    def atan(self, input: TensorTy) -> TensorTy:
        result = self.builder.create_atan(input.handle)
        return self.tensor(result, input.type)

    def atanh(self, input: TensorTy) -> TensorTy:
        result = self.builder.create_atanh(input.handle)
        return self.tensor(result, input.type)

    def sin(self, input: TensorTy) -> TensorTy:
        result = self.builder.create_sin(input.handle)
        return self.tensor(result, input.type)

    def sinh(self, input: TensorTy) -> TensorTy:
        result = self.builder.create_sinh(input.handle)
        return self.tensor(result, input.type)

    def cos(self, input: TensorTy) -> TensorTy:
        result = self.builder.create_cos(input.handle)
        return self.tensor(result, input.type)

    def cosh(self, input: TensorTy) -> TensorTy:
        result = self.builder.create_cosh(input.handle)
        return self.tensor(result, input.type)

    def tan(self, input: TensorTy) -> TensorTy:
        result = self.builder.create_tan(input.handle)
        return self.tensor(result, input.type)

    def tanh(self, input: TensorTy) -> TensorTy:
        result = self.builder.create_tanh(input.handle)
        return self.tensor(result, input.type)

    def exp(self, input: TensorTy) -> TensorTy:
        result = self.builder.create_exp(input.handle)
        return self.tensor(result, input.type)

    def exp2(self, input: TensorTy) -> TensorTy:
        result = self.builder.create_exp2(input.handle)
        return self.tensor(result, input.type)

    def log(self, input: TensorTy) -> TensorTy:
        result = self.builder.create_log(input.handle)
        return self.tensor(result, input.type)

    def log2(self, input: TensorTy) -> TensorTy:
        result = self.builder.create_log2(input.handle)
        return self.tensor(result, input.type)

    def rsqrt(self, input: TensorTy) -> TensorTy:
        result = self.builder.create_rsqrt(input.handle)
        return self.tensor(result, input.type)

    def pow(self, base: TensorTy, exponent: TensorTy) -> TensorTy:
        result = self.builder.create_pow(base.handle, exponent.handle)
        return self.tensor(result, base.type)

    def extern_call(self, symbol: str, args: List[TensorTy], ret_types: List[tl.dtype],
                    is_pure: bool = True) -> TensorTy:
        arg_handles = [arg.handle for arg in args]
        builder = self.builder
        ret_ir_types = [t.to_ir(builder) for t in ret_types]
        result = builder.create_extern_call(symbol, arg_handles, ret_ir_types, is_pure)
        return self.tensor(result, ret_types[0])


def get_thrive_semantic(_semantic):
    from triton._C.libtriton.triton_thrive import ThriveBuilder

    dummy_val = _semantic.builder.get_int32(0)
    ctx = dummy_val.get_context()
    builder = ThriveBuilder(ctx)

    ip = _semantic.builder.get_insertion_point()
    builder.restore_insertion_point(ip)

    return ThriveSemantic(builder)
