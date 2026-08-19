# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
# Copyright 2025-     FlagOS Contributors
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

from dataclasses import dataclass
from typing import TypeAlias


# data types
# ---------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IntegerType:
    bitwidth: int
    is_signed: bool


@dataclass(frozen=True)
class FloatType:
    bitwidth_exponent: int
    bitwidth_mantissa: int
    is_signed: bool
    unsigned_zero: bool = False

    @property
    def bitwidth(self):
        return int(self.is_signed) + self.bitwidth_exponent + self.bitwidth_mantissa


BIT = IntegerType(1, is_signed=False)
UINT8 = IntegerType(8, is_signed=False)
FP4 = FloatType(bitwidth_exponent=2, bitwidth_mantissa=1, is_signed=True)
FP8_E4M3FN = FloatType(bitwidth_exponent=4, bitwidth_mantissa=3, is_signed=True)
FP8_E4M3FNUZ = FloatType(bitwidth_exponent=4, bitwidth_mantissa=3, is_signed=True, unsigned_zero=True)
FP8_E5M2 = FloatType(bitwidth_exponent=5, bitwidth_mantissa=2, is_signed=True)
BF16 = FloatType(bitwidth_exponent=8, bitwidth_mantissa=7, is_signed=True)
FP16 = FloatType(bitwidth_exponent=5, bitwidth_mantissa=10, is_signed=True)
FP32 = FloatType(bitwidth_exponent=8, bitwidth_mantissa=23, is_signed=True)
FP64 = FloatType(bitwidth_exponent=11, bitwidth_mantissa=52, is_signed=True)
INT16 = IntegerType(16, is_signed=True)
INT32 = IntegerType(32, is_signed=True)
INT64 = IntegerType(64, is_signed=True)

DataType: TypeAlias = IntegerType | FloatType
