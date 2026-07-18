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

import torch
from dataclasses import dataclass

MAX_FINITE_FLOAT8E5 = 57344.0
MAX_FINITE_FLOAT8E4NV = 448.0
MAX_FINITE_FLOAT8E4B8 = 240.0


@dataclass(frozen=True)
class BaseFlexData:
    dtype: torch.dtype | None = None

    def view(self, x: torch.Tensor):
        if self.dtype is None:
            return x
        return x.view(self.dtype)

    def reinterpret(self, x):
        if self.dtype is None or x.dtype.itemsize > 1:
            return x
        return x.view(self.dtype)


@dataclass(frozen=True)
class InFlexData(BaseFlexData):
    scale: torch.Tensor | None = None

    @property
    def is_per_batch(self):
        return False if self.scale is None else len(self.scale) > 1


@dataclass(frozen=True)
class OutFlexData(BaseFlexData):
    expected_scale: torch.Tensor | None = None
    actual_scale: torch.Tensor | None = None
    checksum_scale: torch.Tensor | None = None

    @property
    def is_per_batch(self):
        return False if self.expected_scale is None else len(self.expected_scale) > 1

    def __iter__(self):
        yield self.expected_scale
        yield self.actual_scale
        yield self.checksum_scale
