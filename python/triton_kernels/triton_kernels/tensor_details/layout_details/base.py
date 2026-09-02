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

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LayoutTransformation(ABC):

    shape: list[int]
    is_fp4: bool

    @property
    def storage_shape(self) -> list[int]:
        """Physical storage shape produced by this transformation."""
        raise NotImplementedError

    def _validate_storage_shape(self, data):
        assert list(data.shape) == self.storage_shape
        return data

    @abstractmethod
    def swizzle_data(self, data):
        pass

    @abstractmethod
    def unswizzle_data(self, data):
        pass


@dataclass(frozen=True)
class Layout(ABC):

    def can_preserve_storage_as(self, other: "Layout", rank: int) -> bool:
        """Whether existing storage is already valid for `other`."""
        return self == other

    def storage_shape(self, shape: list[int], is_fp4: bool) -> list[int]:
        """Return the physical storage shape for a logical tensor shape."""
        return self.make_transformation(shape, is_fp4).storage_shape

    @abstractmethod
    def make_transformation(self, shape: list[int], is_fp4: bool) -> LayoutTransformation:
        pass

    @abstractmethod
    def swizzle_block_shape(self, block_shape):
        pass
