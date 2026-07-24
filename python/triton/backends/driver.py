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

from abc import ABCMeta, abstractmethod
from typing import Callable, List, Protocol, Sequence


class Benchmarker(Protocol):

    def __call__(self, kernel_call: Callable, *, quantiles: List[float], **kwargs) -> Sequence[float]:
        pass


class DriverBase(metaclass=ABCMeta):

    @classmethod
    @abstractmethod
    def is_active(self):
        pass

    @abstractmethod
    def map_python_to_cpp_type(self, ty: str) -> str:
        """
        Converts a Triton type string to its corresponding C++ type string for this backend.

        Args:
            ty (str): The Triton type string. e.g., 'i32', '*fp16', 'fp32'.

        Returns:
            str: The C++ type string.
        """
        pass

    @abstractmethod
    def get_current_target(self):
        pass

    @abstractmethod
    def get_active_torch_device(self):
        pass

    @abstractmethod
    def get_benchmarker(self) -> Benchmarker:
        """
        Return the benchmarking function that this backend should use by default.
        """
        raise NotImplementedError

    def __init__(self) -> None:
        pass


class GPUDriver(DriverBase):

    def __init__(self):
        # TODO: support other frameworks than torch
        import torch
        self.get_device_capability = torch.cuda.get_device_capability
        try:
            from torch._C import _cuda_getCurrentRawStream
            self.get_current_stream = _cuda_getCurrentRawStream
        except ImportError:
            self.get_current_stream = lambda idx: torch.cuda.current_stream(idx).cuda_stream
        self.get_current_device = torch.cuda.current_device
        self.set_current_device = torch.cuda.set_device

    # TODO: remove once TMA is cleaned up
    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args
