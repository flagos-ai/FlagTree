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
        try:
            import torch_ptpu
            _is_ptpu = True
        except ImportError as e:
            _is_ptpu = False
        if _is_ptpu:
            self.get_device_capability = torch.ptpu.get_device_capability
            self.get_current_stream = lambda dev_idx: torch.ptpu.current_stream(dev_idx).ptpu_stream
            self.get_current_device = torch.ptpu.current_device
            self.set_current_device = torch.ptpu.set_device
            return

        try:
            from torch._C import _cuda_getCurrentRawStream
            _is_cuda = True
        except ImportError as e:
            _cuda_getCurrentRawStream = None
            _is_cuda = True if torch.version.cuda else False
        if _is_cuda:
            self.get_device_capability = torch.cuda.get_device_capability
            if _cuda_getCurrentRawStream is not None:
                self.get_current_stream = _cuda_getCurrentRawStream
            else:
                self.get_current_stream = lambda dev_idx: torch.cuda.current_stream(dev_idx).cuda_stream
            self.get_current_device = torch.cuda.current_device
            self.set_current_device = torch.cuda.set_device
            return

        try:
            import torch_dipu
            _is_dipu = True
        except ImportError as e:
            _is_dipu = False
        if _is_dipu:
            self.get_device_capability = torch.cuda.get_device_capability
            self.get_current_stream = lambda dev_idx: torch.cuda.current_stream(dev_idx).dipu_stream
            self.get_current_device = torch.cuda.current_device
            self.set_current_device = torch.cuda.set_device
            return

    # TODO: remove once TMA is cleaned up
    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args
