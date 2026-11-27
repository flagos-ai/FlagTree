# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""One H004 last-reader arena shared by serial layer closures on one stream."""
from triton.flagmega.errors import RuntimeContractError


class SharedGdnPhasePool:
    def __init__(self, tensor, *, stream):
        from triton.flagmega.rules.ntt.r004_column_phase_bindings import POOL_BYTES
        if (str(tensor.dtype) != 'torch.uint8' or tensor.numel() != POOL_BYTES
                or not tensor.is_contiguous() or tensor.data_ptr() % 256
                or tensor.device.type != 'cuda'):
            raise RuntimeContractError('H004 shared arena requires its exact aligned device U8 span.')
        identity = getattr(stream, 'cuda_stream', stream)
        if type(identity) is not int or identity <= 0:
            raise RuntimeContractError('H004 shared arena requires one explicit CUDA stream.')
        self.tensor = tensor
        self.stream = identity

    def check_stream(self, stream):
        if getattr(stream, 'cuda_stream', stream) != self.stream:
            raise RuntimeContractError('Shared H004 arena may only be used on its owning stream.')
