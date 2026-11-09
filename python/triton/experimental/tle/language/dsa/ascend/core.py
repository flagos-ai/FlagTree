# Copyright 2026- Xcoresigma Technology Co., Ltd

from dataclasses import dataclass

from triton.language.extra.cann.extension.core import ascend_address_space, sub_vec_id, sub_vec_num, sync_block_set, sync_block_wait, sync_block_all
import triton.language.extra.cann.extension as ascend_langugage_cann_extension
from triton.language.extra.cann.extension import compile_hint, custom as raw, multibuffer

UB = ascend_address_space.UB
L1 = ascend_address_space.L1
L0A = ascend_address_space.L0A
L0B = ascend_address_space.L0B
L0C = ascend_address_space.L0C

sub_vec_id = sub_vec_id
sub_vec_num = sub_vec_num
sync_block_set = sync_block_set
sync_block_wait = sync_block_wait
sync_block_all = sync_block_all
compile_hint = compile_hint
raw = raw
multibuffer = multibuffer
PIPE = ascend_langugage_cann_extension.PIPE


@dataclass(frozen=True)
class SyncSpec:
    __triton_compile_time_value__ = True
    backend = "ascend"

    sender: str
    receiver: str
    sender_pipe: PIPE
    receiver_pipe: PIPE

    def __post_init__(self):
        if self.sender not in ("cube", "vector"):
            raise ValueError(f"sender must be 'cube' or 'vector', got {self.sender!r}")
        if self.receiver not in ("cube", "vector"):
            raise ValueError(f"receiver must be 'cube' or 'vector', got {self.receiver!r}")
        if self.sender == self.receiver:
            raise ValueError("sender and receiver must be different")
        if not isinstance(self.sender_pipe, PIPE):
            raise TypeError("sender_pipe must be an instance of PIPE")
        if not isinstance(self.receiver_pipe, PIPE):
            raise TypeError("receiver_pipe must be an instance of PIPE")
