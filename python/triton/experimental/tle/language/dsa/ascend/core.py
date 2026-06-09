from triton.language.extra.cann.extension.core import ascend_address_space, sub_vec_id, sub_vec_num, sync_block_set, sync_block_wait, sync_block_all
import triton.language.extra.cann.extension as ascend_langugage_cann_extension
from triton.language.extra.cann.extension import compile_hint, multibuffer

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
multibuffer = multibuffer
PIPE = ascend_langugage_cann_extension.PIPE
