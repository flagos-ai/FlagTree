"""Typed shared mbarrier pointer view for native bulk TMA inline PTX."""
import triton.language as tl
from triton.language.core import builtin


@builtin
def barrier_address(bar, _semantic=None):
    if not getattr(bar, 'is_slot', False):
        raise ValueError('bulk TMA requires one indexed mbarrier slot')
    zero = _semantic.to_tensor(0)
    dtype = tl.pointer_type(tl.int64, 3)
    op = _semantic.builder.create_local_pointers(dtype.to_ir(_semantic.builder), bar.handle, zero.handle)
    return tl.tensor(op.get_result(0), dtype)
