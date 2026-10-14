"""Prove that all active local values use the two prefetched adjacent heads."""
from itertools import product

from triton.flagmega.codegen.triton import kernel_call_renderers as renderers
from triton.flagmega.ir import DistributedType, VectorType, local_shard_descriptor, local_tensor_type, type_from_data


def two_heads_cover(value_type, head_dim):
    if not isinstance(value_type, DistributedType) or isinstance(value_type.tensor.dtype, VectorType):
        return False
    local = local_tensor_type(value_type)
    if local.rank != 2 or not local.shape[1].is_fixed or not value_type.tensor.shape[1].is_fixed:
        return False
    extent = value_type.tensor.shape[1].fixed_value
    for owner in product(*(range(n) for n in value_type.placement.hierarchy)):
        descriptor = local_shard_descriptor(value_type, owner).axes[1]
        origin = descriptor.map_local_to_global(0)
        if not origin.is_fixed:
            return False
        base_head = origin.fixed_value // head_dim
        for index in range(local.shape[1].fixed_value):
            coordinate = descriptor.map_local_to_global(index)
            if not coordinate.is_fixed:
                return False
            if 0 <= coordinate.fixed_value < extent and coordinate.fixed_value // head_dim not in (base_head,
                                                                                                   base_head + 1):
                return False
    return True


class HeadCacheRenderer:

    def __init__(self, original):
        self.original = original

    def __call__(self, raw):
        call = self.original(raw)
        abi = renderers._buffer(raw, 'outputs', 'result_0')['abi']
        data = abi.get('distributed_type')
        call['two_head_cache_proven'] = data is not None and two_heads_cover(type_from_data(data), call['value_dim'])
        return call


def configure():
    original = renderers._FAMILY_ENCODERS['gdn_recurrent']
    if not isinstance(original, HeadCacheRenderer):
        renderers._FAMILY_ENCODERS['gdn_recurrent'] = HeadCacheRenderer(original)
