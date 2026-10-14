"""A single block per owner balances packed IQ4 rows on a ragged CTA grid."""
from math import prod

from triton.flagmega.ir import DType, SBP, VectorType


class BalancedIQ4Rows:

    def __init__(self, original, gdn_linear_owners=False):
        self.original = original
        self.gdn_linear_owners = gdn_linear_owners

    @property
    def identity(self):
        return f'local-iq4-single-block-rows/v2(align=8,gdn_linear={self.gdn_linear_owners},{self.original.identity})'

    def get_candidates(self, context):
        tensor = context.tensor_type
        lanes = prod(tensor.dtype.lanes) if isinstance(tensor.dtype, VectorType) else 1
        scalar_extent = None if context.maximum_extent is None else context.maximum_extent * lanes
        raw_iq4 = (tensor.dtype == DType.UINT8 and tensor.rank == 2 and tensor.shape[1].is_fixed
                   and tensor.shape[1].fixed_value in (2720, 3264, 9248))
        if self.gdn_linear_owners and scalar_extent in (6144, 10240) and (
            (raw_iq4 and context.tensor_axis == 0) or (not raw_iq4 and context.tensor_axis == tensor.rank - 1)):
            capacity = scalar_extent // 32
            assert capacity % lanes == 0
            return (SBP.split_block_cyclic(context.hierarchy_axes, capacity // lanes), )
        if (tensor.dtype != DType.UINT8 or tensor.rank != 2 or not tensor.shape[1].is_fixed
                or tensor.shape[1].fixed_value not in (2720, 3264, 9248)):
            return self.original.get_candidates(context)
        axes = context.hierarchy_axes
        if context.tensor_axis != 0 or context.maximum_extent is None:
            return ()
        owners = prod(context.placement.hierarchy[axis] for axis in axes)
        extent = context.maximum_extent
        capacity = ((extent + owners - 1) // owners + 7) // 8 * 8
        # Legal existing BlockCyclic semantics admit ragged extents. Since the
        # entire matrix fits one round, every owner gets one consecutive span.
        assert extent <= owners * capacity
        return (SBP.split_block_cyclic(axes, capacity), )


def configure(target, *, gdn_linear_owners=False):
    policy = target.distribution_policy
    policy._split_candidate_provider = BalancedIQ4Rows(policy._split_candidate_provider, gdn_linear_owners)
