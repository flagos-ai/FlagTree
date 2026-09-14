"""Local extent-driven tile choices over the normal elementwise catalog."""

from dataclasses import replace
from math import prod

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.candidates import ElementwiseCandidateProvider
from triton.flagmega.ir.distributed_type import local_shape


def scalar_capacity(value_type):
    tensor = fm.logical_type(value_type)
    if not isinstance(tensor, fm.TensorType):
        return None
    shape = local_shape(value_type) if isinstance(value_type, fm.DistributedType) else tensor.shape
    if not all(dim.is_fixed for dim in shape):
        return None
    lanes = prod(tensor.dtype.lanes) if isinstance(tensor.dtype, fm.VectorType) else 1
    return prod(dim.fixed_value for dim in shape) * lanes


def candidate_tile(candidate):
    return int(candidate.parameters["vector_schedule"]["physical"]["elements_per_program"])


class LocalExtentElementwiseProvider(ElementwiseCandidateProvider):
    def __init__(self, max_elements):
        if type(max_elements) is not int or max_elements not in (512, 1024, 2048):
            raise ValueError("Reviewed tile caps are 512, 1024 or 2048 scalar elements")
        self.max_elements = max_elements

    def propose(self, node, context):
        proposal = super().propose(node, context)
        if proposal is None:
            return None
        capacity = scalar_capacity(node.type)
        if capacity is None:
            return proposal
        desired = min(self.max_elements, 1 << (max(1, capacity) - 1).bit_length())
        original = next(c for c in proposal.candidates if c.id == proposal.default_candidate)
        eligible = tuple(c for c in proposal.candidates if candidate_tile(original) <= candidate_tile(c) <= desired)
        if not eligible:
            return proposal
        selected = max(eligible, key=candidate_tile)
        return replace(proposal, default_candidate=selected.id)
