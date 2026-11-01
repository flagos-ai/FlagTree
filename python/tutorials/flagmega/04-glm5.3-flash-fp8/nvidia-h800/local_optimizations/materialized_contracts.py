"""Expose owned state and keep complete reduction rounding in its producer."""

from triton.flagmega.passes.functions.persistent_state_abi import specialize_persistent_state_abi
from triton.flagmega.passes.functions.reduction_rounding import fuse_materialized_reduction_rounding


def prepare(module):
    # This workload owns its runtime state ABI. The generic compiler keeps
    # public interfaces unchanged unless their owner requests specialization.
    states = module.metadata.get("composite_abi", {}).get("tensor_states", ())
    pairs = tuple(pair for state in states
                  for pair in zip(state["input"], state["output"], strict=True))
    if pairs:
        module = specialize_persistent_state_abi(module, pairs)
    return fuse_materialized_reduction_rounding(module)
