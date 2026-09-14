"""Measured H800/batch-one schedule; not a global SM90 default."""

from dataclasses import replace

from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.targets.nvidia import sm90_triton_implementation_model


def create_target(*, glu_reduction_group=32):
    if glu_reduction_group not in {32, 64, 128}:
        raise ValueError("The reviewed GLU reduction experiments use 32/64/128 elements")
    model = sm90_triton_implementation_model()

    def configure(implementation):
        parameters = dict(implementation.parameters)
        if implementation.contract.get("epilogue") == "residual_norm_stats":
            parameters["reduction_unroll"] = 8
        if implementation.family == "gather_reduce_norm_apply":
            parameters["reduction_width"] = 128
        if implementation.family == "dense_matmul_glu" and "reduction_group" in parameters:
            parameters["reduction_group"] = glu_reduction_group
        if implementation.family == "elementwise" and "elements_per_program" in parameters:
            # Scalar IR describes element semantics, not one CUDA lane. The
            # explicit casts in this profile cover whole hidden/head vectors.
            parameters["elements_per_program"] = 256
        return replace(implementation, parameters=parameters)

    return NvidiaSm90Target(triton_implementation_model=replace(
        model, implementations=tuple(configure(value) for value in model.implementations)))
