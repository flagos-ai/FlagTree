# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Testing hook matching nncase ForceBoxing."""

from triton.flagmega.ir.ops.core import attribute_parameter, input_parameter, op_definition
from triton.flagmega.ir.ops.distributed.boxing import Boxing
from triton.flagmega.ir.type_pattern import is_ir_type


@op_definition(
    "distributed.force_boxing",
    namespace="distributed",
    functional_name="force_boxing",
    display_name="Distributed.ForceBoxing",
)
class ForceBoxing(Boxing):
    value = input_parameter(is_ir_type())
    new_type = attribute_parameter(positional=True)


__all__ = ["ForceBoxing"]
