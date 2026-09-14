# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The exact type relation shared by shape/dtype-preserving pointwise ops.

Broadcasting remains explicit IR. This relation never supplies a reduction,
conversion, or implicit broadcast to make incompatible operands agree.
"""

from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import DistributedType
from triton.flagmega.ir.ops.core import OpDefinition


class SameTypePointwiseOp(OpDefinition):
    @classmethod
    def distributed_input_type_tuples(cls, choices, attrs):
        if not choices:
            return
        common = set(choices[0]).intersection(*(set(values) for values in choices[1:]))
        for value in choices[0]:
            if value in common and getattr(value, "partial", None) is None:
                yield (value,) * len(choices)

    @classmethod
    def distributed_output_type_candidates(cls, choices, output_type, attrs):
        return tuple(dict.fromkeys(value for values in choices for value in values
                                   if isinstance(value, DistributedType) and value.partial is None
                                   and value.tensor == tensor_of(output_type)))

    @classmethod
    def infer_distributed_input_types(cls, output_type, logical_input_types, attrs):
        if (not isinstance(output_type, DistributedType) or output_type.partial is not None
                or not logical_input_types
                or any(tensor_of(value) != output_type.tensor for value in logical_input_types)):
            return ()
        return ((output_type,) * len(logical_input_types),)
