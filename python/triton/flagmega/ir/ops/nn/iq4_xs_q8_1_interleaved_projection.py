# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
from triton.flagmega.ir.memory_effect import MemoryEffect
from triton.flagmega.ir.ops.core import input_parameter, attribute_parameter, op_definition
from triton.flagmega.ir.ops.math._iq4_q8 import Q8Consumer, bind_q8_attributes
from triton.flagmega.ir.type_pattern import is_tensor, has_rank
@op_definition('nn.iq4_xs_q8_1_interleaved_projection', namespace='nn', functional_name='iq4_xs_q8_1_interleaved_projection', display_name='NN.IQ4XSQ8_1InterleavedProjection')
@bind_q8_attributes
class IQ4XSQ8_1InterleavedProjection(Q8Consumer):
    q8=input_parameter(is_tensor() & has_rank(3),memory_effect=MemoryEffect.CHIP_READ)
    weight=input_parameter(is_tensor() & has_rank(2),memory_effect=MemoryEffect.CHIP_READ)
    weight_scale=input_parameter(is_tensor() & has_rank(2),memory_effect=MemoryEffect.NONE)
    feature_shape=attribute_parameter(default=(24,256))
    field_count=attribute_parameter(default=2)
    result_memory_effects=(MemoryEffect.CHIP_WRITE,MemoryEffect.CHIP_WRITE)
    @classmethod
    def normalize_attrs(cls,attributes):
        a=super().normalize_attrs(attributes); a['feature_shape']=tuple(a['feature_shape']); return a
