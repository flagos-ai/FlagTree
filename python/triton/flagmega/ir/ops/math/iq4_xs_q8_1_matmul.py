# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
from triton.flagmega.ir.memory_effect import MemoryEffect
from triton.flagmega.ir.ops.core import input_parameter, op_definition
from triton.flagmega.ir.ops.math._iq4_q8 import Q8Consumer, bind_q8_attributes
from triton.flagmega.ir.type_pattern import is_tensor, has_rank
@op_definition('math.iq4_xs_q8_1_matmul', namespace='math', functional_name='iq4_xs_q8_1_matmul', display_name='Math.IQ4XSQ8_1MatMul')
@bind_q8_attributes
class IQ4XSQ8_1MatMul(Q8Consumer):
    q8=input_parameter(is_tensor() & has_rank(3),memory_effect=MemoryEffect.CHIP_READ)
    weight=input_parameter(is_tensor() & has_rank(2),memory_effect=MemoryEffect.CHIP_READ)
    weight_scale=input_parameter(is_tensor() & has_rank(2),memory_effect=MemoryEffect.NONE)
    result_memory_effects=(MemoryEffect.CHIP_WRITE,)
