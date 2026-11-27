# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
from triton.flagmega.ir.memory_effect import MemoryEffect
from triton.flagmega.ir.ops.core import input_parameter, attribute_parameter, op_definition
from triton.flagmega.ir.ops.math._iq4_q8 import Q8Consumer, bind_q8_attributes
from triton.flagmega.ir.type_pattern import is_tensor, has_rank
@op_definition('nn.iq4_xs_q8_1_glu', namespace='nn', functional_name='iq4_xs_q8_1_glu', display_name='NN.IQ4XSQ8_1Glu')
@bind_q8_attributes
class IQ4XSQ8_1Glu(Q8Consumer):
    activation=attribute_parameter(default="silu")
    gate_weight_encoding=attribute_parameter(default="IQ4_XS")
    up_weight_encoding=attribute_parameter(default="IQ4_XS")
    @classmethod
    def normalize_attrs(cls,attributes):
        a=super().normalize_attrs(attributes)
        if a["activation"]!="silu" or a["gate_weight_encoding"]!="IQ4_XS" or a["up_weight_encoding"]!="IQ4_XS":
            from triton.flagmega.errors import IRSchemaError
            raise IRSchemaError("Q8 GLU requires SiLU and original IQ4 gate/up order.")
        return a
    q8=input_parameter(is_tensor() & has_rank(3),memory_effect=MemoryEffect.CHIP_READ)
    gate_weight=input_parameter(is_tensor() & has_rank(2),memory_effect=MemoryEffect.CHIP_READ)
    gate_weight_scale=input_parameter(is_tensor() & has_rank(2),memory_effect=MemoryEffect.NONE)
    up_weight=input_parameter(is_tensor() & has_rank(2),memory_effect=MemoryEffect.CHIP_READ)
    up_weight_scale=input_parameter(is_tensor() & has_rank(2),memory_effect=MemoryEffect.NONE)
    result_memory_effects=(MemoryEffect.CHIP_WRITE,)
