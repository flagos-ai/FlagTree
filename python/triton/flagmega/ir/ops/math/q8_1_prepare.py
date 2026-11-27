# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
from triton.flagmega.ir.model import DType, Effect, EffectKind, DistributedType, tensor_type
from triton.flagmega.ir.distributed_inference import tensor_of, all_broadcast, broadcast_ir_type
from triton.flagmega.ir.memory_effect import MemoryEffect
from triton.flagmega.ir.ops.core import input_parameter, op_definition
from triton.flagmega.ir.ops.math._iq4_q8 import Q8Contract, bind_q8_attributes
from triton.flagmega.ir.type_pattern import is_tensor, has_rank
from triton.flagmega.errors import IRSchemaError

@op_definition('math.q8_1_prepare', namespace='math', functional_name='q8_1_prepare', display_name='Math.Q8_1Prepare')
@bind_q8_attributes
class Q8_1Prepare(Q8Contract):
    value = input_parameter(is_tensor() & has_rank(2), memory_effect=MemoryEffect.CHIP_READ)
    result_memory_effects = (MemoryEffect.CHIP_WRITE,)
    @classmethod
    def infer_type(cls,inputs,a):
        t=inputs[0].type
        # A static token bucket may carry T activation rows; K, the source dtype
        # and the record geometry stay exactly as the single-row contract.
        rows=tensor_of(t).shape[0]
        if tensor_of(t) != tensor_type(DType(a['source_dtype']),(rows,a['logical_k'])):
            raise IRSchemaError('Q8 preparation requires original ready scalar [T,K] dtype.')
        output=tensor_type(DType.UINT8,(rows,a['group_count'],36))
        if isinstance(t,DistributedType):
            if not all_broadcast(t): raise IRSchemaError('Q8 source must have complete K.')
            return broadcast_ir_type(output,t.placement)
        return output
    @classmethod
    def infer_effect(cls,inputs,a):
        return Effect(EffectKind.READ_WRITE,'q8-activation:'+a['source_version_key'])
    @classmethod
    def evaluate(cls,node,arguments,context):
        from triton.flagmega.evaluator.iq4_q8 import evaluate_prepare
        return evaluate_prepare(node,arguments[0],context)
