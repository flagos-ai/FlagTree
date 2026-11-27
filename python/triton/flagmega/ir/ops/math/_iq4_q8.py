# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""r003-v2 semantic contracts. No native encoder or device helper lives here."""
from math import prod
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.model import DType, TupleType, DistributedType, tensor_type
from triton.flagmega.ir.types import VectorType
from triton.flagmega.ir.distributed_inference import tensor_of, all_broadcast, broadcast_ir_type
from triton.flagmega.ir.ops.core import OpDefinition, attribute_parameter

SCHEMA = 'flagmega.iq4-q8-graph/r003-v2'
PLAN = 'eaa88d099e7d4c9cb33b24793abbaa6e1b42159900c032c40b958ffd1d8bebc4'
NUM = '2d0483edc3787d3dab325f3665e205bec952216a146937152300d3fc4a26ccb1'
INTERFACE = 'K4L32-Q8_1-DP4A/r003-v2'
ABI = 'Q8_1-bytes36-le/v1'
OPS = {'math.q8_1_prepare': 'q8_1_prepare', 'math.iq4_xs_q8_1_matmul': 'iq4_q8_projection',
       'nn.iq4_xs_q8_1_glu': 'iq4_q8_glu', 'nn.iq4_xs_q8_1_interleaved_projection': 'iq4_q8_interleaved'}

class Q8Contract(OpDefinition):
    graph_schema = attribute_parameter(default=SCHEMA)
    plan_hash = attribute_parameter(default=PLAN)
    numerical_profile = attribute_parameter(default='H003-N-v2')
    numerical_profile_hash = attribute_parameter(default=NUM)
    native_interface = attribute_parameter(default=INTERFACE)
    q8_abi = attribute_parameter(default=ABI)
    logical_k = attribute_parameter()
    padded_k = attribute_parameter()
    group_size = attribute_parameter(default=32)
    group_count = attribute_parameter()
    record_bytes = attribute_parameter(default=36)
    source_dtype = attribute_parameter()
    source_version_key = attribute_parameter()
    const_evaluable = False

    @classmethod
    def normalize_attrs(cls, attributes):
        a = super().normalize_attrs(attributes)
        frozen = dict(graph_schema=SCHEMA, plan_hash=PLAN, numerical_profile='H003-N-v2',
                      numerical_profile_hash=NUM, native_interface=INTERFACE, q8_abi=ABI, group_size=32, record_bytes=36)
        if any(a[k] != v for k,v in frozen.items()):
            raise IRSchemaError('Q8 semantic identity does not match r003-v2/H003-N-v2.')
        for k in ('logical_k','padded_k','group_count'):
            if type(a[k]) is not int or a[k] <= 0:
                raise IRSchemaError(f'Q8 {k} must be a positive integer.')
        if a['padded_k'] != 32*((a['logical_k']+31)//32) or a['group_count']*32 != a['padded_k']:
            raise IRSchemaError('Q8 K/Kp/G disagree.')
        if a['source_dtype'] not in ('bfloat16','float32') or not isinstance(a['source_version_key'],str) or not a['source_version_key']:
            raise IRSchemaError('Q8 needs explicit ready dtype and source version.')
        return a

class Q8Consumer(Q8Contract):
    weight_encoding = attribute_parameter(default='IQ4_XS')
    weight_block_n = attribute_parameter()
    weight_block_k = attribute_parameter()
    output_data_type = attribute_parameter()
    output_lanes = attribute_parameter(default=())
    logical_n = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes):
        a = super().normalize_attrs(attributes)
        if a['weight_encoding'] != 'IQ4_XS' or a['logical_k'] not in (5120,6144,17408) or a['padded_k'] != a['logical_k']:
            raise IRSchemaError('Q8 consumer requires frozen IQ4_XS full K.')
        for k in ('logical_n','weight_block_n','weight_block_k'):
            if type(a[k]) is not int or a[k] <= 0: raise IRSchemaError(f'Invalid Q8 {k}.')
        if a['output_data_type'] not in ('bfloat16','float32'): raise IRSchemaError('Invalid Q8 result dtype.')
        lanes = tuple(a['output_lanes'])
        if cls.op_name != 'nn.iq4_xs_q8_1_interleaved_projection':
            if any(type(l) is not int or l <= 0 for l in lanes) or a['logical_n'] % prod(lanes):
                raise IRSchemaError('Invalid Q8 output lanes.')
        else: lanes = tuple(tuple(l) for l in lanes)
        a['output_lanes'] = lanes
        return a

    @classmethod
    def infer_type(cls, inputs, a):
        kinds = tuple(n.type for n in inputs)
        q8 = tensor_of(kinds[0])
        # Only the leading token axis may exceed one; weights, scales and the
        # per-row Q8 record stay frozen at the single-row contract.
        rows = q8.shape[0] if q8.rank == 3 else 1
        if q8 != tensor_type(DType.UINT8,(rows,a['group_count'],36)):
            raise IRSchemaError('Q8 requires UINT8 scalar [T,G,36], no VectorType.')
        n,k = a['logical_n'],a['logical_k']
        for w,s in zip(inputs[1::2],inputs[2::2],strict=True):
            if tensor_of(w.type) != tensor_type(DType.UINT8,(n,136*(k//256))):
                raise IRSchemaError('Q8 IQ4 weight byte extent disagrees with explicit N/K.')
            if tensor_of(s.type) != tensor_type(DType.FLOAT32,((n+a['weight_block_n']-1)//a['weight_block_n'],(k+a['weight_block_k']-1)//a['weight_block_k'])):
                raise IRSchemaError('Q8 placeholder scale shape/type disagrees with logical N/K.')
        dtype = DType(a['output_data_type']); lanes=a['output_lanes']
        if cls.op_name == 'nn.iq4_xs_q8_1_interleaved_projection':
            if (a['feature_shape'],a['field_count'],lanes,n,dtype) != ((24,256),2,((8,),(8,)),12288,DType.BFLOAT16):
                raise IRSchemaError('Q8 interleaved needs two original BF16<8> whole-head fields.')
            output = TupleType(tuple(tensor_type(VectorType(dtype,(8,)),(rows,24,32)) for _ in range(2)))
        else:
            if cls.op_name == 'nn.iq4_xs_q8_1_glu' and dtype != DType.BFLOAT16:
                raise IRSchemaError('Q8 GLU output must retain BF16.')
            output=tensor_type(VectorType(dtype,lanes) if lanes else dtype,(rows,n//prod(lanes)))
        if not any(isinstance(t,DistributedType) for t in kinds): return output
        if not all(isinstance(t,DistributedType) and all_broadcast(t) for t in kinds) or len({t.placement for t in kinds}) != 1:
            raise IRSchemaError('r003 single-chip Q8 requires complete broadcast operands; split K/half-head rejected.')
        return broadcast_ir_type(output,kinds[0].placement)

    @classmethod
    def evaluate(cls,node,arguments,context):
        from triton.flagmega.evaluator.iq4_q8 import evaluate_consumer
        return evaluate_consumer(node,arguments,context)

def bind_q8_attributes(cls):
    """Give each concrete OpDefinition its own schema descriptors."""
    from copy import copy
    from triton.flagmega.ir.ops.core import ParameterInfo
    for base in reversed(cls.__mro__[1:]):
        for name,value in base.__dict__.items():
            if isinstance(value,ParameterInfo) and name not in cls.__dict__:
                parameter=copy(value);parameter.owner=None
                setattr(cls,name,parameter)
    return cls
