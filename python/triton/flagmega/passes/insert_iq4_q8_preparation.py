# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Insert one dynamic preparation per original projection (337, no sharing credit)."""
from dataclasses import replace
from triton.flagmega.ir import VectorType, TupleType
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.rules.neutral._utility import make_node
from triton.flagmega.ir.ops.math._iq4_q8 import SCHEMA, PLAN, NUM, INTERFACE, ABI

class InsertIQ4Q8Preparation:
    def __call__(self,module):
        if any(n.op == 'math.q8_1_prepare' for n in module.nodes):
            raise IRVerificationError('Q8 insertion requires fresh pre-insertion IR.')
        original=module.node_map; result=[]; origins=[]
        for node in module.nodes:
            glu=node.op=='nn.matmul_glu' and node.attrs.get('gate_weight_encoding')=='IQ4_XS'
            linear=node.op in ('math.block_scaled_matmul','nn.interleaved_projection') and node.attrs.get('weight_encoding')=='IQ4_XS'
            if not (glu or linear): result.append(node); continue
            if node.attrs.get('native_compute_profile','f32')!='f32' or node.attrs.get('activation_quantization','none')!='none':
                raise IRVerificationError('Q8 cannot replace a different activation/numerical boundary.',node_id=node.id)
            if glu and (node.attrs.get('up_weight_encoding')!='IQ4_XS' or node.attrs.get('gate_clamp_max') is not None or node.attrs.get('up_clamp') is not None):
                raise IRVerificationError('Q8 GLU requires frozen IQ4 gate/up order and no clamp.',node_id=node.id)
            source=original[node.inputs[0]]; t=tensor_of(source.type)
            if t.rank!=2 or t.shape[0].fixed_value!=1 or isinstance(t.dtype,VectorType):
                raise IRVerificationError('Pre-vectorization Q8 source must be original ready scalar.',node_id=node.id)
            k=t.shape[1].fixed_value; weight=original[node.inputs[1]]; n=tensor_of(weight.type).shape[0].fixed_value
            version=f'{source.id}:current-call-step/rewrite-every-replay:{node.id}'
            common=dict(graph_schema=SCHEMA,plan_hash=PLAN,numerical_profile='H003-N-v2',numerical_profile_hash=NUM,
                        native_interface=INTERFACE,q8_abi=ABI,logical_k=k,padded_k=32*((k+31)//32),group_count=(k+31)//32,
                        group_size=32,record_bytes=36,source_dtype=t.dtype.value,source_version_key=version)
            md={**node.metadata,'iq4_q8_origin_id':node.id,'iq4_q8_source_ssa':source.id,
                'iq4_q8_original_result_type':str(node.type),'iq4_q8_sharing':'unshared-337',
                'iq4_q8_original_source_type':str(source.type),'iq4_q8_ready_publication':source.id}
            prep=make_node('math.q8_1_prepare',node.id+'.q8_prepare',(source,),common,md)
            result.append(prep)
            inter=node.op=='nn.interleaved_projection'
            if inter: dtype='bfloat16'; lanes=((8,),(8,))
            else:
                out=tensor_of(node.type); dtype=out.dtype.elem_type.value if isinstance(out.dtype,VectorType) else out.dtype.value
                lanes=out.dtype.lanes if isinstance(out.dtype,VectorType) else ()
            attrs={**common,'weight_encoding':'IQ4_XS','weight_block_n':node.attrs['weight_block_n'],
                   'weight_block_k':node.attrs['weight_block_k'],'output_data_type':dtype,'output_lanes':lanes,'logical_n':n}
            inputs=(prep,*(original[i] for i in (node.inputs[1],node.inputs[3],node.inputs[2],node.inputs[4]))) if glu else (prep,*(original[i] for i in node.inputs[1:]))
            op='nn.iq4_xs_q8_1_glu' if glu else 'nn.iq4_xs_q8_1_interleaved_projection' if inter else 'math.iq4_xs_q8_1_matmul'
            new=make_node(op,node.id+'.q8' if inter else node.id,inputs,attrs,md)
            if inter:
                # Preserve the original public result at insertion; full packing
                # cancels these storage views against original field lane demands.
                result.append(new); fields=[]
                for index in range(2):
                    field=make_node('builtin.get_item',new.id+f'.field{index}',(new,),{'index':index},md); result.append(field)
                    old=tensor_of(node.type.fields[index])
                    if not isinstance(old.dtype,VectorType):
                        field=make_node('tensors.unpack',field.id+'.scalar',(field,),{'axes':(2,)},md);result.append(field)
                    elif old != tensor_of(new.type.fields[index]):
                        raise IRVerificationError('Original Q field has unsupported lane contract.',node_id=node.id)
                    fields.append(field)
                new=make_node('builtin.tuple',node.id,tuple(fields),{},md)
            if new.type!=node.type: raise IRVerificationError('Q8 rewrite changed original result type.',node_id=node.id)
            result.append(new); origins.append(node.id)
        return replace(module,nodes=tuple(result),metadata={**module.metadata,'iq4_q8_insertion':
            {'schema':SCHEMA,'plan_hash':PLAN,'numerical_profile_hash':NUM,'origins':origins,'sharing':'unshared','launch_credit':0}})

def insert_iq4_q8_preparation(module): return InsertIQ4Q8Preparation()(module)
