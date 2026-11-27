# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Small native byte fixtures exercise numerical owners, not runtime dequantization."""
import numpy as np
import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.evaluator import TorchEvaluator, DictWeightResolver
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.nn.embedding import embedding
from python.test.flagmega.codegen.cpp.test_native_quantized_kernels import packed_rows


def projection_graph(encoding, *, k=256, lanes=(), output_dtype=None, glu=False):
    blockbytes = 136 if encoding == 'IQ4_XS' else 144
    class Graph(fm.Module):
        def forward(self):
            x = self.input('x', fm.tensor_type('bfloat16', (1,k)), id='x')
            w = self.input('w', fm.tensor_type('uint8', (4,k//256*blockbytes)), id='w')
            s = self.input('s', fm.tensor_type('float32', (1,k//128)), id='s')
            if glu:
                u = self.input('u', w.type, id='u')
                y = fm.F.nn.matmul_glu(x,w,u,s,s,activation='silu',weight_block_n=128,weight_block_k=128,
                    gate_weight_encoding=encoding,up_weight_encoding=encoding,name='y')
                self.function('main',(x,w,u,s),(y,))
            else:
                y = fm.F.math.block_scaled_matmul(x,w,s,weight_block_n=128,weight_block_k=128,
                    weight_encoding=encoding,output_lanes=lanes,output_data_type=output_dtype,name='y')
                self.function('main',(x,w,s),(y,))
    return Graph(dialect='high_level',stage='imported',entry='main').build()


@pytest.mark.parametrize('encoding',['IQ4_XS','Q4_K'])
@pytest.mark.parametrize('k',[256,512])
@pytest.mark.parametrize('output_dtype',[None,'float32'])
@pytest.mark.parametrize('lanes',[(),(2,2)])
def test_native_linear_logical_k_decoding_final_dtype_and_lanes(encoding,k,output_dtype,lanes):
    raw, expected_weight = packed_rows(encoding,blocks=k//256)
    x = (torch.randn((1,k),generator=torch.Generator().manual_seed(717))*0.03).bfloat16()
    module = projection_graph(encoding,k=k,lanes=lanes,output_dtype=output_dtype)
    actual, = TorchEvaluator(DictWeightResolver({})).run(module,{'x':x,'w':torch.from_numpy(raw),
        's':torch.full((1,k//128),37.)})
    expected = (x.float()@torch.from_numpy(expected_weight).T).to(torch.float32 if output_dtype else torch.bfloat16)
    assert torch.equal(actual.reshape(1,4),expected)


@pytest.mark.parametrize('encoding',['IQ4_XS','Q4_K'])
@pytest.mark.parametrize('lanes',[(),(8,),(2,4)])
def test_native_embedding_decodes_requested_nonidentity_rows_and_padding(encoding,lanes):
    raw,decoded = packed_rows(encoding,rows=5)
    ids=torch.tensor([4,0,2],dtype=torch.int32)
    actual=embedding(ids,torch.from_numpy(raw),padding_idx=-5,weight_encoding=encoding,output_lanes=lanes)
    expected=torch.from_numpy(decoded[[4,0,2]]).bfloat16();expected[1]=0
    assert torch.equal(actual.reshape(3,512),expected)


@pytest.mark.parametrize('encoding',['IQ4_XS','Q4_K'])
def test_native_glu_keeps_fp32_projections_until_final_store(encoding):
    raw,decoded=packed_rows(encoding)
    up=np.roll(raw,1,axis=0).copy();up_decoded=np.roll(decoded,1,axis=0).copy()
    x=(torch.randn((1,512),generator=torch.Generator().manual_seed(392))*0.1).bfloat16()
    actual,=TorchEvaluator(DictWeightResolver({})).run(projection_graph(encoding,k=512,glu=True),
        {'x':x,'w':torch.from_numpy(raw),'u':torch.from_numpy(up),'s':torch.ones(1,4)})
    gate=x.float()@torch.from_numpy(decoded).T;up=x.float()@torch.from_numpy(up_decoded).T
    expected=(torch.nn.functional.silu(gate)*up).bfloat16()
    early=torch.nn.functional.silu(gate.bfloat16())*up.bfloat16()
    assert torch.equal(actual,expected)
    assert not torch.equal(expected,early)


def test_native_glu_rounding_witness():
    # Eight constant IQ4 groups: code8=1, signed scale1, d=2^-12.
    raw=np.zeros((4,136),np.uint8);raw[:,:2]=np.frombuffer(np.float16(2**-12).tobytes(),np.uint8)
    raw[:,2:4]=np.frombuffer(np.uint16(0xaaaa).tobytes(),np.uint8);raw[:,4:8]=0x11;raw[:,8:]=0x88
    up=raw.copy();up[:,:2]=np.frombuffer(np.float16(2**-10).tobytes(),np.uint8)
    x=torch.full((1,256),1.5,dtype=torch.bfloat16);x[0,0]=-7.5
    result,=TorchEvaluator(DictWeightResolver({})).run(projection_graph('IQ4_XS',glu=True),
        {'x':x,'w':torch.from_numpy(raw),'u':torch.from_numpy(up),'s':torch.ones(1,2)})
    gate=torch.tensor(0.091552734375);up=torch.tensor(0.3662109375)
    expected=(torch.nn.functional.silu(gate)*up).bfloat16()
    assert torch.equal(result,expected.expand_as(result))
    assert expected != torch.nn.functional.silu(gate.bfloat16())*up.bfloat16()


@pytest.mark.parametrize('encoding',['IQ4_XS','Q4_K'])
def test_native_recurrent_projections_round_before_gates_with_nonzero_state(encoding):
    from triton.flagmega.ir.ops.nn._gdn_state import GatedDeltaNetStateConfig,create_gdn_state
    from triton.flagmega.ir.ops.nn.gdn_recurrent_core import gated_delta_net_recurrent_core
    config=GatedDeltaNetStateConfig(1,2,4,8,8,4,256)
    left,right=create_gdn_state(config),create_gdn_state(config)
    rng=torch.Generator().manual_seed(825)
    initial=torch.randn(left.recurrent.shape,generator=rng)*0.1
    left.recurrent.copy_(initial);right.recurrent.copy_(initial)
    raw,decoded=packed_rows(encoding,rows=4,blocks=1)
    up=np.roll(raw,1,axis=0).copy();ud=np.roll(decoded,1,axis=0).copy()
    source=(torch.randn((2,256),generator=rng)*0.03).bfloat16()
    b=(source.float()@torch.from_numpy(decoded).T).bfloat16()
    a=(source.float()@torch.from_numpy(ud).T).bfloat16()
    # Non-native reference gets an exact basis representation of the same
    # independently computed, already BF16-rounded projection sums.
    basis=torch.zeros_like(source);basis[0,0]=1;basis[1,1]=1
    bw=torch.zeros(4,256,dtype=torch.bfloat16);bw[:,:2]=b.T
    aw=torch.zeros_like(bw);aw[:,:2]=a.T
    qkv=torch.randn((2,64),generator=rng).bfloat16();z=torch.randn((2,32),generator=rng).bfloat16()
    attrs=dict(num_key_heads=2,num_value_heads=4,key_head_dim=8,value_head_dim=8,epsilon=1e-6,
               round_beta=True,value_head_order='tiled')
    common=dict(qkv=qkv,z=z,a_log=torch.zeros(4),dt_bias=torch.tensor([.1,-.2,.3,-.4]).bfloat16(),norm_weight=torch.ones(8),torch=torch)
    native,_=gated_delta_net_recurrent_core(state=left,projection_input=source,b_weight=torch.from_numpy(raw),
                    a_weight=torch.from_numpy(up),attrs={**attrs,'b_weight_encoding':encoding,'a_weight_encoding':encoding},**common)
    expected,_=gated_delta_net_recurrent_core(state=right,projection_input=basis,b_weight=bw,a_weight=aw,attrs=attrs,**common)
    assert torch.equal(native,expected)
    assert torch.equal(left.recurrent,right.recurrent)
    assert torch.equal(left.convolution,right.convolution)


def test_fp8_glu_keeps_dynamic_quantization_and_bf16_intermediate_rounding():
    from types import SimpleNamespace
    from triton.flagmega.ir.ops.nn.matmul_glu import MatMulGlu
    from triton.flagmega.ir.ops.math._block_scaled import dynamic_block_quant_dequant
    rng=torch.Generator().manual_seed(513)
    x=(torch.randn((1,128),generator=rng)*0.2).bfloat16()
    w=torch.randn((4,128),generator=rng).to(torch.float8_e4m3fn)
    u=torch.randn((4,128),generator=rng).to(torch.float8_e4m3fn)
    scales=torch.tensor([[.03]])
    attrs=MatMulGlu.normalize_attrs(dict(activation='silu',weight_block_n=128,weight_block_k=128,
                                        gate_clamp_max=.25,up_clamp=.5))
    node=SimpleNamespace(type=fm.tensor_type('bfloat16',(1,4)),attrs=attrs)
    ctx=SimpleNamespace(torch=torch,torch_dtype=lambda d:getattr(torch,d.value))
    actual=MatMulGlu.evaluate(node,(x,w,u,scales,scales),ctx)
    activation=dynamic_block_quant_dequant(x,block_k=128)
    gate=(activation@(w.float()*.03).T).bfloat16().clamp(max=.25)
    up=(activation@(u.float()*.03).T).bfloat16().clamp(min=-.5,max=.5)
    assert torch.equal(actual,torch.nn.functional.silu(gate)*up)


@pytest.mark.parametrize('attrs',[{'gate_weight_encoding':'IQ4_XS','up_weight_encoding':'Q4_K'},
                                 {'gate_weight_encoding':'IQ4_XS','up_weight_encoding':'IQ4_XS','gate_clamp_max':1.}])
def test_native_glu_unsupported_mixed_storage_or_clamp_rejected(attrs):
    from triton.flagmega.ir.ops.nn.matmul_glu import MatMulGlu
    with pytest.raises(IRSchemaError):
        MatMulGlu.normalize_attrs(dict(activation='silu',weight_block_n=128,weight_block_k=128,**attrs))


def test_packed_fp8_storage_cannot_claim_native_byte_encoding():
    from triton.flagmega.ir.ops.math.packed_block_scaled_matmul import PackedBlockScaledMatMul
    with pytest.raises(IRSchemaError,match='vector-FP8.*native GGUF'):
        PackedBlockScaledMatMul.normalize_attrs(dict(weight_block_n=128,weight_block_k=128,k_pack=2,k_vector=16,
                                                    weight_encoding='IQ4_XS'))


@pytest.mark.parametrize('encoding',['IQ4_XS','Q4_K'])
def test_coarse_native_gdn_projects_qkv_z_and_output_from_native_bytes(encoding):
    from triton.flagmega.ir.ops.nn._gdn_state import GatedDeltaNetStateConfig,create_gdn_state
    from triton.flagmega.ir.ops.nn.gated_delta_net import gated_delta_net
    from triton.flagmega.ir.ops.nn.gdn_convolution import gated_delta_net_convolution
    from triton.flagmega.ir.ops.nn.gdn_recurrent_core import gated_delta_net_recurrent_core
    config=GatedDeltaNetStateConfig(1,2,4,8,64,4,256)
    left,right=create_gdn_state(config),create_gdn_state(config)
    rng=torch.Generator().manual_seed(418)
    initial=torch.randn(left.recurrent.shape,generator=rng)*0.01
    left.recurrent.copy_(initial);right.recurrent.copy_(initial)
    fixture={name:packed_rows(encoding,rows=n,blocks=1) for name,n in [('qkv',288),('z',256),('b',4),('a',4),('out',256)]}
    raw={name:torch.from_numpy(parts[0]) for name,parts in fixture.items()}
    decoded={name:torch.from_numpy(parts[1]) for name,parts in fixture.items()}
    x=(torch.randn((2,256),generator=rng)*0.01).bfloat16()
    attrs=dict(num_key_heads=2,num_value_heads=4,key_head_dim=8,value_head_dim=64,conv_kernel_size=4,
               epsilon=1e-6,weight_block_n=128,weight_block_k=128,
               **{f'{name}_weight_encoding':encoding for name in ['qkv','z','b','a','output']})
    conv=(torch.randn((288,4),generator=rng)*0.01).bfloat16()
    common=dict(a_log=torch.zeros(4),dt_bias=torch.zeros(4,dtype=torch.bfloat16),norm_weight=torch.ones(64),torch=torch)
    native,_=gated_delta_net(hidden=x,state=left,qkv_weight=raw['qkv'],qkv_scale=torch.full((3,2),37.),
        z_weight=raw['z'],z_scale=torch.full((2,2),37.),b_weight=raw['b'],a_weight=raw['a'],conv_weight=conv,
        output_weight=raw['out'],output_scale=torch.full((2,2),37.),attrs=attrs,**common)
    qkv=(x.float()@decoded['qkv'].T).bfloat16();z=(x.float()@decoded['z'].T).bfloat16()
    convolved,_=gated_delta_net_convolution(qkv=qkv,state=right,conv_weight=conv,conv_kernel_size=4,torch=torch)
    bp=(x.float()@decoded['b'].T).bfloat16();ap=(x.float()@decoded['a'].T).bfloat16()
    basis=torch.zeros_like(x);basis[0,0]=1;basis[1,1]=1
    bw=torch.zeros(4,256,dtype=torch.bfloat16);bw[:,:2]=bp.T
    aw=torch.zeros_like(bw);aw[:,:2]=ap.T
    plain_attrs={k:v for k,v in attrs.items() if not k.endswith('_encoding')}
    gated,_=gated_delta_net_recurrent_core(state=right,qkv=convolved,z=z,projection_input=basis,b_weight=bw,a_weight=aw,attrs=plain_attrs,**common)
    expected=(gated.float()@decoded['out'].T).bfloat16()
    assert torch.equal(native,expected)
    assert torch.equal(left.recurrent,right.recurrent)
    assert torch.equal(left.convolution,right.convolution)
