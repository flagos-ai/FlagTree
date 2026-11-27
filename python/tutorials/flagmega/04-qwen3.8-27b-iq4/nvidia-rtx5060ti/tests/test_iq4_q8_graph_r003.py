"""CPU graph contract/admission regressions; no device calls."""
from dataclasses import replace
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from triton.flagmega import ir as fm
from triton.flagmega import Compiler,CompileOptions
from triton.flagmega.errors import CodegenError,IRSchemaError
from triton.flagmega.ir.ops.math._iq4_q8 import SCHEMA,PLAN,NUM,INTERFACE,ABI
from agent_optimizations.iq4_streaming.graph import prepare_q8_distribution,configure_target,verify_q8_nodes,Q8Config

def semantic_graph():
    class Graph(fm.Module):
        def forward(self):
            x=self.input('x',fm.tensor_type('bfloat16',(1,5120)))
            w=self.input('w',fm.tensor_type('uint8',(6144,2720)))
            s=self.input('s',fm.tensor_type('float32',(48,40)))
            q=fm.F.math.q8_1_prepare(x,logical_k=5120,padded_k=5120,group_count=160,source_dtype='bfloat16',source_version_key='x:step',
                metadata={'iq4_q8_origin_id':'projection','iq4_q8_ready_publication':'x'})
            y=fm.F.math.iq4_xs_q8_1_matmul(q,w,s,logical_k=5120,padded_k=5120,group_count=160,source_dtype='bfloat16',source_version_key='x:step',
                weight_block_n=128,weight_block_k=128,logical_n=6144,output_data_type='bfloat16',
                metadata={'iq4_q8_origin_id':'projection','iq4_q8_ready_publication':'x'})
            self.function('main',(x,w,s),(y,))
    return Graph(dialect='high_level',stage='unused_functions_removed',entry='main').build()

def test_semantics_serialization_effects(tmp_path):
    m=semantic_graph();fm.verify_module(m)
    q=next(n for n in m.nodes if n.op=='math.q8_1_prepare')
    assert q.type==fm.tensor_type('uint8',(1,160,36)) and not q.effect.is_pure
    from triton.flagmega.ir.ops.core import get_definition
    assert not get_definition(q.op).const_evaluable
    fm.emit_module(m,tmp_path/'module.py');loaded=fm.load_module(tmp_path/'module.py')
    assert loaded.semantic_hash==m.semantic_hash
    y=m.node_map[m.functions[0].outputs[0]]
    assert y.type==fm.tensor_type('bfloat16',(1,6144))
    assert get_definition(y.op).weight_scale.memory_effect.mode==0

def test_same_config_rechecks_every_node():
    m=prepare_q8_distribution(semantic_graph())
    assert prepare_q8_distribution(m) is m
    for index,n in enumerate(m.nodes):
        if n.op not in ('math.q8_1_prepare','math.iq4_xs_q8_1_matmul'):continue
        for key in ('kernel_maxnreg','kernel_enable_static_shared','kernel_num_warps','iq4_q8_plan_hash'):
            bad=replace(m,nodes=tuple(replace(v,metadata={k:x for k,x in v.metadata.items() if k!=key}) if i==index else v for i,v in enumerate(m.nodes)))
            with pytest.raises(CodegenError):prepare_q8_distribution(bad)

def test_target_installed_catalog_rechecked():
    c=Compiler(CompileOptions(target='nvidia-sm120',codegen_backend='triton'))
    target=configure_target(c.target,Q8Config())
    assert configure_target(target) is target
    from triton.flagmega.codegen.triton.implementation import TritonImplementationModel
    old=target.triton_implementation_model
    target.triton_implementation_model=TritonImplementationModel(tuple(i for i in old.implementations if i.family!='q8_1_prepare'),{k:v for k,v in old.preferences.items() if k!="q8_1_prepare"},name=old.name)
    with pytest.raises(CodegenError):configure_target(target)
