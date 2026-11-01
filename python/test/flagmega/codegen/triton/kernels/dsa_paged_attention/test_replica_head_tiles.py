# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""DSA head tasks preserve online MMA arithmetic and chip visibility."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import PagedAttentionStateConfig, create_paged_attention_state
from triton.flagmega.options import CompileOptions
from triton.flagmega.runtime import load
from triton.flagmega.targets import NvidiaSm90Target

from python.test.flagmega.codegen.triton.kernels.dense_matmul.test_replica_owner_groups import fixture


@pytest.fixture(scope='module')
def optimization():
    root = (Path(__file__).resolve().parents[6]
            / 'tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800/local_optimizations')
    name = '_flagmega_dsa_head_tests'
    package = ModuleType(name)
    package.__path__ = [str(root)]
    sys.modules[name] = package
    previous = TritonTemplateRegistry.render
    try:
        spec = importlib.util.spec_from_file_location(name + '.dsa_work', root / 'dsa_work.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        TritonTemplateRegistry.render = previous
    try:
        yield module
    finally:
        for key in tuple(sys.modules):
            if key == name or key.startswith(name + '.'):
                sys.modules.pop(key)


@pytest.mark.parametrize('cause', ['query', 'state', 'partial', 'input_partial', 'exclusive', 'private'])
def test_head_tasks_reject_unproved_sharing(optimization, cause):
    call, context = fixture(owner_axes=())
    abi = call['outputs'][0]['buffers'][0]['abi']
    if cause in ('query', 'state'):
        call['query' if cause == 'query' else 'slot_mapping'] = 'workspace + shard_coord1 * 32'
    elif cause == 'private':
        abi['memory_sharing_scope'] = 'block'
    elif cause == 'input_partial':
        call['inputs'][0]['buffers'][0]['abi']['distributed_backing_type'] = {
            'partial': {'axes': [1]}, 'axis_policies': ()}
    else:
        abi['distributed_type'][cause] = {'axes': [1]}
    assert optimization.head_group(call, context) is None


def _graph(config, heads, mesh, traversal):
    class Graph(fm.Module):
        def forward(self):
            query = self.input('query', fm.tensor_type(fm.vector_type('bfloat16', (8,)),
                (1, heads, config.head_dim // 8)), id='query')
            topk = self.input('topk', fm.tensor_type('int32', (1, 2051)), id='topk')
            state = self.input('state', config.ref_type, id='state')
            layer = fm.F.builtin.scalar_const(fm.tensor_type('int32', ()), 0)
            result = fm.F.nn.dsa_paged_attention(query, topk, state, layer, scale=1.,
                layout=('seq', 'head', 'dim'), hidden_size=heads * config.head_dim,
                probability_rounding='online_bf16', index_traversal=traversal)
            sentinel = self.input('sentinel', fm.tensor_type('float32', (128,)), id='sentinel')
            split = fm.DistributedType(sentinel.type, (fm.SBP.split_contiguous((0, 1), 1),), mesh)
            local = fm.F.distributed.force_boxing(sentinel, split)
            copied = fm.F.distributed.force_boxing(fm.F.math.add(local, local), sentinel.type)
            self.function('main', (query, topk, state, sentinel), (result, copied))
    return Graph(dialect='distributed', stage='frozen_constants', entry='main',
                 metadata={'auto_distribution': {'placement': mesh.to_data()}}).build()


@pytest.mark.parametrize('heads,dim,warps,traversal', [
    (64,512,8,'compact'), (17,48,8,'compact'), (64,32,16,'compact'), (17,48,8,'slots')])
def test_head_tasks_keep_mma_order_and_complete_outputs(tmp_path, monkeypatch, optimization, heads, dim, warps, traversal):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9,0):
        pytest.skip('requires SM90')
    mesh = fm.Placement((8,16), 'yx', 'bb')
    config = PagedAttentionStateConfig(1,1,dim,num_blocks=16)
    compiler = Compiler(CompileOptions(placement=mesh))
    base = NvidiaSm90Target().triton_implementation_model
    catalog = replace(base, implementations=tuple(
        replace(i, parameters={**i.parameters,'decode_schedule':'bf16_mma','compute_num_warps':warps})
        if i.id=='tir.dsa_paged_attention.decode' else i for i in base.implementations))
    compiler.target = NvidiaSm90Target(triton_implementation_model=catalog)
    module = compiler.compile(_graph(config,heads,mesh,traversal)).module
    baseline = write_artifact(module,tmp_path/'baseline',target='nvidia-sm90',emit_executable=True)
    with monkeypatch.context() as patch:
        patch.setattr(TritonTemplateRegistry,'render',optimization.override)
        candidate = write_artifact(module,tmp_path/'candidate',target='nvidia-sm90',emit_executable=True)
    runtimes = [load(p,device='cuda:0') for p in (baseline,candidate)]
    state = create_paged_attention_state(config,device='cuda')
    query = torch.zeros((1,heads,dim//8,8),dtype=torch.bfloat16,device='cuda')
    topk = torch.full((1,2051),-1,dtype=torch.int32,device='cuda')
    sentinel = torch.arange(128,dtype=torch.float32,device='cuda')
    size = query.numel()
    storage = [[torch.full((size+32,),123.,dtype=torch.bfloat16,device='cuda'),
                torch.full((160,),123.,device='cuda')] for _ in runtimes]
    outputs = [(a[:size].reshape(query.shape),b[:128]) for a,b in storage]
    arguments = []
    for runtime,out in zip(runtimes,outputs,strict=True):
        buffers = {}
        for value,ids in runtime.buffer_plan.function_map[module.entry].parameters:
            if isinstance(module.node_map[value].type,fm.RefType):
                buffers.update(zip(ids,(state.kv_caches,state.query_start_loc,state.seq_lens,
                                        state.slot_mapping,state.block_table),strict=True))
            else:
                buffers[ids[0]]={'query':query,'topk':topk,'sentinel':sentinel}[module.node_map[value].attrs['name']]
        for (_,ids),actual in zip(runtime.buffer_plan.function_map[module.entry].outputs,out,strict=True):
            buffers[ids[0]]=actual
        arguments.append(tuple(buffers[s['buffer']] for s in runtime.external_arguments))

    def run(expected=None):
        for runtime,values in zip(runtimes,arguments,strict=True):
            runtime.run_into(*values)
            assert runtime.resource_report['grid']==[128,1,1]
            assert runtime.resource_report['spill_bytes']==0
        torch.cuda.synchronize()
        torch.testing.assert_close(outputs[1][0],outputs[0][0],rtol=0,atol=0)
        for out,protected in zip(outputs,storage,strict=True):
            if expected is not None:
                torch.testing.assert_close(out[0],torch.full_like(out[0],expected),rtol=0,atol=0)
            torch.testing.assert_close(out[1],sentinel*2,rtol=0,atol=0)
            assert torch.all(protected[0][size:]==123.) and torch.all(protected[1][128:]==123.)

    try:
        for runtime,values in zip(runtimes,arguments,strict=True):
            runtime.prepare(*values)
        state.seq_lens.fill_(3);state.slot_mapping.fill_(2)
        for pos,value in enumerate((1,2,4)):
            state.kv_caches[0,0,1,pos,0].fill_(value)
        topk[0,[0,64,2048,2049]]=torch.tensor([2,99,0,2],dtype=torch.int32,device='cuda')
        run(3.)  # Duplicates remain, future token 99 is filtered: (4+1+4)/3.
        generator=torch.Generator(device='cuda').manual_seed(501)
        query.copy_(torch.randn(query.shape,device='cuda',generator=generator)/8)
        state.kv_caches[0,0,0,:3].copy_(torch.randn(state.kv_caches[0,0,0,:3].shape,device='cuda',generator=generator)/8)
        run()
        query.zero_();state.kv_caches.zero_();topk.fill_(-1)
        query.flatten(2)[:,:,0]=4096;query.flatten(2)[:,:,8]=1;query.flatten(2)[:,:,16]=-4096
        key=state.kv_caches[0,0,0,0,0].flatten();key[0]=4096;key[8]=1;key[16]=4096
        state.kv_caches[0,0,1,0,0].fill_(1)
        topk[0,[0,2048]]=torch.tensor([0,1],dtype=torch.int32,device='cuda')
        run(.5)  # First K16 rounds 2**24+1 before the next K16 cancels it.
        topk.fill_(-1)
        run(0.)
    finally:
        for runtime in runtimes:
            runtime.close()
