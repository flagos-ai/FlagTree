"""Exact complete DSA score regression, using the actual rendered helpers.

On an idle H800 in the FlagTree environment:
  python repro_dsa_pool_score_batch.py --score-ctas 96
  python repro_dsa_pool_score_batch.py --score-ctas 64
Only synthetic BF16 state/query/weights and FP32 APE are used.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile

from local_optimizations.joint_dsa_absorbed_pool_scores import helpers
from local_optimizations.native_dsa_pool_scores import KEY
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry


def kernel_source(score_ctas):
    assert score_ctas in (64, 96)
    registry = TritonTemplateRegistry()
    raw = registry.environment.loader.get_source(registry.environment, KEY)[0]
    source = helpers(raw)
    original_start = source.index('@triton.jit(noinline=True)\ndef {{ call.symbol }}__pool_scores(')
    original_end = source.index('@triton.jit(noinline=True)\ndef {{ call.symbol }}__pool_scores__long96(', original_start)
    batch_start = source.index('@triton.jit(noinline=True)\ndef {{ call.symbol }}__pool_scores__batched(')
    batch_end = source.index('@triton.jit(noinline=True)\ndef {{ call.symbol }}__pool_scores__batched_long96(', batch_start)
    macro = raw[:raw.index('{% for call in render_calls %}')]
    call = dict(symbol='repro', signature='Result, State, Q, Weights, Ape, Meta',
                layer_id='0', state_layer_ids=(0,), state='State',
                state_layer_stride=1052672, state_token_stride=257, state_value_stride=1,
                state_capacity=4096, state_tile=4096, sequence_lengths='Meta',
                index_dim=128, index_heads=32, q_width=4096, q='Q', q_stride=1,
                weights='Weights', weights_stride=1, compress_ape='Ape',
                ape_row_stride=128, ape_dim_stride=1, pool=4, pool_tile=1024,
                result='Result', index_topk=2048, visibility_publish=True,
                value_type='tl.bfloat16', scale=0.08838834764831845,
                joint_dsa_pool_scores=dict(index='tl.program_id(0)',count=score_ctas))
    text = registry.environment.from_string(
        '{% import "entrypoints/_common.py.jinja" as entry with context %}\n'+macro
        +source[original_start:original_end]+source[batch_start:batch_end]).render(
            call=call,distributed_entry=False,grid_mesh=False,mesh_rank=3,
            mesh_hierarchy=(1,1,1),mesh_levels='cbb')
    prefix = 'import triton\nimport triton.language as tl\nimport triton.experimental.tle.language as tle\n'
    wrapper = '\n@triton.jit\ndef baseline(Result, State, Q, Weights, Ape, Meta):\n'
    wrapper += f'    if tl.program_id(0) < {score_ctas}:\n        repro__pool_scores(Result, State, Q, Weights, Ape, Meta)\n'
    wrapper += '\n@triton.jit\ndef candidate(Result, State, Q, Weights, Ape, Meta, B:tl.constexpr):\n'
    wrapper += f'    if tl.program_id(0) < {score_ctas}:\n        repro__pool_scores__batched(Result, State, Q, Weights, Ape, Meta)\n'
    return prefix+text+wrapper


def main():
    import torch
    from triton.flagmega.runtime.nvidia_launch import compilation_options
    from triton.flagmega.runtime.prepared import ResourceContract, prepare_jit_kernel
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--score-ctas',type=int,choices=(64,96),default=96)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='dsa-score-batch8-') as directory:
        path=Path(directory)/'kernels.py';path.write_text(kernel_source(args.score_ctas))
        spec=importlib.util.spec_from_file_location('batch8_score_repro',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        torch.set_num_threads(1);torch.cuda.set_device(0)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());torch.cuda.set_stream(stream)
        state=torch.empty((4096,257),dtype=torch.bfloat16,device='cuda')
        query=torch.empty(4096,dtype=torch.bfloat16,device='cuda')
        weights=torch.empty(32,dtype=torch.bfloat16,device='cuda')
        ape=torch.empty((4,128),dtype=torch.float32,device='cuda')
        meta=torch.zeros(16,dtype=torch.int32,device='cuda')
        contract=ResourceContract(compute_num_warps=8,resident_blocks_per_sm=1,requires_grid_wide_sync=False)
        options=compilation_options(contract);options.update(enable_fp_fusion=False,launch_cooperative_grid=False)
        buffers=[torch.full((2083,),-991271,dtype=torch.int32,device='cuda') for _ in range(2)]
        outputs=[b[16:2067] for b in buffers];programs=[];graphs=[]
        for output in outputs:output[2048:2050].zero_()
        for i,kernel in enumerate((module.baseline,module.candidate)):
            params=(outputs[i],state,query,weights,ape,meta)+( (8,) if i else () )
            program=prepare_jit_kernel(kernel,params,(),grid=(128,1,1),contract=contract,allocation_device='cuda:0',**options)
            assert program.resource_report['spill_bytes']==0
            with torch.cuda.graph(graph:=torch.cuda.CUDAGraph(),stream=stream):program.launch()
            programs.append(program);graphs.append(graph)
        generator=torch.Generator(device='cuda');cases=0
        counts=(0,1,2,8,16,63,64,65,95,96,97,127,128,129,255,256,257,511,512,513,527,528,529,767,768,769,1023,1024)
        for count in counts:
            for condition in range(4):
                for missing in range(1 if count==0 else 4):
                    seed=113731+count*17+condition*113+missing
                    generator.manual_seed(seed)
                    first=min((0,7,31,127)[condition],4096-count*4)
                    visible=max(count*4-missing,0);base=first+visible-1
                    state.normal_(generator=generator);state[:,:128].mul_((.01,1,32,100)[condition]);state[:,128:256].mul_((.1,1,100,1000)[condition]);state[:,256].fill_(1)
                    if condition==3:state[::19,256].zero_()
                    query.normal_(generator=generator);weights.normal_(generator=generator);ape.normal_(generator=generator)
                    meta[:3].copy_(torch.tensor([base,first,visible],dtype=torch.int32,device='cuda'))
                    immutable=[v.clone() for v in (state,query,weights,ape,meta)]
                    for replay in range(3):
                        for buffer,output in zip(buffers,outputs,strict=True):
                            buffer.fill_(-991271);output[2048:2050].view(torch.int32).copy_(meta[1:3])
                        for program,graph in zip(programs,graphs,strict=True):
                            if replay<2:graph.replay()
                            else:program.launch()
                        torch.cuda.synchronize()
                        assert torch.equal(outputs[0].view(torch.uint8),outputs[1].view(torch.uint8)),(count,condition,missing,replay,seed)
                        assert all(bool(b[:16].eq(-991271).all()) and bool(b[2067:].eq(-991271).all()) for b in buffers)
                        assert all(torch.equal(o[1024:2048],torch.full_like(o[1024:2048],-991271)) and torch.equal(o[2048:2050].view(torch.int32),meta[1:3]) for o in outputs)
                        assert all(torch.equal(a.view(torch.uint8),b.view(torch.uint8)) for a,b in zip((state,query,weights,ape,meta),immutable,strict=True))
                    cases+=1
        print(json.dumps(dict(passed=True,cases=cases,score_ctas=args.score_ctas,all1024_FP32_bytes_equal=True,two_graph_and_eager=True,inputs_and_canaries_unchanged=True,full_model_acceptance=False)))


if __name__=='__main__':main()
