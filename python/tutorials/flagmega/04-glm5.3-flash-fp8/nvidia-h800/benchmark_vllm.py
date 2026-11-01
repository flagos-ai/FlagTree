"""Unmodified official vLLM engine, default optimization, no trace hooks."""
import argparse, dataclasses, json, os, time
from pathlib import Path
import torch, vllm
from vllm import LLM, SamplingParams
from local_optimizations.scenarios import (OFFICIAL_ONE_TOKEN, make_prompts,
                                         validate_lengths)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--prompt-lengths',type=int,nargs='+',default=[1])
    parser.add_argument('--validate-only',action='store_true')
    a=parser.parse_args()
    validate_lengths(a.prompt_lengths)
    prompts=make_prompts(a.checkpoint,a.prompt_lengths)
    report=dict(runtime='vllm',passed=False,versions=dict(vllm=vllm.__version__,torch=torch.__version__,cuda=torch.version.cuda),requests=[],warmup_requests=2,measured_requests=10,prompt=[11],output_tokens=8,enforce_eager=False,prefix_caching=False,checkpoint_revision='eb9eb208eb0d988989d07a6a12d0fdeb5f52574a',cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    assert vllm.__version__=='0.31.0'
    report.update(scenarios=[],validation_only=a.validate_only,prefill_mode='native_batched',measured_requests=0 if a.validate_only else 10)
    if a.prompt_lengths!=[1]:report.pop('prompt')
    engine=None;start=time.perf_counter()
    try:
        # The full weights occupy ~75 GiB/card. Bound prefill activation memory
        # with the native scheduler's chunks while retaining the full context.
        engine=LLM(model=str(a.checkpoint),tensor_parallel_size=4,dtype='bfloat16',max_model_len=4096,max_num_seqs=1,max_num_batched_tokens=1 if a.prompt_lengths==[1] else 512,enable_prefix_caching=False,enforce_eager=False,skip_tokenizer_init=True,language_model_only=True,gpu_memory_utilization=.97,seed=0,disable_log_stats=False)
        config=engine.llm_engine.vllm_config;text=config.model_config.hf_text_config
        assert config.parallel_config.tensor_parallel_size==4 and config.model_config.max_model_len==4096
        assert text.num_hidden_layers==45 and text.hidden_size==4096 and text.n_routed_experts==288 and text.num_experts_per_tok==8
        comp=config.compilation_config
        report['configuration']=dict(compilation_mode=str(comp.mode),cudagraph_mode=str(comp.cudagraph_mode),cudagraph_capture_sizes=comp.cudagraph_capture_sizes,max_num_seqs=config.scheduler_config.max_num_seqs,max_num_batched_tokens=config.scheduler_config.max_num_batched_tokens,attention_backend=str(config.attention_config.backend))
        report['engine_initialization_seconds']=time.perf_counter()-start
        print(json.dumps(dict(event='engine_ready',**report['configuration'],seconds=report['engine_initialization_seconds'])),flush=True)
        params=SamplingParams(temperature=0,max_tokens=8,min_tokens=8,ignore_eos=True,detokenize=False,seed=0)
        for length in a.prompt_lengths:
            expected=OFFICIAL_ONE_TOKEN if length==1 else None
            scenario=dict(prompt_length=length,prompt=prompts[length],output_tokens=8,requests=[])
            report['scenarios'].append(scenario)
            for request in range(2 if a.validate_only else 12):
                wall_start_utc=time.time()
                begin=time.perf_counter()
                result=engine.generate([{'prompt_token_ids':prompts[length]}],params,use_tqdm=False)
                elapsed=(time.perf_counter()-begin)*1000
                assert len(result)==1 and len(result[0].outputs)==1
                output=result[0];tokens=list(output.outputs[0].token_ids)
                if expected is None:expected=tokens
                assert len(tokens)==8 and tokens==expected,(length,request,tokens,expected)
                assert output.num_cached_tokens in (None,0),output.num_cached_tokens
                m=output.metrics
                assert m is not None and m.first_token_latency>0 and m.last_token_ts>m.first_token_ts,m
                row=dict(prompt_length=length,wall_start_utc=wall_start_utc,wall_end_utc=time.time(),request=request,warmup=request<2,tokens=tokens,wall_request_ms=elapsed,ttft_ms=m.first_token_latency*1000,decode_tpot_ms=(m.last_token_ts-m.first_token_ts)*1000/7,engine_inference_ms=(m.last_token_ts-m.scheduled_ts)*1000,request_metrics=dataclasses.asdict(m),num_cached_tokens=output.num_cached_tokens)
                scenario['requests'].append(row)
                if a.prompt_lengths==[1]:report['requests'].append(row)
                print(json.dumps(row),flush=True)
                (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        report['passed']=True
    finally:
        report['total_seconds']=time.perf_counter()-start
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if engine is not None:engine.llm_engine.engine_core.shutdown()

if __name__=='__main__':main()
