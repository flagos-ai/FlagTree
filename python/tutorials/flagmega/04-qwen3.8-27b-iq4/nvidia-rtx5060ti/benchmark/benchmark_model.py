"""Frozen full64 Triton host decode timing; ordinary cohort only."""
import argparse, json, os, statistics, time, traceback
from pathlib import Path
from reference_identity import file_pin, save_json
from timing_isolation import timing_metadata, monotonic_ns, WINDOW_SCHEMA, validate_round_result, byte_pin
class NumericalError(RuntimeError): pass
class DecodeContext:
    """One runtime, its bound stream, and its external state storage."""

    def __init__(self, artifact, *, token_id=1, device='cuda:0'):
        import torch
        from triton.flagmega.ir import VectorType
        from triton.flagmega.runtime import load

        self.torch = torch
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        torch.cuda.set_stream(self.stream)
        self.runtime = load(artifact, device=device)
        self.values = {}
        for argument in self.runtime.external_arguments:
            buffer = self.runtime.buffer_plan.buffer_map[argument['buffer']]
            dtype = buffer.dtype
            lanes = dtype.lanes if isinstance(dtype, VectorType) else ()
            scalar = dtype.elem_type if lanes else dtype
            self.values[buffer.id] = torch.zeros((*buffer.shape, *lanes),
                                                 dtype=getattr(torch, scalar.value), device=device)
        self.values['input_ids'].fill_(token_id)
        self.values['paged_state.query_start_loc'].copy_(torch.tensor([0, 1], dtype=torch.int32, device=device))
        table = self.values['paged_state.block_table']
        table.copy_(torch.arange(table.numel(), dtype=table.dtype, device=device).reshape_as(table))
        self.arguments = tuple(self.values[a['buffer']] for a in self.runtime.external_arguments)
        outputs = self.runtime.buffer_plan.function_map[self.runtime.ir_module.entry].outputs
        self.logits, self.token = (self.values[outputs[i][1][0]] for i in (0, 1))
        if self.token.numel() != 1 or self.values['input_ids'].numel() != 1:
            raise ValueError('This contract is single-sequence decode-1')
        self.initial = {name: value.clone() for name, value in self.values.items()}
        self.runtime.prepare(*self.arguments)
        self.stream.synchronize()

    def reset(self):
        for name, value in self.values.items():
            value.copy_(self.initial[name])

    def launch(self):
        self.values['paged_state.slot_mapping'].copy_(self.values['paged_state.seq_lens'])
        self.runtime.run_into(*self.arguments)
        self.values['input_ids'].copy_(self.token.reshape_as(self.values['input_ids']))

    def check_state(self, steps):
        length = int(self.values['paged_state.seq_lens'].item())
        if length != steps:
            raise NumericalError(f'Sequence length is {length}, expected {steps}')
        if int(self.values['paged_state.slot_mapping'].item()) != steps - 1:
            raise NumericalError('Slot mapping did not follow sequence progression')
        for name, value in self.values.items():
            if value.is_floating_point() and not self.torch.isfinite(value).all().item():
                raise NumericalError(f'Non-finite external state/output {name}')

    def capture(self):
        self.reset()
        self.stream.synchronize()
        graph = self.torch.cuda.CUDAGraph()
        with self.torch.cuda.graph(graph, stream=self.stream):
            self.launch()
        self.graph = graph
        return graph

    def close(self):
        self.stream.synchronize()
        # Destroy graph references before their captured argument storage.
        self.graph = None
        self.runtime.close()



def measure(args):
    import torch
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    golden=json.loads((args.golden/'metadata.json').read_text())
    c=None
    report=dict(schema='flagmega.triton-decode-benchmark/v2',status='running',backend='triton',steps=args.steps,repeats=3,
                cohort='ordinary-host-token',measurement_id=args.output.parent.name,artifact=file_pin(args.artifact/'artifact.json'),
                harness=file_pin(__file__),golden=file_pin(args.golden/'metadata.json'),scenario=args.scenario,round_timing=timing_metadata(),
                boundary='CUDA graph replay with slot/token feedback + GPU greedy + one CPU token return; injection/reset/JIT/diagnostics excluded')
    sidecar=args.output.with_suffix('.rounds.jsonl')
    def save(): save_json(args.output,report)
    try:
        if args.native_module:
            import importlib
            importlib.import_module(args.native_module).register()
        if args.native_file:
            import importlib.util
            native_pin = file_pin(args.native_file)
            spec = importlib.util.spec_from_file_location('_measurement_native', args.native_file)
            native = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(native)
            native.register()
            report['native_encoder'] = native_pin
        c=DecodeContext(args.artifact,token_id=1)
        meta=c.runtime.ir_module.metadata
        assert c.runtime.manifest['codegen_backend']=='triton'
        assert meta['num_hidden_layers']==64 and tuple(meta['imported_layer_indices'])==tuple(range(64))
        recurrent=[v for k,v in c.values.items() if 'recurrent' in k]
        assert recurrent and all(v.dtype==torch.float32 for v in recurrent)
        report.update(layers=64,resource_report=c.runtime.resource_report,torch_version=torch.__version__,triton_gpu=torch.cuda.get_device_name())
        prompt_count=golden.get('prompt_token_count',0)
        injected=torch.tensor(golden['input_tokens'][:prompt_count],dtype=c.values['input_ids'].dtype,device='cuda')
        def inject(step):
            if step<prompt_count: c.values['input_ids'].copy_(injected[step].reshape_as(c.values['input_ids']))
        expected=[]; consumed=[]
        c.reset(); c.stream.synchronize()
        for step in range(args.steps):
            inject(step); consumed.append(int(c.values['input_ids'].item())); c.launch(); c.stream.synchronize()
            if not torch.isfinite(c.logits).all().item(): raise NumericalError(f'Nonfinite logits at {step}')
            expected.append(int(c.token.item()))
        c.check_state(args.steps)
        start=max(0,prompt_count-1)
        report.update(tokens=expected,input_tokens=consumed,prompt_token_count=prompt_count,finite=True,
                      prefix=dict(start_step=start,required=10,actual=expected[start:start+10],golden=golden['tokens'][start:start+10],
                                  passed=expected[start:start+10]==golden['tokens'][start:start+10],independent_generation=True))
        save()
        if not report['prefix']['passed']:
            report['numerical_status']='failed'
            raise NumericalError(f"First 10 independently generated tokens differ: {report['prefix']}")
        graph=c.capture()
        c.reset(); c.stream.synchronize(); graph_tokens=[]
        for step in range(args.steps):
            inject(step); graph.replay(); returned=int(c.token.item()); graph_tokens.append(returned)
            if not torch.isfinite(c.logits).all().item(): raise NumericalError(f'Nonfinite graph logits {step}')
        c.check_state(args.steps)
        assert graph_tokens==expected, 'Graph/eager trajectory mismatch'
        report.update(numerical_status='passed',graph_trajectory_equal=True)
        if args.numerical_only:
            c.reset();c.stream.synchronize();reset_tokens=[]
            for step in range(args.steps):
                inject(step);graph.replay();reset_tokens.append(int(c.token.item()))
                if not torch.isfinite(c.logits).all().item(): raise NumericalError(f'Nonfinite reset graph logits {step}')
            c.check_state(args.steps)
            assert reset_tokens==expected, 'Reset graph trajectory mismatch'
            report.update(status='numerical-only-passed',repeats_reset_state=True,reset_trajectory_equal=True,
                          performance_measured=False,resource_report=c.runtime.resource_report)
            save();print(json.dumps(report['prefix'],indent=2),flush=True);return
        batches=[]; report['rounds']=[]
        for repeat in range(4):
            c.reset(); c.stream.synchronize(); samples=[]; trajectory=[]
            window_start=monotonic_ns()
            for step in range(args.steps):
                inject(step); c.stream.synchronize()
                before=time.perf_counter(); graph.replay(); token=int(c.token.item()); after=time.perf_counter()
                samples.append((after-before)*1000); trajectory.append(token)
            window_end=monotonic_ns()
            window=dict(schema=WINDOW_SCHEMA,repeat=repeat,warmup=repeat==0,start_ns=str(window_start),end_ns=str(window_end),
                        sample_count=args.steps,sample_offset=None if repeat==0 else (repeat-1)*args.steps)
            report['round_timing']['rounds'].append(window)
            row=dict(repeat=repeat,warmup=repeat==0,sample_count=args.steps,latency_ms=samples,tokens=trajectory,trajectory_equal=trajectory==expected,
                     round_window=window,clock=report['round_timing']['clock'],policy=report['round_timing']['policy'],policy_limits=report['round_timing']['policy_limits'])
            with sidecar.open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
            if repeat: batches.append(samples);report['rounds'].append(row)
            report['latency_ms']=batches;save()
            c.check_state(args.steps)
            assert trajectory==expected, f'Trajectory changed round {repeat}'
        report.update(mean_ms=statistics.mean(x for row in batches for x in row),round_mean_ms=[statistics.mean(row) for row in batches],
                      warmup_rounds=1,repeats_reset_state=True,timed_trajectories_equal=True,graph_trajectory_equal=True,
                      numerical_status='passed' if report['prefix']['passed'] else 'failed',peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      rounds_sidecar=byte_pin(sidecar.read_bytes()),resource_report=c.runtime.resource_report,status='measured')
        assert report['artifact']==file_pin(args.artifact/'artifact.json')
        validate_round_result(report,rounds_raw=sidecar.read_bytes());save()
        print(json.dumps({k:report[k] for k in ('mean_ms','round_mean_ms','numerical_status','prefix')},indent=2),flush=True)
    except BaseException as e:
        report.update(status='failed',error=repr(e),traceback=traceback.format_exc());save();raise
    finally:
        if c is not None: c.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact',type=Path,required=True);p.add_argument('--golden',type=Path,required=True)
    p.add_argument('--scenario',choices=['synthetic32','chinese64'],required=True);p.add_argument('--steps',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--numerical-only',action='store_true',help='Validate eager/graph/reset trajectories without timing')
    native = p.add_mutually_exclusive_group()
    native.add_argument('--native-module',help='Register the artifact native family encoders before load')
    native.add_argument('--native-file',type=Path,help='Register frozen artifact-native encoders from a source snapshot')
    p.add_argument('--task',type=Path,default=Path(__file__).resolve().parents[1]/'.local/task.yaml')
    a=p.parse_args()
    if a.output.exists(): p.error('Use fresh output; failed evidence is immutable')
    task=json.loads(a.task.read_text())
    if a.steps != task['workload']['scenarios'][a.scenario]['steps']:p.error('steps must match frozen scenario')
    golden=json.loads((a.golden/'metadata.json').read_text())
    if golden['checkpoint_sha256']!=task['workload']['checkpoint_sha256']:p.error('golden checkpoint differs from task')
    a.output.parent.mkdir(parents=True,exist_ok=True);measure(a)
if __name__=='__main__': main()
