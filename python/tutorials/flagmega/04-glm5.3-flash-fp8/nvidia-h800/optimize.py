"""Rebuild the full measured model from original IR and publish a runnable artifact."""
import argparse,ast,datetime,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent

SOURCE_DIRECTORIES = ('local_optimizations', 'patches')
SOURCE_FILES = ('optimize.py', 'accuracy.py', 'benchmark_gpu.py',
                'benchmark_vllm.py', 'reproduce.py', 'render_results.py',
                'generated_kernels.py')

def source_inputs(here):
    paths = [here/name for name in SOURCE_FILES]
    for name in SOURCE_DIRECTORIES:
        paths.extend(p for p in (here/name).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts)
    return {str(p.relative_to(here)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}

def compiler_inputs():
    from triton import flagmega
    root = Path(flagmega.__file__).resolve().parent
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.jinja'}
            and '__pycache__' not in p.parts}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--readonly-cache',type=Path,required=True)
    parser.add_argument('--checkpoint-digests',type=Path)
    args=parser.parse_args()
    out=args.output.resolve()
    if out.exists():raise ValueError('Use a fresh build output directory')
    inputs=source_inputs(HERE)
    compiler_sources=compiler_inputs()
    begin=time.monotonic()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    code=subprocess.call([sys.executable,str(HERE/'local_optimizations/compile_model.py'),'--output',str(out)],env=env)
    if code:raise SystemExit(code)
    for directory in SOURCE_DIRECTORIES:
        shutil.copytree(HERE/directory,out/directory,
                        ignore=shutil.ignore_patterns('__pycache__'))
    for name in SOURCE_FILES:
        shutil.copy2(HERE/name,out/name)
    sys.path.insert(0,str(out))
    from local_optimizations.kernel_recipe import activate,fm
    activate()
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.importer import DirectoryCheckpoint
    module=fm.load_module(out/'dag-final.py')
    contract=json.loads((HERE/'local_optimizations/delivery-contract.json').read_text())
    if module.semantic_hash!=contract['expected_semantic_hash']:
        raise ValueError('Rebuilt typed module differs from the measured contract')
    counts={op:sum(n.op.split('.')[-1].lower()==op for n in module.nodes)
            for op in ('pack','slice','cast','concat','unpack')}
    if any(counts.values()):raise ValueError(('Compute-body representation gate',counts))
    artifact=write_artifact(module,out/'artifact',target='nvidia-sm90',emit_executable=True,
        checkpoint=DirectoryCheckpoint(args.checkpoint,integrity_report=args.checkpoint_digests),
        rdata_cache_dir=args.readonly_cache)
    source=artifact/'generated_kernels.py'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if digest!=contract['expected_source_sha256'] or source.read_bytes()!=(HERE/'generated_kernels.py').read_bytes():
        raise ValueError('Normal code generation differs from the measured executable source')
    manifest=json.loads((artifact/'artifact.json').read_text())
    if manifest['codegen']['source_sha256']!=digest:raise ValueError('Manifest source identity mismatch')
    (out/'dag-audit.json').write_text(json.dumps(dict(semantic_hash=module.semantic_hash,
        compute_body_five_op_counts=counts,functions=len(module.execution_functions),
        typed_module_verified=True,numerical_acceptance=False,performance_acceptance=False),indent=2)+'\n')
    receipt=dict(original_preallocation_rebuilt=True,all_normal_planners_enabled=True,
        source_inputs_unchanged=source_inputs(HERE)==inputs,
        compiler_sources_unchanged=compiler_inputs()==compiler_sources,
        compiler_source_files=compiler_sources,
        source_sha256=digest,semantic_hash=module.semantic_hash,artifact=str(artifact),
        readonly_cache=str(args.readonly_cache.resolve()),checkpoint_digest_report_used=args.checkpoint_digests is not None,
        seconds=time.monotonic()-begin,timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        gpu_acceptance_pending=True,performance_acceptance=False,files=inputs)
    if not receipt['source_inputs_unchanged'] or not receipt['compiler_sources_unchanged']:
        raise ValueError('Reproduction or compiler input changed while building')
    (out/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in {'files','compiler_source_files'}}),flush=True)

if __name__=='__main__':main()

