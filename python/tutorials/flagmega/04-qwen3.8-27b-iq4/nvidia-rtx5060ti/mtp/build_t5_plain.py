"""Build a multi-token verify artifact on the accepted r031 path only.

T=2 builds select R050 projections and the R049 GLU; T=3..5 preserve their existing GLU path.
T=1 retains the accepted q8_r003_v2 selection and launch recipe. The persistent
and pipeline renderers derive all scheduling constants from the artifact ABI.

Usage: build_t5_plain.py OUT_DIR MIN MAX SEGMENT
"""
import json
import sys
import time
import traceback
from pathlib import Path

from triton.flagmega import Compiler, CompileOptions
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.importer import GGUFCheckpoint, checkpoint_from_path
from triton.flagmega.importer.qwen3_5_dense import Qwen35DenseImporter
from triton.flagmega.ir import emit_module
from triton.flagmega.rules.ntt.r004_native_model import prepare_native_model
from triton.flagmega.rules.ntt.r031_model_policy import prepare_model_column_distribution

HERE = Path(__file__).resolve().parent
TUTORIAL = HERE.parents[5]
if str(TUTORIAL) not in sys.path:
    sys.path.insert(1, str(TUTORIAL))
import agent_optimizations.iq4_streaming.graph as G

GGUF = '/home/sunnycase/.cache/flagtree-gguf-validation/qwen3.8-27b-IQ4_XS-pure.gguf'
RDATA_CACHE = TUTORIAL / '.local/cache/weights'

OUT = Path(sys.argv[1])
R048_BUILD = OUT.name.startswith('build-r048') and len(sys.argv) == 5
MIN, MAX, SEGMENT = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
OUT.mkdir(parents=True, exist_ok=True)
status = {'schema': 'flagmega.multitoken-artifact/v1', 'checkpoint': GGUF,
          'tokens_min': MIN, 'tokens_max': MAX, 'segment': SEGMENT,
          'q8_path': ('r046_glu_staging_v1' if len(sys.argv) == 9 else 'r047_glu_occupancy_v1' if len(sys.argv) == 10 and sys.argv[5] == 'cp_async' else 'r038_glu_global_prefetch_v1' if len(sys.argv) == 6 and sys.argv[5] == 'shell' else 'r048_glu_larger_tiles_v1' if len(sys.argv) == 10 and sys.argv[5] != 'consumer' else 'r048_glu_larger_tiles_v1' if R048_BUILD else 'r050_projection_consumer_prefetch_v1+r049_glu_consumer_prefetch_v1') if MAX == 2 else 'r038_glu_global_prefetch_v1' if MAX > 1 else 'accepted-q8_r003_v2',
          'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}


def save():
    (OUT / 'build-status.json').write_text(json.dumps(status, indent=2, default=str))


save()
try:
    ckpt = GGUFCheckpoint(GGUF)
    imported = Qwen35DenseImporter(ckpt, num_tokens=(MIN, MAX)).import_module()
    status['imported'] = {'hash': imported.semantic_hash, 'stage': imported.stage}
    save()

    started = time.time()
    compiler = Compiler(CompileOptions(target='nvidia-sm120', codegen_backend='triton',
                                       shape_bucket_segment=SEGMENT,
                                       work_dir=OUT / 'compiler-boundary'))
    compiler.target = G.configure_q8_target(compiler.target)
    m = compiler.compile(imported, stop_after='remove-unused-functions').module
    status['boundary'] = {'hash': m.semantic_hash, 'tokens_per_call': m.metadata.get('tokens_per_call'),
                          'seconds': round(time.time() - started, 2)}
    save()

    m = G.prepare_q8_distribution(m, G.Q8Config())
    m = prepare_model_column_distribution(m)
    m = prepare_native_model(m)
    status['configured'] = {'hash': m.semantic_hash, 'tokens_per_call': m.metadata.get('tokens_per_call')}
    save()

    started = time.time()
    c = Compiler(CompileOptions(target='nvidia-sm120', codegen_backend='triton',
                                work_dir=OUT / 'compiler-full'))
    if m.metadata['tokens_per_call'] > 1:
        if m.metadata['tokens_per_call'] == 2:
            if len(sys.argv) == 6 and sys.argv[5] == 'shell':
                from persistent_glu import configure, persistent_view
            elif R048_BUILD:
                from larger_tiles_glu import configure, persistent_view
            elif len(sys.argv) == 5 or (len(sys.argv) == 10 and sys.argv[5] == 'consumer'):
                from consumer_prefetch_projections import configure, persistent_view
            elif len(sys.argv) == 9:
                from staging_glu import configure, persistent_view
            elif len(sys.argv) == 10 and sys.argv[5] == 'cp_async':
                from occupancy_glu import configure, persistent_view
            else:
                from larger_tiles_glu import configure, persistent_view
        else:
            from persistent_glu import configure, persistent_view
        staging_config = (sys.argv[5], int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])) if len(sys.argv) == 9 else (sys.argv[5], *(int(v) for v in sys.argv[6:10])) if len(sys.argv) == 10 and sys.argv[5] == 'cp_async' else tuple(int(v) for v in sys.argv[6:10]) if len(sys.argv) == 10 and sys.argv[5] == 'consumer' else tuple(int(v) for v in sys.argv[5:10]) if len(sys.argv) == 10 else None
        c.target = configure(c.target, m.metadata['tokens_per_call'], staging_config) if m.metadata['tokens_per_call'] == 2 and not (len(sys.argv) == 6 and sys.argv[5] == 'shell') else configure(c.target, m.metadata['tokens_per_call'])
        m = persistent_view(m)
    else:
        c.target = G.configure_q8_target(c.target)
    compiled = c.compile(m).module
    status['compiled'] = {'hash': compiled.semantic_hash, 'stage': compiled.stage,
                          'nodes': len(compiled.nodes), 'seconds': round(time.time() - started, 2)}
    emit_module(compiled, OUT / 'compiled.py')
    save()

    started = time.time()
    artifact = write_artifact(compiled, OUT / 'artifact', target='nvidia-sm120',
                              checkpoint=checkpoint_from_path(GGUF), emit_executable=True,
                              rdata_cache_dir=RDATA_CACHE)
    if OUT.name.startswith('build-r054') and MAX == 2:
        from r054_package import compile_candidate, package
        candidate_codegen = compile_candidate(c, m, OUT)
        package(artifact, candidate_codegen)
    manifest = json.loads((Path(artifact) / 'artifact.json').read_text())
    binding = dict(manifest['metadata']['shape_bucket_binding']['bindings'])
    assert binding == {'tokens': compiled.metadata['tokens_per_call']}
    status['artifact'] = {'path': str(artifact), 'semantic_hash': manifest['semantic_hash'],
                          'stage': manifest['stage'], 'tokens_per_call': manifest['metadata']['tokens_per_call'],
                          'seconds': round(time.time() - started, 2)}
    status['status'] = 'PASS'
    status['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    save()
    print(json.dumps(status['artifact'], indent=2), flush=True)
except BaseException:
    status['status'] = 'FAILED'
    status['traceback'] = traceback.format_exc()
    status['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    save()
    print(status['traceback'], flush=True)
    raise
