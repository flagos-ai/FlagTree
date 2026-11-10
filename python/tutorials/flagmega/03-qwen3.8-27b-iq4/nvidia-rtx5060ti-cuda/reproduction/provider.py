"""Bind the original ordinary method to a freshly reviewed public artifact.

All inputs are explicit. Independent native review and actual normal prepare
remain separate gates; this module does not issue an acceptance receipt.
"""
from pathlib import Path
import hashlib
import json
import math
import os

HERE = Path(__file__).resolve().parent
DELIVERY = HERE.parent
REPO = DELIVERY.parents[4]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pin(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return dict(path=str(path), nbytes=path.stat().st_size, sha256=digest.hexdigest())


def checked(row):
    require(pin(row['path']) == row, 'Changed bound input: ' + row['path'])
    return Path(row['path']).read_bytes()


def verify_binding(binding, scenario):
    from .native_condition import verify_current
    require(binding.get('schema') == 'flagmega.public-t27-execution-binding/v2', 'Current public build binding required')
    side = binding['side']
    require(json.loads(checked(binding['side_input'])) == side, 'Changed independent side input')
    artifact = side['artifact']
    for key in ('manifest', 'source', 'library'):
        require(pin(artifact[key]['path']) == artifact[key], 'Changed current artifact: ' + key)
    manifest = json.loads(checked(artifact['manifest']))
    expected = json.loads(checked(side['expected_normal_descriptor']))
    programs = expected['cpp_programs']
    cg = manifest['codegen']
    require(manifest['codegen_backend'] == 'cpp' and cg.get('operator_profile') is None,
            'Ordinary CPP artifact required')
    require(manifest['semantic_hash'] == side['module_semantic_hash'] == expected['module_semantic_hash'],
            'Current typed module differs')
    require(cg['source_sha256'] == artifact['source']['sha256'] == expected['source_sha256']
            and cg['library_sha256'] == artifact['library']['sha256'], 'Current artifact/source/library differs')
    require(cg['cpp_programs'] == programs and cg['cpp_counter_epoch_plan'] == expected['cpp_counter_epoch_plan'],
            'Current complete ordinary/epoch descriptor differs')
    require(side['plan_sha256'] == programs['plan_sha256'] and
            side['counter_epoch_plan_sha256'] == cg['cpp_counter_epoch_plan']['plan_sha256'],
            'Current plan identity differs')
    require((len(programs['entries']), len(programs['invocations']), side['reset_count']) ==
            (33, 869, {'baseline275':275, 'candidate49':49}[side['schedule']]), 'Wrong complete schedule')
    verify_current(json.loads(checked(side['native_review'])), side, programs, checked, require)
    require(Path(binding['public_repo']).resolve() == REPO, 'Wrong public import root')
    from .build_source import verify_build_source
    verify_build_source(binding, side)
    from normal_compile import RECIPE
    require(Path(side['recipe']['path']).resolve() == RECIPE and pin(RECIPE) == side['recipe'],
            'Wrong public recipe identity')
    require(manifest['rdata'] == side['readonly']['descriptor'], 'Wrong readonly descriptor')
    readonly = Path(side['readonly']['path'])
    require(readonly.resolve() == (Path(artifact['root']) / manifest['rdata']['image']).resolve(),
            'Wrong current readonly image path')
    stat = readonly.stat()
    identity = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_nlink, stat.st_mode]
    require(identity == side['readonly']['identity'], 'Readonly identity changed; current admission required')
    golden = binding['goldens'][scenario]
    for name in ('metadata.json', 'logits.f32'):
        row = golden['pins'][name]
        require(Path(row['path']).resolve() == (Path(golden['root']) / name).resolve(), 'Wrong golden file binding')
        require(pin(row['path']) == row, 'Changed original reference golden')
    for name in ('monitor', 'helper'):
        require(pin(binding['monitor_expected'][name]['path']) == binding['monitor_expected'][name],
                'Changed original monitor/helper')
    require(Path(binding['monitor_expected']['monitor']['path']).resolve() == HERE / 'monitor.py'
            and Path(binding['monitor_expected']['helper']['path']).resolve() == DELIVERY / 'timing_isolation.py',
            'Non-public monitor/helper import')
    context = Path(side['resource_context']) / 'O-native-api-candidate.json'
    require(json.loads(context.read_text()) == {'static_resources': side['static_resources']},
            'Original resource context must bind this actual static resource report')
    return side, manifest, programs


def validate_resources(result, side, programs):
    # Same namespace meanings as V76 qualify.py. The two correspondence rows
    # are inherited from the original GLU 10->10 and 26->25 contract.
    mapping = {'definitions': [dict(parent=10, successor=10, disposition='same_semantic_calls_actual_replanned_ABI'),
                               dict(parent=26, successor=25, disposition='same_semantic_calls_actual_replanned_ABI')]}
    entries = programs['entries']
    namespace = dict(result=result, common_resources=None, entries=entries,
        limits=entries[0]['launch_resource_limits'], F=Path(side['resource_context']),
        mapping=mapping, side=side, math=math,
        read=lambda p: json.loads(Path(p).read_text()), checked=lambda row: json.loads(checked(row)))
    path = HERE / 'methods/ordinary-resources.py'
    exec(compile(path.read_bytes(), str(path), 'exec'), namespace)
    return dict(resource_lineage=namespace['resource_lineage'], glu_rows=namespace['glu_rows'],
                statements=21, entries=33, original_predicates=True)


def install(benchmark, side, programs, binding_pin, output):
    from .host_adapter import install_benchmark_adapter
    from .resource_guard import install as install_guard
    from normal_compile import register_recipe
    register_recipe(side['recipe']['path'])
    install_guard(side)
    execution_binding = dict(binding=binding_pin, artifact=side['artifact'],
        native_review=side['native_review'], counter_epoch_plan_sha256=side['counter_epoch_plan_sha256'],
        ordinary_program_plan_sha256=side['plan_sha256'], reset_count=side['reset_count'])
    install_benchmark_adapter(benchmark, execution_binding)
    owned_run = benchmark.run
    def run(args, report, save):
        owned_run(args, report, save)
        resources = validate_resources(report, side, programs)
        benchmark.save_json(Path(output).with_suffix('.resources.json'), resources)
    benchmark.run = run
