"""Join a successful public build to its complete current source closure.

This produces source/build provenance only. The existing independent native
review must separately bind the emitted record; this file never issues one.
"""
import importlib.util
import json
import math
from pathlib import Path

from .provider import REPO, DELIVERY, checked, pin, require

SCHEMA = 'flagmega.public-build-source/v1'
FREEZE_NAME = 'public-source-freeze.json'


def _relative(value, root):
    require(isinstance(value, str) and value and not Path(value).is_absolute(), 'Non-relative source member')
    parts = value.split('/')
    require(all(part and part not in ('.', '..', '.local', '__pycache__') for part in parts)
            and Path(value).as_posix() == value, 'Private or escaped source member')
    path = root / value
    resolved = path.resolve()
    require(resolved.is_relative_to(root.resolve()) and '.local' not in resolved.relative_to(root.resolve()).parts,
            'Private or escaped source target')
    return path


def _public_sources():
    path = DELIVERY / 'compile.py'
    spec = importlib.util.spec_from_file_location('_flagmega_public_build_compile', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.REPO == REPO and module.HERE == DELIVERY, 'Wrong canonical compile entry')
    identity, sources = module.source_identity()
    require(isinstance(sources, dict) and sources, 'Missing canonical public source set')
    for relative in sources:
        _relative(relative, REPO)
    return module, identity, sources


def _local_file(root, relative):
    path = root / relative
    require(not path.is_symlink() and path.resolve().is_relative_to(root), 'Escaped build member: ' + relative)
    require(path.is_file(), 'Missing build member: ' + relative)
    return path


def _arg(argv, name):
    require(isinstance(argv, list) and argv.count(name) == 1 and argv.index(name) + 1 < len(argv),
            'Missing or repeated build argument: ' + name)
    return argv[argv.index(name) + 1]


def build_source_record(build_pin):
    """Recompute the producer record from explicit completed build outputs."""
    build_path = Path(build_pin['path'])
    require(build_path.is_absolute() and build_path == build_path.resolve()
            and build_path.name == 'build.json' and not build_path.is_symlink(), 'Expected canonical build.json')
    root = build_path.parent
    build = json.loads(checked(build_pin))
    compiler, identity, sources = _public_sources()
    require(build['status'] == 'verified' and build['source_identity'] == identity, 'Unsuccessful or stale public build')
    require(build['target'] == 'nvidia-sm120' and build['codegen_backend'] == 'cpp'
            and build['placement'] == 'yx=4x24/bb' and build['layer'] is None
            and build['resumed_from'] is None, 'Not an original full public reconstruction')
    schedule = build['schedule']
    require(schedule in ('baseline275', 'candidate49'), 'Wrong ordinary schedule')
    require(build['checkpoint_sha256'] == compiler.SHA256 and len(build['checkpoint_identity']) == 5
            and build['checkpoint_bytes'] == build['checkpoint_identity'][2], 'Original checkpoint provenance differs')
    require(Path(build['checkpoint']).is_absolute(), 'Noncanonical checkpoint path')
    sources_path = _local_file(root, 'sources.json')
    saved_sources = json.loads(sources_path.read_text())
    require(saved_sources == sources, 'Incomplete, extra, stale, or mismatched public source closure')
    snapshot = root / 'source'
    require(snapshot.is_dir() and not snapshot.is_symlink(), 'Missing original source snapshot')
    snapshot_members = {p.relative_to(snapshot).as_posix() for p in snapshot.rglob('*') if p.is_file()}
    require(snapshot_members == set(sources), 'Incomplete or extra build source snapshot')
    for relative, digest in sources.items():
        path = _relative(relative, snapshot)
        require(pin(path)['sha256'] == digest, 'Changed build source snapshot: ' + relative)
    recipe_path = DELIVERY / 'local_optimizations/recipes/u4_l_fullhead48.json'
    recipe = json.loads(recipe_path.read_text())
    require(build['recipe'] == recipe == json.loads(_local_file(root, 'recipe.json').read_text()), 'Wrong build recipe')
    steps = build['commands']
    require([row['name'] for row in steps] == ['import', 'verify-ir', 'inspect-ir', 'compile', 'verify-artifact'],
            'Incomplete original reconstruction commands')
    require(all(type(row['exit']) is int and row['exit'] == 0 and isinstance(row['seconds'], (float, int))
                and math.isfinite(row['seconds']) and row['seconds'] >= 0 for row in steps), 'Failed build command')
    argv = steps[3]['argv']
    require(len(argv) >= 3 and argv[1:3] == ['-m', 'normal_compile'], 'Non-public normal compiler command')
    expected_args = {'--imported': str(root / 'imported.py'), '--checkpoint': build['checkpoint'],
        '--output': str(root / 'artifact'), '--work-dir': str(root / 'dumps'),
        '--recipe': str(recipe_path), '--schedule': schedule}
    require(all(_arg(argv, key) == value for key, value in expected_args.items()), 'Other-build compiler inputs/outputs')
    cache = _arg(argv, '--rdata-cache-dir')
    require(Path(cache).is_absolute() and str(Path(cache).resolve()) == cache, 'Noncanonical fresh cache path')
    for row, key, value in ((steps[0], '--model', build['checkpoint']), (steps[0], '--output', str(root / 'imported.py')),
                            (steps[0], '--mode', 'decode-1'), (steps[0], '--numerical-profile', 'nncase'),
                            (steps[0], '--revision', compiler.REVISION)):
        require(_arg(row['argv'], key) == value, 'Wrong original import command')
    require('--full-model' in steps[0]['argv'] and '--layer' not in steps[0]['argv'], 'Partial original import')
    for row, command, target in ((steps[1], 'verify', root / 'imported.py'),
                                 (steps[2], 'inspect', root / 'imported.py'),
                                 (steps[4], 'verify', root / 'artifact')):
        require(row['argv'][-3:] == [command, str(target), '--json'], 'Wrong build verification target')
    reconstruction_path = _local_file(root, 'dumps/reconstruction.json')
    reconstruction = json.loads(reconstruction_path.read_text())
    require(reconstruction['schema'] == 'flagmega.original-gguf-normal-reconstruction/v1'
            and reconstruction['schedule'] == schedule and reconstruction['original_checkpoint'] == build['checkpoint']
            and reconstruction['expected_checkpoint_sha256'] == compiler.SHA256, 'Wrong original reconstruction provenance')
    require(all(reconstruction[key] is True for key in ('normal_import', 'normal_Compiler', 'normal_writer'))
            and all(reconstruction[key] is False for key in ('old_physical_input', 'old_rdata_dependency'))
            and reconstruction['payload_clone_policy'] == 'require-reflink', 'Wrong normal reconstruction method')
    imports = reconstruction['imports']
    require(isinstance(imports, dict) and imports, 'Missing actual compiler import provenance')
    for name, relative in imports.items():
        if name == 'triton.flagmega' or name.startswith('triton.flagmega.'):
            base = 'python/' + name.replace('.', '/')
        else:
            require(name == 'local_optimizations' or name.startswith('local_optimizations.'), 'Unexpected compiler import namespace')
            base = DELIVERY.relative_to(REPO).as_posix() + '/' + name.replace('.', '/')
        require(relative in sources and relative in (base + '.py', base + '/__init__.py'), 'Escaped/private compiler import')
    artifact_root = root / 'artifact'
    manifest_path = _local_file(root, 'artifact/artifact.json')
    manifest = json.loads(manifest_path.read_text())
    cg = manifest['codegen']
    require(manifest['codegen_backend'] == 'cpp' and cg.get('operator_profile') is None, 'Nonordinary artifact')
    artifact_members = {p.name: pin(p) for p in artifact_root.iterdir() if p.is_file() and p.suffix in ('.json', '.cpp', '.so')}
    require(build['artifact_pins'] == {name: value['sha256'] for name, value in artifact_members.items()}, 'Build/artifact pins differ')
    def member(name):
        require(name in artifact_members and Path(name).name == name, 'Missing native build member')
        _local_file(artifact_root, name)
        return artifact_members[name]
    artifact = dict(root=str(artifact_root), manifest=member('artifact.json'), source=member(cg['source']), library=member(cg['library']))
    require(cg['source_sha256'] == artifact['source']['sha256'] and cg['library_sha256'] == artifact['library']['sha256'],
            'Manifest/native source identity differs')
    programs, epoch = cg['cpp_programs'], cg['cpp_counter_epoch_plan']
    resets = 49 if schedule == 'candidate49' else 275
    require((len(programs['entries']), len(programs['invocations']), len(epoch['resets'])) == (33, 869, resets)
            and (reconstruction['entries'], reconstruction['invocations'], reconstruction['resets']) == (33, 869, resets),
            'Wrong full ordinary reconstruction census')
    require(reconstruction['module_semantic_hash'] == manifest['semantic_hash']
            and reconstruction['ordinary_program_plan_sha256'] == programs['plan_sha256']
            and reconstruction['counter_epoch_plan_sha256'] == epoch['plan_sha256'], 'Reconstruction/artifact identity differs')
    return dict(schema=SCHEMA, public_repo=str(REPO), build=build_pin, public_sources=pin(sources_path),
        source_identity=identity, source_files=sources, snapshot_root=str(snapshot),
        reconstruction=pin(reconstruction_path), artifact=artifact, schedule=schedule,
        module_semantic_hash=manifest['semantic_hash'], ordinary_program_plan_sha256=programs['plan_sha256'],
        counter_epoch_plan_sha256=epoch['plan_sha256'], rdata_cache_dir=cache,
        evidence_kind='producer_source_and_build_binding_only')


def verify_build_source(binding, side):
    """Join producer provenance to the separately supplied native review input."""
    record = build_source_record(binding['public_build'])
    expected_path = Path(binding['public_build']['path']).parent / FREEZE_NAME
    require(side['source_freeze']['path'] == str(expected_path), 'Other-build source freeze')
    require(json.loads(checked(side['source_freeze'])) == record, 'Native admission uses a different build/source freeze')
    require(binding['public_sources'] == record['public_sources'], 'Other-build public sources')
    require({key: side['artifact'][key] for key in record['artifact']} == record['artifact'], 'Side uses another build artifact')
    require(side['schedule'] == record['schedule'] and side['module_semantic_hash'] == record['module_semantic_hash']
            and side['plan_sha256'] == record['ordinary_program_plan_sha256']
            and side['counter_epoch_plan_sha256'] == record['counter_epoch_plan_sha256'], 'Side/build execution identity differs')
    return record
