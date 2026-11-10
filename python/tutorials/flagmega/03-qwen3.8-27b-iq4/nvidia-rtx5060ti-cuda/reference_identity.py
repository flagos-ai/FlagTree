"""Pin clean llama.cpp inputs, built libraries and the runner's actual linkage.

This module does not launch a model or initialize CUDA. Receipts are issued by
build_reference.py after a normal recorded build, and revalidated before and
after each reference execution.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess

LLAMA_COMMIT = '5ce06fc71f5e938301f7dc51697a59ae308bd6b7'
UPSTREAM = 'https://github.com/ggml-org/llama.cpp.git'
RECEIPT_NAME = 'reference-build-receipt.json'
EXPECTED_CONFIG = {
    'CMAKE_BUILD_TYPE': 'Release', 'CMAKE_CUDA_ARCHITECTURES': '120a',
    'BUILD_SHARED_LIBS': 'ON', 'GGML_CUDA': 'ON', 'GGML_CPU': 'ON',
    'GGML_NATIVE': 'OFF', 'GGML_BACKEND_DL': 'OFF',
    'GGML_CUDA_GRAPHS': 'ON', 'GGML_CUDA_FA': 'ON',
    'GGML_CUDA_FA_ALL_QUANTS': 'OFF',
    'GGML_CUDA_FA_QUANTS': 'q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16',
    'GGML_CUDA_FORCE_MMQ': 'OFF', 'GGML_CUDA_FORCE_CUBLAS': 'OFF',
    'LLAMA_BUILD_TESTS': 'OFF', 'LLAMA_BUILD_EXAMPLES': 'OFF', 'LLAMA_BUILD_COMMON': 'OFF',
    'LLAMA_BUILD_TOOLS': 'OFF', 'LLAMA_BUILD_MTMD': 'OFF',
    'CMAKE_C_FLAGS': '', 'CMAKE_CXX_FLAGS': '', 'CMAKE_CUDA_FLAGS': '',
    'CMAKE_C_FLAGS_RELEASE': '-O3 -DNDEBUG', 'CMAKE_CXX_FLAGS_RELEASE': '-O3 -DNDEBUG',
    'CMAKE_CUDA_FLAGS_RELEASE': '-O3 -DNDEBUG',
    'CMAKE_SHARED_LINKER_FLAGS': '', 'CMAKE_SHARED_LINKER_FLAGS_RELEASE': '',
}
LIBRARY_FAMILIES = {'libllama', 'libggml', 'libggml-base', 'libggml-cpu', 'libggml-cuda'}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_pin(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return {'path': str(path), 'nbytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def check_pin(pin):
    if not isinstance(pin, dict) or set(pin) != {'path', 'nbytes', 'sha256'} or file_pin(pin['path']) != pin:
        raise ValueError('Pinned file changed: ' + str(pin.get('path') if isinstance(pin, dict) else pin))


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def command_output(command, **kwargs):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, **kwargs).strip()


def source_identity(source, *, expected_commit=LLAMA_COMMIT):
    """Check actual tracked bytes against Git blobs, not just index status."""
    source = Path(source).resolve(strict=True)
    git = ['git', '-C', str(source), '-c', 'core.fsmonitor=false']
    root = Path(command_output([*git, 'rev-parse', '--show-toplevel'])).resolve()
    if root != source:
        raise ValueError('llama source must be the Git worktree root')
    commit = command_output([*git, 'rev-parse', 'HEAD'])
    if commit != expected_commit:
        raise ValueError(f'Expected pinned llama.cpp commit {expected_commit}, got {commit}')
    dirty = subprocess.check_output([*git, 'status', '--porcelain=v1', '-z', '--untracked-files=all', '--ignore-submodules=none'])
    if dirty:
        raise ValueError('llama.cpp source is dirty: ' + dirty.decode(errors='replace').replace('\0', '\n'))
    tree = subprocess.check_output([*git, 'ls-tree', '-r', '-z', '--full-tree', 'HEAD'])
    object_format = command_output([*git, 'rev-parse', '--show-object-format'])
    if object_format not in ('sha1', 'sha256'):
        raise ValueError('Unsupported Git object format')
    inventory = []
    for record in tree.split(b'\0'):
        if not record:
            continue
        header, name = record.split(b'\t', 1)
        mode, kind, blob = header.decode().split()
        relative = os.fsdecode(name)
        path = source / relative
        if kind != 'blob' or mode not in ('100644', '100755', '120000'):
            raise ValueError(f'Unsupported tracked source entry {relative}: {mode}/{kind}')
        if mode == '120000':
            if not path.is_symlink():
                raise ValueError('Tracked source symlink changed: ' + relative)
            data = os.fsencode(os.readlink(path))
        else:
            if path.is_symlink() or not path.is_file():
                raise ValueError('Tracked source file changed: ' + relative)
            executable = bool(path.stat().st_mode & stat.S_IXUSR)
            if executable != (mode == '100755'):
                raise ValueError('Tracked executable mode changed: ' + relative)
            data = path.read_bytes()
        actual = hashlib.new(object_format, b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if actual != blob:
            raise ValueError('Tracked source bytes differ from pinned commit: ' + relative)
        inventory.append((relative, mode, blob))
    # Recheck status/HEAD after the byte walk; a changed checkout is not admitted.
    if command_output([*git, 'rev-parse', 'HEAD']) != commit or subprocess.check_output(
            [*git, 'status', '--porcelain=v1', '-z', '--untracked-files=all', '--ignore-submodules=none']):
        raise ValueError('Source changed during admission')
    remote = subprocess.run([*git, 'remote', 'get-url', 'origin'], capture_output=True, text=True)
    return {'path': str(source), 'commit': commit, 'tree': command_output([*git, 'rev-parse', 'HEAD^{tree}']),
            'upstream': UPSTREAM, 'origin': remote.stdout.strip() if remote.returncode == 0 else None,
            'dirty_status': '', 'tracked_files': len(inventory), 'tracked_bytes_verified': True,
            'tracked_inventory_sha256': hashlib.sha256(json.dumps(inventory, separators=(',', ':')).encode()).hexdigest()}


def cmake_cache(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith(('#', '//')):
            continue
        match = re.fullmatch(r'([^:=]+):([^=]+)=(.*)', line)
        if match:
            key, _, value = match.groups()
            if key in result:
                raise ValueError('Duplicate CMake cache entry: ' + key)
            result[key] = value
    return result


def validate_config(cache, source):
    if Path(cache.get('CMAKE_HOME_DIRECTORY', '')).resolve() != Path(source).resolve():
        raise ValueError('CMake build belongs to a different source worktree')
    mismatches = {key: {'expected': value, 'actual': cache.get(key)}
                  for key, value in EXPECTED_CONFIG.items() if cache.get(key) != value}
    if mismatches:
        raise ValueError('Reference build has a different configuration: ' + json.dumps(mismatches, sort_keys=True))
    return {key: cache[key] for key in EXPECTED_CONFIG}


def tool_identity(path):
    resolved = shutil.which(str(path)) if not Path(path).is_absolute() else str(path)
    if not resolved:
        raise ValueError('Build tool not found: ' + str(path))
    return {'binary': file_pin(resolved), 'version': command_output([resolved, '--version'])}


def toolchain_identity(cache):
    return {key: tool_identity(cache[key]) for key in
            ('CMAKE_C_COMPILER', 'CMAKE_CXX_COMPILER', 'CMAKE_CUDA_COMPILER', 'CMAKE_COMMAND')}


def library_identity(build):
    build = Path(build).resolve(strict=True)
    aliases, objects = {}, {}
    for path in sorted((build / 'bin').glob('*.so*')):
        family = path.name.split('.so', 1)[0]
        if family not in LIBRARY_FAMILIES:
            if family.startswith(('libllama', 'libggml')):
                raise ValueError('Unexpected reference library family: ' + family)
            continue
        target = path.resolve(strict=True)
        if not target.is_relative_to(build):
            raise ValueError('Reference library escapes its admitted build: ' + str(path))
        if str(target) not in objects:
            objects[str(target)] = file_pin(target)
        pin = objects[str(target)]
        aliases[path.name] = pin
    if {name.split('.so', 1)[0] for name in aliases} != LIBRARY_FAMILIES:
        raise ValueError('Missing required llama/ggml CPU/CUDA shared libraries')
    if len(objects) != len(LIBRARY_FAMILIES):
        raise ValueError('Ambiguous reference library versions in build/bin')
    return {'aliases': aliases, 'objects': sorted(objects.values(), key=lambda x: x['path'])}


def build_snapshot(source, build):
    build = Path(build).resolve(strict=True)
    cache_path = build / 'CMakeCache.txt'
    cache = cmake_cache(cache_path)
    configuration = validate_config(cache, source)
    return {'build': str(build), 'source': source_identity(source), 'cmake_cache': file_pin(cache_path),
            'configuration': configuration, 'toolchain': toolchain_identity(cache),
            'libraries': library_identity(build)}


def manual_build_evidence(directory, *, source, build):
    """Admit an explicitly attested fresh manual build without claiming to run it."""
    directory = Path(directory).resolve(strict=True)
    record_path = directory / 'manual-build.json'
    record = json.loads(record_path.read_text())
    if (Path(record.get('source', '')).resolve() != Path(source).resolve()
            or Path(record.get('build', '')).resolve() != Path(build).resolve()
            or record.get('commit') != LLAMA_COMMIT
            or record.get('fresh_source') is not True or record.get('fresh_build') is not True
            or any(type(record.get(key)) is not int or record[key] != 0
                   for key in ('checkout_returncode', 'configure_returncode', 'build_returncode'))
            or not all(isinstance(record.get(key), str) and record[key] for key in
                       ('actor', 'checkout_session', 'build_session'))):
        raise ValueError('Manual adoption requires explicit successful fresh-checkout/build tool evidence for these exact paths and commit')
    configure = directory / 'configure.log'
    compilation = directory / 'build.log'
    config_text, build_text = configure.read_text(), compilation.read_text()
    if ('Configuring done' not in config_text or 'Generating done' not in config_text
            or 'Build files have been written to: ' + str(Path(build).resolve()) not in config_text):
        raise ValueError('Manual configure log does not establish successful generation of this build directory')
    if not all(re.search(r'Built target ' + re.escape(family.removeprefix('lib')) + r'\s*$', build_text, re.MULTILINE)
               for family in LIBRARY_FAMILIES):
        raise ValueError('Manual build log does not cover the complete llama/ggml library set')
    if 'Building CUDA object ' not in build_text or 'Building CXX object src/CMakeFiles/llama.dir/' not in build_text:
        raise ValueError('Manual adoption needs the fresh CUDA and model compilation log, not an incremental no-op')
    return {'directory': str(directory), 'attestation': file_pin(record_path), 'record': record,
            'configure_log': file_pin(configure), 'build_log': file_pin(compilation),
            'limitation': 'No pre-build tracked-byte snapshot was saved. Fresh pinned checkout and successful build tool sessions are the trusted historical provenance; current tracked bytes are independently checked.',
            'executed_by_wrapper': False}


def validate_build(source, build, receipt=None):
    receipt = Path(receipt) if receipt else Path(build) / RECEIPT_NAME
    document = json.loads(receipt.read_text())
    if (document.get('schema') != 'flagmega.llama-reference-build/v1' or document.get('status') != 'verified'
            or document.get('expected_commit') != LLAMA_COMMIT):
        raise ValueError('A verified pinned reference build receipt is required; run build_reference.py')
    for step in document.get('commands', []):
        if step['returncode'] != 0:
            raise ValueError('Reference build receipt includes a failed command')
        check_pin(step['log'])
    if [step['step'] for step in document.get('commands', [])] != ['configure', 'build']:
        raise ValueError('Reference receipt must include both normal configure and build commands')
    actual = build_snapshot(source, build)
    if actual != document.get('snapshot'):
        raise ValueError('Reference source/config/toolchain/libraries no longer match the build receipt')
    mode = document.get('build_mode')
    if mode == 'adopted_trusted_manual_fresh':
        adoption = document.get('adoption', {})
        if manual_build_evidence(adoption.get('directory', ''), source=source, build=build) != adoption:
            raise ValueError('Manual build evidence changed after adoption')
        if document.get('source_before') is not None:
            raise ValueError('Manual adoption cannot invent a pre-build tracked-byte snapshot')
    elif mode in ('clean_first', 'incremental_admitted'):
        if document.get('source_before') != actual['source']:
            raise ValueError('Reference clean source was not stable throughout the recorded build')
        if mode == 'clean_first' and '--clean-first' not in document['commands'][1]['command']:
            raise ValueError('Initial wrapper receipt did not rebuild every object')
        if mode == 'incremental_admitted' and not document.get('previous_admission'):
            raise ValueError('Incremental wrapper receipt lacks its prior admission')
    else:
        raise ValueError('Unknown reference build provenance mode')
    return {'receipt': file_pin(receipt), 'snapshot': actual}


def loader_environment():
    # Upstream interprets presence (even an empty value) as disabling graphs.
    if 'GGML_CUDA_DISABLE_GRAPHS' in os.environ:
        raise ValueError('GGML_CUDA_DISABLE_GRAPHS violates the graph-enabled reference contract')
    for name in ('LD_PRELOAD', 'LD_AUDIT', 'LD_TRACE_LOADED_OBJECTS'):
        if os.environ.get(name):
            raise ValueError(name + ' is incompatible with admitted reference linkage')
    return {name: value for name, value in os.environ.items()
            if name.startswith('GGML_CUDA_') or name in ('LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES', 'CUDA_MODULE_LOADING')}


def parse_linkage(text, libraries):
    aliases = libraries['aliases']
    resolved = {}
    for line in text.splitlines():
        if 'not found' in line:
            raise ValueError('Runner has an unresolved shared library: ' + line.strip())
        match = re.fullmatch(r'\s*(\S+)\s+=>\s+(.+?)\s+\(0x[0-9a-fA-F]+\)\s*', line)
        if not match:
            continue
        name, path = match.groups()
        if not name.startswith(('libllama', 'libggml')):
            continue
        if name in resolved or name not in aliases:
            raise ValueError('Unexpected or duplicated llama/ggml dependency: ' + name)
        pin = file_pin(path)
        if pin != aliases[name]:
            raise ValueError('Runner resolves outside admitted llama/ggml build bytes: ' + name)
        resolved[name] = pin
    actual_objects = {row['path'] for row in resolved.values()}
    if actual_objects != {row['path'] for row in libraries['objects']}:
        raise ValueError('Runner linkage does not cover the complete admitted llama/ggml library set')
    return resolved


def runner_identity(binary, cpp_source, admission):
    environment = loader_environment()
    result = subprocess.run(['ldd', str(Path(binary).resolve())], capture_output=True, text=True,
                            env={**os.environ, 'LC_ALL': 'C'})
    if result.returncode:
        raise ValueError('ldd failed: ' + result.stdout + result.stderr)
    return {'binary': file_pin(binary), 'source': file_pin(cpp_source),
            'resolved_reference_libraries': parse_linkage(result.stdout, admission['snapshot']['libraries']),
            'loader_environment': environment, 'ldd_stdout': result.stdout, 'ldd_stderr': result.stderr}


def prepare_runner(*, source, build, output, cpp_source, binary_name, receipt=None):
    """Build a runner and save its admitted inputs before any GPU execution."""
    admission = validate_build(source, build, receipt)
    loader_environment()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    binary = output / binary_name
    compiler = admission['snapshot']['toolchain']['CMAKE_CXX_COMPILER']['binary']['path']
    libraries = Path(build).resolve() / 'bin'
    cpp_pin = file_pin(cpp_source)
    command = [compiler, '-O3', '-std=c++17', str(Path(cpp_source).resolve()),
               '-I' + str(Path(source).resolve() / 'include'), '-I' + str(Path(source).resolve() / 'ggml/include'),
               '-L' + str(libraries), '-Wl,-rpath,' + str(libraries), '-lllama', '-o', str(binary.resolve())]
    log = output / 'runner-build.log'
    with log.open('w') as stream:
        compiled = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    build_record = {'command': command, 'returncode': compiled.returncode, 'log': file_pin(log)}
    save_json(output / 'runner-build.json', build_record)
    compiled.check_returncode()
    if validate_build(source, build, receipt) != admission or file_pin(cpp_source) != cpp_pin:
        raise ValueError('Reference inputs changed while compiling its runner')
    identity = runner_identity(binary, cpp_source, admission)
    # Preserve receipt bytes with each execution, independent of external cleanup.
    shutil.copy2(admission['receipt']['path'], output / RECEIPT_NAME)
    preflight = {'schema': 'flagmega.llama-runner-provenance/v1', 'status': 'ready', 'created_utc': utc_now(),
                 'reference_build': admission, 'build_receipt_copy': file_pin(output / RECEIPT_NAME),
                 'runner': identity, 'runner_build': build_record,
                 'helper': file_pin(__file__), 'runtime_validation': 'not_run'}
    save_json(output / 'provenance.json', preflight)
    return binary, command, preflight


def execute_checked(command, *, source, build, output, cpp_source, preflight, receipt=None):
    """Save raw child output, then reject timings if any admitted input drifted."""
    output = Path(output)
    admission = validate_build(source, build, receipt)
    if admission != preflight['reference_build']:
        raise ValueError('Reference build changed before runner execution')
    before = runner_identity(command[0], cpp_source, admission)
    def stable(identity):
        return {key: value for key, value in identity.items() if key not in ('ldd_stdout', 'ldd_stderr')}
    if stable(before) != stable(preflight['runner']):
        raise ValueError('Runner linkage/source/binary changed before execution')
    preflight.update(status='running', runtime_validation='pending', started_utc=utc_now(), run_command=command)
    save_json(output / 'provenance.json', preflight)
    with (output / 'llama.log').open('w') as log:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=log, text=True)
    (output / 'stdout.log').write_text(completed.stdout)
    preflight.update(returncode=completed.returncode, ended_utc=utc_now(),
                     stdout=file_pin(output / 'stdout.log'), stderr=file_pin(output / 'llama.log'))
    try:
        after = validate_build(source, build, receipt)
        post_runner = runner_identity(command[0], cpp_source, after)
        if after != admission or stable(post_runner) != stable(before):
            raise ValueError('Reference source/config/libraries/runner changed during execution')
        preflight.update(status='verified' if completed.returncode == 0 else 'child_failed',
                         runtime_validation='stable_before_after', post_runner=post_runner)
    except BaseException as error:
        preflight.update(status='rejected', runtime_validation='failed', error=f'{type(error).__name__}: {error}')
        save_json(output / 'provenance.json', preflight)
        raise
    save_json(output / 'provenance.json', preflight)
    completed.check_returncode()
    return completed
