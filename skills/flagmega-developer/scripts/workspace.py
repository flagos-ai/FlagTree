#!/usr/bin/env python3
"""Record execution, sync latest working views, and seal reviewed snapshots.

sync --kind artifacts/figures/readme --record REQUEST.json needs no release.
Working requests pin real files/status evidence and the current view hash; root
working views remain authoritative when a reviewed snapshot is published.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit
import dashboard as dashboard_table

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / 'assets/workspace-templates/deliverables.json').read_text())
AGENTS = CATALOG['agents']
ARTIFACT_SUFFIXES = {'.py', '.il', '.mlir', '.ttir', '.ttgir', '.ll', '.llir', '.ptx', '.sass'}
WORKING_METADATA = {'artifacts': '.local/views/root-artifacts.json',
                    'figures': 'figures/manifest.json', 'readme': 'readme.manifest.json'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _immutable_snapshot_hash(candidate):
    """Identify an existing payload only through its committed snapshot receipt."""
    require(candidate.is_file() and not candidate.is_symlink(), 'snapshot candidate is not a regular file')
    for parent in reversed(candidate.parents):
        if parent.name == 'files':
            archive = parent.parent
            if (archive / 'manifest.json').is_file() and (archive / 'receipt.json').is_file():
                manifest, receipt = read(archive / 'manifest.json'), read(archive / 'receipt.json')
                require(receipt.get('status') == 'synced'
                        and sha(archive / 'manifest.json') == receipt.get('root_view_sha256'),
                        'snapshot archive is not committed')
                relative = str(candidate.relative_to(parent))
                row = next((item for item in manifest['files'] if item['path'] == relative), None)
                require(row is not None, 'snapshot candidate is not in the committed manifest')
                return row['sha256']
        if parent.name == 'input-snapshots':
            frame = parent.parent
            require((frame / 'render-receipt.json').is_file(), 'render snapshot is not committed')
            receipt = read(frame / 'render-receipt.json')
            require(receipt.get('status') == 'synced', 'render snapshot is not committed')
            relative = str(candidate.relative_to(frame.parents[2]))
            row = next((item for item in receipt['input_pins'] if item['path'] == relative), None)
            require(row is not None, 'snapshot candidate is not in the committed render receipt')
            return row['sha256']
    raise ValueError('snapshot candidate must belong to a committed immutable frame')


def copy_immutable_snapshot(source, destination, *, immutable_candidate=None, expected_sha256=None):
    """Reuse a verified immutable copy while keeping every active file independent."""
    source, destination = Path(source), Path(destination)
    require(not destination.exists(), 'snapshot destination already exists')
    if immutable_candidate is not None:
        candidate = Path(immutable_candidate)
        sealed_hash = _immutable_snapshot_hash(candidate)
        require(expected_sha256 is None or expected_sha256 == sealed_hash, 'snapshot candidate hash mismatch')
        require(not source.samefile(candidate), 'snapshot candidate aliases the active source')
        before = candidate.stat()
        require(sha(candidate) == sealed_hash, 'immutable snapshot content changed')
        with source.open('rb') as active, candidate.open('rb') as frozen:
            while True:
                block = active.read(1024 * 1024)
                require(block == frozen.read(1024 * 1024), 'snapshot source content differs')
                if not block:
                    break
        after = candidate.stat()
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
                'immutable snapshot changed while verifying')
        os.link(candidate, destination, follow_symlinks=False)
    else:
        shutil.copyfile(source, destination)
        require(expected_sha256 is None or sha(destination) == expected_sha256,
                'source changed while copying snapshot')


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def json_text(data):
    return json.dumps(data, indent=2, allow_nan=False) + '\n'


def new_file(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(text)


def contained(base, value):
    path = (base / value).resolve()
    require(path.is_relative_to(base.resolve()), f'path leaves {base}: {value}')
    return path


def identifier(value, prefix=None):
    pattern = prefix + r'[0-9]{3,}' if prefix else r'[A-Za-z0-9][A-Za-z0-9_.-]*'
    require(isinstance(value, str) and re.fullmatch(pattern, value), f'invalid ID: {value}')
    return value


def canonical(value):
    name = value.replace('_', ' ')
    name = 'reviewer' if name == 'reviwer' else name
    require(name in AGENTS, 'unknown agent')
    return name


def agent(args):
    return canonical(args.agent)


def revision_dir(args):
    if args.report_id:
        require(not args.baseline and not args.milestone and not args.revision, 'live report cannot name a mutable/frozen revision')
        return args.work_dir / '.local/reports' / identifier(args.report_id)
    if args.baseline:
        require(not args.milestone and not args.revision, 'baseline cannot name a milestone/revision')
        return args.work_dir / '.local/baseline'
    identifier(args.milestone, 'm')
    identifier(args.revision, 'r')
    return args.work_dir / '.local/milestones' / args.milestone / 'revisions' / args.revision


def resolve_entry(args, entry):
    scope = entry['scope']
    if scope == 'work':
        base = args.work_dir
    elif scope == 'milestone':
        base = args.work_dir / '.local/milestones' / identifier(args.milestone, 'm')
    elif scope == 'revision':
        base = revision_dir(args)
    elif scope.startswith('capture-'):
        base = revision_dir(args) / 'profiles' / scope.split('-', 1)[1] / identifier(args.capture)
    else:
        require(scope == 'assignment', 'unknown scope')
        recipient = canonical(args.recipient or args.agent)
        require(args.kind == 'request' or recipient == agent(args), 'an agent can prepare only its own assignment outputs')
        dispatch = recipient.replace(' ', '_')
        base = (revision_dir(args) / 'agents' if args.milestone else args.work_dir / '.local/agents')
        base = base / dispatch / identifier(args.assignment)
    return contained(base, entry['path'].format(cache_kind=args.cache_kind))


def plan_owner(args):
    path = revision_dir(args) / 'revision.json'
    owner = read(path)['plan_owner'] if path.exists() else args.plan_owner
    require(owner in {'graph expert', 'kernel expert'}, 'plan owner must be a graph/kernel expert')
    return owner


def prepare(args):
    require(args.kind in CATALOG['entries'], 'unknown delivery kind; inspect workspace.py catalog')
    entry = CATALOG['entries'][args.kind]
    require(not entry.get('generated'), 'this delivery is generated by ' + entry['generator'])
    if args.report_id:
        require(args.kind in {'figure-data', 'figure-graph', 'figure-operators', 'figure-history',
                             'figure-milestones', 'figure-backends', 'readme', 'release',
                             'validation', 'review', 'job', 'request', 'response', 'handoff'},
                'live reports contain figures/readme/audits, not model artifacts or revision control')
    owners = entry['owners']
    if 'plan_owner' in owners:
        owners = [plan_owner(args)]
    require('assigned agent' in owners or agent(args) in owners, f'{args.kind} belongs to {owners}')
    if args.kind == 'cache-manifest':
        expected = ['graph profiler'] if args.cache_kind == 'comparison' else ['graph optimizer']
        require(agent(args) in expected, f'{args.cache_kind} cache belongs to {expected}')
    path = resolve_entry(args, entry)
    if args.output:
        path = contained(path.parent, args.output)
    if 'template' in entry:
        template_path = ROOT / entry['template']
        text = template_path.read_text()
        if path.suffix == '.json':
            data = json.loads(text)
        else:
            def relocate(match):
                url = urlsplit(match.group(2))
                if url.scheme or not url.path:
                    return match.group(0)
                target = (template_path.parent / url.path).resolve()
                if not target.exists():
                    return match.group(0)
                link = os.path.relpath(target, path.parent) + ('#' + url.fragment if url.fragment else '')
                return '[' + match.group(1) + '](' + link + ')'
            text = re.sub(r'\[([^\]\n]+)\]\(([^)\n]+)\)', relocate, text)
            new_file(path, text)
            return path
    else:
        data = deepcopy(entry['body'])
    data.update({'run_id': args.run_id, 'milestone_id': args.milestone,
                 'revision_id': args.revision, 'report_id': args.report_id, 'agent_name': agent(args), 'agent_id': args.agent_id})
    if args.kind == 'revision':
        data['plan_owner'] = args.plan_owner
        for directory in ('artifacts/generated', 'artifacts/ir', 'figures'):
            (path.parent / directory).mkdir(parents=True, exist_ok=True)
    if args.kind == 'milestone':
        for directory in ('artifacts/generated', 'artifacts/ir', 'figures', 'revisions'):
            (path.parent / directory).mkdir(parents=True, exist_ok=True)
    if args.kind == 'request':
        require(args.recipient, 'request requires --recipient')
        data['recipient'] = canonical(args.recipient)
    if args.assignment:
        data['assignment_id'] = args.assignment
    new_file(path, json_text(data))
    return path


def initialize(args):
    require(agent(args) == 'graph optimizer', 'graph optimizer initializes the physical workspace')
    for directory in ('artifacts/generated', 'artifacts/ir', 'figures', 'tests', 'benchmark', '.local/cache/weights',
                      '.local/cache/kernels', '.local/cache/comparison', '.local/baseline', '.local/milestones'):
        (args.work_dir / directory).mkdir(parents=True, exist_ok=True)
    ignore = args.work_dir / '.gitignore'
    previous = ignore.read_text() if ignore.exists() else ''
    if '/.local' not in previous.splitlines():
        with ignore.open('a') as stream:
            stream.write(('\n' if previous and not previous.endswith('\n') else '') + '/.local\n')
    if not (args.work_dir / '.local/workspace.json').exists():
        args.kind = 'workspace'
        prepare(args)
    return args.work_dir


def files(directory, destination, kind):
    result = []
    if kind == 'artifacts':
        require(sorted(p.name for p in directory.iterdir()) == ['generated', 'ir']
                and all((directory / name).is_dir() and not (directory / name).is_symlink()
                        for name in ('generated', 'ir')), 'artifacts contain exactly generated/ir directories')
    for path in sorted(directory.rglob('*')):
        require(not path.is_symlink(), f'symlink is not a frozen file: {path}')
        if not path.is_file() or path.resolve() == destination.resolve():
            continue
        if kind == 'artifacts':
            artifact_path(Path('artifacts') / path.relative_to(directory))
        result.append({'path': str(path.relative_to(directory)), 'size_bytes': path.stat().st_size,
                       'sha256': sha(path)})
    return result


def manifest(args):
    owner = agent(args)
    require(owner != 'orchestrator', 'delegate file inspection/manifest generation')
    directory = contained(args.work_dir, args.directory)
    output = contained(args.work_dir, args.output)
    require(directory.is_dir(), 'manifest directory is missing')
    if args.manifest_kind == 'artifacts':
        require(owner == 'graph optimizer', 'graph optimizer exports full artifacts')
        require(output.is_relative_to(args.work_dir / '.local')
                and 'artifacts' not in output.relative_to(args.work_dir / '.local').parts,
                'artifact manifests stay under .local outside artifacts')
    else:
        require(directory.is_relative_to(args.work_dir / '.local'), 'native/cache/raw payloads stay local')
    data = {'schema': 'flagmega.files/v1', 'kind': args.manifest_kind, 'agent_name': owner,
            'created_at_utc': utc(), 'directory': str(directory.relative_to(args.work_dir)),
            'input_identities': read(args.inputs) if args.inputs else {},
            'files': files(directory, output, args.manifest_kind)}
    new_file(output, json_text(data))
    return output


def run_job(args):
    require(agent(args) != 'orchestrator', 'orchestrator dispatches; the assigned worker executes')
    job = read(args.job)
    require(job.get('agent_name') == agent(args), 'job actor does not match its owner')
    identifier(job.get('job_id'))
    argv = job.get('argv')
    require(isinstance(argv, list) and argv and all(isinstance(s, str) and s for s in argv), 'argv must be a nonempty list')
    log_dir = args.work_dir / '.local/jobs' / job['job_id']
    for item in job.get('input_files', []):
        path = contained(args.work_dir, item['path'])
        require(path.is_file() and sha(path) == item['sha256'], 'job input changed')
    log_dir.mkdir(parents=True, exist_ok=False)
    cwd = (args.work_dir / job.get('cwd', '.')).resolve()
    receipt = {'schema': 'flagmega.job-receipt/v1', 'job_id': job['job_id'],
               'agent_name': agent(args), 'job_sha256': sha(args.job), 'argv': argv,
               'cwd': str(cwd), 'started_at_utc': utc(), 'outputs': [],
               'measurement_ids': job.get('measurement_ids', []), 'input_identities': job.get('input_identities', {})}
    with (log_dir / 'stdout.log').open('wb') as out, (log_dir / 'stderr.log').open('wb') as err:
        try:
            receipt['exit_code'] = subprocess.run(argv, cwd=cwd, stdout=out, stderr=err, check=False).returncode
        except OSError as error:
            err.write(str(error).encode())
            receipt['exit_code'] = 127
    receipt['ended_at_utc'] = utc()
    for item in job.get('expected_outputs', []):
        path = contained(args.work_dir, item['path'])
        valid = path.is_file() and path.stat().st_size >= item.get('min_bytes', 1)
        receipt['outputs'].append({'path': item['path'], 'valid': valid,
                                   'sha256': sha(path) if path.is_file() else None})
    receipt['status'] = 'completed' if receipt['exit_code'] == 0 and all(i['valid'] for i in receipt['outputs']) else 'failed'
    receipt['logs'] = [{'path': str((log_dir / name).relative_to(args.work_dir)), 'sha256': sha(log_dir / name)}
                       for name in ('stdout.log', 'stderr.log')]
    new_file(log_dir / 'receipt.json', json_text(receipt))
    return log_dir / 'receipt.json', 0 if receipt['status'] == 'completed' else 1


def append_measurement(args):
    require(agent(args) == 'reviewer', 'only reviewer appends the ledger')
    event = read(args.record)
    require(event.get('agent_name') == 'reviewer' or event.get('recorded_by') == 'reviewer', 'missing reviewer provenance')
    event_id = event.get('event_id') or event.get('measurement_id')
    require(isinstance(event_id, str) and event_id, 'event/measurement ID is required')
    require(event.get('validity') != 'pending', 'record the actual outcome, not a pending scaffold')
    ledger = args.work_dir / '.local/metrics/performance-history.jsonl'
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        prefix = stream.read()
        require(not prefix or prefix.endswith('\n'), 'truncated ledger prefix')
        previous = [json.loads(line) for line in prefix.splitlines()]
        require(all(isinstance(row, dict) for row in previous), 'invalid ledger row')
        require(event_id not in {row.get('event_id') or row.get('measurement_id') for row in previous}, 'duplicate event ID; append a distinct correction event')
        stream.seek(0, 2)
        stream.write(json.dumps(event, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return ledger


def working_path(base, relative):
    """Managed paths must not traverse symlinks, even within the workspace."""
    require(isinstance(relative, str) and relative, 'working path is required')
    path = Path(relative)
    require(not path.is_absolute() and '..' not in path.parts, 'working path must be workspace-relative')
    result = base
    for part in path.parts:
        result = result / part
        require(not result.is_symlink(), f'working path is a symlink: {result}')
    return contained(base, relative)


@contextmanager
def view_lock(work, kind):
    path = working_path(work, '.local/views/root-' + kind + '.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def check_pins(work, pins):
    require(isinstance(pins, list) and pins, 'working status needs actual input pins')
    for pin in pins:
        path = working_path(work, pin['path'])
        require(path.is_file() and sha(path) == pin['sha256'], 'working input pin changed')


def artifact_metadata_status(status):
    """Keep execution evidence external to the model source/IR view."""
    result = deepcopy(status)
    dashboard = status.get('dashboard')
    if isinstance(dashboard, dict):
        result['dashboard'] = {key: dashboard[key] for key in
                               ('path', 'sha256', 'snapshot', 'parser_sha256')
                               if key in dashboard}
    return result


def artifact_metadata_file(item):
    """File/stage identity is sufficient; execution overlays stay local."""
    return {key: item[key] for key in ('path', 'source', 'sha256', 'size_bytes',
                                      'source_id', 'retained_from_sync_id',
                                      'stage_info', 'source_map') if key in item}


def artifact_path(relative):
    relative = Path(relative)
    require(len(relative.parts) >= 3 and relative.parts[0] == 'artifacts'
            and relative.parts[1] in {'generated', 'ir'}, 'artifacts accept generated/ir destinations only')
    require(relative.suffix in ARTIFACT_SUFFIXES, 'artifacts contain a non-IR/codegen payload')
    require(relative.parts[1] != 'generated' or relative.suffix == '.py',
            'generated contains emitted code; compiler dumps belong to ir')
    require('ir' not in relative.parts[2:] if relative.parts[1] == 'generated' else True,
            'compiler dumps belong to ir, not generated')


def artifact_tree_matches(root, rows):
    if not root.is_dir() or root.is_symlink():
        return False
    if sorted(p.name for p in root.iterdir()) != ['generated', 'ir']:
        return False
    expected = {str(Path(row['path']).relative_to('artifacts')): row['sha256'] for row in rows}
    expected_dirs = {'generated', 'ir'}
    for name in expected:
        expected_dirs.update(str(p) for p in Path(name).parents if str(p) != '.')
    actual = {}
    directories = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            return False
        name = str(path.relative_to(root))
        if path.is_dir():
            directories.add(name)
        elif path.is_file():
            actual[name] = sha(path)
        else:
            return False
    return actual == expected and directories == expected_dirs


def sync_artifact_frame(work, record, record_path, previous, rows, sources, managed, archive, pointer):
    """Replace the selected payload, preserve displaced evidence, and share unchanged payload frames."""
    for name in rows:
        artifact_path(name)
    same_payload = sorted((r['path'], r['sha256']) for r in rows.values()) == sorted(
        (r['path'], r['sha256']) for r in previous.get('files', []))
    payload_archive = previous.get('payload_archive', previous.get('archive')) if same_payload else None
    replace_tree = not artifact_tree_matches(work / 'artifacts', list(rows.values()))
    data = {'schema': 'flagmega.working-view/v1', 'mode': 'working', 'kind': 'artifacts', 'scope': 'root',
            'sync_id': record['sync_id'], 'generation': previous.get('generation', 0) + 1,
            'agent_name': 'graph optimizer', 'agent_id': record['agent_id'], 'source_id': record['source_id'],
            'observed_at_utc': record.get('observed_at_utc') or utc(), 'status': artifact_metadata_status(record['status']),
            'input_pins': record['input_pins'], 'source_manifest': record.get('source_manifest', previous.get('source_manifest') if record.get('retain_previous') else None),
            'request_sha256': sha(record_path), 'files': list(rows.values()), 'managed_files': list(rows.values()),
            'archive': str(archive.relative_to(work)),
            'payload_archive': payload_archive or str(archive.relative_to(work)),
            'acceptance_note': 'Latest actual model and labelled working components; acceptance requires separate review.'}
    archive.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(dir=archive.parent, prefix='.' + record['sync_id'] + '-'))
    pointer_before = pointer.read_bytes() if pointer.exists() else None
    retired = stage / 'retired/artifacts'
    replaced = False
    finalized = False
    temporary_pointer = stage.with_name(stage.name + '.pointer')
    try:
        if not payload_archive:
            for name, source in sources.items():
                target = stage / 'files' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                copy_immutable_snapshot(source, target, expected_sha256=rows[name]['sha256'])
        if replace_tree:
            current = stage / 'current/artifacts'
            for kind in ('generated', 'ir'):
                (current / kind).mkdir(parents=True, exist_ok=True)
            for name, source in sources.items():
                destination = stage / 'current' / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                require(sha(destination) == rows[name]['sha256'], 'source changed while staging working view')
        check_pins(work, record['input_pins'])
        for name, row in managed.items():
            source = working_path(work, name)
            require(source.is_file() and sha(source) == row['sha256'], 'managed file changed during staging')
        (stage / 'manifest.json').write_text(json_text(data))
        if replace_tree:
            retired.parent.mkdir(parents=True, exist_ok=True)
            if (work / 'artifacts').exists():
                (work / 'artifacts').replace(retired)
            try:
                current.replace(work / 'artifacts')
            except BaseException:
                if retired.exists():
                    retired.replace(work / 'artifacts')
                raise
            replaced = True
            shutil.rmtree(stage / 'current')
        stage.replace(archive)
        finalized = True
        retired = archive / 'retired/artifacts'
        receipt = {'schema': 'flagmega.working-sync-receipt/v1', 'status': 'synced', 'kind': 'artifacts',
                   'sync_id': record['sync_id'], 'generation': data['generation'], 'agent_name': 'graph optimizer',
                   'agent_id': record['agent_id'], 'root_view': str(pointer.relative_to(work)),
                   'root_view_sha256': sha(archive / 'manifest.json'), 'archive': str(archive.relative_to(work)),
                   'payload_archive': data['payload_archive'], 'payload_changed': not same_payload,
                   'retired_artifacts': str(retired.relative_to(work)) if retired.exists() else None,
                   'files': data['files'], 'synced_at_utc': utc()}
        new_file(archive / 'receipt.json', json_text(receipt))
        temporary_pointer.write_text(json_text(data))
        temporary_pointer.replace(pointer)
        return archive / 'receipt.json'
    except BaseException:
        if replaced:
            shutil.rmtree(work / 'artifacts')
            if retired.exists():
                retired.replace(work / 'artifacts')
        if pointer_before is None:
            pointer.unlink(missing_ok=True)
        else:
            pointer.write_bytes(pointer_before)
        if finalized:
            shutil.rmtree(archive)
        raise
    finally:
        temporary_pointer.unlink(missing_ok=True)
        shutil.rmtree(stage, ignore_errors=True)


def sync(args):
    """Sync a hash-pinned working request, including failed/pending states.

    Requests use flagmega.working-sync/v1 with sync_id, kind, agent_name,
    agent_id, source_id, status, base_view_sha256, input_pins and files.
    Each file names source, root destination path and sha256. retain_previous
    supports subset/status-only updates without inventing a new artifact.
    """
    work, kind = args.work_dir, args.kind
    require(kind in WORKING_METADATA, 'sync requires --kind artifacts/figures/readme')
    owner = 'graph optimizer' if kind == 'artifacts' else 'graph profiler'
    require(agent(args) == owner, f'{kind} working view belongs to {owner}')
    record = read(args.record)
    require(record.get('schema') == 'flagmega.working-sync/v1', 'invalid working request schema')
    require(record.get('kind') == kind and record.get('agent_name') == owner, 'working request owner/kind mismatch')
    require(record.get('agent_id') and (not args.agent_id or record['agent_id'] == args.agent_id), 'working context mismatch')
    sync_id = identifier(record.get('sync_id'))
    require(isinstance(record.get('source_id'), str) and record['source_id']
            and isinstance(record.get('status'), dict) and record['status'],
            'working source identity and structured actual status are required')
    require(type(record.get('retain_previous', False)) is bool, 'retain_previous must be boolean')
    require('base_view_sha256' in record, 'working request needs base_view_sha256 CAS pin')
    chosen = record.get('files')
    require(isinstance(chosen, list) and (chosen or record.get('retain_previous')), 'empty update must retain previous produced files')
    pointer = working_path(work, '.local/views/root-' + kind + '.json')
    archive = working_path(work, '.local/working/' + kind + '/' + sync_id)
    with view_lock(work, kind):
        require(not archive.exists(), 'working sync ID already archived; history is immutable')
        require(record['base_view_sha256'] == (sha(pointer) if pointer.exists() else None), 'working view changed; stale update rejected')
        check_pins(work, record.get('input_pins'))
        previous = read(pointer) if pointer.exists() else {}
        require(chosen or previous.get('files'), 'status-only update needs an actual previous produced view')
        source_manifest = record.get('source_manifest')
        if source_manifest is None and record.get('retain_previous'):
            source_manifest = previous.get('source_manifest')
        if source_manifest is not None:
            check_pins(work, [source_manifest])
            require(source_manifest in record['input_pins'], 'selected source manifest must be pinned as an input')
        managed = {row['path']: row for row in previous.get('managed_files', previous.get('files', []))}
        for name, row in managed.items():
            path = working_path(work, name)
            require(path.is_file() and sha(path) == row['sha256'], 'managed working file was changed or removed')
        rows, sources = {}, {}
        for item in chosen:
            source = working_path(work, item['source'])
            destination = working_path(work, item['path'])
            relative = Path(item['path'])
            require(item['path'] != WORKING_METADATA[kind], 'working metadata destination is reserved')
            if kind == 'readme':
                require(item['path'] == 'readme.md' and len(chosen) == 1 and source.suffix == '.md', 'working readme must name readme.md')
            else:
                require(relative.parts[0] == kind, 'working file is outside the selected view')
                if kind == 'artifacts':
                    artifact_path(relative)
                require((source.suffix in ARTIFACT_SUFFIXES and relative.suffix in ARTIFACT_SUFFIXES)
                        if kind == 'artifacts' else source.suffix == relative.suffix == '.svg',
                        'working views accept IR/kernel sources or SVG figures, never weights/binaries')
            require(source.is_file() and sha(source) == item['sha256'], 'working source changed')
            require(item['path'] not in rows, 'duplicate working destination')
            require(not destination.exists() or item['path'] in managed, 'cannot overwrite an unmanaged file')
            rows[item['path']] = {**item, 'source_id': record['source_id'], 'size_bytes': source.stat().st_size}
            sources[item['path']] = source
        if record.get('retain_previous'):
            for item in previous.get('files', []):
                if item['path'] not in rows:
                    rows[item['path']] = {**item, 'source_id': item.get('source_id', previous.get('source_id')),
                                          'retained_from_sync_id': previous.get('sync_id')}
                    sources[item['path']] = working_path(work, item['path'])
        if kind == 'artifacts':
            rows = {name: artifact_metadata_file(item) for name, item in rows.items()}
            return sync_artifact_frame(work, record, args.record, previous, rows, sources, managed, archive, pointer)
        metadata_path = working_path(work, WORKING_METADATA[kind])
        require(not metadata_path.exists() or WORKING_METADATA[kind] in managed, 'cannot overwrite unmanaged working metadata')
        data = {'schema': 'flagmega.working-view/v1', 'mode': 'working', 'kind': kind, 'scope': 'root',
                'sync_id': sync_id, 'generation': previous.get('generation', 0) + 1,
                'agent_name': owner, 'agent_id': record['agent_id'], 'source_id': record['source_id'],
                'observed_at_utc': record.get('observed_at_utc') or utc(),
                'status': artifact_metadata_status(record['status']) if kind == 'artifacts' else record['status'],
                'input_pins': record['input_pins'], 'request_sha256': sha(args.record),
                'source_manifest': source_manifest,
                'files': list(rows.values()), 'archive': str(archive.relative_to(work)),
                'acceptance_note': 'Latest actual working state; acceptance and achieved history require separate review.'}
        archive.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(dir=archive.parent, prefix='.' + sync_id + '-'))
        staged, backups, committed = [], {}, []
        pointer_before = pointer.read_bytes() if pointer.exists() else None
        finalized = False
        temporary_pointer = None
        try:
            for name, source in sources.items():
                target = stage / 'files' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                candidate = None
                if name in managed and rows[name]['sha256'] == managed[name]['sha256'] and previous.get('archive'):
                    candidate = working_path(work, previous['archive']) / 'files' / name
                copy_immutable_snapshot(source, target, immutable_candidate=candidate,
                                        expected_sha256=rows[name]['sha256'])
                require(sha(target) == rows[name]['sha256'], 'source changed while staging working view')
            staged_metadata = stage / 'files' / WORKING_METADATA[kind]
            staged_metadata.parent.mkdir(parents=True, exist_ok=True)
            staged_metadata.write_text(json_text(data))
            current = [*data['files'], {'path': WORKING_METADATA[kind], 'sha256': sha(staged_metadata), 'generated_metadata': True}]
            data['managed_files'] = current
            (stage / 'manifest.json').write_text(json_text(data))
            # Validate the complete frame before touching any managed root file.
            check_pins(work, record['input_pins'])
            for item in chosen:
                require(sha(working_path(work, item['source'])) == item['sha256'], 'working source changed during staging')
            for name, row in managed.items():
                source = working_path(work, name)
                require(sha(source) == row['sha256'], 'managed file changed during staging')
                backup = stage / 'rollback' / name
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, backup)
                backups[name] = backup
            for item in current:
                destination = working_path(work, item['path'])
                destination.parent.mkdir(parents=True, exist_ok=True)
                fd, name = tempfile.mkstemp(dir=destination.parent, prefix='.working-')
                os.close(fd)
                temporary = Path(name)
                staged.append((temporary, destination))
                shutil.copyfile(stage / 'files' / item['path'], temporary)
                require(sha(temporary) == item['sha256'], 'working staged copy changed')
            for temporary, destination in staged:
                temporary.replace(destination)
                committed.append(str(destination.relative_to(work)))
            for name in managed.keys() - {item['path'] for item in current}:
                working_path(work, name).unlink()
                committed.append(name)
            # The pointer is the atomic commit marker. Archives are complete,
            # immutable file copies; readers never need a live source symlink.
            stage.replace(archive)
            finalized = True
            backups = {name: archive / 'rollback' / name for name in backups}
            receipt = {'schema': 'flagmega.working-sync-receipt/v1', 'status': 'synced', 'kind': kind,
                       'sync_id': sync_id, 'generation': data['generation'], 'agent_name': owner,
                       'agent_id': record['agent_id'], 'root_view': str(pointer.relative_to(work)),
                       'root_view_sha256': sha(archive / 'manifest.json'), 'archive': str(archive.relative_to(work)),
                       'files': current, 'synced_at_utc': utc()}
            new_file(archive / 'receipt.json', json_text(receipt))
            temporary_pointer = stage.with_name(stage.name + '.pointer')
            temporary_pointer.write_text(json_text(data))
            temporary_pointer.replace(pointer)
            shutil.rmtree(archive / 'rollback', ignore_errors=True)
            return archive / 'receipt.json'
        except BaseException:
            # Retain the old committed view on staging/commit errors, including
            # removals. The archive has not become a published history version.
            for name in reversed(committed):
                destination = working_path(work, name)
                backup = backups.get(name)
                if backup and backup.exists():
                    shutil.copyfile(backup, destination)
                else:
                    destination.unlink(missing_ok=True)
            if pointer_before is None:
                pointer.unlink(missing_ok=True)
            else:
                pointer.write_bytes(pointer_before)
            if finalized:
                shutil.rmtree(archive)
            raise
        finally:
            for temporary, _ in staged:
                temporary.unlink(missing_ok=True)
            if temporary_pointer is not None:
                temporary_pointer.unlink(missing_ok=True)
            shutil.rmtree(stage, ignore_errors=True)


def publish(args):
    with view_lock(args.work_dir, args.publish_kind):
        return publish_locked(args)


def watched_source_matches(current, spec):
    """Labels and manifest paths cannot identify a produced source cohort."""
    expected_files = sorted((item['path'], item['sha256']) for item in spec['files'])
    current_files = sorted((item['path'], item['sha256']) for item in current.get('files', []))
    manifest_hash = spec['source_manifest']['sha256']
    if 'source_manifest' in current:
        selected = current['source_manifest']
        manifest_matches = isinstance(selected, dict) and selected.get('sha256') == manifest_hash
    else:
        # Bootstrap a legacy view once. New sync frames always declare selection
        # explicitly, so an ancestor pin cannot identify a later produced view.
        manifest_matches = any(pin['sha256'] == manifest_hash for pin in current.get('input_pins', []))
    return current_files == expected_files and manifest_matches


def watch(args, once=False):
    """Refresh actual status when its bytes change; retain the produced files.

    flagmega.working-watch/v2 pins watch_id, kind, agent_name, agent_id,
    source_id, dashboard_path, source_manifest {path, sha256}, and the same files
    list used by sync. refresh executes one event hook; watch polls until its
    stop file, timeout, superseding source, or max-update limit is reached.
    """
    spec = read(args.record)
    kind, work = args.kind, args.work_dir
    owner = 'graph optimizer' if kind == 'artifacts' else 'graph profiler'
    require(kind in WORKING_METADATA and agent(args) == owner, 'watch role/kind mismatch')
    require(spec.get('schema') == 'flagmega.working-watch/v2', 'invalid working watch schema; use dashboard v2')
    require(spec.get('kind') == kind and spec.get('agent_name') == owner, 'watch owner/kind mismatch')
    require(spec.get('agent_id') and (not args.agent_id or spec['agent_id'] == args.agent_id), 'watch context mismatch')
    identifier(spec.get('watch_id'))
    require(spec.get('source_id') and isinstance(spec.get('files'), list) and spec['files'], 'watch needs produced source identity/files')
    require(args.watch_interval > 0 and (args.watch_timeout is None or args.watch_timeout > 0)
            and (args.max_updates is None or args.max_updates > 0), 'watch bounds must be positive')
    require(spec.get('dashboard_path') == 'dashboard.md' and 'state_path' not in spec,
            'watch reads only the root dashboard.md management table')
    dashboard_path = working_path(work, spec['dashboard_path'])
    stop = working_path(work, args.stop_file) if args.stop_file else None
    home = working_path(work, '.local/watchers/' + spec['watch_id'])
    home.mkdir(parents=True, exist_ok=True)
    lock = (home / 'watch.lock').open('a')
    service_lock = None
    service_path = working_path(work, args.service_record) if args.service_record else None
    service = None
    def service_status(status, **details):
        if service is None:
            return
        service.update({'status': status, 'observed_at_utc': utc(), **details})
        temporary = service_path.with_suffix('.json.updating')
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json_text(service))
        temporary.replace(service_path)
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('this working watcher/event hook is already running')
        if not once:
            service_lock_path = working_path(work, '.local/views/watch-' + kind + '.lock')
            service_lock_path.parent.mkdir(parents=True, exist_ok=True)
            service_lock = service_lock_path.open('a')
            try:
                fcntl.flock(service_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('a working watcher for this view is already running')
        fixed_spec = home / 'spec.json'
        if fixed_spec.exists():
            require(sha(fixed_spec) == sha(args.record), 'watch ID belongs to an immutable different spec')
        else:
            shutil.copyfile(args.record, fixed_spec)
        started = time.monotonic()
        if service_path:
            proc_fields = Path('/proc/self/stat').read_text().split(') ', 1)[1].split()
            service = {'schema': 'flagmega.working-watch-service/v1', 'watch_id': spec['watch_id'],
                       'agent_name': owner, 'agent_id': spec['agent_id'], 'kind': kind,
                       'pid': os.getpid(), 'proc_start_ticks': int(proc_fields[19]),
                       'argv': sys.argv, 'started_at_utc': utc(), 'spec_sha256': sha(fixed_spec),
                       'script_sha256': sha(Path(__file__)), 'stop_file': args.stop_file,
                       'dashboard_parser_sha256': sha(Path(dashboard_table.__file__)),
                       'source_manifest_sha256': spec['source_manifest']['sha256'],
                       'logs': [str((home / name).relative_to(work)) for name in ('stdout.log', 'stderr.log')]}
            service_status('running')
        events, outcome = [], 'unchanged'
        while True:
            if stop and stop.exists():
                outcome = 'stopped'
                break
            pointer = working_path(work, '.local/views/root-' + kind + '.json')
            pointer_bytes = pointer.read_bytes() if pointer.exists() else None
            current = json.loads(pointer_bytes) if pointer_bytes is not None else {}
            if current.get('mode') == 'working' and not watched_source_matches(current, spec):
                outcome = 'superseded'
                break
            check_pins(work, [spec['source_manifest']])
            for item in spec['files']:
                require(sha(working_path(work, item['source'])) == item['sha256'], 'watch produced source changed')
            dashboard_bytes = dashboard_path.read_bytes()
            table = dashboard_table.parse(dashboard_bytes.decode('utf-8'))
            dashboard_hash = hashlib.sha256(dashboard_bytes).hexdigest()
            prior_dashboard = next((pin['sha256'] for pin in current.get('input_pins', [])
                                    if pin['path'] == spec['dashboard_path']), None)
            layout_changed = kind == 'artifacts' and not artifact_tree_matches(work / 'artifacts', current.get('files', []))
            if dashboard_hash != prior_dashboard or layout_changed:
                snapshots = home / 'dashboards'
                snapshots.mkdir(exist_ok=True)
                snapshot = snapshots / (dashboard_hash + '.md')
                if not snapshot.exists():
                    with snapshot.open('xb') as stream:
                        stream.write(dashboard_bytes)
                require(sha(snapshot) == dashboard_hash, 'archived dashboard snapshot changed')
                sequence = len(list((home / 'events').glob('*.json'))) if (home / 'events').exists() else 0
                event_id = spec['watch_id'] + '-' + str(sequence).zfill(6)
                request = home / 'events' / (event_id + '.json')
                record = {'schema': 'flagmega.working-sync/v1', 'sync_id': event_id, 'kind': kind,
                          'agent_name': owner, 'agent_id': spec['agent_id'],
                          'source_id': current.get('source_id', spec['source_id']),
                          'source_manifest': spec['source_manifest'],
                          'status': {**current.get('status', {}), **spec.get('status', {}),
                                     'last_produced_source': current.get('source_id', spec['source_id']),
                                     'dashboard': {**table, 'path': spec['dashboard_path'], 'sha256': dashboard_hash,
                                                   'snapshot': str(snapshot.relative_to(work)),
                                                   'parser_sha256': sha(Path(dashboard_table.__file__))}},
                          'base_view_sha256': hashlib.sha256(pointer_bytes).hexdigest() if pointer_bytes is not None else None,
                          'input_pins': [spec['source_manifest'], {'path': spec['dashboard_path'], 'sha256': dashboard_hash},
                                         {'path': str(snapshot.relative_to(work)), 'sha256': dashboard_hash}],
                          'files': [] if current.get('mode') == 'working' else spec['files'],
                          'retain_previous': current.get('mode') == 'working'}
                new_file(request, json_text(record))
                event_args = deepcopy(args)
                event_args.record = request
                try:
                    receipt = sync(event_args)
                except ValueError:
                    # A concurrent authoritative transition is a new event, not
                    # permission to commit stale status. Keep the rejected input.
                    if sha(dashboard_path) != dashboard_hash or (pointer.exists() and sha(pointer) != record['base_view_sha256']):
                        continue
                    raise
                events.append({'receipt': str(receipt.relative_to(work)), 'sha256': sha(receipt)})
                outcome = 'refreshed'
                service_status('running', successful_updates=len(events), last_dashboard_sha256=dashboard_hash,
                               last_sync_receipt=str(receipt.relative_to(work)))
                print(receipt, flush=True)
            if once or (args.max_updates is not None and len(events) >= args.max_updates):
                break
            if args.watch_timeout is not None and time.monotonic() - started >= args.watch_timeout:
                outcome = 'timeout'
                break
            deadline = time.monotonic() + args.watch_interval
            while time.monotonic() < deadline and not (stop and stop.exists()):
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(min(0.5, remaining))
        runs = home / 'runs'
        run_id = str(len(list(runs.glob('*.json'))) if runs.exists() else 0).zfill(6)
        receipt = runs / (run_id + '.json')
        new_file(receipt, json_text({'schema': 'flagmega.working-watch-receipt/v1', 'watch_id': spec['watch_id'],
                 'agent_name': owner, 'agent_id': spec['agent_id'], 'kind': kind,
                 'status': outcome, 'events': events, 'spec_sha256': sha(fixed_spec),
                 'elapsed_seconds': time.monotonic() - started, 'ended_at_utc': utc()}))
        service_status(outcome, ended_at_utc=utc(), terminal_receipt=str(receipt.relative_to(work)))
        return receipt
    except BaseException as error:
        service_status('failed', error=str(error), ended_at_utc=utc())
        raise
    finally:
        lock.close()
        if service_lock is not None:
            service_lock.close()


def seal_reviewed(work, kind, rev, release_path, chosen):
    """Keep accepted file copies independent of the current working view."""
    release_hash = sha(release_path)
    archive = working_path(work, '.local/reviewed/' + kind + '/' + release_hash)
    if archive.exists():
        require(sha(archive / 'release.json') == release_hash, 'reviewed archive release changed')
        require(read(archive / 'manifest.json')['files'] == chosen, 'reviewed archive identity changed')
        for item in chosen:
            require(sha(archive / 'files' / item['path']) == item['sha256'], 'reviewed history changed')
        return archive
    archive.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(dir=archive.parent, prefix='.sealing-'))
    try:
        for item in chosen:
            destination = stage / 'files' / item['path']
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(contained(rev, item['path']), destination)
            require(sha(destination) == item['sha256'], 'released file changed during sealing')
        shutil.copyfile(release_path, stage / 'release.json')
        require(sha(stage / 'release.json') == release_hash, 'release changed during sealing')
        new_file(stage / 'manifest.json', json_text({'schema': 'flagmega.reviewed-view/v1',
                 'kind': kind, 'release_sha256': release_hash, 'files': chosen,
                 'source_revision': str(rev.relative_to(work)), 'sealed_at_utc': utc()}))
        stage.replace(archive)
        return archive
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def publish_locked(args):
    expected = 'graph optimizer' if args.publish_kind == 'artifacts' else 'graph profiler'
    require(agent(args) == expected, f'{args.publish_kind} publication belongs to {expected}')
    rev = revision_dir(args)
    require(not args.report_id or args.publish_kind in {'figures', 'readme'}, 'live reports publish figures/readme only')
    release_path = contained(rev, 'results/release.json')
    release = read(release_path)
    require(release.get('status') == 'approved' and release.get('agent_name') == 'reviewer', 'publication needs reviewer approval')
    require(release.get('agent_id') and release.get('publication_status'), 'publication needs actual reviewer context and status')
    require(release.get('validation_receipt') and release.get('signoff_receipts'), 'publication needs validation/signoff receipt pins')
    for pin in [release['validation_receipt'], *release['signoff_receipts']]:
        evidence = contained(rev, pin['path'])
        require(evidence.is_file() and sha(evidence) == pin['sha256'], 'released validation/signoff evidence changed')
    require(release.get('milestone_id') == args.milestone and release.get('revision_id') == args.revision and
            release.get('report_id') == args.report_id, 'release identity mismatch')
    milestone = None if args.baseline or args.report_id else rev.parent.parent
    control = {} if milestone is None else read(milestone / 'milestone.json')
    frozen = control.get('status') == 'achieved'
    if frozen:
        pin = control.get('release_receipt') or {}
        require(control.get('achieved_revision') == args.revision and pin.get('sha256') == sha(release_path), 'achieved milestone accepts only its pinned winning release')
    chosen = release.get('files', {}).get(args.publish_kind, [])
    require(chosen, 'release has no files for this publication')
    output = contained(args.work_dir, args.output) if args.output else rev / 'results' / ('publication-' + args.publish_kind + '.json')
    require(not output.exists(), 'publication receipt exists; use a versioned --output for a follow-up')
    plans = []
    views = {'root': {'base': args.work_dir, 'files': []}}
    root_pointer = args.work_dir / '.local/views' / ('root-' + args.publish_kind + '.json')
    working_root = root_pointer.exists() and read(root_pointer).get('mode') == 'working'
    if working_root:
        views.pop('root')
    if milestone and args.publish_kind != 'readme':
        views[args.milestone] = {'base': milestone, 'files': []}
    for item in chosen:
        source = contained(rev, item['path'])
        require(source.is_file() and not source.is_symlink() and sha(source) == item['sha256'], 'released file changed')
        if args.publish_kind == 'readme':
            require(item['path'] == 'results/readme.md' and len(chosen) == 1, 'readme release must name results/readme.md')
            relative = Path('readme.md')
            destinations = [] if working_root else [args.work_dir / relative]
        else:
            relative = Path(item['path'])
            require(relative.parts[0] == args.publish_kind, 'released file is outside the selected view')
            if args.publish_kind == 'artifacts':
                artifact_path(relative)
            if args.publish_kind == 'figures':
                require(source.suffix == '.svg', 'figure views must be SVG')
            destinations = ([] if working_root else [args.work_dir / relative]) + ([milestone / relative] if milestone else [])
        for destination in destinations:
            if frozen and destination.is_relative_to(milestone) and destination.exists():
                require(sha(destination) == item['sha256'], 'cannot modify an achieved milestone snapshot')
            plans.append((source, destination, item['sha256']))
            scope = 'root' if destination == args.work_dir / relative else args.milestone
            views[scope]['files'].append({'path': str(relative), 'sha256': item['sha256']})
    if args.publish_kind == 'artifacts':
        reviewed_archive = seal_reviewed(args.work_dir, 'artifacts', rev, release_path, chosen)
        publish_artifact_views(args.work_dir, views, plans, sha(release_path), frozen)
        receipt = {'schema': 'flagmega.publication/v1', 'agent_name': agent(args), 'kind': 'artifacts',
                   'milestone_id': args.milestone, 'revision_id': args.revision, 'report_id': args.report_id,
                   'release_sha256': sha(release_path), 'publication_status': release['publication_status'],
                   'published_at_utc': utc(), 'files': chosen, 'root_view_updated': not working_root,
                   'reviewed_archive': str(reviewed_archive.relative_to(args.work_dir)),
                   'root_update_note': 'working view remains authoritative' if working_root else 'reviewed root view'}
        new_file(output, json_text(receipt))
        return output
    # Retire only unchanged files managed by an earlier publication, preserving revision history.
    stale = []
    for scope, view in views.items():
        pointer = args.work_dir / '.local/views' / (scope + '-' + args.publish_kind + '.json')
        view['pointer'] = pointer
        previous = read(pointer)['files'] if pointer.exists() else []
        current_paths = {item['path'] for item in view['files']}
        if args.report_id and args.publish_kind == 'figures':
            # A live history refresh updates a subset; retain the other reviewed root figures.
            for item in previous:
                if item['path'] not in current_paths:
                    retained = contained(view['base'], item['path'])
                    require(retained.is_file() and sha(retained) == item['sha256'], 'retained root figure changed; review it before merging live reports')
                    view['files'].append(item)
                    current_paths.add(item['path'])
        for item in previous:
            if item['path'] in current_paths:
                continue
            prior = contained(view['base'], item['path'])
            if prior.exists():
                require(not (frozen and scope != 'root'), 'cannot retire files in an achieved milestone')
                require(prior.is_file() and sha(prior) == item['sha256'], 'previously published file was edited; retain it for review')
                stale.append(prior)
    staged = []
    try:
        for source, destination, expected_sha in plans:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.publishing-', delete=False) as stream:
                temporary = Path(stream.name)
            staged.append((temporary, destination))
            shutil.copyfile(source, temporary)
            require(sha(temporary) == expected_sha, 'source changed while preparing publication')
        reviewed_archive = seal_reviewed(args.work_dir, args.publish_kind, rev, release_path, chosen)
        for temporary, destination in staged:
            temporary.replace(destination)
        for prior in stale:
            prior.unlink()
        for scope, view in views.items():
            data = {'schema': 'flagmega.view/v1', 'agent_name': agent(args), 'kind': args.publish_kind,
                    'scope': scope, 'milestone_id': args.milestone, 'revision_id': args.revision, 'report_id': args.report_id,
                    'release_sha256': sha(release_path), 'files': view['files']}
            view['pointer'].parent.mkdir(parents=True, exist_ok=True)
            temporary = view['pointer'].with_suffix('.json.updating')
            temporary.write_text(json_text(data))
            temporary.replace(view['pointer'])
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    receipt = {'schema': 'flagmega.publication/v1', 'agent_name': agent(args), 'kind': args.publish_kind,
               'milestone_id': args.milestone, 'revision_id': args.revision, 'report_id': args.report_id,
               'release_sha256': sha(release_path), 'publication_status': release['publication_status'],
               'published_at_utc': utc(), 'files': chosen, 'root_view_updated': not working_root,
               'reviewed_archive': str(reviewed_archive.relative_to(args.work_dir)),
               'root_update_note': 'working view remains authoritative' if working_root else 'legacy reviewed root view'}
    new_file(output, json_text(receipt))
    return output


def publish_artifact_views(work, views, plans, release_hash, frozen):
    """Reviewed active views obey the same layout; superseded trees stay local."""
    for scope, view in views.items():
        pointer = working_path(work, '.local/views/' + scope + '-artifacts.json')
        base = view['base']
        root = base / 'artifacts'
        if frozen and scope != 'root':
            require(artifact_tree_matches(root, view['files']), 'cannot modify an achieved milestone snapshot')
            continue
        archive = working_path(work, '.local/working/artifacts/reviewed-' + release_hash + '-' + scope)
        if archive.exists():
            require(artifact_tree_matches(root, view['files']), 'reviewed artifact view changed after installation')
            continue
        archive.mkdir(parents=True)
        current, retired = archive / 'current/artifacts', archive / 'retired/artifacts'
        pointer_before = pointer.read_bytes() if pointer.exists() else None
        replaced = False
        temporary = archive / 'pointer.updating'
        try:
            for kind in ('generated', 'ir'):
                (current / kind).mkdir(parents=True)
            for source, destination, expected_hash in plans:
                if not destination.is_relative_to(base / 'artifacts'):
                    continue
                target = current / destination.relative_to(base / 'artifacts')
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                require(sha(target) == expected_hash, 'released source changed while preparing publication')
            retired.parent.mkdir(parents=True)
            if root.exists():
                root.replace(retired)
            try:
                current.replace(root)
            except BaseException:
                if retired.exists():
                    retired.replace(root)
                raise
            replaced = True
            data = {'schema': 'flagmega.view/v1', 'agent_name': 'graph optimizer', 'kind': 'artifacts',
                    'scope': scope, 'release_sha256': release_hash, 'files': view['files']}
            pointer.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json_text(data))
            temporary.replace(pointer)
            (archive / 'manifest.json').write_text(json_text(data))
            shutil.rmtree(archive / 'current')
        except BaseException:
            if replaced:
                shutil.rmtree(root)
                if retired.exists():
                    retired.replace(root)
            if pointer_before is None:
                pointer.unlink(missing_ok=True)
            else:
                pointer.write_bytes(pointer_before)
            shutil.rmtree(archive)
            raise


def snapshot(args):
    require(agent(args) == 'reviewer', 'reviewer freezes tracking phase snapshots')
    rev = revision_dir(args)
    sequence = identifier(args.sequence)
    tracking = rev / 'tracking.md'
    text = tracking.read_text()
    require('## 4. ' in text, 'tracking has no signature section boundary')
    body = text.split('## 4. ', 1)[0]
    destination = rev / 'results/tracking-snapshots' / (sequence + '.md')
    new_file(destination, body)
    new_file(destination.with_suffix('.json'), json_text({
        'schema': 'flagmega.tracking-snapshot/v1', 'agent_name': 'reviewer',
        'agent_id': args.agent_id, 'sequence': sequence,
        'milestone_id': args.milestone, 'revision_id': args.revision,
        'tracking_sha256_at_capture': sha(tracking), 'snapshot_sha256': sha(destination),
        'created_at_utc': utc()}))
    return destination


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('action', choices=['init', 'prepare', 'manifest', 'run', 'append-measurement', 'sync', 'watch', 'refresh', 'publish', 'snapshot', 'catalog'])
    result.add_argument('--work-dir', type=Path)
    result.add_argument('--agent', default='graph_optimizer')
    result.add_argument('--agent-id')
    result.add_argument('--run-id')
    result.add_argument('--milestone')
    result.add_argument('--revision')
    result.add_argument('--baseline', action='store_true')
    result.add_argument('--report-id', help='live figure/readme packet under .local/reports; never changes frozen revisions')
    result.add_argument('--kind', choices=sorted(set(CATALOG['entries']) | {'artifacts', 'native', 'cache', 'raw', 'figures'}),
                        help='delivery for prepare/catalog; payload for manifest; view for sync/publish')
    result.add_argument('--plan-owner', choices=['graph expert', 'kernel expert'], default='graph expert')
    result.add_argument('--recipient')
    result.add_argument('--assignment')
    result.add_argument('--sequence')
    result.add_argument('--capture')
    result.add_argument('--cache-kind', choices=['weights', 'kernels', 'comparison'], default='weights')
    result.add_argument('--directory')
    result.add_argument('--output')
    result.add_argument('--inputs', type=Path)
    result.add_argument('--manifest-kind', choices=['artifacts', 'native', 'cache', 'raw'], default='artifacts')
    result.add_argument('--job', type=Path)
    result.add_argument('--record', type=Path)
    result.add_argument('--watch-interval', type=float, default=2.0, help='watch polling interval in seconds')
    result.add_argument('--watch-timeout', type=float, help='optional bounded watcher lifetime in seconds')
    result.add_argument('--max-updates', type=int, help='stop after this many successful state updates')
    result.add_argument('--stop-file', help='workspace-relative file whose presence stops the watcher')
    result.add_argument('--service-record', help='workspace-relative owned service PID/start/status receipt')
    result.add_argument('--publish-kind', choices=['artifacts', 'figures', 'readme'])
    return result


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        if args.action == 'catalog':
            print(json_text(CATALOG['entries'][args.kind] if args.kind else {
                'agents': AGENTS, 'delivery_kinds': sorted(CATALOG['entries'])}))
            return 0
        require(args.work_dir is not None, '--work-dir is required')
        args.work_dir = args.work_dir.resolve()
        if args.action == 'prepare':
            require(args.kind, 'prepare requires --kind')
            path = prepare(args)
        elif args.action == 'init':
            path = initialize(args)
        elif args.action == 'manifest':
            require(args.directory and args.output, 'manifest requires --directory and --output')
            if args.kind:
                require(args.kind in {'artifacts', 'native', 'cache', 'raw'}, 'invalid manifest kind')
                args.manifest_kind = args.kind
            path = manifest(args)
        elif args.action == 'run':
            require(args.job, 'run requires --job')
            path, code = run_job(args)
            print(path)
            return code
        elif args.action == 'snapshot':
            path = snapshot(args)
        elif args.action == 'append-measurement':
            require(args.record, 'append-measurement requires --record')
            path = append_measurement(args)
        elif args.action == 'sync':
            require(args.record, 'sync requires --record with pinned files/status and base view hash')
            path = sync(args)
        elif args.action in {'watch', 'refresh'}:
            require(args.record, 'watch/refresh requires a pinned working-watch spec')
            path = watch(args, once=args.action == 'refresh')
        else:
            args.publish_kind = args.kind or args.publish_kind
            require(args.publish_kind in {'artifacts', 'figures', 'readme'}, 'publish requires --kind artifacts/figures/readme')
            path = publish(args)
        print(path)
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        cli.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    sys.exit(main())
