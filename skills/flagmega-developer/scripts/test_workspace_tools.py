"""Exercise real file/command/provenance behavior in temporary workspaces (no GPU)."""
import json
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import workspace as ws
import dashboard
import render_workspace_figures as figures


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name).resolve()

    def args(self, agent='graph_optimizer', **values):
        args = ws.parser().parse_args(['prepare', '--work-dir', str(self.work), '--agent', agent])
        for name, value in values.items():
            setattr(args, name, value)
        return args

    def revision(self, owner='graph expert'):
        ws.initialize(self.args())
        ws.prepare(self.args('orchestrator', kind='milestone', milestone='m000'))
        args = self.args('orchestrator', kind='revision', milestone='m000', revision='r000', plan_owner=owner)
        ws.prepare(args)
        return ws.revision_dir(args)

    def write(self, path, body):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(ws.json_text(body))
        return path

    def test_artifact_layout_rejects_legacy_destination_before_commit(self):
        args = self.working_request('legacy-path')
        record = ws.read(args.record)
        record['files'][0]['path'] = 'artifacts/old-candidate/source.py'
        args.record.write_text(ws.json_text(record))
        with self.assertRaisesRegex(ValueError, 'generated/ir'):
            ws.sync(args)
        self.assertFalse((self.work / '.local/views/root-artifacts.json').exists())

    def test_artifact_layout_archives_unmanaged_history_and_refresh_keeps_two_directories(self):
        args = self.working_request('layout-first')
        record = ws.read(args.record)
        record['files'][0]['path'] = 'artifacts/ir/full-model/model.il'
        args.record.write_text(ws.json_text(record))
        old = self.work / 'artifacts/old-candidate/helper.py'
        old.parent.mkdir(parents=True)
        old.write_text('historical helper fixture')
        first = ws.sync(args)
        self.assertEqual(sorted(p.name for p in (self.work / 'artifacts').iterdir()), ['generated', 'ir'])
        self.assertEqual((first.parent / 'retired/artifacts/old-candidate/helper.py').read_text(), old.read_text() if old.exists() else 'historical helper fixture')
        pointer = self.work / '.local/views/root-artifacts.json'
        active = self.work / record['files'][0]['path']
        original_inode = active.stat().st_ino
        second = ws.sync(self.working_request('layout-status', files=[], retain_previous=True,
                                              status={'build': 'completed', 'review': 'passed'}))
        self.assertFalse((second.parent / 'files').exists(), 'unchanged payload must not create another snapshot')
        self.assertEqual(active.stat().st_ino, original_inode)
        self.assertEqual(ws.read(pointer)['payload_archive'], str(first.parent.relative_to(self.work)))
        self.assertEqual(sorted(p.name for p in (self.work / 'artifacts').iterdir()), ['generated', 'ir'])

    def test_artifact_manifest_output_is_external(self):
        ws.initialize(self.args())
        source = self.work / 'artifacts/generated/model.py'
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text('emitted fixture')
        with self.assertRaisesRegex(ValueError, 'outside artifacts'):
            ws.manifest(self.args(directory='artifacts', output='artifacts/manifest.json'))

    def test_reviewed_artifact_publication_rejects_legacy_manifest(self):
        rev, source, release = self.release()
        metadata = self.write(rev / 'artifacts/manifest.json', {'historical': True})
        approval = ws.read(release)
        approval['files']['artifacts'].append({'path': 'artifacts/manifest.json', 'sha256': ws.sha(metadata)})
        release.write_text(ws.json_text(approval))
        with self.assertRaisesRegex(ValueError, 'generated/ir'):
            ws.publish(self.args(milestone='m000', revision='r000', publish_kind='artifacts'))
        self.assertFalse((self.work / '.local/views/root-artifacts.json').exists())

    def test_build_exporter_keeps_runtime_ir_outside_generated(self):
        path = ws.ROOT.parents[1] / 'python/tutorials/flagmega/01-qwen3-1.7b-bf16/nvidia-h800/benchmark/build_artifacts.py'
        spec = importlib.util.spec_from_file_location('build_artifacts_fixture', path)
        exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exporter)
        source = self.work / '.local/runtime'
        for name, text in [('program_sources/model.py', 'actual emitted fixture'), ('ir/final.py', 'IR fixture'),
                           ('assets/weights.py', 'packed data fixture'), ('artifact.json', '{}')]:
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        records = []
        exporter.export_sources(source, self.work / 'artifacts/generated', records)
        self.assertEqual([str(p.relative_to(self.work / 'artifacts')) for p in (self.work / 'artifacts').rglob('*') if p.is_file()],
                         ['generated/program_sources/model.py'])
        self.assertEqual(records[0]['sha256'], ws.sha(source / 'program_sources/model.py'))


    def test_flow_owner_cannot_run_or_create_technical_deliveries(self):
        with self.assertRaises(ValueError):
            ws.initialize(self.args('orchestrator'))
        with self.assertRaises(ValueError):
            ws.prepare(self.args('graph_optimizer', kind='state'))
        with self.assertRaises(ValueError):
            ws.run_job(self.args('orchestrator', job=self.work / 'missing.json'))
        self.assertFalse(list(self.work.iterdir()))

    def test_revision_plan_owner_tracking_and_relocated_template_links(self):
        self.revision('kernel expert')
        common = dict(milestone='m000', revision='r000')
        with self.assertRaises(ValueError):
            ws.prepare(self.args('graph_expert', kind='plan', **common))
        plan = ws.prepare(self.args('kernel_expert', kind='plan', **common))
        tracking = ws.prepare(self.args('reviewer', kind='tracking', **common))
        for doc in (plan, tracking):
            for target in re.findall(r'\]\(([^)]+)\)', doc.read_text()):
                self.assertTrue((doc.parent / target.split('#')[0]).resolve().exists(), target)
        handoff = ws.prepare(self.args('kernel_expert', kind='handoff', assignment='a001', **common))
        self.assertEqual(ws.read(handoff)['signoffs'], [])
        self.assertEqual(ws.read(handoff)['status'], 'pending')

    def test_baseline_assignment_path_alias_and_actor_scope(self):
        ws.initialize(self.args())
        output = ws.prepare(self.args('reviwer', kind='handoff', assignment='a001', baseline=True))
        self.assertEqual(output, self.work / '.local/agents/reviewer/a001/handoff.json')
        self.assertEqual(ws.read(output)['agent_name'], 'reviewer')
        with self.assertRaises(ValueError):
            ws.prepare(self.args('graph_optimizer', kind='handoff', assignment='a002', recipient='kernel_profiler'))
        self.assertFalse((self.work / '.local/agents/kernel_profiler/a002/handoff.json').exists())

    def test_multiple_jobs_use_versioned_owned_outputs(self):
        self.revision()
        common = dict(kind='job', milestone='m000', revision='r000')
        first = ws.prepare(self.args('graph_optimizer', output='build-job.json', **common))
        second = ws.prepare(self.args('graph_optimizer', output='rebuild-job.json', **common))
        self.assertNotEqual(first, second)
        self.assertTrue(first.is_file() and second.is_file())
        with self.assertRaises(ValueError):
            ws.prepare(self.args('graph_optimizer', output='../../../../outside-job.json', **common))

    def test_snapshot_excludes_signatures_and_never_overwrites(self):
        rev = self.revision()
        common = dict(milestone='m000', revision='r000')
        ws.prepare(self.args('reviewer', kind='tracking', **common))
        args = self.args('reviewer', sequence='000', **common)
        path = ws.snapshot(args)
        original = path.read_bytes()
        self.assertNotIn('## 4.', path.read_text())
        with (rev / 'tracking.md').open('a') as stream:
            stream.write('\nNew signature receipt\n')
        self.assertEqual(path.read_bytes(), original)
        with self.assertRaises(FileExistsError):
            ws.snapshot(args)
        self.assertEqual(ws.read(path.with_suffix('.json'))['snapshot_sha256'], ws.sha(path))

    def test_artifact_manifest_rejects_weights_and_hashes_actual_files(self):
        rev = self.revision()
        artifacts = rev / 'artifacts'
        (artifacts / 'ir/model.il').write_text('real text fixture')
        forbidden = artifacts / 'weights.safetensors'
        forbidden.write_bytes(b'weight fixture')
        args = self.args(directory=str(artifacts.relative_to(self.work)), output=str((rev / 'results/artifact-manifest.json').relative_to(self.work)))
        with self.assertRaises(ValueError):
            ws.manifest(args)
        self.assertFalse((artifacts / 'manifest.json').exists())
        forbidden.unlink()
        result = ws.manifest(args)
        row = ws.read(result)['files'][0]
        self.assertEqual(row['sha256'], ws.sha(artifacts / 'ir/model.il'))
        self.assertEqual(row['size_bytes'], (artifacts / 'ir/model.il').stat().st_size)

    def test_command_exit_and_missing_outputs_have_real_retained_receipts(self):
        spec = {'job_id': 'j001', 'agent_name': 'graph profiler', 'argv': [sys.executable, '-c', 'print("actual stdout")'], 'cwd': '.', 'expected_outputs': [{'path': '.local/missing.raw'}]}
        job = self.write(self.work / 'job.json', spec)
        path, code = ws.run_job(self.args('graph_profiler', job=job))
        receipt = ws.read(path)
        self.assertEqual(receipt['exit_code'], 0)
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(code, 1)
        self.assertIn('actual stdout', (path.parent / 'stdout.log').read_text())
        with self.assertRaises(FileExistsError):
            ws.run_job(self.args('graph_profiler', job=job))
        spec.update(job_id='j002', argv=[sys.executable, '-c', 'raise SystemExit(7)'], expected_outputs=[])
        self.write(job, spec)
        path, code = ws.run_job(self.args('graph_profiler', job=job))
        self.assertEqual(ws.read(path)['exit_code'], 7)
        self.assertEqual(code, 1)

    def test_tracked_renderer_wrapper_runs_the_fixed_renderer(self):
        ws.initialize(self.args())
        script = ws.prepare(self.args('graph_profiler', kind='benchmark-render'))
        output = self.work / 'figures/example.svg'
        result = subprocess.run([sys.executable, '-B', str(script), '--renderer', 'graph',
                                 '--skill-dir', str(ws.ROOT), str(ws.ROOT / 'assets/workspace-figures/history.json'),
                                 '--output', str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        ET.fromstring(output.read_text())
        self.assertIn('SYNTHETIC EXAMPLE', output.read_text())

    def test_real_entrypoint_binding_forwards_args_and_exit_code(self):
        ws.initialize(self.args())
        script = ws.prepare(self.args('graph_profiler', kind='benchmark-model'))
        (self.work / 'fixture_binding.py').write_text('def run(args):\n    print(args)\n    return 6\n')
        result = subprocess.run([sys.executable, '-B', str(script), '--binding', 'fixture_binding:run', '--batch', '2'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 6)
        self.assertIn("['--batch', '2']", result.stdout)
        unbound = subprocess.run([sys.executable, '-B', str(script)], capture_output=True, text=True)
        self.assertNotEqual(unbound.returncode, 0)

    def test_ledger_duplicate_or_truncated_prefix_never_changes_history(self):
        record = self.write(self.work / 'event.json', {'measurement_id': 'q001', 'agent_name': 'reviewer', 'validity': 'invalid', 'reason': 'fixture execution failure'})
        ledger = ws.append_measurement(self.args('reviewer', record=record))
        original = ledger.read_bytes()
        with self.assertRaises(ValueError):
            ws.append_measurement(self.args('reviewer', record=record))
        self.assertEqual(ledger.read_bytes(), original)
        ledger.write_text('{"truncated":')
        with self.assertRaises(ValueError):
            ws.append_measurement(self.args('reviewer', record=record))
        self.assertEqual(ledger.read_text(), '{"truncated":')

    def release(self):
        rev = self.revision()
        source = rev / 'artifacts/ir/model.il'
        source.write_text('immutable selected IR')
        release = {'status': 'approved', 'agent_name': 'reviewer', 'agent_id': 'fixture-only-reviewer', 'milestone_id': 'm000', 'revision_id': 'r000', 'publication_status': 'no_gain', 'files': {'artifacts': [{'path': 'artifacts/ir/model.il', 'sha256': ws.sha(source)}]}}
        validation = self.write(rev / 'results/validation.json', {'fixture': True, 'status': 'no_gain'})
        signoff = self.write(rev / 'results/reviewer-signoff.json', {'fixture': True, 'publication_review': 'approved'})
        release['validation_receipt'] = {'path': 'results/validation.json', 'sha256': ws.sha(validation)}
        release['signoff_receipts'] = [{'path': 'results/reviewer-signoff.json', 'sha256': ws.sha(signoff)}]
        path = self.write(rev / 'results/release.json', release)
        return rev, source, path

    def test_baseline_publication_does_not_require_or_create_milestone(self):
        ws.initialize(self.args())
        base = self.work / '.local/baseline'
        source = base / 'artifacts/ir/model.il'
        source.parent.mkdir(parents=True)
        source.write_text('baseline fixture IR')
        validation = self.write(base / 'results/validation.json', {'fixture': True})
        signoff = self.write(base / 'results/signoff.json', {'fixture': True})
        self.write(base / 'results/release.json', {
            'status': 'approved', 'agent_name': 'reviewer', 'agent_id': 'fixture-only',
            'milestone_id': None, 'revision_id': None, 'publication_status': 'baseline',
            'validation_receipt': {'path': 'results/validation.json', 'sha256': ws.sha(validation)},
            'signoff_receipts': [{'path': 'results/signoff.json', 'sha256': ws.sha(signoff)}],
            'files': {'artifacts': [{'path': 'artifacts/ir/model.il', 'sha256': ws.sha(source)}]}})
        ws.publish(self.args(baseline=True, publish_kind='artifacts'))
        self.assertEqual((self.work / 'artifacts/ir/model.il').read_text(), 'baseline fixture IR')
        self.assertFalse(list((self.work / '.local/milestones').iterdir()))

    def test_latest_view_removes_prior_published_files_and_preserves_history(self):
        rev, source, release = self.release()
        ws.publish(self.args(milestone='m000', revision='r000', publish_kind='artifacts'))
        unrelated = self.work / 'artifacts/ir/manual.il'
        unrelated.write_text('unrelated user source')
        ws.prepare(self.args('orchestrator', kind='revision', milestone='m000', revision='r001'))
        new_rev = rev.parent / 'r001'
        new_source = new_rev / 'artifacts/ir/new-model.il'
        new_source.write_text('new revision fixture')
        approval = ws.read(release)
        approval['revision_id'] = 'r001'
        approval['files']['artifacts'] = [{'path': 'artifacts/ir/new-model.il', 'sha256': ws.sha(new_source)}]
        for pin in [approval['validation_receipt'], *approval['signoff_receipts']]:
            self.write(new_rev / pin['path'], ws.read(rev / pin['path']))
        self.write(new_rev / 'results/release.json', approval)
        ws.publish(self.args(milestone='m000', revision='r001', publish_kind='artifacts'))
        self.assertFalse((self.work / 'artifacts/ir/model.il').exists())
        self.assertFalse((rev.parent.parent / 'artifacts/ir/model.il').exists())
        self.assertTrue(source.exists())
        self.assertFalse(unrelated.exists())
        self.assertTrue(any(p.read_text() == 'unrelated user source' for p in (self.work / '.local/working/artifacts').rglob('manual.il')))
        self.assertEqual((self.work / 'artifacts/ir/new-model.il').read_text(), new_source.read_text())

    def test_live_report_updates_root_after_closure_without_touching_snapshot(self):
        rev, source, release = self.release()
        archived = rev.parent.parent / 'figures/milestones.svg'
        archived.parent.mkdir(exist_ok=True)
        archived.write_text('frozen snapshot')
        graph = rev / 'figures/graph.svg'
        graph.parent.mkdir(exist_ok=True)
        graph.write_text('retained graph fixture')
        approval = ws.read(release)
        approval['files']['figures'] = [{'path': 'figures/graph.svg', 'sha256': ws.sha(graph)}]
        self.write(release, approval)
        ws.publish(self.args('graph_profiler', milestone='m000', revision='r000', publish_kind='figures'))
        control_path = rev.parent.parent / 'milestone.json'
        control = ws.read(control_path)
        control.update(status='achieved', achieved_revision='r000')
        self.write(control_path, control)
        report_args = dict(report_id='after-m000')
        packet = ws.revision_dir(self.args('graph_profiler', **report_args))
        live_svg = packet / 'figures/milestones.svg'
        live_svg.parent.mkdir(parents=True)
        live_svg.write_text('updated live chart fixture')
        validation = self.write(packet / 'results/validation.json', {'fixture': True})
        signoff = self.write(packet / 'results/signoff.json', {'fixture': True})
        self.write(packet / 'results/release.json', {
            'status': 'approved', 'agent_name': 'reviewer', 'agent_id': 'fixture-only',
            'milestone_id': None, 'revision_id': None, 'report_id': 'after-m000', 'publication_status': 'achieved',
            'validation_receipt': {'path': 'results/validation.json', 'sha256': ws.sha(validation)},
            'signoff_receipts': [{'path': 'results/signoff.json', 'sha256': ws.sha(signoff)}],
            'files': {'figures': [{'path': 'figures/milestones.svg', 'sha256': ws.sha(live_svg)}]}})
        ws.publish(self.args('graph_profiler', publish_kind='figures', **report_args))
        self.assertEqual((self.work / 'figures/milestones.svg').read_text(), 'updated live chart fixture')
        self.assertEqual(archived.read_text(), 'frozen snapshot')
        self.assertEqual((self.work / 'figures/graph.svg').read_text(), 'retained graph fixture')
        self.assertEqual(ws.read(control_path), control)
        with self.assertRaises(ValueError):
            ws.publish(self.args(publish_kind='artifacts', **report_args))

    def test_documented_manifest_and_publish_kind_flags_execute(self):
        rev, source, release = self.release()
        script = str(ws.ROOT / 'scripts/workspace.py')
        output = rev / 'results/artifact-manifest.json'
        result = subprocess.run([sys.executable, '-B', script, 'manifest', '--work-dir', str(self.work),
                                 '--agent', 'graph_optimizer', '--kind', 'artifacts',
                                 '--directory', str((rev / 'artifacts').relative_to(self.work)),
                                 '--output', str(output.relative_to(self.work))], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, '-B', script, 'publish', '--work-dir', str(self.work),
                                 '--agent', 'graph_optimizer', '--kind', 'artifacts',
                                 '--milestone', 'm000', '--revision', 'r000'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.work / 'artifacts/ir/model.il').read_text(), source.read_text())

    def test_release_pins_checked_before_copy_and_flow_never_promoted(self):
        rev, source, release = self.release()
        args = self.args(milestone='m000', revision='r000', publish_kind='artifacts')
        original = source.read_text()
        source.write_text('changed input')
        with self.assertRaises(ValueError):
            ws.publish(args)
        self.assertFalse((self.work / 'artifacts/ir/model.il').exists())
        source.write_text(original)
        result = ws.publish(args)
        self.assertEqual((self.work / 'artifacts/ir/model.il').read_text(), original)
        self.assertEqual(ws.read(result)['publication_status'], 'no_gain')
        self.assertEqual(ws.read(rev.parent.parent / 'milestone.json')['status'], 'open')
        self.assertEqual(ws.read(rev.parent.parent / 'milestone.json')['achieved_revision'], None)
        source.write_text('later mutation')
        with self.assertRaises(ValueError):
            ws.publish(args)
        self.assertEqual((self.work / 'artifacts/ir/model.il').read_text(), original)


    def dashboard(self, status='进行中', description='actual fixture work'):
        text = '| ' + ' | '.join(dashboard.COLUMNS) + ' |\n'
        text += '| ' + ' | '.join(['---'] * len(dashboard.COLUMNS)) + ' |\n'
        text += '| W001 | 2026-11-14 09:39:37 | ' + status + ' | 2 | ' + description + ' | graph optimizer | [证据](.local/evidence.json) | 下一步 |\n'
        path = self.work / 'dashboard.md'
        temporary = path.with_suffix('.updating')
        temporary.write_text(text)
        temporary.replace(path)
        return path

    def working_request(self, sync_id='w000', kind='artifacts', files=None, **changes):
        owner = 'graph optimizer' if kind == 'artifacts' else 'graph profiler'
        evidence = self.work / 'dashboard.md'
        if not evidence.exists():
            self.dashboard('待办')
        if files is None:
            suffix = '.il' if kind == 'artifacts' else '.svg' if kind == 'figures' else '.md'
            source = self.work / '.local/produced' / ('source' + suffix)
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text('actual source fixture')
            target = ('artifacts/ir/model' + suffix if kind == 'artifacts' else kind + '/model' + suffix) if kind != 'readme' else 'readme.md'
            files = [{'source': str(source.relative_to(self.work)), 'path': target, 'sha256': ws.sha(source)}]
        pointer = self.work / '.local/views' / ('root-' + kind + '.json')
        record = {'schema': 'flagmega.working-sync/v1', 'sync_id': sync_id, 'kind': kind,
                  'agent_name': owner, 'agent_id': 'fixture-context', 'source_id': 'b001',
                  'status': {'build': 'completed', 'review': 'pending'},
                  'base_view_sha256': ws.sha(pointer) if pointer.exists() else None,
                  'input_pins': [{'path': 'dashboard.md', 'sha256': ws.sha(evidence)}], 'files': files}
        record.update(changes)
        path = self.write(self.work / '.local/requests' / (sync_id + '.json'), record)
        return self.args(owner.replace(' ', '_'), kind=kind, record=path)

    def test_working_sync_no_release_pending_failed_and_immutable_history(self):
        args = self.working_request()
        receipt = ws.sync(args)
        pointer = self.work / '.local/views/root-artifacts.json'
        archive = receipt.parent
        archived = {str(p.relative_to(archive)): p.read_bytes() for p in archive.rglob('*') if p.is_file()}
        self.assertEqual(ws.read(pointer)['status']['review'], 'pending')
        self.assertFalse((self.work / '.local/baseline/results/release.json').exists())
        failed = self.working_request('w001', files=[], retain_previous=True, source_id='b002-attempt',
                                      status={'build': 'failed', 'review': 'unreviewed'})
        ws.sync(failed)
        current = ws.read(pointer)
        self.assertEqual(current['source_id'], 'b002-attempt')
        self.assertEqual(current['files'][0]['source_id'], 'b001')
        self.assertEqual((self.work / 'artifacts/ir/model.il').read_text(), 'actual source fixture')
        self.assertEqual(current['status']['build'], 'failed')
        self.assertEqual(archived, {str(p.relative_to(archive)): p.read_bytes() for p in archive.rglob('*') if p.is_file()})
        with self.assertRaises(ValueError):
            ws.sync(args)

    def test_working_status_frames_share_only_immutable_payload_blocks(self):
        first = ws.sync(self.working_request())
        original = first.parent / 'files/artifacts/ir/model.il'
        first_metadata = (first.parent / 'manifest.json').read_bytes()
        second = ws.sync(self.working_request('same-payload', files=[], retain_previous=True,
                                              status={'build': 'completed', 'review': 'passed'}))
        self.assertFalse((second.parent / 'files').exists())
        self.assertEqual(ws.read(second)['payload_archive'], str(first.parent.relative_to(self.work)))
        active = self.work / 'artifacts/ir/model.il'
        source = self.work / '.local/produced/source.il'
        self.assertNotEqual(original.stat().st_ino, active.stat().st_ino)
        self.assertNotEqual(original.stat().st_ino, source.stat().st_ino)
        self.assertEqual((first.parent / 'manifest.json').read_bytes(), first_metadata)
        source.write_text('new actual payload')
        third = ws.sync(self.working_request('changed-payload', files=[{
            'source': str(source.relative_to(self.work)), 'path': 'artifacts/ir/model.il',
            'sha256': ws.sha(source)}]))
        changed = third.parent / 'files/artifacts/ir/model.il'
        self.assertNotEqual(changed.stat().st_ino, original.stat().st_ino)
        self.assertEqual(original.read_text(), 'actual source fixture')
        self.assertEqual(original.read_text(), 'actual source fixture')

    def test_working_sync_rejects_stale_state_source_and_base_pins(self):
        args = self.working_request()
        state = self.work / 'dashboard.md'
        original = state.read_bytes()
        self.dashboard('进行中')
        with self.assertRaisesRegex(ValueError, 'input pin changed'):
            ws.sync(args)
        state.write_bytes(original)
        source = self.work / ws.read(args.record)['files'][0]['source']
        source.write_text('changed produced source')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            ws.sync(args)
        source.write_text('actual source fixture')
        stale = self.working_request('w001')
        ws.sync(args)
        before = (self.work / '.local/views/root-artifacts.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'stale update'):
            ws.sync(stale)
        self.assertEqual((self.work / '.local/views/root-artifacts.json').read_bytes(), before)

    def test_working_sync_protects_managed_manual_and_reserved_files(self):
        args = self.working_request()
        manual = self.work / 'artifacts/ir/model.il'
        manual.parent.mkdir(parents=True)
        manual.write_text('user file')
        with self.assertRaisesRegex(ValueError, 'unmanaged'):
            ws.sync(args)
        manual.unlink()
        ws.sync(args)
        manual.write_text('user edit')
        with self.assertRaisesRegex(ValueError, 'managed working file'):
            ws.sync(self.working_request('w001', files=[], retain_previous=True))
        manual.write_text('actual source fixture')
        reserved = ws.read(args.record)['files'][0]
        reserved['path'] = 'artifacts/manifest.json'
        with self.assertRaisesRegex(ValueError, 'generated/ir'):
            ws.sync(self.working_request('w002', files=[reserved]))

    def test_working_sync_filters_payload_owner_paths_and_structured_status(self):
        args = self.working_request()
        with self.assertRaises(ValueError):
            ws.sync(self.args('graph_profiler', kind='artifacts', record=args.record))
        for changes in ({'status': 'pending'}, {'retain_previous': 'yes'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ws.sync(self.working_request('bad', **changes))
        record = ws.read(args.record)
        weight = self.work / '.local/weights.safetensors'
        weight.write_text('weights')
        for source, target in ((str(weight.relative_to(self.work)), 'artifacts/weights.il'),
                               (record['files'][0]['source'], '../model.il')):
            with self.subTest(target=target), self.assertRaises(ValueError):
                ws.sync(self.working_request('bad-file', files=[{'source': source, 'path': target,
                                                             'sha256': ws.sha(self.work / source)}]))
        link = self.work / '.local/link.il'
        link.symlink_to(self.work / record['files'][0]['source'])
        with self.assertRaisesRegex(ValueError, 'symlink'):
            ws.sync(self.working_request('link', files=[{'source': '.local/link.il', 'path': 'artifacts/ir/link.il', 'sha256': ws.sha(link)}]))

    def test_working_sync_retires_only_managed_files(self):
        ws.sync(self.working_request())
        manual = self.work / 'artifacts/ir/manual.il'
        manual.write_text('manual')
        source = self.work / '.local/new.il'
        source.write_text('new produced source')
        ws.sync(self.working_request('w001', files=[{'source': '.local/new.il', 'path': 'artifacts/ir/new.il', 'sha256': ws.sha(source)}]))
        self.assertFalse((self.work / 'artifacts/ir/model.il').exists())
        self.assertFalse(manual.exists())
        self.assertEqual((self.work / '.local/working/artifacts/w001/retired/artifacts/ir/manual.il').read_text(), 'manual')

    def test_working_sync_late_commit_errors_restore_complete_frame_and_pointer(self):
        ws.sync(self.working_request())
        pointer = self.work / '.local/views/root-artifacts.json'
        frame = {row['path']: (self.work / row['path']).read_bytes() for row in ws.read(pointer)['managed_files']}
        old_pointer = pointer.read_bytes()
        source = self.work / '.local/new.il'
        source.write_text('new source')
        for failure in ('receipt', 'pointer'):
            args = self.working_request('late-' + failure, files=[{'source': '.local/new.il', 'path': 'artifacts/ir/new.il', 'sha256': ws.sha(source)}])
            original_new_file, original_replace = ws.new_file, Path.replace
            def fail_receipt(path, text):
                if path.name == 'receipt.json':
                    raise OSError('injected late receipt failure')
                return original_new_file(path, text)
            def fail_pointer(path, target):
                if Path(target) == pointer:
                    raise OSError('injected late pointer failure')
                return original_replace(path, target)
            patch = mock.patch.object(ws, 'new_file', side_effect=fail_receipt) if failure == 'receipt' else mock.patch.object(Path, 'replace', fail_pointer)
            with self.subTest(failure=failure), patch, self.assertRaisesRegex(OSError, 'injected late'):
                ws.sync(args)
            self.assertEqual(pointer.read_bytes(), old_pointer)
            self.assertEqual(frame, {name: (self.work / name).read_bytes() for name in frame})
            self.assertFalse((self.work / 'artifacts/ir/new.il').exists())
            self.assertFalse((self.work / '.local/working/artifacts' / ('late-' + failure)).exists())

    def test_older_reviewed_release_seals_history_without_rolling_back_working_root(self):
        rev, source, release = self.release()
        ws.sync(self.working_request(files=[{'source': str(source.relative_to(self.work)), 'path': 'artifacts/ir/model.il', 'sha256': ws.sha(source)}]))
        ws.sync(self.working_request('new-status', files=[], retain_previous=True, status={'build': 'completed', 'review': 'pending-new-attempt'}))
        pointer = self.work / '.local/views/root-artifacts.json'
        frame = pointer.read_bytes(), (self.work / '.local/views/root-artifacts.json').read_bytes(), (self.work / 'artifacts/ir/model.il').read_bytes()
        receipt = ws.publish(self.args(milestone='m000', revision='r000', publish_kind='artifacts'))
        self.assertFalse(ws.read(receipt)['root_view_updated'])
        archived = self.work / ws.read(receipt)['reviewed_archive']
        self.assertEqual((archived / 'files/artifacts/ir/model.il').read_bytes(), source.read_bytes())
        self.assertEqual(frame, (pointer.read_bytes(), (self.work / '.local/views/root-artifacts.json').read_bytes(), (self.work / 'artifacts/ir/model.il').read_bytes()))
        before = (archived / 'manifest.json').read_bytes()
        ws.publish(self.args(milestone='m000', revision='r000', publish_kind='artifacts', output=str((rev / 'results/publication-followup.json').relative_to(self.work))))
        self.assertEqual((archived / 'manifest.json').read_bytes(), before)
        self.assertEqual(pointer.read_bytes(), frame[0])

    def test_working_sync_figures_and_readme_metadata(self):
        for kind in ('figures', 'readme'):
            with self.subTest(kind=kind):
                receipt = ws.sync(self.working_request('working-' + kind, kind=kind))
                self.assertTrue(receipt.is_file())
                self.assertEqual(ws.read(self.work / ws.WORKING_METADATA[kind])['kind'], kind)

    def working_watch(self, watch_id='artifact-watch', **changes):
        args = self.working_request('watch-source')
        request = ws.read(args.record)
        source_manifest = self.write(self.work / '.local/produced/manifest.json', {'files': request['files']})
        spec = {'schema': 'flagmega.working-watch/v2', 'watch_id': watch_id, 'kind': 'artifacts',
                'agent_name': 'graph optimizer', 'agent_id': 'fixture-context', 'source_id': 'b001',
                'dashboard_path': 'dashboard.md', 'source_manifest': {'path': str(source_manifest.relative_to(self.work)),
                                                                       'sha256': ws.sha(source_manifest)},
                'files': request['files']}
        spec.update(changes)
        args.record = self.write(self.work / '.local/requests' / (watch_id + '.json'), spec)
        args.watch_interval = 0.01
        args.watch_timeout = 1
        return args

    def test_refresh_updates_current_state_without_heartbeat_history(self):
        args = self.working_watch()
        first = ws.watch(args, once=True)
        pointer = self.work / '.local/views/root-artifacts.json'
        old = pointer.read_bytes()
        self.assertEqual(ws.read(first)['status'], 'refreshed')
        second = ws.watch(args, once=True)
        self.assertEqual(ws.read(second)['status'], 'unchanged')
        self.assertEqual(pointer.read_bytes(), old)
        self.dashboard('失败', '真实失败')
        ws.watch(args, once=True)
        current = ws.read(self.work / '.local/views/root-artifacts.json')
        reference = current['status']['dashboard']
        self.assertNotIn('rows', reference)
        self.assertEqual(reference['sha256'], ws.sha(self.work / 'dashboard.md'))
        self.assertEqual(dashboard.load(self.work / reference['snapshot'])['rows'][0]['status'], '失败')
        self.assertEqual(current['files'][0]['source_id'], 'b001')
        snapshots = self.work / '.local/watchers/artifact-watch/dashboards'
        self.assertEqual(len(list(snapshots.glob('*.md'))), 2)
        self.assertEqual((self.work / 'artifacts/ir/model.il').read_text(), 'actual source fixture')

    def test_refresh_archives_forbidden_paths_even_without_dashboard_change(self):
        args = self.working_watch()
        ws.watch(args, once=True)
        old = self.work / 'artifacts/ir/old-candidate/index.json'
        old.parent.mkdir(parents=True)
        old.write_text('old index fixture')
        result = ws.watch(args, once=True)
        self.assertEqual(ws.read(result)['status'], 'refreshed')
        view = ws.read(self.work / '.local/views/root-artifacts.json')
        archive = self.work / view['archive']
        self.assertFalse((archive / 'files').exists())
        self.assertEqual((archive / 'retired/artifacts/ir/old-candidate/index.json').read_text(), 'old index fixture')
        self.assertEqual(sorted(p.name for p in (self.work / 'artifacts').iterdir()), ['generated', 'ir'])
        again = ws.watch(args, once=True)
        self.assertEqual(ws.read(again)['status'], 'unchanged')

    def test_watcher_observes_real_transition_and_stops_by_owned_control(self):
        args = self.working_watch()
        args.max_updates = 2
        def transition(_delay):
            self.dashboard('进行中', '等待采集')
        with mock.patch.object(ws.time, 'sleep', side_effect=transition):
            result = ws.watch(args)
        self.assertEqual(len(ws.read(result)['events']), 2)
        current = ws.read(self.work / '.local/views/root-artifacts.json')
        reference = current['status']['dashboard']
        self.assertNotIn('rows', reference)
        self.assertEqual(reference['sha256'], ws.sha(self.work / 'dashboard.md'))
        self.assertEqual(dashboard.load(self.work / reference['snapshot'])['rows'][0]['description'], '等待采集')
        args.stop_file = '.local/watchers/artifact-watch/stop'
        (self.work / args.stop_file).write_text('stop owned service')
        self.assertEqual(ws.read(ws.watch(args))['status'], 'stopped')

    def test_watcher_rejects_stale_source_and_does_not_roll_back_newer_source(self):
        args = self.working_watch()
        ws.watch(args, once=True)
        source = self.work / '.local/produced/source.il'
        source.write_text('changed source')
        with self.assertRaisesRegex(ValueError, 'produced source changed'):
            ws.watch(args, once=True)
        source.write_text('actual source fixture')
        new_source = self.work / '.local/b002.il'
        new_source.write_text('actual b002 bytes')
        files = [{'source': '.local/b002.il', 'path': 'artifacts/ir/model.il', 'sha256': ws.sha(new_source)}]
        manifest = self.write(self.work / '.local/b002-manifest.json', {'files': files})
        selected = {'path': '.local/b002-manifest.json', 'sha256': ws.sha(manifest)}
        ws.sync(self.working_request('new-source', files=files, source_id='b002', source_manifest=selected,
                                     input_pins=[selected, {'path': 'dashboard.md', 'sha256': ws.sha(self.work / 'dashboard.md')}]))
        pointer = self.work / '.local/views/root-artifacts.json'
        old = pointer.read_bytes()
        self.assertEqual(ws.read(ws.watch(args, once=True))['status'], 'superseded')
        self.assertEqual(pointer.read_bytes(), old)

    def alias_source_transition(self, change_files):
        args = self.working_watch()
        ws.watch(args, once=True)
        spec = ws.read(args.record)
        new_files = spec['files']
        if change_files:
            source = self.work / '.local/new-compiled-source.il'
            source.write_text('different actual compiled source bytes')
            new_files = [{'source': str(source.relative_to(self.work)), 'path': 'artifacts/ir/model.il', 'sha256': ws.sha(source)}]
        new_manifest = self.write(self.work / '.local/new-manifest.json', {'files': new_files, 'semantic_hash': 'different'})
        manifest_pin = {'path': str(new_manifest.relative_to(self.work)), 'sha256': ws.sha(new_manifest)}
        state_path = self.work / 'dashboard.md'
        ws.sync(self.working_request('new-compiled-cohort', files=new_files, source_id=spec['source_id'],
                                     source_manifest=manifest_pin,
                                     input_pins=[manifest_pin, {'path': 'dashboard.md', 'sha256': ws.sha(state_path)}]))
        pointer = self.work / '.local/views/root-artifacts.json'
        produced = pointer.read_bytes()
        self.dashboard('待审', '新产物后续测量待审')
        result = ws.watch(args, once=True)
        self.assertEqual(ws.read(result)['status'], 'superseded')
        self.assertEqual(pointer.read_bytes(), produced)
        self.assertIn(manifest_pin, ws.read(pointer)['input_pins'])
        self.assertNotIn(spec['source_manifest'], ws.read(pointer)['input_pins'])

    def test_watcher_same_label_different_bytes_and_manifest_superseded(self):
        self.alias_source_transition(change_files=True)

    def test_watcher_same_label_different_manifest_identical_files_superseded(self):
        self.alias_source_transition(change_files=False)

    def test_watcher_manifest_path_and_label_alias_with_identical_bytes_is_same_source(self):
        args = self.working_watch()
        ws.watch(args, once=True)
        spec = ws.read(args.record)
        alias = self.work / '.local/aliased-manifest.json'
        alias.write_bytes((self.work / spec['source_manifest']['path']).read_bytes())
        ws.sync(self.working_request('same-content-alias', files=spec['files'], source_id='renamed-label',
                                     source_manifest={'path': '.local/aliased-manifest.json', 'sha256': ws.sha(alias)},
                                     input_pins=[{'path': '.local/aliased-manifest.json', 'sha256': ws.sha(alias)},
                                                 {'path': 'dashboard.md', 'sha256': ws.sha(self.work / 'dashboard.md')}]))
        self.dashboard('待审', '相同来源，名称改变')
        result = ws.watch(args, once=True)
        self.assertEqual(ws.read(result)['status'], 'refreshed')
        self.assertEqual(ws.read(self.work / '.local/views/root-artifacts.json')['source_id'], 'renamed-label')

    def test_watcher_pointer_replaced_during_dashboard_read_cannot_attach_old_pins(self):
        args = self.working_watch()
        ws.watch(args, once=True)
        self.dashboard('待审', '读取时新构建提交')
        source = self.work / '.local/compiled-C.il'
        source.write_text('actual new source C')
        new_files = [{'source': '.local/compiled-C.il', 'path': 'artifacts/ir/model.il', 'sha256': ws.sha(source)}]
        new_manifest = self.write(self.work / '.local/compiled-C.json', {'files': new_files})
        new_request = self.working_request('concurrent-C', files=new_files,
                                           source_manifest={'path': '.local/compiled-C.json', 'sha256': ws.sha(new_manifest)},
                                           input_pins=[{'path': '.local/compiled-C.json', 'sha256': ws.sha(new_manifest)},
                                                       {'path': 'dashboard.md', 'sha256': ws.sha(self.work / 'dashboard.md')}])
        original_parse = dashboard.parse
        pointer = self.work / '.local/views/root-artifacts.json'
        committed = []
        def concurrent_commit(text):
            if not committed:
                ws.sync(new_request)
                committed.append(pointer.read_bytes())
            return original_parse(text)
        with mock.patch.object(dashboard, 'parse', side_effect=concurrent_commit):
            result = ws.watch(args, once=True)
        self.assertEqual(ws.read(result)['status'], 'superseded')
        self.assertEqual(pointer.read_bytes(), committed[0])
        self.assertEqual(ws.read(pointer)['files'][0]['sha256'], ws.sha(source))

    def test_concurrent_dashboard_replacement_retries_and_preserves_rejected_snapshot(self):
        args = self.working_watch()
        ws.watch(args, once=True)
        self.dashboard('进行中', '读取旧表时发生替换')
        rejected_hash = ws.sha(self.work / 'dashboard.md')
        original_parse = dashboard.parse
        replaced = []
        def replace_dashboard(text):
            parsed = original_parse(text)
            if not replaced:
                self.dashboard('待审', '新的完整表已提交')
                replaced.append(True)
            return parsed
        with mock.patch.object(dashboard, 'parse', side_effect=replace_dashboard):
            receipt = ws.watch(args, once=True)
        self.assertEqual(ws.read(receipt)['status'], 'refreshed')
        metadata = ws.read(self.work / '.local/views/root-artifacts.json')['status']['dashboard']
        self.assertEqual(metadata['sha256'], ws.sha(self.work / 'dashboard.md'))
        self.assertNotIn('rows', metadata)
        self.assertEqual(dashboard.load(self.work / metadata['snapshot'])['rows'][0]['status'], '待审')
        rejected = self.work / '.local/watchers/artifact-watch/dashboards' / (rejected_hash + '.md')
        self.assertEqual(ws.sha(rejected), rejected_hash)
        request = self.work / '.local/watchers/artifact-watch/events/artifact-watch-000001.json'
        self.assertTrue(request.exists())
        self.assertFalse((self.work / '.local/working/artifacts/artifact-watch-000001').exists())

    def test_old_manifest_dependency_cannot_identify_new_selected_manifest(self):
        args = self.working_watch()
        ws.watch(args, once=True)
        spec = ws.read(args.record)
        new_manifest = self.write(self.work / '.local/child-manifest.json', {'files': spec['files'], 'parent': spec['source_manifest']})
        selected = {'path': '.local/child-manifest.json', 'sha256': ws.sha(new_manifest)}
        ws.sync(self.working_request('child-cohort', files=spec['files'], source_manifest=selected,
                                     input_pins=[selected, spec['source_manifest'],
                                                 {'path': 'dashboard.md', 'sha256': ws.sha(self.work / 'dashboard.md')}]))
        pointer = self.work / '.local/views/root-artifacts.json'
        child_frame = pointer.read_bytes()
        self.dashboard('待审', '新manifest保留旧manifest依赖')
        self.assertEqual(ws.read(ws.watch(args, once=True))['status'], 'superseded')
        self.assertEqual(pointer.read_bytes(), child_frame)
        self.assertEqual(ws.read(pointer)['source_manifest'], selected)

    def test_new_full_view_without_manifest_selection_is_not_assumed_to_be_old_source(self):
        args = self.working_watch()
        ws.watch(args, once=True)
        spec = ws.read(args.record)
        ws.sync(self.working_request('unspecified-selection', files=spec['files'], input_pins=[spec['source_manifest'],
                                      {'path': 'dashboard.md', 'sha256': ws.sha(self.work / 'dashboard.md')}]))
        self.assertEqual(ws.read(ws.watch(args, once=True))['status'], 'superseded')

    def test_managed_watcher_cli_survives_launcher_and_records_actual_process(self):
        args = self.working_watch()
        home = self.work / '.local/watchers/artifact-watch'
        home.mkdir(parents=True)
        service_path = home / 'service.json'
        stop = home / 'stop'
        argv = [sys.executable, '-B', str(ws.ROOT / 'scripts/workspace.py'), 'watch',
                '--work-dir', str(self.work), '--agent', 'graph_optimizer', '--kind', 'artifacts',
                '--record', str(args.record), '--watch-interval', '0.02', '--watch-timeout', '3',
                '--stop-file', str(stop.relative_to(self.work)), '--service-record', str(service_path.relative_to(self.work))]
        with (home / 'stdout.log').open('w') as stdout, (home / 'stderr.log').open('w') as stderr:
            process = subprocess.Popen(argv, stdout=stdout, stderr=stderr, start_new_session=True)
        self.addCleanup(lambda: process.poll() is None and process.kill())
        deadline = time.monotonic() + 2
        while not (self.work / '.local/views/root-artifacts.json').exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIsNone(process.poll(), (home / 'stderr.log').read_text())
        service = ws.read(service_path)
        self.assertEqual(service['pid'], process.pid)
        self.assertEqual(service['status'], 'running')
        self.assertEqual(service['proc_start_ticks'], int(Path('/proc/' + str(process.pid) + '/stat').read_text().split(') ', 1)[1].split()[19]))
        competing_spec = ws.read(args.record)
        competing_spec['watch_id'] = 'different-id-same-view'
        competing_record = self.write(self.work / '.local/requests/competing-watch.json', competing_spec)
        competing_argv = argv[:]
        competing_argv[competing_argv.index('--record') + 1] = str(competing_record)
        duplicate = subprocess.run(competing_argv, capture_output=True, text=True)
        self.assertEqual(duplicate.returncode, 2)
        self.assertIn('a working watcher for this view is already running', duplicate.stderr)
        self.assertIsNone(process.poll())
        state = self.work / 'dashboard.md'
        self.dashboard('失败', '真实新尝试失败')
        while ws.read(self.work / '.local/views/root-artifacts.json')['status']['dashboard']['sha256'] != ws.sha(state) and time.monotonic() < deadline:
            time.sleep(0.01)
        reference = ws.read(self.work / '.local/views/root-artifacts.json')['status']['dashboard']
        self.assertNotIn('rows', reference)
        self.assertEqual(reference['sha256'], ws.sha(state))
        self.assertEqual(dashboard.load(self.work / reference['snapshot'])['rows'][0]['status'], '失败')
        stop.write_text('owned stop control')
        self.assertEqual(process.wait(timeout=2), 0, (home / 'stderr.log').read_text())
        self.assertEqual(ws.read(service_path)['status'], 'stopped')

    def test_dashboard_catalog_and_watcher_never_create_json_management_state(self):
        self.assertNotIn('state', ws.CATALOG['entries'])
        ws.initialize(self.args())
        path = ws.prepare(self.args('orchestrator', kind='dashboard'))
        self.assertEqual(path, self.work / 'dashboard.md')
        self.assertTrue(dashboard.load(path)['rows'])
        args = self.working_watch()
        ws.watch(args, once=True)
        self.assertFalse((self.work / '.local/state.json').exists())
        metadata = ws.read(self.work / '.local/views/root-artifacts.json')
        view = metadata['status']['dashboard']
        self.assertNotIn('rows', view)
        self.assertNotIn('table', view)
        self.assertEqual(view['path'], 'dashboard.md')
        snapshot = self.work / view['snapshot']
        self.assertEqual(snapshot.read_bytes(), path.read_bytes())
        self.assertEqual(view['sha256'], ws.sha(path))
        self.assertIn({'path': 'dashboard.md', 'sha256': ws.sha(path)}, metadata['input_pins'])
        args = self.working_watch('legacy-rejected', schema='flagmega.working-watch/v1', state_path='.local/state.json')
        with self.assertRaisesRegex(ValueError, 'dashboard v2'):
            ws.watch(args, once=True)
        self.assertFalse((self.work / '.local/state.json').exists())


class DashboardTests(unittest.TestCase):
    def table(self, row):
        return '# 当前工作\n\n| ' + ' | '.join(dashboard.COLUMNS) + ' |\n| ' + ' | '.join(['---'] * 8) + ' |\n' + row + '\n'

    def test_chinese_status_time_links_and_escaped_pipes_preserve_cells(self):
        text = self.table(r'| W001 | 2026-11-14 09:39:37 / 09:40:00 | 待审 | 2/10 | 编译 A\|C；路径 \\tmp | graph optimizer | [真实来源](.local/A\|C.md) | 下一步：复审 |')
        parsed = dashboard.parse(text)
        row = parsed['rows'][0]
        self.assertEqual(row['id'], 'W001')
        self.assertEqual(row['time_utc'], '2026-11-14 09:39:37 / 09:40:00')
        self.assertEqual(row['status'], '待审')
        self.assertEqual(row['flow_steps'], '2/10')
        self.assertEqual(row['description'], r'编译 A|C；路径 \tmp')
        self.assertEqual(row['evidence'], '[真实来源](.local/A|C.md)')
        self.assertIn(r'A\|C', row['raw_cells'][4])
        self.assertEqual(len(row['cells']), 8)
        self.assertEqual(parsed['table'], text.split('\n\n', 1)[1].rstrip('\n'))

    def test_malformed_duplicate_and_missing_status_are_rejected(self):
        valid = '| W001 | 2026-11-14 09:39:37 | 进行中 | 2 | 实际工作 | optimizer | [证据](x) | 待审 |'
        for text in (self.table(valid + '\n' + valid), self.table(valid.replace('进行中', '')), self.table(valid.replace('实际工作', 'A|C')), '# 没有表'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                dashboard.parse(text)

    def test_load_binds_actual_bytes_and_never_writes_a_mirror(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'dashboard.md'
            path.write_text(self.table('| W001 | 2026-11-14 09:39:37 | 完成 | 2 | 原样读取 | optimizer | [证据](x) | 下一步 |'))
            result = dashboard.load(path)
            self.assertEqual(result['source_sha256'], ws.sha(path))
            self.assertEqual(list(Path(directory).iterdir()), [path])


class FigureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def example(self, name):
        return ws.read(ws.ROOT / 'assets/workspace-figures' / (name + '.json'))

    def test_all_templates_render_valid_svg_and_label_synthetic_data(self):
        for name in ('graph', 'operators', 'history', 'milestones', 'backends'):
            data = self.example(name)
            figures.validate(data, self.base / 'input.json')
            svg = figures.render(data)
            ET.fromstring(svg)
            self.assertIn('SYNTHETIC EXAMPLE', svg)
            if name == 'backends':
                self.assertIn('Unavailable: not measured', svg)
            if name == 'graph':
                self.assertIn('unavailable', svg)
                self.assertIn('inflight_envelope', svg)

    def test_series_missing_value_is_gap_not_zero_or_line_bridge(self):
        data = self.example('history')
        data.pop('milestones', None)
        data['rows'] = [{'label': 'x', 'x': 0, 'value': 100, 'status': 'qualified'}, {'label': 'x', 'x': 1, 'value': None, 'reason': 'not_run'}, {'label': 'x', 'x': 2, 'value': 110, 'status': 'qualified'}]
        tree = ET.fromstring(figures.render(data))
        ns = {'s': 'http://www.w3.org/2000/svg'}
        self.assertEqual(len(tree.findall('.//s:circle', ns)), 2)
        self.assertFalse([line for line in tree.findall('.//s:line', ns) if line.attrib.get('stroke-width') == '1.5'])

    def milestone_data(self, lower=False):
        data = self.example('history')
        data['direction'] = 'lower' if lower else 'higher'
        if lower:
            data['metric'], data['unit'] = 'Latency', 'ms'
            data['rows'][2].update(value=75, low=73, high=77)
        data['milestones'] = [{
            'milestone_id': 'm000', 'revision_id': 'r002', 'series': 'FlagMega',
            'point_x': 2, 'baseline_point_x': 0, 'baseline_id': 'baseline',
            'baseline_value': 100, 'value': 75 if lower else 115,
            'main_improvement': 'Fuse producer and consumer to remove an intermediate buffer',
            'status': 'achieved'}]
        return data

    def test_compact_point_table_preserves_validated_directional_gain_in_data(self):
        for lower, expected in [(False, .15), (True, .25)]:
            data = self.milestone_data(lower)
            figures.validate(data, self.base / 'input.json')
            svg = figures.render(data)
            ET.fromstring(svg)
            self.assertNotIn('Milestone improvements', svg)
            self.assertNotIn('Fuse producer and consumer', svg)
            self.assertAlmostEqual(figures.milestone_gain(data, data['milestones'][0]), expected)
            self.assertEqual(svg.count('class="performance-point-table-row"'), 3)

    def test_point_table_uses_exactly_changed_curve_points(self):
        data = self.example('history')
        data['milestones'] = []
        data['rows'] = [{'label':'baseline', 'x':0, 'value':495.141973, 'status':'measured',
                         'point_id':'baseline', 'main_improvement':'起始基线'},
                        {'label':'baseline', 'x':1, 'value':495.141973, 'status':'qualified'},
                        {'label':'baseline', 'x':2, 'value':None, 'reason':'failed'},
                        {'label':'baseline', 'x':3, 'value':494.0, 'status':'no_gain',
                         'point_id':'r003', 'main_improvement':'保留舍入并提前吸收类型转换'},
                        {'label':'baseline', 'x':4, 'value':494.0, 'status':'rejected'}]
        data['attempts_table'] = [{'id':'failed-build', 'source_id':'CPU', 'status':'failed'}]
        svg = figures.render(data)
        self.assertEqual(svg.count('class="performance-point-marker"'), 2)
        self.assertEqual(svg.count('class="performance-point-table-row"'), 2)
        self.assertNotIn('failed-build', svg)
        self.assertNotIn('No milestone entries yet', svg)
        self.assertIn('495.141973', svg)
        self.assertIn('494.000000', svg)
        self.assertIn('主要优化点', svg)
        ns = {'s':'http://www.w3.org/2000/svg'}
        rows = [g for g in ET.fromstring(svg).findall('.//s:g', ns)
                if g.attrib.get('class') == 'performance-point-table-row']
        self.assertEqual([[t.text for t in g.findall('s:text', ns)] for g in rows],
                         [['baseline', '495.141973', '起始基线'],
                          ['r003', '494.000000', '保留舍入并提前吸收类型转换']])

    def test_baseline_one_point_one_row_and_missing_points_create_no_table_rows(self):
        data = self.example('history')
        data['milestones'] = []
        data['rows'] = [{'label':'baseline', 'x':0, 'value':495.141973, 'status':'measured'}]
        svg = figures.render(data)
        self.assertEqual(svg.count('class="performance-point-marker"'), 1)
        self.assertEqual(svg.count('class="performance-point-table-row"'), 1)
        data.update(pending_only=True, pending_reason='missing real measurement', synthetic=False)
        data['rows'][0].update(value=None, reason='measurement pending')
        svg = figures.render(data)
        self.assertNotIn('performance-point-table-row', svg)

    def test_milestone_table_rejects_metric_mismatch_or_false_win(self):
        data = self.milestone_data()
        data['milestones'][0]['value'] = 150
        with self.assertRaises(ValueError):
            figures.validate(data, self.base / 'input.json')
        data = self.milestone_data()
        data['milestones'][0]['baseline_value'] = 200
        with self.assertRaises(ValueError):
            figures.validate(data, self.base / 'input.json')
        data = self.milestone_data()
        data['rows'][2].update(value=99, low=98, high=100)
        data['milestones'][0]['value'] = 99
        with self.assertRaises(ValueError):
            figures.validate(data, self.base / 'input.json')

    def test_milestone_table_cannot_hide_or_downgrade_achieved_curve_points(self):
        for status in ('open', 'pending', 'no_gain'):
            data = self.milestone_data()
            data['milestones'][0].update(status=status, revision_id=None)
            with self.assertRaises(ValueError):
                figures.validate(data, self.base / 'input.json')
        for absent in (False, True):
            data = self.milestone_data()
            if absent:
                data.pop('milestones')
            else:
                data['milestones'] = []
            with self.assertRaises(ValueError):
                figures.validate(data, self.base / 'input.json')

    def test_graph_timeline_keeps_full_model_window_and_bars_sort_by_time(self):
        data = self.example('graph')
        data['window_ns'] = [0, 20000]
        figures.validate(data, self.base / 'input.json')
        svg = figures.render(data)
        self.assertIn('Full model window: 20000 ns', svg)
        tree = ET.fromstring(svg)
        ns = {'s': 'http://www.w3.org/2000/svg'}
        first = next(rect for rect in tree.findall('.//s:rect', ns) if rect.attrib.get('fill') == figures.COLORS['tensor'])
        self.assertAlmostEqual(float(first.attrib['width']), (figures.WIDTH - figures.LEFT - figures.RIGHT) * 4000 / 20000)
        bars = self.example('operators')
        figures.validate(bars, self.base / 'input.json')
        svg = figures.render(bars)
        self.assertLess(svg.index('attention'), svg.index('gate/up'))
        self.assertLess(svg.index('gate/up'), svg.index('unattributed'))

    def test_full_model_tick_labels_keep_precision_with_large_integer_origins(self):
        data = self.example('graph')
        origin = 1 << 60
        data['window_ns'] = [origin, origin + 12000]
        for row in data['rows']:
            for interval in row['intervals']:
                interval['start_ns'] += origin
                interval['end_ns'] += origin
        figures.validate(data, self.base / 'input.json')
        tree = ET.fromstring(figures.render(data))
        texts = [node.text for node in tree.findall('.//{http://www.w3.org/2000/svg}text')]
        for tick in ('0', '3000', '6000', '9000', '12000'):
            self.assertIn(tick, texts)

    def test_actual_figure_rejects_changed_provenance_and_nonfinite_values(self):
        data = self.example('operators')
        raw = self.base / 'raw.json'
        raw.write_text('[1,2,3]')
        data['synthetic'] = False
        data['provenance'] = [{'path': 'raw.json', 'sha256': ws.sha(raw)}]
        for row in data['rows']:
            if row['value'] is not None:
                row['measurement_ids'] = ['fixture-only-q001']
        figures.validate(data, self.base / 'input.json')
        raw.write_text('[4,5,6]')
        with self.assertRaises(ValueError):
            figures.validate(data, self.base / 'input.json')
        data['synthetic'] = True
        data['rows'][0]['value'] = float('nan')
        with self.assertRaises(ValueError):
            figures.validate(data, self.base / 'input.json')

    def test_timeline_rejects_fake_observations_and_wrong_clock(self):
        data = self.example('graph')
        data['rows'][0]['intervals'][0]['evidence'] = 'observed_activity'
        with self.assertRaises(ValueError):
            figures.validate(data, self.base / 'input.json')
        data = self.example('graph')
        data['timebase']['alignment'] = 'wg_only'
        with self.assertRaises(ValueError):
            figures.validate(data, self.base / 'input.json')


if __name__ == '__main__':
    unittest.main()
