"""Publish the dashboard bytes actually parsed while root keeps updating it."""
from pathlib import Path
import importlib.util
import tempfile
import sys
import unittest

WORK = Path(__file__).resolve().parents[1]
RESULT = WORK / '.local/milestones/m000/revisions/r007/results/figure-refresh-repair/dashboard-race'
sys.path.insert(0, str(WORK / 'benchmark'))
import render_results as owner
if '--before' in sys.argv:
    before_spec = importlib.util.spec_from_file_location('original_render_owner', RESULT / 'render_results.before.py')
    owner = importlib.util.module_from_spec(before_spec)
    before_spec.loader.exec_module(owner)
sys.path.insert(0, str(owner.find_skill() / 'scripts'))
spec = importlib.util.spec_from_file_location('snapshot_workspace', owner.find_skill() / 'scripts/workspace.py')
workspace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workspace)


class DashboardSnapshotRace(unittest.TestCase):
    def test_root_update_after_parse_keeps_one_coherent_published_snapshot(self):
        with tempfile.TemporaryDirectory(dir=RESULT) as temp:
            work = Path(temp)
            original = (RESULT / 'actual-dashboard-before.md').read_bytes()
            live = work / 'dashboard.md'
            live.write_bytes(original)
            inputs = owner.Inputs(work)
            parsed = owner.read_dashboard(inputs, owner.find_skill())
            captured_hash = inputs.pins['dashboard.md']['sha256']
            live.write_bytes(original + b'\n<!-- root publishes another real state -->\n')
            frame = work / '.local/reports/live-race-fixture'
            owner.archive_render_inputs(inputs, frame, workspace, {})
            snapshot = frame / 'input-snapshots/dashboard.md'
            self.assertEqual(snapshot.read_bytes(), original)
            self.assertEqual(owner.sha(snapshot), captured_hash)
            self.assertTrue(parsed['rows'])
            self.assertNotEqual(snapshot.stat().st_ino, live.stat().st_ino)
            # Another root update between archiving and sync must not change
            # either the parsed table or its exact publication pin.
            live.write_bytes(original + b'\n<!-- next root update during sync -->\n')
            pin = owner.publication_input_pin(inputs, inputs.pins['dashboard.md'])
            self.assertEqual(pin['path'], str(snapshot.relative_to(work)))
            self.assertEqual(pin['sha256'], captured_hash)
            self.assertEqual(snapshot.read_bytes(), original)

    def test_changed_native_source_still_fails_original_copy_guard(self):
        with tempfile.TemporaryDirectory(dir=RESULT) as temp:
            work = Path(temp)
            source = work / 'native.py'
            source.write_bytes(b'original native source')
            inputs = owner.Inputs(work)
            inputs.raw('native.py')
            source.write_bytes(b'changed native source')
            with self.assertRaises(ValueError):
                owner.archive_render_inputs(inputs, work / 'frame', workspace, {})

    def test_tampered_captured_snapshot_still_fails_publication_guard(self):
        with tempfile.TemporaryDirectory(dir=RESULT) as temp:
            work = Path(temp)
            (work / 'dashboard.md').write_bytes((RESULT / 'actual-dashboard-before.md').read_bytes())
            inputs = owner.Inputs(work)
            owner.read_dashboard(inputs, owner.find_skill())
            frame = work / '.local/reports/live-race-fixture'
            owner.archive_render_inputs(inputs, frame, workspace, {})
            (frame / 'input-snapshots/dashboard.md').write_bytes(b'wrong captured bytes')
            with self.assertRaises(ValueError):
                owner.publication_input_pin(inputs, inputs.pins['dashboard.md'])


if __name__ == '__main__':
    if '--before' in sys.argv:
        suite = unittest.TestSuite([DashboardSnapshotRace('test_root_update_after_parse_keeps_one_coherent_published_snapshot')])
        result = unittest.TextTestRunner().run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    unittest.main()
