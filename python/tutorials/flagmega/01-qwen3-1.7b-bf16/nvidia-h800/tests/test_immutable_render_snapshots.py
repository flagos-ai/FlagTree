"""Snapshot reuse must preserve history while avoiding mutable-source aliases."""
import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

WORK = Path(__file__).resolve().parents[1]
RESULT = WORK / '.local/milestones/m000/revisions/r004/results/figure-storage'
sys.path.insert(0, str(WORK / 'benchmark'))
import render_results as owner

sys.path.insert(0, str(owner.find_skill() / 'scripts'))
spec = importlib.util.spec_from_file_location('snapshot_workspace', owner.find_skill() / 'scripts/workspace.py')
workspace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workspace)


def archive(inputs, directory, candidates):
    if '--before' not in sys.argv:
        return owner.archive_render_inputs(inputs, directory, workspace, candidates)
    # Execute the original render_live copying loop on the same real input bytes.
    tree = ast.parse((RESULT / 'render_results.before.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'render_live')
    loop = next(n for n in ast.walk(function) if isinstance(n, ast.For) and
                isinstance(n.target, ast.Tuple) and [v.id for v in n.target.elts] == ['original', 'pin'])
    inputs.archive_pins = {}
    exec(compile(ast.Module(body=[loop], type_ignores=[]), 'original-render-input-loop', 'exec'),
         {'work': inputs.work, 'inputs': inputs, 'directory': directory, 'sha': owner.sha})


class ImmutableRenderSnapshots(unittest.TestCase):
    def test_equal_inputs_share_only_previous_immutable_snapshot(self):
        with tempfile.TemporaryDirectory(dir=RESULT) as temp:
            work = Path(temp)
            source = work / 'mutable.py'
            original = (RESULT / 'render_results.before.py').read_bytes()
            source.write_bytes(original)
            inputs = owner.Inputs(work)
            inputs.raw('mutable.py')
            first = work / '.local/reports/live-fixture-0'
            archive(inputs, first, {})
            owner.dump(first / 'render-receipt.json', {'schema': 'flagmega.live-render/v1',
                       'status': 'synced', 'input_pins': list(inputs.archive_pins.values())})
            snapshot0 = first / 'input-snapshots/mutable.py'
            self.assertNotEqual(source.stat().st_ino, snapshot0.stat().st_ino)
            candidates = {'mutable.py': {'path': snapshot0, 'sha256': owner.sha(snapshot0)}}
            second = work / '.local/reports/live-fixture-1'
            archive(inputs, second, candidates)
            snapshot1 = second / 'input-snapshots/mutable.py'
            self.assertEqual(snapshot0.stat().st_ino, snapshot1.stat().st_ino)
            source.write_bytes(original + b'\n# changed mutable source\n')
            self.assertEqual(snapshot0.read_bytes(), original)
            self.assertEqual(snapshot1.read_bytes(), original)
            inputs.raw('mutable.py')
            archive(inputs, work / '.local/reports/live-fixture-2', candidates)
            snapshot2 = work / '.local/reports/live-fixture-2/input-snapshots/mutable.py'
            self.assertNotEqual(snapshot2.stat().st_ino, source.stat().st_ino)
            self.assertNotEqual(snapshot2.stat().st_ino, snapshot0.stat().st_ino)
            self.assertEqual(snapshot2.read_bytes(), source.read_bytes())

    def test_changed_candidate_is_rejected_without_fallback(self):
        with tempfile.TemporaryDirectory(dir=RESULT) as temp:
            work = Path(temp)
            source = work / 'mutable.py'
            source.write_bytes(b'original input')
            inputs = owner.Inputs(work)
            inputs.raw('mutable.py')
            first = work / '.local/reports/live-fixture-0'
            archive(inputs, first, {})
            owner.dump(first / 'render-receipt.json', {'schema': 'flagmega.live-render/v1',
                       'status': 'synced', 'input_pins': list(inputs.archive_pins.values())})
            prior = first / 'input-snapshots/mutable.py'
            candidates = {'mutable.py': {'path': prior, 'sha256': owner.sha(prior)}}
            prior.write_bytes(b'corrupted immutable input')
            with self.assertRaises((ValueError, AssertionError)):
                archive(inputs, work / 'next-frame', candidates)

    def test_actual_candidates_come_from_committed_render_pins(self):
        inputs = owner.Inputs(WORK)
        candidates = owner.previous_render_snapshots(inputs)
        self.assertTrue(candidates)
        pin = candidates['benchmark/render_results.py']
        self.assertIn('/input-snapshots/', str(pin['path']))
        self.assertEqual(owner.sha(pin['path']), pin['sha256'])
        self.assertNotEqual(pin['path'].stat().st_ino, (WORK / 'benchmark/render_results.py').stat().st_ino)


if __name__ == '__main__':
    before = '--before' in sys.argv
    if before:
        sys.argv.remove('--before')
        # The original loop intentionally fails the equal-input reuse assertion.
        sys.argv.append('--before')
        suite = unittest.TestSuite([ImmutableRenderSnapshots('test_equal_inputs_share_only_previous_immutable_snapshot')])
        result = unittest.TextTestRunner().run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    unittest.main()
