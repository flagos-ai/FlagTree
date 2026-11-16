"""An actually committed figure half must not seal an incomplete render."""
from pathlib import Path
import copy
import importlib.util
import json
import sys
import unittest
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[1]
RESULT = WORK / '.local/milestones/m000/revisions/r007/results/figure-refresh-repair'
sys.path.insert(0, str(WORK / 'benchmark'))
import render_results as owner


class CompletedRenderSelection(unittest.TestCase):
    def test_actual_partial_figure_head_retains_completed_input_cache(self):
        original_read = owner.Inputs.read
        partial = json.loads((RESULT / 'partial-figures-before.json').read_text())
        partial_frame = (WORK / partial['files'][0]['source']).parent
        self.assertFalse((partial_frame / 'render-receipt.json').exists())
        def read(inputs, relative):
            if str(relative) == 'figures/manifest.json':
                return copy.deepcopy(partial)
            return original_read(inputs, relative)
        inputs = owner.Inputs(WORK)
        with patch.object(owner.Inputs, 'read', read):
            candidates = owner.previous_render_snapshots(inputs)
        self.assertTrue(candidates)
        sample = candidates['benchmark/render_results.py']
        self.assertFalse(sample['path'].is_relative_to(partial_frame))
        self.assertEqual(owner.sha(sample['path']), sample['sha256'])

    def test_corrupt_complete_render_pin_is_rejected(self):
        original_read = owner.Inputs.read
        def read(inputs, relative):
            value = original_read(inputs, relative)
            if str(relative) == '.local/reports/live-watcher/completed-render.json':
                value = copy.deepcopy(value)
                value['receipt']['sha256'] = '0' * 64
            return value
        with patch.object(owner.Inputs, 'read', read), self.assertRaises(ValueError):
            owner.previous_render_snapshots(owner.Inputs(WORK))

    def test_restart_without_first_refresh_keeps_completed_commit(self):
        original_read = owner.Inputs.read
        def read(inputs, relative):
            if str(relative) == '.local/reports/live-watcher/status.json':
                return {'agent_id': '/fixture/restart', 'last_success': None}
            return original_read(inputs, relative)
        with patch.object(owner.Inputs, 'read', read):
            self.assertTrue(owner.previous_render_snapshots(owner.Inputs(WORK)))


if __name__ == '__main__':
    if '--before' in sys.argv:
        suite = unittest.TestSuite([CompletedRenderSelection('test_actual_partial_figure_head_retains_completed_input_cache')])
        result = unittest.TextTestRunner().run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    unittest.main()
