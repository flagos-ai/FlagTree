"""Replay pinned actual paired receipts; no GPU or invented measurements."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import unittest

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'benchmark'))
import render_results as owner


def producer(revision):
    path = Path('.local/milestones/m000/revisions') / revision / 'results/paired-performance/producer-handoff-000.json'
    return ({}, str(path), json.loads((WORK / path).read_text()))


class PairedRevisionPublication(unittest.TestCase):
    def test_actual_canonical_stage_selects_original_NUM(self):
        artifacts = json.loads((WORK / '.local/milestones/m000/revisions/r003/results/figure-publication/actual-selected-artifacts.json').read_text())
        hashes = {p['sha256'] for p in artifacts['files'] if Path(p['path']).name == 'generated_kernels.py'}
        inputs = owner.Inputs(WORK)
        runtime = owner.selected_runtime(WORK, inputs, artifacts, artifacts['source_id'], hashes)
        self.assertIsNotNone(runtime)
        self.assertEqual(owner.sha(runtime), '280c246aa2a7931e5078a202a3296a814b283067d981f2ac09707e3c4c4a20d3')
        path = '.local/milestones/m000/revisions/r003/results/g3-independent-review/publication/NUM-CU-publication-v1.json'
        receipt = owner.selected_num_receipt(inputs, [({}, path, inputs.read(path))], owner.sha(runtime), hashes)
        self.assertTrue(receipt['selection']['source_native_full5_terminal_checked'])
        self.assertIsNone(owner.selected_runtime(WORK, owner.Inputs(WORK), artifacts, 'wrong-source', hashes))

    def test_actual_direct_revision_and_signed_review(self):
        review_path = '.local/milestones/m000/revisions/r003/agents/dispatch/review_r003_paired_performance/handoff.json'
        review = ({}, review_path, json.loads((WORK / review_path).read_text()))
        inputs = owner.Inputs(WORK)
        result = owner.paired_performance_history(inputs, [producer('r003'), review])
        self.assertEqual(len(result), 1, inputs.errors)
        point = result[0]['revision_primary']
        self.assertEqual(point['unit_key'], 'm000/r003')
        self.assertAlmostEqual(point['value'], 511.8056202855138)
        self.assertEqual(point['status'], 'qualified')
        self.assertEqual(len(result[0]['points']), 4)
        self.assertFalse(point['final_milestone_completed'])
        host = next(c for c in result[0]['contrasts'] if c['prompt_length'] == 128 and c['cohort'] == 'ordinary_no_events')
        self.assertLess(host['two_sided_95pct_interval'][0], 0)
        self.assertGreater(host['two_sided_95pct_interval'][1], 0)

    def test_declared_revision_cannot_relabel_actual_artifact(self):
        row, path, actual = producer('r003')
        invalid = copy.deepcopy(actual)
        invalid['revision_id'] = 'r999'
        body = {k: v for k, v in invalid.items() if k != 'signature'}
        invalid['signature']['body_sha256'] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        inputs = owner.Inputs(WORK)
        self.assertEqual(owner.paired_performance_history(inputs, [(row, path, invalid)]), [])
        self.assertTrue(any('direct revision' in e['error'] for e in inputs.errors), inputs.errors)

    def test_legacy_adopted_r001_keeps_its_original_point(self):
        inputs = owner.Inputs(WORK)
        result = owner.paired_performance_history(inputs, [producer('r001')])
        self.assertEqual(len(result), 1, inputs.errors)
        self.assertEqual(result[0]['revision_primary']['unit_key'], 'm000/r001')
        self.assertAlmostEqual(result[0]['revision_primary']['value'], 508.7959850118029)


if __name__ == '__main__':
    unittest.main()
