"""The real dense 9068 trace must expose callsites beyond hover-only titles."""
from pathlib import Path
import copy
import importlib.util
import json
import sys
import unittest
import xml.etree.ElementTree as ET

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'benchmark'))
import render_results as owner


class ReadableGraph(unittest.TestCase):
    def test_actual_r001_source_and_NUM_select_without_accepting_wrong_source(self):
        data, inputs = owner.collect(WORK)
        expected = '08431b497d6f4ff43a81a2e85af7325c41e4db69e4a579141bb658b5e81fe3f2'
        self.assertEqual(data['source_binding']['runtime_manifest_sha256'], expected)
        self.assertEqual(data['numerical'], 'independently passed for selected source')
        self.assertEqual(data['num_receipt']['selection']['original_verdict']['sha256'],
                         '50b5abac89654591e8a1b3db511e352f949dcf60c522388278ca5e92ae23b765')
        self.assertEqual(data['num_receipt']['selection']['actual_report']['sha256'],
                         'e519322a8be4580595442f60f8e9d6e5ff0dc0a789c49f84bde17b2dd28cc107')
        path = '.local/milestones/m000/revisions/r001/results/down-lhs-cu-diagnostic/actual-NUM-CU-verdict-003.json'
        receipt = inputs.read(path)
        self.assertIsNone(owner.selected_num_receipt(inputs, [(None, path, receipt)],
                          '486ad3c3c78dea4e246366c1400fb28a7f646bebfd6272d0a8dbb3b2084b951d',
                          set(data['source_binding']['generated_source_sha256'])))
        self.assertIsNone(owner.selected_runtime(WORK, inputs, data['artifacts'], data['selected_source'],
                          {'fb6068cfb67603d976e522eca268b2d4009dcda3b921e6516f545c6f32996ad5'}))

    def test_dense_real_trace_has_visible_mapping_and_same_clock_detail(self):
        chart = json.loads((WORK / '.local/milestones/m000/revisions/r000/profiles/graph/tir-primary128-001/figure-graph.json').read_text())
        original = copy.deepcopy(chart)
        renderer = owner.find_skill() / 'scripts/render_workspace_figures.py'
        sys.path.insert(0, str(renderer.parent))
        spec = importlib.util.spec_from_file_location('t293_canonical_renderer', renderer)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        method = getattr(owner, 'render_graph_presentation', lambda d, r, p: r.render(d))
        svg = ET.fromstring(method(chart, module, WORK / '.local/milestones/m000/revisions/r000/profiles/graph/tir-primary128-001/figure-graph.json'))
        ns = {'s': 'http://www.w3.org/2000/svg'}
        visible = ' '.join(t.text or '' for t in svg.findall('.//s:text', ns))
        callsites = {e['callsite'] for row in chart['rows'] for e in row['intervals']}
        self.assertEqual(len(callsites), 21)
        for callsite in sorted(callsites):
            self.assertIn(callsite, visible, 'An exported image must identify actual callsites without hover')
        self.assertIn('Layer 0 detail', visible)
        self.assertIn('callsite colors', visible)
        detail = svg.find("s:svg[@data-graph-panel='layer0-detail']", ns)
        self.assertIsNotNone(detail)
        detail_meta = json.loads(detail.find('s:metadata', ns).text)
        self.assertEqual(detail_meta['observed_window_raw_units'], [960, 90624])
        self.assertEqual(detail_meta['shared_span_dataset'], chart['shared_span_dataset'])
        self.assertEqual(detail_meta['timebase'], chart['timebase'])
        self.assertEqual(chart, original, 'Presentation must not rewrite the full measured chart')


if __name__ == '__main__':
    unittest.main()
