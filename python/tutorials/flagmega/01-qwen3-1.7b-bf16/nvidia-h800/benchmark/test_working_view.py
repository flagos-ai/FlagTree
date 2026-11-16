"""Actual registered FlashRT GPU report -> canonical view regression, CPU only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import render_results as renderer

class ActualFlashRTProjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work=Path(renderer.__file__).resolve().parents[1]
        cls.data,cls.inputs=renderer.collect(cls.work)
        cls.data['agent_id']='/fixture/actual-FlashRT-projection'
        cls.inputs.archive_pins=dict(cls.inputs.pins)
        cls.expected={}
        for entry in json.loads((cls.work/'benchmark/current-native-backend-input.json').read_text())['cohorts']:
            if entry['backend']=='flashrt':
                report=json.loads((cls.work/entry['report']['path']).read_text())
                cls.expected[entry['cohort']]=next(s for s in report['scenarios'] if s['prompt_length']==128)
        assert set(cls.expected)=={'ordinary_no_events','event_instrumented'}
    def charts(self,data=None):
        directory=self.work/'.local/agents/dispatch/repair_flashrt_registered_backend_figure_projection'
        with tempfile.TemporaryDirectory(dir=directory) as temp:
            return renderer.build_charts(data or self.data,self.inputs,renderer.find_skill(),Path(temp))
    def test_actual_current_flashrt_HOST_DEVICE_and_instrumentedHOST_rows(self):
        charts=self.charts()
        hosts=[r for r in charts['backend-comparison']['rows'] if r['label'].startswith('FlashRT')]
        self.assertEqual(len(hosts),2,'actual ordinary and instrumented HOST rows must both be present')
        ordinary=next(r for r in hosts if 'instrumented' not in r['label'])
        instrumented=next(r for r in hosts if 'instrumented' in r['label'])
        device=next(r for r in charts['backend-device-comparison']['rows'] if r['label'].startswith('FlashRT'))
        self.assertEqual(ordinary['value'],self.expected['ordinary_no_events']['host_tokens_per_second'])
        self.assertEqual(instrumented['value'],self.expected['event_instrumented']['host_tokens_per_second'])
        self.assertEqual(device['value'],self.expected['event_instrumented']['device_tokens_per_second'])
        self.assertEqual(ordinary['measurement_channel'],'HOST')
        self.assertEqual(device['measurement_channel'],'DEVICE')
        self.assertNotEqual(ordinary['measurement_ids'],instrumented['measurement_ids'])
        for name in ['backend-comparison','backend-device-comparison']:
            text=' '.join(charts[name]['working_status']['visible_context'])
            self.assertIn('FlashRT-derived',text)
            self.assertIn('a007ef0',text)
            self.assertIn('independent',text)
    def test_absent_current_flashrt_stays_pending_and_keeps_other_values(self):
        data=copy.deepcopy(self.data);data['native_backend_states'].pop('flashrt')
        charts=self.charts(data)
        for name in ['backend-comparison','backend-device-comparison']:
            rows=[r for r in charts[name]['rows'] if r['label'].startswith('FlashRT')]
            self.assertTrue(rows)
            self.assertTrue(all(r['value'] is None for r in rows))
    def test_actual_history_remains_single_c13_point(self):
        charts=self.charts()
        for name in ['performance-vs-milestone','performance-vs-iteration']:
            self.assertEqual(len(charts[name]['rows']),1)
            self.assertEqual(charts[name]['rows'][0]['measurement_ids'],['c0013'])

if __name__=='__main__':unittest.main()
