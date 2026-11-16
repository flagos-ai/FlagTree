"""Actual 9068 caller projection after a new ordinary source has no recapture."""
from pathlib import Path
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

WORK=Path(__file__).resolve().parents[1]
RESULT=WORK/'.local/milestones/m000/revisions/r000/results/repair-current-trace-clock-publication'
sys.path.insert(0,str(WORK/'benchmark'))
import render_results as owner


class TraceClockPublication(unittest.TestCase):
    def test_actual_unrecaptured_source_preserves_original_callers_and_history(self):
        data,inputs=owner.collect(WORK)
        data['agent_id']='/fixture/t287-cpu-regression'
        view=data.get('tir_views')
        self.assertIsNotNone(view,'Changing ordinary source must retain the pinned measured diagnostic view')
        self.assertFalse(view['matching_selected_source'])
        self.assertEqual(view['descriptor']['dataset']['sha256'],
                         '9068ec93d1fdccb9d32c090b9579c9ffdda8b02d777bfcca165572bf0c0dcc78')
        inputs.archive_pins=dict(inputs.pins)
        # Freeze the original T287 two-point history scenario; r001's later real
        # measurement has its own adoption regression.
        data['paired_performance']=[p for p in data['paired_performance'] if p['revision_primary']['revision_id']=='r000']
        with tempfile.TemporaryDirectory(dir=RESULT) as temp:
            charts=owner.build_charts(data,inputs,owner.find_skill(),Path(temp))
            graph,operators=charts['graph-trace'],charts['operator-latency']
            self.assertEqual(graph['scope'],'observed_TIR_callers')
            self.assertEqual(graph['unit'],'raw globaltimer units')
            self.assertEqual(graph['rows'],view['charts']['graph']['rows'])
            self.assertEqual(operators['rows'],view['charts']['operators']['rows'])
            self.assertEqual(len(graph['rows']),12)
            self.assertEqual(len(operators['rows']),21)
            self.assertEqual(len(charts['performance-vs-iteration']['rows']),2)
            self.assertEqual(len(charts['performance-vs-milestone']['rows']),1)
            self.assertIn('Historical source',graph['title'])
            self.assertTrue(any('matching_selected_source=false' in s for s in graph['working_status']['visible_context']))
            self.assertFalse(graph['timebase']['nominal_ns_verified'])
            self.assertIsNone(graph['timebase']['actual_resolution_ns'])
            self.assertFalse(graph['timebase']['cross_execution_alignment'])

    def test_current_diagnostic_binding_still_rejects_a_different_trace(self):
        inputs=owner.Inputs(WORK)
        artifacts=json.loads((WORK/'artifacts/manifest.json').read_text())
        selected=Path(artifacts['source_manifest']['path'])
        selected=selected if selected.is_absolute() else WORK/selected
        real_read=owner.Inputs.read
        def read(reader,relative):
            value=real_read(reader,relative)
            if (WORK/relative).resolve()==selected.resolve():
                value=copy.deepcopy(value)
                candidate=WORK/'.local/milestones/m000/revisions/r000/build/down-lhs-c000/runtime-artifact/artifact.json'
                value['diagnostic_runtime_manifest']={'path':str(candidate),'sha256':owner.sha(candidate)}
            return value
        with patch.object(owner.Inputs,'read',read):
            self.assertIsNone(owner.current_tir_views(inputs,artifacts))


if __name__=='__main__':unittest.main()
