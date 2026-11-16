"""Affected CPU metadata/plot tests using the frozen actual T241 packet."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

WORK = Path(__file__).resolve().parents[2]
ASSIGNMENT = WORK / '.local/agents/dispatch/refresh_current_hardware_figure_priority'
parser = argparse.ArgumentParser()
parser.add_argument('--renderer', type=Path, default=WORK / 'benchmark/render_results.py')
parser.add_argument('--case')
ARGS, REST = parser.parse_known_args()
spec = importlib.util.spec_from_file_location('actual_hardware_projection_subject', ARGS.renderer)
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


def signed(body):
    value = copy.deepcopy(body)
    value.pop('signature', None)
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
    value['signature'] = {'agent_id': '/root/kernel_profiler', 'body_sha256': hashlib.sha256(raw).hexdigest()}
    return value


class CurrentHardwarePriority(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ASSIGNMENT / 'input-snapshots/current-working-data.json').read_text())
        cls.handoff_path = '.local/agents/dispatch/execute_same_model_current_hardware_capture/handoff.json'
        cls.handoff = json.loads((WORK / cls.handoff_path).read_text())
        cls.metrics = json.loads(Path(cls.handoff['metrics']['path']).read_text())
        cls.runtime_hash = cls.data['source_binding']['runtime_manifest_sha256']
        cls.program_hashes = set(cls.data['source_binding']['generated_source_sha256'])

    def read(self, mutate=None, runtime=None, corrupt_signature=False, corrupt_pin=False):
        handoff, metrics = copy.deepcopy(self.handoff), copy.deepcopy(self.metrics)
        shadows = {}
        if mutate:
            mutate(handoff, metrics)
            raw = json.dumps(metrics, indent=2).encode()
            shadows[str(Path(handoff['metrics']['path']).relative_to(WORK))] = raw
            handoff['metrics'].update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
            handoff = signed(handoff)
            shadows[self.handoff_path] = json.dumps(handoff, indent=2).encode()
        if corrupt_pin:
            handoff['metrics']['bytes'] += 1
            handoff = signed(handoff)
        if corrupt_signature:
            handoff['signature']['body_sha256'] = '0' * 64
        if corrupt_pin or corrupt_signature:
            shadows[self.handoff_path] = json.dumps(handoff, indent=2).encode()
        class ShadowInputs(renderer.Inputs):
            def raw(inputs, relative):
                if str(relative) in shadows:
                    raw = shadows[str(relative)]
                    inputs.pins[str(relative)] = {'path': str(relative), 'sha256': hashlib.sha256(raw).hexdigest()}
                    return raw
                return super().raw(relative)
        inputs = ShadowInputs(WORK)
        method = getattr(renderer, 'current_hardware_diagnostics', None)
        evidence = [({'id': 'T241'}, self.handoff_path, handoff)]
        packet = method(inputs, evidence, runtime or self.runtime_hash, self.program_hashes) if method else {'current': None, 'historical': []}
        return packet, inputs

    def charts(self, data):
        inputs = renderer.Inputs(WORK)
        class ActualPins(dict):
            def __missing__(pins, key):
                value = {'path': key, 'sha256': renderer.sha(WORK / key)}
                pins[key] = value
                return value
        inputs.archive_pins = ActualPins()
        with tempfile.TemporaryDirectory(dir=ASSIGNMENT) as temp:
            return renderer.build_charts(data, inputs, renderer.find_skill(), Path(temp))

    def test_current_hardware_has_priority(self):
        packet, _ = self.read()
        data = copy.deepcopy(self.data)
        data['current_hardware'] = packet['current']
        charts = self.charts(data)
        row = next(r for r in charts['hbm-ncu-reported']['rows'] if '128' in r['label'])
        actual = self.metrics['rows'][1]['NCU_reported_DRAM_percent']
        self.assertEqual(row['value'], actual)
        self.assertIn('T241', row['measurement_ids'][0])
        self.assertIsNotNone(packet['current'])

    def test_same_H800_route_is_not_a_new_source_gate(self):
        packet, _ = self.read()
        self.assertTrue(packet['current']['binding']['matching_selected_runtime'])
        self.assertEqual(packet['current']['rows'][1]['physical_index'], 1)

    def test_changed_handoff_signature_is_rejected(self):
        packet, inputs = self.read(corrupt_signature=True)
        self.assertIsNone(packet['current'])
        self.assertTrue(inputs.errors)

    def test_changed_metrics_full_pin_is_rejected(self):
        packet, inputs = self.read(corrupt_pin=True)
        self.assertIsNone(packet['current'])
        self.assertTrue(inputs.errors)

    def test_other_source_is_retained_as_history(self):
        packet, inputs = self.read(runtime='0' * 64)
        self.assertIsNone(packet['current'])
        self.assertEqual(len(packet['historical']), 1)
        self.assertFalse(inputs.errors)

    def rejected(self, mutate):
        packet, inputs = self.read(mutate)
        self.assertIsNone(packet['current'])
        self.assertTrue(inputs.errors)

    def test_mixed_source_is_rejected(self):
        self.rejected(lambda h, m: m['rows'][1].update(source_id='foreign:' + '0' * 64))

    def test_mixed_native_is_rejected(self):
        self.rejected(lambda h, m: m['rows'][1].update(cubin_sha256='0' * 64))

    def test_changed_duration_unit_is_rejected(self):
        self.rejected(lambda h, m: m['rows'][1]['elapsed_unit_evidence'].update(actual_CSV_unit='s'))

    def test_changed_byte_time_ratio_is_rejected(self):
        self.rejected(lambda h, m: m['rows'][1].update(actual_total_bytes_by_fixed_GPU_elapsed_Bps=1))

    def test_incomplete_contexts_are_rejected(self):
        self.rejected(lambda h, m: m['rows'].pop())

    def test_historical_P2_and_performance_rows_are_preserved(self):
        packet, _ = self.read()
        before = copy.deepcopy(self.data)
        after = copy.deepcopy(self.data)
        after['current_hardware'] = packet['current']
        first, second = self.charts(before), self.charts(after)
        self.assertEqual(before['latest_diagnostic'], after['latest_diagnostic'])
        for name in ['performance-vs-milestone', 'performance-vs-iteration', 'backend-comparison', 'backend-device-comparison', 'graph-trace', 'operator-latency']:
            self.assertEqual(first[name]['rows'], second[name]['rows'], name)
            self.assertEqual(first[name]['kind'], second[name]['kind'], name)


if __name__ == '__main__':
    unittest.main(argv=[__file__] + ([f'CurrentHardwarePriority.{ARGS.case}'] if ARGS.case else REST))
