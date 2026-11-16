"""CPU projection regressions using immutable actual native GPU reports."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import render_results as renderer


class CurrentBackendStates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(renderer.__file__).resolve().parents[1]
        cls.registration = json.loads((cls.work / "benchmark/current-native-backend-input.json").read_text())

    def collect_with(self, registration):
        original = renderer.Inputs.read
        def read(inputs, relative):
            if str(relative) == "benchmark/current-native-backend-input.json":
                return registration
            return original(inputs, relative)
        with patch.object(renderer.Inputs, "read", read):
            return renderer.collect(self.work)

    def test_actual_completed_cohorts_override_old_failure_and_CPU_progress(self):
        data, _ = self.collect_with(self.registration)
        for entry in self.registration["cohorts"]:
            state = data["native_backend_states"][entry["backend"]]
            report = json.loads((self.work / entry["report"]["path"]).read_text())
            actual = next(s for s in report["scenarios"] if s["prompt_length"] == 128)
            cohort = state["current_cohorts"][entry["cohort"]]
            self.assertEqual(cohort["HOST_TPS"], actual["host_tokens_per_second"])
            self.assertEqual(cohort.get("DEVICE_TPS"), actual.get("device_tokens_per_second"))
            self.assertEqual(cohort["measurement_id"], entry["measurement_id"])
            self.assertNotIn("actual GPU pending", state["reason"])

    def test_actual_DEVICE_chart_keeps_each_registered_DEVICE_value(self):
        data, inputs = self.collect_with(self.registration)
        data["agent_id"] = "/fixture/current-native-backend-projection"
        inputs.archive_pins = dict(inputs.pins)
        with tempfile.TemporaryDirectory(dir=self.work / ".local/agents/dispatch/repair_current_native_backend_working_projection") as temp:
            charts = renderer.build_charts(data, inputs, renderer.find_skill(), Path(temp))
        rows = {r["label"]: r for r in charts["backend-device-comparison"]["rows"]}
        for entry in self.registration["cohorts"]:
            if entry["cohort"] != "event_instrumented":
                continue
            report = json.loads((self.work / entry["report"]["path"]).read_text())
            expected = next(s for s in report["scenarios"] if s["prompt_length"] == 128)["device_tokens_per_second"]
            label = {"sglang": "SGLang", "vllm": "vLLM"}[entry["backend"]]
            self.assertEqual(rows[label]["value"], expected)

    def reject(self, change):
        registry = copy.deepcopy(self.registration)
        registry["cohorts"] = [registry["cohorts"][0]]
        change(registry["cohorts"][0])
        data, inputs = self.collect_with(registry)
        self.assertNotIn("current_cohorts", data["native_backend_states"].get(registry["cohorts"][0]["backend"], {}))
        self.assertTrue(any("registered native cohort" in row["error"] for row in inputs.errors))

    def test_wrong_cohort_is_not_relabelled(self):
        self.reject(lambda e: e.update(cohort="event_instrumented"))

    def test_wrong_physical_UUID_is_not_relabelled(self):
        self.reject(lambda e: e.update(physical_gpu_uuid="GPU-00000000-0000-0000-0000-000000000000"))

    def test_wrong_source_hash_is_not_an_alias(self):
        self.reject(lambda e: e["source_manifest"].update(sha256="0" * 64))

    def test_wrong_job_identity_is_not_a_current_report(self):
        self.reject(lambda e: e.update(measurement_id="actual-negative-mismatched-job"))

    def test_actual_failed_report_cannot_supply_current_values(self):
        data, _ = renderer.collect(self.work)
        historical = data["native_backend_states"]["sglang"]["historical_state"]["report"]
        self.reject(lambda e: e.update(report=historical))

    def test_duplicate_current_cohort_is_rejected(self):
        registry = copy.deepcopy(self.registration)
        registry["cohorts"] = [registry["cohorts"][0], registry["cohorts"][0]]
        data, inputs = self.collect_with(registry)
        self.assertNotIn("current_cohorts", data["native_backend_states"].get(registry["cohorts"][0]["backend"], {}))
        self.assertTrue(any("duplicate" in row["error"] for row in inputs.errors))


if __name__ == "__main__":
    unittest.main()
