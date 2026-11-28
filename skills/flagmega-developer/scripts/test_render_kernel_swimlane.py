# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Check timing, evidence and input-boundary behavior of the swimlane renderer."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import render_kernel_swimlane as renderer

SCRIPT = Path(renderer.__file__)
EXAMPLE = SCRIPT.parents[1] / "assets/kernel-swimlane/example.json"
SVG = "{http://www.w3.org/2000/svg}"


def fixture():
    return {
        "schema_version":
        1, "title":
        "Timing <&> test", "synthetic":
        False, "window_ns": [0, 10], "panels": [{
            "label":
            "Baseline", "invocation":
            "launch-1", "scope":
            "CTA 0", "timebase": {"source": "%globaltimer", "resolution_ns": 1, "alignment": "validated_common"},
            "lanes": [{
                "id":
                "wg", "label":
                "WG0", "kind":
                "wg", "scope":
                "warps 0–3", "status":
                "captured", "intervals":
                [{"id": "a", "start_ns": 1, "end_ns": 5, "kind": "simt", "label": "a", "evidence": "observed_activity"},
                 {"id": "b", "start_ns": 3, "end_ns": 7, "kind": "simt", "label": "b", "evidence":
                  "observed_activity"}, {
                      "id": "wait", "start_ns": 7, "end_ns": 9, "kind": "wait", "label": "wait", "evidence":
                      "observed_activity"
                  }, {
                      "id": "point", "start_ns": 0, "end_ns": 0, "kind": "sync", "label": "point", "evidence":
                      "observed_activity"
                  }]
            }, {
                "id":
                "hw", "label":
                "TMA lifetime", "kind":
                "hardware", "scope":
                "SM 0; envelope only", "status":
                "captured", "intervals": [{
                    "id": "flight", "start_ns": 1, "end_ns": 8, "kind": "transfer", "label": "flight", "evidence":
                    "inflight_envelope"
                }]
            }, {
                "id": "missing", "label": "Tensor Core", "kind": "hardware", "scope": "SM 0", "status": "unavailable",
                "intervals": []
            }], "dependencies": [{"from": "flight", "to": "wait"}], "bubbles": [{
                "interval": "wait", "classification": "unknown", "critical_path": None, "reason":
                "Unresolved causal attribution", "evidence": "fixture"
            }]
        }]
    }


class SwimlaneTests(unittest.TestCase):

    def test_union_lengths_do_not_count_overlap_or_envelopes_as_work(self):
        result = renderer.summarize(fixture())["panels"][0]
        wg, hw, missing = result["lanes"]
        self.assertEqual(wg["direct_work_union_ns"], 6)
        self.assertEqual(wg["wait_union_ns_by_evidence"], {"observed_activity": 2})
        self.assertEqual(wg["uncovered_ns"], 2)
        self.assertEqual(hw["interval_union_ns_by_evidence"], {"inflight_envelope": 7})
        self.assertIsNone(hw["direct_work_union_ns"])
        self.assertIsNone(missing["uncovered_ns"])
        self.assertIsNone(missing["direct_work_union_ns"])
        self.assertEqual(result["bubbles"][0]["duration_ns"], 2)

    def test_points_and_subpixel_spans_keep_real_geometry(self):
        data = fixture()
        data["window_ns"] = [0, 10000]
        root = ET.fromstring(renderer.render(data))
        events = {e.attrib["data-id"]: e for e in root.iter() if "data-id" in e.attrib}
        self.assertEqual(events["point"].tag, SVG + "line")
        self.assertEqual(events["point"].attrib["x1"], events["point"].attrib["x2"])
        self.assertLess(float(events["a"].attrib["width"]), 1)
        self.assertAlmostEqual(float(events["a"].attrib["width"]), 4 * 1210 / 10000)
        self.assertNotEqual(events["a"].attrib["y"], events["b"].attrib["y"])
        self.assertEqual(events["a"].attrib["data-start-ns"], "1")
        self.assertEqual(root.find(SVG + "title").text, data["title"])

    def test_comparisons_keep_shared_scale_and_distinct_invocations(self):
        data = fixture()
        candidate = deepcopy(data["panels"][0])
        candidate["invocation"] = "launch-2"
        data["panels"].append(candidate)
        root = ET.fromstring(renderer.render(data))
        same_time = [e for e in root.iter() if e.attrib.get("data-id") == "a"]
        self.assertEqual(same_time[0].attrib["x"], same_time[1].attrib["x"])
        self.assertEqual({e.attrib["data-invocation"] for e in same_time}, {"launch-1", "launch-2"})

    def test_cropping_does_not_relocate_dependency_completion(self):
        data = fixture()
        data["window_ns"] = [0, 8]
        root = ET.fromstring(renderer.render(data))
        self.assertFalse([e for e in root.iter() if e.attrib.get("class") == "dependency"])
        self.assertEqual(renderer.summarize(data)["panels"][0]["bubbles"][0]["duration_ns"], 1)

    def test_missing_hardware_and_unvalidated_axis_are_rejected(self):
        data = fixture()
        data["panels"][0]["timebase"]["alignment"] = "wg_only"
        with self.assertRaisesRegex(ValueError, "validated_common"):
            renderer.validate(data)
        data = fixture()
        data["panels"][0]["lanes"][1]["status"] = "unavailable"
        with self.assertRaisesRegex(ValueError, "cannot have intervals"):
            renderer.validate(data)

    def test_synthetic_example_is_deterministic_and_cannot_claim_measurement(self):
        data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        first = renderer.render(data)
        self.assertEqual(first, renderer.render(data))
        ET.fromstring(first)
        data["panels"][0]["lanes"][0]["intervals"][0]["evidence"] = "observed_activity"
        with self.assertRaisesRegex(ValueError, "synthetic intervals"):
            renderer.validate(data)

    def test_invalid_cli_input_leaves_no_svg(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.json", Path(directory) / "out.svg"
            data = fixture()
            data["panels"][0]["lanes"][0]["intervals"][0]["end_ns"] = -1
            source.write_text(json.dumps(data))
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(source), "--output",
                 str(output)], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("interval times", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
