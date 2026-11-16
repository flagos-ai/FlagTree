"""Scoped real-recording checks; supply an authentic frozen figure input."""
import copy
import json
import os
from pathlib import Path
import unittest

import render_workspace_figures as renderer


@unittest.skipUnless(os.environ.get("FLAGMEGA_RECORDED_LEADER_FIXTURE"), "requires actual recorded leader fixture")
class RecordedLeaderFigures(unittest.TestCase):
    def setUp(self):
        self.path = Path(os.environ["FLAGMEGA_RECORDED_LEADER_FIXTURE"])
        self.data = json.loads(self.path.read_text())

    def test_actual_recording_renders_without_invented_resolution(self):
        renderer.validate(self.data, self.path)
        svg = renderer.render(self.data)
        self.assertIn("Observed role-leader window", svg)
        self.assertIn("resolution/calibration pending", svg)
        self.assertNotIn("Full model window:", svg)

    def test_whole_model_still_requires_known_resolution_and_alignment(self):
        self.data.update(scope="whole_model", window_ns=self.data.pop("observed_window_ns"))
        with self.assertRaisesRegex(ValueError, "clock source/resolution"):
            renderer.validate(self.data, self.path)

    def test_partial_recording_requires_source_pin(self):
        self.data["leader_capture"].pop("source_pin")
        with self.assertRaises(ValueError):
            renderer.validate(self.data, self.path)

    def test_unhealthy_recording_cannot_render(self):
        self.data["leader_capture"]["health_validated"] = False
        with self.assertRaises(ValueError):
            renderer.validate(self.data, self.path)

    def test_partial_cannot_claim_busy_or_cross_CTA_alignment(self):
        for mutate in [lambda d:d["leader_capture"].update(observed_engine_busy=True),
                       lambda d:d["rows"][0]["intervals"][0].update(cta=1)]:
            data = copy.deepcopy(self.data)
            mutate(data)
            with self.assertRaises(ValueError):
                renderer.validate(data, self.path)


if __name__ == "__main__":
    unittest.main()
