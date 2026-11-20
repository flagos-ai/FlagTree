"""Conservative read issue ordering; reused from the measured native prototype."""
import importlib
from pathlib import Path
import sys
from types import ModuleType
import unittest

backend = Path(__file__).resolve().parents[4] / "third_party/hcu/backend"
package = "hcu_ready_lds_issue_test_backend"
if package not in sys.modules:
    module = ModuleType(package)
    module.__path__ = [str(backend)]
    sys.modules[package] = module
ready = importlib.import_module(package + ".ready_lds_issue")

LOADS = [
    "buffer_load_dwordx4 v[0:3], v[8:9], s[4:7], 0 idxen offen",
    "buffer_load_dwordx4 v[4:7], v[10:11], s[4:7], 0 idxen offen",
    "ds_read_b128 v[32:35], v100",
    "ds_read_b128 v[36:39], v100 offset:16",
]


def kernel(ops):
    instructions = [
        *ops, "s_waitcnt vmcnt(0) lgkmcnt(0)", "v_mmac_f32_16x16x16_bf16 v[40:43], v[32:33], v[0:1], v[40:43]",
        "s_endpgm"
    ]
    return "kernel:\n" + "".join(f"  {inst}\n" for inst in instructions) + ".Lfunc_end0:\n"


class TestReadyLdsIssue(unittest.TestCase):

    def test_independent_reads_move_and_are_idempotent(self):
        original = kernel(LOADS)
        changed, info = ready.schedule_ready_lds_issue(original, "kernel")
        self.assertEqual(len(info["sites"]), 1)
        self.assertEqual(changed, kernel(LOADS[2:] + LOADS[:2]))
        self.assertEqual(ready.schedule_ready_lds_issue(changed, "kernel")[0], changed)

    def test_control_and_dependency_boundaries_keep_original(self):
        negatives = [
            kernel([*LOADS[:3], LOADS[3].replace("v[36:39]", "v[0:3]")]),
            kernel([*LOADS[:1], LOADS[1] + " glc", *LOADS[2:]]),
            kernel([*LOADS[:2], LOADS[2] + " gds", LOADS[3]]),
            kernel([*LOADS[:2], "s_waitcnt vmcnt(0)", *LOADS[2:]]),
            kernel([*LOADS[:2], "s_barrier", *LOADS[2:]]),
            kernel(["s_cbranch_scc0 .Ltarget", *LOADS[:2], ".Ltarget:", *LOADS[2:]]),
            kernel(LOADS).replace("s_waitcnt vmcnt(0) lgkmcnt(0)", "s_waitcnt vmcnt(0)"),
        ]
        for original in negatives:
            with self.subTest(assembly=original):
                changed, info = ready.schedule_ready_lds_issue(original, "kernel")
                self.assertEqual(changed, original)
                self.assertFalse(info["sites"])

    def test_model_version_mismatch_keeps_original(self):
        original = kernel(LOADS)
        before = ready.isa.MODEL_API_VERSION
        try:
            ready.isa.MODEL_API_VERSION = -1
            changed, info = ready.schedule_ready_lds_issue(original, "kernel")
        finally:
            ready.isa.MODEL_API_VERSION = before
        self.assertEqual(changed, original)
        self.assertFalse(info["sites"])
