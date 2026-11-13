"""CPU-only completion regressions; import neither Triton nor a GPU runtime."""
import importlib.util
from pathlib import Path
import unittest


def load_checker(path):
    spec = importlib.util.spec_from_file_location("mma_phase_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check_native_readiness


check = load_checker(Path(__file__).resolve().parents[4] /
                     "third_party/hcu/backend/mma_phase.py")


def audit(*instructions):
    return check("kernel:\n  " + "\n  ".join(instructions) +
                 "\n  s_endpgm\n.Lfunc_end0:\n", "kernel")


class TestXadNativeReadiness(unittest.TestCase):
    def test_explicit_sources_and_destination_alias(self):
        for instruction in (
            "v_xad_u32 v83, v4, 32, s4",  # Observed gfx936 instruction.
            "v_xad_u32 v0, v1, v2, v3",
            "v_xad_u32 v0, s1, s2, s3",
            "v_xad_u32 v0, -2147483648, 4294967295, 0xffffffff",
            "v_xad_u32 v0, v0, 1, s4",
            "v_xad_u32 v0, 1, v0, s4",
            "v_xad_u32 v0, 1, s4, v0",
        ):
            with self.subTest(instruction=instruction):
                self.assertTrue(audit(instruction)["accepted"])

    def test_each_source_waits_for_ds_and_smem(self):
        for kind, load, source in (
            ("DS", "ds_read_b32 v4, v10", "v4"),
            ("SMEM", "s_load_dword s4, s[8:9], 0", "s4"),
        ):
            for index in range(3):
                sources = ["v1", "32", "s1"]
                sources[index] = source
                instruction = "v_xad_u32 v83, " + ", ".join(sources)
                with self.subTest(kind=kind, source=index):
                    result = audit(load, instruction)
                    self.assertFalse(result["accepted"])
                    self.assertEqual(result["reason"], "async register used before explicit completion")
                    self.assertEqual(result["op"], "v_xad_u32")
                    self.assertTrue(audit(load, "s_waitcnt lgkmcnt(0)", instruction)["accepted"])

    def test_pending_destination_cannot_be_overwritten(self):
        for sources in ("v1, 32, s1", "v83, 32, s1", "32, v83, s1", "32, s1, v83"):
            with self.subTest(sources=sources):
                instruction = "v_xad_u32 v83, " + sources
                result = audit("ds_read_b32 v83, v10", instruction)
                self.assertFalse(result["accepted"])
                self.assertIn(("v", 83), result["registers"])
                self.assertTrue(audit("ds_read_b32 v83, v10", "s_waitcnt lgkmcnt(0)", instruction)["accepted"])

    def test_vm_source_and_destination_keep_existing_completion_rules(self):
        for instruction in ("v_xad_u32 v83, v4, 32, s1", "v_xad_u32 v4, v1, 32, s1"):
            with self.subTest(instruction=instruction):
                load = "buffer_load_dword v4, v10, s[8:11], 0 offen"
                self.assertFalse(audit(load, instruction)["accepted"])
                self.assertTrue(audit(load, "s_waitcnt vmcnt(0)", instruction)["accepted"])

    def test_partial_wait_must_complete_the_actual_source(self):
        loads = ("ds_read_b32 v4, v10", "ds_read_b32 v5, v11")
        self.assertTrue(audit(*loads, "s_waitcnt lgkmcnt(1)", "v_xad_u32 v83, v4, 1, s1")["accepted"])
        self.assertFalse(audit(*loads, "s_waitcnt lgkmcnt(1)", "v_xad_u32 v83, v5, 1, s1")["accepted"])

    def test_malformed_xad_is_rejected(self):
        for args in (
            "", "v0, v1, 2", "v0, v1, 2, s1, v3", "s0, v1, 2, s1",
            "v[0:1], v1, 2, s1", "v0, v[1:2], 2, s1", "v0, v1, s[2:3], s1",
            "v0, v1, 1.0, s1", "v0, v1, -v2, s1", "v0, v1, abs(v2), s1",
            "v0, v1, 2, vcc", "v0, v1, 2, exec", "v0, v1, 2, s1 clamp",
            "v0, v1, 2, s1 junk", "v0, v1, 4294967296, s1",
            "v0, v1, -2147483649, s1", "v0, v1, 0x100000000, s1",
            "v0, v1, 0x, s1", "v0, v1, , s1", "v0, v1, 2, s1,",
        ):
            with self.subTest(args=args):
                result = audit("v_xad_u32 " + args)
                self.assertFalse(result["accepted"])
                self.assertEqual(result["reason"], "unsupported v_xad_u32 operands")

    def test_unknown_opcode_and_xad_variants_remain_rejected(self):
        for op in ("v_xad_u64", "v_xad_u32_e64", "v_xad_u32_fake", "v_unknown_u32"):
            with self.subTest(op=op):
                result = audit(op + " v0, v1, 2, s1")
                self.assertFalse(result["accepted"])
                self.assertEqual(result["reason"], "unsupported vector opcode")

    def test_cfg_path_with_incomplete_source_is_not_hidden(self):
        result = audit("ds_read_b32 v4, v10", "s_cbranch_scc1 .Ljoin",
                       "s_waitcnt lgkmcnt(0)", ".Ljoin:", "v_xad_u32 v83, v4, 32, s1")
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "async register used before explicit completion")


class TestPriorityNativeReadiness(unittest.TestCase):
    def test_priority_keeps_pending_responses(self):
        self.assertTrue(audit("s_setprio 1", "s_setprio 0")["accepted"])
        for load, use, wait in (
            ("ds_read_b32 v4, v10", "v_mov_b32 v8, v4", "s_waitcnt lgkmcnt(0)"),
            ("s_load_dword s4, s[8:9], 0", "s_add_u32 s8, s4, 1", "s_waitcnt lgkmcnt(0)"),
            ("buffer_load_dword v4, v10, s[8:11], 0 offen", "v_mov_b32 v8, v4", "s_waitcnt vmcnt(0)"),
        ):
            with self.subTest(load=load):
                result = audit(load, "s_setprio 1", "s_setprio 0", use)
                self.assertFalse(result["accepted"])
                self.assertEqual(result["reason"], "async register used before explicit completion")
                self.assertTrue(audit(load, "s_setprio 1", wait,
                                      "s_setprio 0", use)["accepted"])

    def test_unknown_priority_operands_and_opcodes_are_rejected(self):
        for operands in ("", "2", "3", "-1", "0x1", "s0", "v0", "0, 1", "1 junk"):
            with self.subTest(operands=operands):
                result = audit("s_setprio " + operands)
                self.assertFalse(result["accepted"])
                self.assertEqual(result["reason"], "unsupported s_setprio operands")
        self.assertFalse(audit("s_setprio_fake 1")["accepted"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
