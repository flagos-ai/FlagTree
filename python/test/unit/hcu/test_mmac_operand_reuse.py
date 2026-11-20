"""Native MMAC scheduling legality without importing a GPU runtime."""
from collections import defaultdict
import importlib
from pathlib import Path
import sys
from types import ModuleType
import unittest

backend = Path(__file__).resolve().parents[4] / "third_party/hcu/backend"
package = "hcu_mmac_operand_reuse_test_backend"
if package not in sys.modules:
    module = ModuleType(package)
    module.__path__ = [str(backend)]
    sys.modules[package] = module
reuse = importlib.import_module(package + ".mmac_operand_reuse")
isa = importlib.import_module(package + ".native_isa_model")


def dot(dst, b, a, dtype="bf16"):
    return f"v_mmac_f32_16x16x16_{dtype} v[{dst}:{dst+3}], v[{b}:{b+1}], v[{a}:{a+1}], v[{dst}:{dst+3}]"


def program(dtype="bf16", prefix=True, extra=()):
    body = [
        dot(4, 36, 128, dtype), "s_waitcnt lgkmcnt(0)",
        dot(8, 40, 128, dtype),
        dot(0, 34, 130, dtype),
        dot(4, 38, 130, dtype),
        dot(8, 42, 130, dtype)
    ]
    if not prefix:
        body.insert(0, dot(0, 32, 128, dtype))
    instructions = [
        "ds_read_b128 v[32:35], v100", "ds_read_b128 v[36:39], v100 offset:16", "ds_read_b128 v[40:43], v100 offset:32",
        "s_waitcnt lgkmcnt(1)"
    ]
    if prefix:
        instructions.append(dot(0, 32, 128, dtype))
    instructions += [
        "s_setprio 1", *body[:1], *extra, *body[1:], "s_setprio 0", "s_waitcnt lgkmcnt(0)", "s_barrier", "s_endpgm"
    ]
    return "kernel:\n  " + "\n  ".join(instructions) + "\n.Lfunc_end0:\n"


def accumulator_updates(assembly):
    result = defaultdict(list)
    for inst in isa.parse_kernel(assembly, "kernel")["instructions"]:
        if inst["op"].startswith("v_mmac_"):
            result[inst["args"].split(",")[0]].append((inst["op"], inst["args"]))
    return dict(result)


class TestMmacOperandReuse(unittest.TestCase):

    def test_unchanged_prefix_keeps_unused_debug_label(self):
        original = program().replace("  s_waitcnt lgkmcnt(1)", ".Ldebug:\n  s_waitcnt lgkmcnt(1)")
        changed, report = reuse.schedule_mmac_operand_reuse(original, "kernel")
        self.assertTrue(report["changed"])
        self.assertIn(".Ldebug:", changed)
        self.assertEqual(accumulator_updates(changed), accumulator_updates(original))

    def test_incoming_prefix_edge_keeps_original(self):
        original = program().replace("kernel:\n", "kernel:\n  s_cbranch_scc1 .Ljoin\n")
        original = original.replace("  s_waitcnt lgkmcnt(1)", ".Ljoin:\n  s_waitcnt lgkmcnt(1)")
        changed, report = reuse.schedule_mmac_operand_reuse(original, "kernel")
        self.assertFalse(report["changed"])
        self.assertEqual(changed, original)

    def test_reorders_both_dtypes_without_changing_each_sum(self):
        for dtype in ("bf16", "f16"):
            for prefix in (False, True):
                with self.subTest(dtype=dtype, prefix=prefix):
                    original = program(dtype, prefix)
                    changed, report = reuse.schedule_mmac_operand_reuse(original, "kernel")
                    self.assertTrue(report["accepted"])
                    self.assertTrue(report["changed"])
                    self.assertNotEqual(changed, original)
                    self.assertEqual(accumulator_updates(changed), accumulator_updates(original))
                    self.assertTrue(reuse.check_native_readiness(changed, "kernel")["accepted"])
                    for op in ("s_setprio 1", "s_setprio 0", "s_barrier"):
                        self.assertEqual(original.count(op), changed.count(op))

    def test_memory_and_side_effects_end_a_cohort(self):
        for inst in ("ds_read_b32 v80, v100", "s_barrier", "s_add_u32 s0, s0, 1",
                     "buffer_store_dword v128, v100, s[8:11], 0 offen"):
            with self.subTest(inst=inst):
                original = program(extra=(inst, ))
                changed, report = reuse.schedule_mmac_operand_reuse(original, "kernel")
                self.assertFalse(report["changed"])
                self.assertEqual(changed, original)

    def test_accumulator_used_as_operand_keeps_original(self):
        original = program().replace(dot(8, 40, 128), dot(8, 40, 4))
        changed, report = reuse.schedule_mmac_operand_reuse(original, "kernel")
        self.assertFalse(report["changed"])
        self.assertEqual(changed, original)

    def test_partially_overlapping_accumulators_keep_original(self):
        original = program().replace(dot(8, 40, 128), dot(2, 40, 128))
        changed, report = reuse.schedule_mmac_operand_reuse(original, "kernel")
        self.assertFalse(report["changed"])
        self.assertEqual(changed, original)

    def test_internal_branch_target_keeps_original(self):
        original = program().replace("  s_setprio 1\n", "  s_cbranch_scc1 .Linside\n  s_setprio 1\n")
        original = original.replace("  " + dot(4, 36, 128), ".Linside:\n  " + dot(4, 36, 128))
        changed, report = reuse.schedule_mmac_operand_reuse(original, "kernel")
        self.assertFalse(report["changed"])
        self.assertEqual(changed, original)

    def test_incomplete_original_input_is_not_repaired_silently(self):
        original = program().replace("s_waitcnt lgkmcnt(1)", "s_waitcnt lgkmcnt(2)")
        original = original.replace(dot(4, 36, 128), dot(4, 40, 128))
        changed, report = reuse.schedule_mmac_operand_reuse(original, "kernel")
        self.assertFalse(report["accepted"])
        self.assertEqual(changed, original)

    def test_missing_cohort_is_a_noop(self):
        original = "kernel:\n  s_endpgm\n.Lfunc_end0:\n"
        changed, report = reuse.schedule_mmac_operand_reuse(original, "kernel")
        self.assertTrue(report["accepted"])
        self.assertFalse(report["changed"])
        self.assertEqual(changed, original)

    def test_unknown_instruction_and_raw_code_fall_back(self):
        for inst in ("v_unknown_u32 v1, v2", ".long 0"):
            original = program(extra=(inst, ))
            changed, report = reuse.schedule_mmac_operand_reuse(original, "kernel")
            self.assertFalse(report["changed"])
            self.assertEqual(changed, original)


if __name__ == "__main__":
    unittest.main()
