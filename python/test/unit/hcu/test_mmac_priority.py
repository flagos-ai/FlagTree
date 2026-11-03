"""CPU-only tests of bounded native priority refinement; no runtime import."""
import ast
from pathlib import Path
import re
import unittest


source = Path(__file__).resolve().parents[4] / "third_party/hcu/backend/compiler_hcu.py"
tree = ast.parse(source.read_text())
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_chunk_mmac_priority")
namespace = {"re": re}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
rewrite = namespace["_chunk_mmac_priority"]
MMAC = "\tv_mmac_f32_16x16x16_bf16 v[0:3], v[4:5], v[6:7], v[0:3]\n"


def kernel(body):
    return "kernel:\n" + body + "\ts_endpgm\n.Lfunc_end0:\n"


def bounded(count):
    return "\ts_setprio 1\n" + MMAC * count + "\ts_setprio 0\n"


def without_priority(text):
    return "".join(line for line in text.splitlines(keepends=True) if not line.strip().startswith("s_setprio"))


class TestMmacPriorityChunk(unittest.TestCase):
    def test_refines_multiple_scopes_and_keeps_every_other_byte(self):
        original = kernel(bounded(16) + "\ts_waitcnt lgkmcnt(0)\n" + bounded(8))
        result, count = rewrite(original, "kernel", 8)
        self.assertEqual(count, 3)
        self.assertEqual(without_priority(result), without_priority(original))
        instructions = [line.strip().split()[0] for line in result.splitlines() if line.startswith("\t")]
        expected = ["v_mmac_f32_16x16x16_bf16", "s_setprio"] + ["v_mmac_f32_16x16x16_bf16"] * 7 + ["s_setprio"]
        self.assertEqual(instructions[:10], expected)
        self.assertEqual(rewrite(result, "kernel", 8), (result, 0))

    def test_waits_lds_barriers_and_vector_operations_keep_exact_order(self):
        operations = ("\ts_waitcnt vmcnt(4)\n\ts_barrier\n\tds_read_b128 v[8:11], v12\n"
                      "\ts_waitcnt lgkmcnt(0)\n\tv_mov_b32 v4, v8\n")
        original = kernel("\ts_setprio 1\n" + operations + MMAC * 16 + "\ts_setprio 0\n")
        result, count = rewrite(original, "kernel", 8)
        self.assertEqual(count, 2)
        self.assertEqual(without_priority(result), without_priority(original))
        self.assertTrue(result.startswith("kernel:\n" + operations + MMAC + "\ts_setprio 1\n"))

    def test_partial_unsupported_region_does_not_hide_other_eligible_scopes(self):
        original = kernel(bounded(5) + bounded(8))
        result, count = rewrite(original, "kernel", 4)
        self.assertEqual(count, 2)
        self.assertIn(bounded(5), result)
        self.assertEqual(without_priority(result), without_priority(original))

    def test_invalid_priority_nesting_and_operands_preserve_entire_function(self):
        for body in ("s_setprio 2\n", "s_setprio 1\ns_setprio 1\n", "s_setprio 0\n", "s_setprio 1\n",
                     "s_setprio 1, 0\n", "s_setprio_mask 1\n"):
            original = kernel(bounded(8) + body)
            self.assertEqual(rewrite(original, "kernel", 8), (original, 0))

    def test_control_flow_and_raw_encodings_keep_region_unchanged(self):
        for instruction in ("s_branch .Ltarget", "s_cbranch_scc1 .Ltarget", "s_setpc_b64 s[0:1]",
                            "s_swappc_b64 s[0:1], s[2:3]", "s_endpgm", ".long 0x12345678", ".inst 0", ".Ltarget:"):
            original = kernel("s_setprio 1\n" + MMAC * 4 + instruction + "\n" + MMAC * 4 + "s_setprio 0\n")
            self.assertEqual(rewrite(original, "kernel", 8), (original, 0))

    def test_numeric_branches_and_section_stacks_reject_entire_function(self):
        for prefix in ("s_branch 4\n", "s_cbranch_scc0 -8\n", ".pushsection .text\n", ".previous\n", ".subsection 1\n"):
            original = kernel(prefix + bounded(8))
            self.assertEqual(rewrite(original, "kernel", 8), (original, 0))

    def test_only_unreferenced_debug_anchor_is_accepted(self):
        body = "s_setprio 1\n.Ltmp1:\n.loc 1 3 4\n" + MMAC * 8 + "s_setprio 0\n"
        original = kernel(body)
        self.assertEqual(rewrite(original, "kernel", 8)[1], 1)
        for reference in (".long .Ltmp1\n", "s_branch .Ltmp1\n", '.section .debug_fake,"a"\n.long .Ltmp1\n'):
            original = kernel(body) + reference
            self.assertEqual(rewrite(original, "kernel", 8), (original, 0))
        original = kernel(body) + '.section .debug_info,""\n.long .Ltmp1\n'
        self.assertEqual(rewrite(original, "kernel", 8)[1], 1)

    def test_target_function_only_and_parameters_are_generic(self):
        original = kernel(bounded(12)) + "other:\n" + bounded(8) + ".Lfunc_end1:\n"
        result, count = rewrite(original, "kernel", 3)
        self.assertEqual(count, 4)
        self.assertEqual(result.split("other:")[1], original.split("other:")[1])
        self.assertEqual(rewrite(original, "missing", 8), (original, 0))
        for chunk in (0, 1, -1, True, 8.0):
            with self.assertRaises(ValueError):
                rewrite(original, "kernel", chunk)


if __name__ == "__main__":
    unittest.main()
