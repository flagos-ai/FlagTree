"""CPU-only regressions for the active first-MMAC priority schedule."""
import ast
from pathlib import Path
import re
import unittest


source = Path(__file__).resolve().parents[4] / 'third_party/hcu/backend/compiler_hcu.py'
function = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_priority_after_first_mmac')
namespace = {'re': re}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
rewrite = namespace['_priority_after_first_mmac']
MMAC = '\tv_mmac_f32_16x16x16_bf16 v[0:3], v[4:5], v[6:7], v[0:3]\n'


def kernel(body):
    return 'kernel:\n' + body + '\ts_endpgm\n.Lfunc_end0:\n'


class TestFirstMmacPriority(unittest.TestCase):
    def test_preserves_all_data_memory_wait_instructions_and_their_order(self):
        prefix = '\ts_waitcnt lgkmcnt(0)\n\ts_add_u32 s4, s4, 16\n'
        original = kernel('\ts_setprio 1\n' + prefix + MMAC + MMAC)
        result, count = rewrite(original, 'kernel')
        self.assertEqual(count, 1)
        self.assertEqual(result, kernel(prefix + MMAC + '\ts_setprio 1\n' + MMAC))

    def test_memory_sync_control_and_named_labels_stop_motion(self):
        for boundary in ('ds_read_b32 v8, v10', 'buffer_load_dword v8, v10, s[12:15], 0 offen',
                         's_barrier', 's_branch .Ltarget', '.Ltarget:'):
            with self.subTest(boundary=boundary):
                original = kernel('\ts_setprio 1\n\t' + boundary + '\n' + MMAC)
                self.assertEqual(rewrite(original, 'kernel'), (original, 0))

    def test_numeric_control_and_section_stack_reject_whole_kernel(self):
        for prefix in ('s_branch 4\n', 's_cbranch_scc0 -8\n', '.pushsection .text\n'):
            original = kernel(prefix + '\ts_setprio 1\n' + MMAC)
            self.assertEqual(rewrite(original, 'kernel'), (original, 0))

    def test_debug_anchors_require_no_executable_reference(self):
        original = kernel('\ts_setprio 1\n.Ltmp1:\n.loc 1 3 4\n' + MMAC)
        self.assertEqual(rewrite(original, 'kernel')[1], 1)
        referenced = original + '.section .rodata\n.long .Ltmp1\n'
        self.assertEqual(rewrite(referenced, 'kernel'), (referenced, 0))

    def test_unrelated_kernel_body_is_unchanged(self):
        other = 'other:\n\ts_setprio 1\n' + MMAC + '.Lfunc_end1:\n'
        result, count = rewrite(kernel('\ts_setprio 1\n' + MMAC) + other, 'kernel')
        self.assertEqual(count, 1)
        self.assertTrue(result.endswith(other))


if __name__ == '__main__':
    unittest.main()
