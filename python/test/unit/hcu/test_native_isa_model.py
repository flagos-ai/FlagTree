"""Native wait semantics without a device or importing the Triton runtime."""
import hashlib
import importlib
from pathlib import Path
import itertools
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


BACKEND = Path(__file__).resolve().parents[4] / 'third_party/hcu/backend'
PACKAGE = '_hcu_native_isa_tests'
package = types.ModuleType(PACKAGE)
package.__path__ = [str(BACKEND)]
sys.modules[PACKAGE] = package
model = importlib.import_module(PACKAGE + '.native_isa_model')
coalesce = importlib.import_module(PACKAGE + '.waitcnt_coalesce').coalesce_lgkm_waits
native = importlib.import_module(PACKAGE + '.mma_phase')


def kernel(body, name='kernel'):
    return '.text\n' + name + ':\n' + body + '\n s_endpgm\n.Lfunc_end0:\n'


class TestNativeISAModel(unittest.TestCase):
    def test_wait_fields_and_hardware_limits(self):
        for lgkm in range(16):
            for vm in range(64):
                self.assertEqual(model.parse_wait(f'lgkmcnt({lgkm}) vmcnt({vm})'),
                                 {'lgkmcnt': lgkm, 'vmcnt': vm})
        for invalid in ('', 'lgkmcnt(16)', 'vmcnt(64)', 'lgkmcnt(-1)',
                        'lgkmcnt(0) lgkmcnt(1)', 'expcnt(0)', 'lgkmcnt(x)'):
            self.assertIsNone(model.parse_wait(invalid), invalid)

    def test_event_classification_retains_unknown_fallback(self):
        cases = {'v_add_u32 v0, v1, 1': 'PURE', 's_nop 0': 'PURE',
                 'ds_read_b32 v0, v1': 'DS_READ', 's_load_dword s0, s[2:3], 0': 'SMEM',
                 'buffer_load_dword v0, v1, s[0:3], 0 offen lds': 'G2L',
                 'buffer_store_dword v0, v1, s[0:3], 0 offen': 'OTHER_VM',
                 's_barrier': 'CTA', 's_barrier junk': 'UNKNOWN',
                 's_branch .Lloop': 'CONTROL', 'ds_write_b32 v0, v1': 'UNKNOWN',
                 'v_perm_b32 v0, v1, v2, v3': 'PURE',
                 'v_perm_b32_fake v0, v1, v2, v3': 'UNKNOWN', 's_waitcnt expcnt(0)': 'UNKNOWN'}
        for text, expected in cases.items():
            op, args = model.parse_instruction(text)
            self.assertEqual(model.classify_instruction(dict(op=op, args=args)), expected)

    def test_exhaustive_small_credit_sequences_and_idempotence(self):
        for count in range(2, 5):
            for values in itertools.product(range(4), repeat=count):
                source = kernel('\n'.join(f' s_waitcnt lgkmcnt({v})\n s_nop 0' for v in values))
                result, report = coalesce(source, 'kernel')
                self.assertTrue(report['accepted'], report)
                self.assertEqual(report['deleted_count'], count - 1)
                self.assertEqual(report['sites'][0]['new_first_credit'], min(values))
                again, second = coalesce(result, 'kernel')
                self.assertEqual(again, result)
                self.assertEqual(second['deleted_count'], 0)

    def test_new_events_and_unknowns_end_segments(self):
        for event in ('buffer_load_dword v0, v1, s[0:3], 0 offen',
                      'buffer_load_dword v0, v1, s[0:3], 0 offen lds',
                      'buffer_store_dword v0, v1, s[0:3], 0 offen',
                      's_load_dword s0, s[2:3], 0', 'ds_read_b32 v0, v1',
                      'ds_write_b32 v0, v1', 's_barrier', 'v_unknown_b32 v0, v1, v2, v3',
                      's_waitcnt vmcnt(0)', 's_waitcnt vmcnt(0) lgkmcnt(0)'):
            source = kernel(' s_waitcnt lgkmcnt(9)\n ' + event + '\n s_waitcnt lgkmcnt(6)')
            result, report = coalesce(source, 'kernel')
            self.assertTrue(report['accepted'], report)
            self.assertEqual(result, source, event)

    def test_alt_matrix_read_is_an_lds_event(self):
        for op in ('ds_read_m32x16_b16', 'ds_read_m32x16_b16_alt'):
            inst = dict(op=op, args='v[4:7], v10 offset:128')
            self.assertEqual(model.classify_instruction(inst), 'DS_READ')
            source = kernel(' s_waitcnt lgkmcnt(9)\n ' + op +
                            ' v[4:7], v10\n s_waitcnt lgkmcnt(6)')
            self.assertEqual(coalesce(source, 'kernel')[0], source)
        self.assertEqual(model.classify_instruction(dict(op='ds_read_m32x16_b16_alt_fake', args='')),
                         'UNKNOWN')

    def test_register_permute_allows_no_new_event_wait_proof(self):
        source = kernel(' s_waitcnt lgkmcnt(9)\n v_perm_b32 v0, v1, v2, v3\n'
                        ' s_waitcnt lgkmcnt(6)')
        result, report = coalesce(source, 'kernel')
        self.assertTrue(report['accepted'], report)
        self.assertEqual(report['deleted_count'], 1)
        self.assertEqual(result, kernel(' s_waitcnt lgkmcnt(6)\n v_perm_b32 v0, v1, v2, v3'))

    def test_incoming_edge_prevents_cross_block_coalescing(self):
        source = kernel(' s_cbranch_scc0 .Lside\n s_waitcnt lgkmcnt(9)\n'
                        ' s_nop 0\n.Lside:\n s_waitcnt lgkmcnt(6)')
        self.assertEqual(coalesce(source, 'kernel')[0], source)

    def test_debug_labels_preserve_bytes_without_creating_cfg_edges(self):
        source = kernel(' s_waitcnt lgkmcnt(9) ; first\n.loc 1 2 3\n.Ldebug:\n'
                        ' s_nop 0\n s_waitcnt lgkmcnt(6) ; second')
        expected = source.replace('lgkmcnt(9)', 'lgkmcnt(6)').replace(' s_waitcnt lgkmcnt(6) ; second\n', '')
        result, report = coalesce(source, 'kernel')
        self.assertTrue(report['accepted'], report)
        self.assertEqual(result, expected)

    def test_unsupported_code_syntax_returns_exact_input(self):
        for body in ('s_branch 1\n1:', 's_branch 1f\n1: s_nop 0',
                     's_setpc_b64 s[0:1]', '.inst 0xbe800000', '.long 0xbe800000',
                     '.include "unknown.s"', '.Linline: s_nop 0',
                     '.Lx:\n.Lx:', 's_branch .Lmissing', 's_join s0'):
            source = kernel(body + '\n s_waitcnt lgkmcnt(9)\n s_waitcnt lgkmcnt(6)')
            result, report = coalesce(source, 'kernel')
            self.assertFalse(report['accepted'], (body, report))
            self.assertEqual(result, source)
        source = kernel(' s_waitcnt lgkmcnt(9)\n s_waitcnt lgkmcnt(6)', 'other')
        self.assertFalse(coalesce(source, 'kernel')[1]['accepted'])

    def test_api_mismatch_returns_input_with_reason(self):
        source = kernel(' s_waitcnt lgkmcnt(9)\n s_waitcnt lgkmcnt(6)')
        with patch.object(model, 'MODEL_API_VERSION', 2):
            result, report = coalesce(source, 'kernel')
        self.assertEqual(result, source)
        self.assertFalse(report['accepted'])
        self.assertIn('API version', report['reason'])

    def test_comment_change_is_observed_without_disabling_pass(self):
        source = kernel(' s_waitcnt lgkmcnt(9)\n s_waitcnt lgkmcnt(6)')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'native_isa_model.py'
            path.write_text(Path(model.__file__).read_text() + '\n# comment-only change\n')
            with patch.object(model, '__file__', str(path)):
                _, report = coalesce(source, 'kernel')
            self.assertTrue(report['accepted'], report)
            self.assertEqual(report['deleted_count'], 1)
            self.assertEqual(report['classifier_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())


    def test_scalar_bitset_exact_opcode_and_register_effects(self):
        for bit in (0, 1):
            for width in (32, 64):
                op = f's_bitset{bit}_b{width}'
                dst = 's4' if width == 32 else 's[4:5]'
                regs = {('s', 4)} if width == 32 else {('s', 4), ('s', 5)}
                self.assertEqual(model.classify_instruction(dict(op=op, args=dst + ', 31')), 'PURE')
                for index in ('0', '31', '63', '-1', '0xffffffff'):
                    self.assertEqual(native._bitset_effects(op, [dst, index]), (regs, regs))
                self.assertEqual(native._bitset_effects(op, [dst, 's8']),
                                 (regs | {('s', 8)}, regs))
                self.assertEqual(native._bitset_effects(op, [dst, 's4']), (regs, regs))
        for op in ('s_bitset2_b32', 's_bitset1_b16', 's_bitset1_b32_fake'):
            self.assertEqual(model.classify_instruction(dict(op=op, args='s4, 1')), 'UNKNOWN')
            self.assertIsNone(native._bitset_effects(op, ['s4', '1']))

    def test_scalar_bitset_malformed_and_special_operands_stay_rejected(self):
        cases = [('s_bitset1_b32', ['v0', '1']), ('s_bitset1_b32', ['m0', '1']),
                 ('s_bitset1_b32', ['s0', 'v1']), ('s_bitset1_b32', ['s0', 'exec']),
                 ('s_bitset1_b32', ['s0', '4294967296']),
                 ('s_bitset1_b32', ['s0', '-2147483649']),
                 ('s_bitset1_b32', ['s0']), ('s_bitset1_b32', ['s0', '1', '2']),
                 ('s_bitset0_b64', ['s[3:4]', '1']),
                 ('s_bitset0_b64', ['s[2:4]', '1']), ('s_bitset0_b64', ['s2', '1'])]
        for op, args in cases:
            self.assertIsNone(native._bitset_effects(op, args), (op, args))
            self.assertFalse(native.check_native_readiness(kernel(' ' + op + ' ' + ', '.join(args)), 'kernel')['accepted'])

    def test_scalar_bitset_waits_for_old_destination_and_index(self):
        for op, dst in [('s_bitset0_b32', 's0'), ('s_bitset1_b32', 's0'),
                        ('s_bitset0_b64', 's[0:1]'), ('s_bitset1_b64', 's[0:1]')]:
            for load, index in [('s_load_dword s0, s[6:7], 0', '1'),
                                ('s_load_dword s4, s[6:7], 0', 's4'),
                                ('s_load_dword s1, s[6:7], 0', '1')]:
                body = ' ' + load + '\n ' + op + ' ' + dst + ', ' + index
                report = native.check_native_readiness(kernel(body), 'kernel')
                expected = not (load.startswith('s_load_dword s1,') and dst == 's0')
                self.assertEqual(not report['accepted'], expected, (body, report))
                waited = body.replace('\n ', '\n s_waitcnt lgkmcnt(0)\n ', 1)
                self.assertTrue(native.check_native_readiness(kernel(waited), 'kernel')['accepted'])

    def test_scalar_bitset_neither_completes_queues_nor_crosses_a_bad_cfg_path(self):
        body = (' s_load_dword s8, s[10:11], 0\n s_bitset1_b32 s0, 28\n'
                ' s_cbranch_scc0 .Luse\n s_waitcnt lgkmcnt(0)\n.Luse:\n v_mov_b32 v0, s8')
        self.assertFalse(native.check_native_readiness(kernel(body), 'kernel')['accepted'])
        body = body.replace('.Luse:\n', '.Luse:\n s_waitcnt lgkmcnt(0)\n')
        self.assertTrue(native.check_native_readiness(kernel(body), 'kernel')['accepted'])

    def test_scalar_bitset_is_a_pure_event_for_wait_coalescing(self):
        source = kernel(' s_waitcnt lgkmcnt(9)\n s_bitset1_b32 s0, 28\n s_waitcnt lgkmcnt(6)')
        result, report = coalesce(source, 'kernel')
        self.assertTrue(report['accepted'], report)
        self.assertEqual(report['deleted_count'], 1)
        self.assertTrue(native.check_native_readiness(result, 'kernel')['accepted'])





if __name__ == '__main__':
    unittest.main()
