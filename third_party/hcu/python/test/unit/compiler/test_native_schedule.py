# native isa model
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

_native_isa_model_BACKEND = Path(__file__).resolve().parents[6] / 'third_party/hcu/backend'
_native_isa_model_PACKAGE = '_hcu_native_isa_tests'
_native_isa_model_package = types.ModuleType(_native_isa_model_PACKAGE)
_native_isa_model_package.__path__ = [str(_native_isa_model_BACKEND)]
sys.modules[_native_isa_model_PACKAGE] = _native_isa_model_package
_native_isa_model_model = importlib.import_module(_native_isa_model_PACKAGE + '.native_isa_model')
_native_isa_model_coalesce = importlib.import_module(_native_isa_model_PACKAGE +
                                                     '.waitcnt_coalesce').coalesce_lgkm_waits
_native_isa_model_native = importlib.import_module(_native_isa_model_PACKAGE + '.mma_phase')


def _native_isa_model_kernel(body, name='kernel'):
    return '.text\n' + name + ':\n' + body + '\n s_endpgm\n.Lfunc_end0:\n'


class TestNativeISAModel(unittest.TestCase):

    def test_wait_fields_and_hardware_limits(self):
        for lgkm in range(16):
            for vm in range(64):
                self.assertEqual(_native_isa_model_model.parse_wait(f'lgkmcnt({lgkm}) vmcnt({vm})'),
                                 {'lgkmcnt': lgkm, 'vmcnt': vm})
        for invalid in ('', 'lgkmcnt(16)', 'vmcnt(64)', 'lgkmcnt(-1)', 'lgkmcnt(0) lgkmcnt(1)', 'expcnt(0)',
                        'lgkmcnt(x)'):
            self.assertIsNone(_native_isa_model_model.parse_wait(invalid), invalid)

    def test_event_classification_retains_unknown_fallback(self):
        cases = {
            'v_add_u32 v0, v1, 1': 'PURE', 's_nop 0': 'PURE', 'ds_read_b32 v0, v1': 'DS_READ',
            's_load_dword s0, s[2:3], 0': 'SMEM', 'buffer_load_dword v0, v1, s[0:3], 0 offen lds': 'G2L',
            'buffer_store_dword v0, v1, s[0:3], 0 offen': 'OTHER_VM', 's_barrier': 'CTA', 's_barrier junk': 'UNKNOWN',
            's_branch .Lloop': 'CONTROL', 'ds_write_b32 v0, v1': 'UNKNOWN', 'v_perm_b32 v0, v1, v2, v3': 'PURE',
            'v_perm_b32_fake v0, v1, v2, v3': 'UNKNOWN', 's_waitcnt expcnt(0)': 'UNKNOWN'
        }
        for text, expected in cases.items():
            op, args = _native_isa_model_model.parse_instruction(text)
            self.assertEqual(_native_isa_model_model.classify_instruction(dict(op=op, args=args)), expected)

    def test_exhaustive_small_credit_sequences_and_idempotence(self):
        for count in range(2, 5):
            for values in itertools.product(range(4), repeat=count):
                source = _native_isa_model_kernel('\n'.join(f' s_waitcnt lgkmcnt({v})\n s_nop 0' for v in values))
                result, report = _native_isa_model_coalesce(source, 'kernel')
                self.assertTrue(report['accepted'], report)
                self.assertEqual(report['deleted_count'], count - 1)
                self.assertEqual(report['sites'][0]['new_first_credit'], min(values))
                again, second = _native_isa_model_coalesce(result, 'kernel')
                self.assertEqual(again, result)
                self.assertEqual(second['deleted_count'], 0)

    def test_new_events_and_unknowns_end_segments(self):
        for event in ('buffer_load_dword v0, v1, s[0:3], 0 offen', 'buffer_load_dword v0, v1, s[0:3], 0 offen lds',
                      'buffer_store_dword v0, v1, s[0:3], 0 offen', 's_load_dword s0, s[2:3], 0', 'ds_read_b32 v0, v1',
                      'ds_write_b32 v0, v1', 's_barrier', 'v_unknown_b32 v0, v1, v2, v3', 's_waitcnt vmcnt(0)',
                      's_waitcnt vmcnt(0) lgkmcnt(0)'):
            source = _native_isa_model_kernel(' s_waitcnt lgkmcnt(9)\n ' + event + '\n s_waitcnt lgkmcnt(6)')
            result, report = _native_isa_model_coalesce(source, 'kernel')
            self.assertTrue(report['accepted'], report)
            self.assertEqual(result, source, event)

    def test_alt_matrix_read_is_an_lds_event(self):
        for op in ('ds_read_m32x16_b16', 'ds_read_m32x16_b16_alt'):
            inst = dict(op=op, args='v[4:7], v10 offset:128')
            self.assertEqual(_native_isa_model_model.classify_instruction(inst), 'DS_READ')
            source = _native_isa_model_kernel(' s_waitcnt lgkmcnt(9)\n ' + op + ' v[4:7], v10\n s_waitcnt lgkmcnt(6)')
            self.assertEqual(_native_isa_model_coalesce(source, 'kernel')[0], source)
        self.assertEqual(_native_isa_model_model.classify_instruction(dict(op='ds_read_m32x16_b16_alt_fake', args='')),
                         'UNKNOWN')

    def test_register_permute_allows_no_new_event_wait_proof(self):
        source = _native_isa_model_kernel(' s_waitcnt lgkmcnt(9)\n v_perm_b32 v0, v1, v2, v3\n'
                                          ' s_waitcnt lgkmcnt(6)')
        result, report = _native_isa_model_coalesce(source, 'kernel')
        self.assertTrue(report['accepted'], report)
        self.assertEqual(report['deleted_count'], 1)
        self.assertEqual(result, _native_isa_model_kernel(' s_waitcnt lgkmcnt(6)\n v_perm_b32 v0, v1, v2, v3'))

    def test_incoming_edge_prevents_cross_block_coalescing(self):
        source = _native_isa_model_kernel(' s_cbranch_scc0 .Lside\n s_waitcnt lgkmcnt(9)\n'
                                          ' s_nop 0\n.Lside:\n s_waitcnt lgkmcnt(6)')
        self.assertEqual(_native_isa_model_coalesce(source, 'kernel')[0], source)

    def test_debug_labels_preserve_bytes_without_creating_cfg_edges(self):
        source = _native_isa_model_kernel(' s_waitcnt lgkmcnt(9) ; first\n.loc 1 2 3\n.Ldebug:\n'
                                          ' s_nop 0\n s_waitcnt lgkmcnt(6) ; second')
        expected = source.replace('lgkmcnt(9)', 'lgkmcnt(6)').replace(' s_waitcnt lgkmcnt(6) ; second\n', '')
        result, report = _native_isa_model_coalesce(source, 'kernel')
        self.assertTrue(report['accepted'], report)
        self.assertEqual(result, expected)

    def test_unsupported_code_syntax_returns_exact_input(self):
        for body in ('s_branch 1\n1:', 's_branch 1f\n1: s_nop 0', 's_setpc_b64 s[0:1]', '.inst 0xbe800000',
                     '.long 0xbe800000', '.include "unknown.s"', '.Linline: s_nop 0', '.Lx:\n.Lx:',
                     's_branch .Lmissing', 's_join s0'):
            source = _native_isa_model_kernel(body + '\n s_waitcnt lgkmcnt(9)\n s_waitcnt lgkmcnt(6)')
            result, report = _native_isa_model_coalesce(source, 'kernel')
            self.assertFalse(report['accepted'], (body, report))
            self.assertEqual(result, source)
        source = _native_isa_model_kernel(' s_waitcnt lgkmcnt(9)\n s_waitcnt lgkmcnt(6)', 'other')
        self.assertFalse(_native_isa_model_coalesce(source, 'kernel')[1]['accepted'])

    def test_api_mismatch_returns_input_with_reason(self):
        source = _native_isa_model_kernel(' s_waitcnt lgkmcnt(9)\n s_waitcnt lgkmcnt(6)')
        with patch.object(_native_isa_model_model, 'MODEL_API_VERSION', 2):
            result, report = _native_isa_model_coalesce(source, 'kernel')
        self.assertEqual(result, source)
        self.assertFalse(report['accepted'])
        self.assertIn('API version', report['reason'])

    def test_comment_change_is_observed_without_disabling_pass(self):
        source = _native_isa_model_kernel(' s_waitcnt lgkmcnt(9)\n s_waitcnt lgkmcnt(6)')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'native_isa_model.py'
            path.write_text(Path(_native_isa_model_model.__file__).read_text() + '\n# comment-only change\n')
            with patch.object(_native_isa_model_model, '__file__', str(path)):
                _, report = _native_isa_model_coalesce(source, 'kernel')
            self.assertTrue(report['accepted'], report)
            self.assertEqual(report['deleted_count'], 1)
            self.assertEqual(report['classifier_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_scalar_bitset_exact_opcode_and_register_effects(self):
        for bit in (0, 1):
            for width in (32, 64):
                op = f's_bitset{bit}_b{width}'
                dst = 's4' if width == 32 else 's[4:5]'
                regs = {('s', 4)} if width == 32 else {('s', 4), ('s', 5)}
                self.assertEqual(_native_isa_model_model.classify_instruction(dict(op=op, args=dst + ', 31')), 'PURE')
                for index in ('0', '31', '63', '-1', '0xffffffff'):
                    self.assertEqual(_native_isa_model_native._bitset_effects(op, [dst, index]), (regs, regs))
                self.assertEqual(_native_isa_model_native._bitset_effects(op, [dst, 's8']), (regs | {('s', 8)}, regs))
                self.assertEqual(_native_isa_model_native._bitset_effects(op, [dst, 's4']), (regs, regs))
        for op in ('s_bitset2_b32', 's_bitset1_b16', 's_bitset1_b32_fake'):
            self.assertEqual(_native_isa_model_model.classify_instruction(dict(op=op, args='s4, 1')), 'UNKNOWN')
            self.assertIsNone(_native_isa_model_native._bitset_effects(op, ['s4', '1']))

    def test_scalar_bitset_malformed_and_special_operands_stay_rejected(self):
        cases = [('s_bitset1_b32', ['v0', '1']), ('s_bitset1_b32', ['m0', '1']), ('s_bitset1_b32', ['s0', 'v1']),
                 ('s_bitset1_b32', ['s0', 'exec']), ('s_bitset1_b32', ['s0', '4294967296']),
                 ('s_bitset1_b32', ['s0', '-2147483649']), ('s_bitset1_b32', ['s0']),
                 ('s_bitset1_b32', ['s0', '1', '2']), ('s_bitset0_b64', ['s[3:4]', '1']),
                 ('s_bitset0_b64', ['s[2:4]', '1']), ('s_bitset0_b64', ['s2', '1'])]
        for op, args in cases:
            self.assertIsNone(_native_isa_model_native._bitset_effects(op, args), (op, args))
            self.assertFalse(
                _native_isa_model_native.check_native_readiness(
                    _native_isa_model_kernel(' ' + op + ' ' + ', '.join(args)), 'kernel')['accepted'])

    def test_scalar_bitset_waits_for_old_destination_and_index(self):
        for op, dst in [('s_bitset0_b32', 's0'), ('s_bitset1_b32', 's0'), ('s_bitset0_b64', 's[0:1]'),
                        ('s_bitset1_b64', 's[0:1]')]:
            for load, index in [('s_load_dword s0, s[6:7], 0', '1'), ('s_load_dword s4, s[6:7], 0', 's4'),
                                ('s_load_dword s1, s[6:7], 0', '1')]:
                body = ' ' + load + '\n ' + op + ' ' + dst + ', ' + index
                report = _native_isa_model_native.check_native_readiness(_native_isa_model_kernel(body), 'kernel')
                expected = not (load.startswith('s_load_dword s1,') and dst == 's0')
                self.assertEqual(not report['accepted'], expected, (body, report))
                waited = body.replace('\n ', '\n s_waitcnt lgkmcnt(0)\n ', 1)
                self.assertTrue(
                    _native_isa_model_native.check_native_readiness(_native_isa_model_kernel(waited),
                                                                    'kernel')['accepted'])

    def test_scalar_bitset_neither_completes_queues_nor_crosses_a_bad_cfg_path(self):
        body = (' s_load_dword s8, s[10:11], 0\n s_bitset1_b32 s0, 28\n'
                ' s_cbranch_scc0 .Luse\n s_waitcnt lgkmcnt(0)\n.Luse:\n v_mov_b32 v0, s8')
        self.assertFalse(
            _native_isa_model_native.check_native_readiness(_native_isa_model_kernel(body), 'kernel')['accepted'])
        body = body.replace('.Luse:\n', '.Luse:\n s_waitcnt lgkmcnt(0)\n')
        self.assertTrue(
            _native_isa_model_native.check_native_readiness(_native_isa_model_kernel(body), 'kernel')['accepted'])

    def test_scalar_bitset_is_a_pure_event_for_wait_coalescing(self):
        source = _native_isa_model_kernel(' s_waitcnt lgkmcnt(9)\n s_bitset1_b32 s0, 28\n s_waitcnt lgkmcnt(6)')
        result, report = _native_isa_model_coalesce(source, 'kernel')
        self.assertTrue(report['accepted'], report)
        self.assertEqual(report['deleted_count'], 1)
        self.assertTrue(_native_isa_model_native.check_native_readiness(result, 'kernel')['accepted'])


if __name__ == '__main__':
    unittest.main()

# mma phase native
import importlib.util


def _mma_phase_native_load_checker(path):
    spec = importlib.util.spec_from_file_location("mma_phase_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check_native_readiness


_mma_phase_native_check = _mma_phase_native_load_checker(
    Path(__file__).resolve().parents[6] / "third_party/hcu/backend/mma_phase.py")


def _mma_phase_native_audit(*instructions):
    return _mma_phase_native_check("kernel:\n  " + "\n  ".join(instructions) + "\n  s_endpgm\n.Lfunc_end0:\n", "kernel")


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
                self.assertTrue(_mma_phase_native_audit(instruction)["accepted"])

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
                    result = _mma_phase_native_audit(load, instruction)
                    self.assertFalse(result["accepted"])
                    self.assertEqual(result["reason"], "async register used before explicit completion")
                    self.assertEqual(result["op"], "v_xad_u32")
                    self.assertTrue(_mma_phase_native_audit(load, "s_waitcnt lgkmcnt(0)", instruction)["accepted"])

    def test_pending_destination_cannot_be_overwritten(self):
        for sources in ("v1, 32, s1", "v83, 32, s1", "32, v83, s1", "32, s1, v83"):
            with self.subTest(sources=sources):
                instruction = "v_xad_u32 v83, " + sources
                result = _mma_phase_native_audit("ds_read_b32 v83, v10", instruction)
                self.assertFalse(result["accepted"])
                self.assertIn(("v", 83), result["registers"])
                self.assertTrue(
                    _mma_phase_native_audit("ds_read_b32 v83, v10", "s_waitcnt lgkmcnt(0)", instruction)["accepted"])

    def test_vm_source_and_destination_keep_existing_completion_rules(self):
        for instruction in ("v_xad_u32 v83, v4, 32, s1", "v_xad_u32 v4, v1, 32, s1"):
            with self.subTest(instruction=instruction):
                load = "buffer_load_dword v4, v10, s[8:11], 0 offen"
                self.assertFalse(_mma_phase_native_audit(load, instruction)["accepted"])
                self.assertTrue(_mma_phase_native_audit(load, "s_waitcnt vmcnt(0)", instruction)["accepted"])

    def test_partial_wait_must_complete_the_actual_source(self):
        loads = ("ds_read_b32 v4, v10", "ds_read_b32 v5, v11")
        self.assertTrue(_mma_phase_native_audit(*loads, "s_waitcnt lgkmcnt(1)", "v_xad_u32 v83, v4, 1, s1")["accepted"])
        self.assertFalse(
            _mma_phase_native_audit(*loads, "s_waitcnt lgkmcnt(1)", "v_xad_u32 v83, v5, 1, s1")["accepted"])

    def test_malformed_xad_is_rejected(self):
        for args in (
                "",
                "v0, v1, 2",
                "v0, v1, 2, s1, v3",
                "s0, v1, 2, s1",
                "v[0:1], v1, 2, s1",
                "v0, v[1:2], 2, s1",
                "v0, v1, s[2:3], s1",
                "v0, v1, 1.0, s1",
                "v0, v1, -v2, s1",
                "v0, v1, abs(v2), s1",
                "v0, v1, 2, vcc",
                "v0, v1, 2, exec",
                "v0, v1, 2, s1 clamp",
                "v0, v1, 2, s1 junk",
                "v0, v1, 4294967296, s1",
                "v0, v1, -2147483649, s1",
                "v0, v1, 0x100000000, s1",
                "v0, v1, 0x, s1",
                "v0, v1, , s1",
                "v0, v1, 2, s1,",
        ):
            with self.subTest(args=args):
                result = _mma_phase_native_audit("v_xad_u32 " + args)
                self.assertFalse(result["accepted"])
                self.assertEqual(result["reason"], "unsupported v_xad_u32 operands")

    def test_unknown_opcode_and_xad_variants_remain_rejected(self):
        for op in ("v_xad_u64", "v_xad_u32_e64", "v_xad_u32_fake", "v_unknown_u32"):
            with self.subTest(op=op):
                result = _mma_phase_native_audit(op + " v0, v1, 2, s1")
                self.assertFalse(result["accepted"])
                self.assertEqual(result["reason"], "unsupported vector opcode")

    def test_cfg_path_with_incomplete_source_is_not_hidden(self):
        result = _mma_phase_native_audit("ds_read_b32 v4, v10", "s_cbranch_scc1 .Ljoin", "s_waitcnt lgkmcnt(0)",
                                         ".Ljoin:", "v_xad_u32 v83, v4, 32, s1")
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "async register used before explicit completion")


class TestPriorityNativeReadiness(unittest.TestCase):

    def test_priority_keeps_pending_responses(self):
        self.assertTrue(_mma_phase_native_audit("s_setprio 1", "s_setprio 0")["accepted"])
        for load, use, wait in (
            ("ds_read_b32 v4, v10", "v_mov_b32 v8, v4", "s_waitcnt lgkmcnt(0)"),
            ("s_load_dword s4, s[8:9], 0", "s_add_u32 s8, s4, 1", "s_waitcnt lgkmcnt(0)"),
            ("buffer_load_dword v4, v10, s[8:11], 0 offen", "v_mov_b32 v8, v4", "s_waitcnt vmcnt(0)"),
        ):
            with self.subTest(load=load):
                result = _mma_phase_native_audit(load, "s_setprio 1", "s_setprio 0", use)
                self.assertFalse(result["accepted"])
                self.assertEqual(result["reason"], "async register used before explicit completion")
                self.assertTrue(_mma_phase_native_audit(load, "s_setprio 1", wait, "s_setprio 0", use)["accepted"])

    def test_unknown_priority_operands_and_opcodes_are_rejected(self):
        for operands in ("", "2", "3", "-1", "0x1", "s0", "v0", "0, 1", "1 junk"):
            with self.subTest(operands=operands):
                result = _mma_phase_native_audit("s_setprio " + operands)
                self.assertFalse(result["accepted"])
                self.assertEqual(result["reason"], "unsupported s_setprio operands")
        self.assertFalse(_mma_phase_native_audit("s_setprio_fake 1")["accepted"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

# mmac operand reuse
from collections import defaultdict
from types import ModuleType

_mmac_operand_reuse_backend = Path(__file__).resolve().parents[6] / "third_party/hcu/backend"
_mmac_operand_reuse_package = "hcu_mmac_operand_reuse_test_backend"
if _mmac_operand_reuse_package not in sys.modules:
    module = ModuleType(_mmac_operand_reuse_package)
    module.__path__ = [str(_mmac_operand_reuse_backend)]
    sys.modules[_mmac_operand_reuse_package] = module
_mmac_operand_reuse_reuse = importlib.import_module(_mmac_operand_reuse_package + ".mmac_operand_reuse")
_mmac_operand_reuse_isa = importlib.import_module(_mmac_operand_reuse_package + ".native_isa_model")


def _mmac_operand_reuse_dot(dst, b, a, dtype="bf16"):
    return f"v_mmac_f32_16x16x16_{dtype} v[{dst}:{dst+3}], v[{b}:{b+1}], v[{a}:{a+1}], v[{dst}:{dst+3}]"


def _mmac_operand_reuse_program(dtype="bf16", prefix=True, extra=()):
    body = [
        _mmac_operand_reuse_dot(4, 36, 128, dtype), "s_waitcnt lgkmcnt(0)",
        _mmac_operand_reuse_dot(8, 40, 128, dtype),
        _mmac_operand_reuse_dot(0, 34, 130, dtype),
        _mmac_operand_reuse_dot(4, 38, 130, dtype),
        _mmac_operand_reuse_dot(8, 42, 130, dtype)
    ]
    if not prefix:
        body.insert(0, _mmac_operand_reuse_dot(0, 32, 128, dtype))
    instructions = [
        "ds_read_b128 v[32:35], v100", "ds_read_b128 v[36:39], v100 offset:16", "ds_read_b128 v[40:43], v100 offset:32",
        "s_waitcnt lgkmcnt(1)"
    ]
    if prefix:
        instructions.append(_mmac_operand_reuse_dot(0, 32, 128, dtype))
    instructions += [
        "s_setprio 1", *body[:1], *extra, *body[1:], "s_setprio 0", "s_waitcnt lgkmcnt(0)", "s_barrier", "s_endpgm"
    ]
    return "kernel:\n  " + "\n  ".join(instructions) + "\n.Lfunc_end0:\n"


def _mmac_operand_reuse_accumulator_updates(assembly):
    result = defaultdict(list)
    for inst in _mmac_operand_reuse_isa.parse_kernel(assembly, "kernel")["instructions"]:
        if inst["op"].startswith("v_mmac_"):
            result[inst["args"].split(",")[0]].append((inst["op"], inst["args"]))
    return dict(result)


class TestMmacOperandReuse(unittest.TestCase):

    def test_unchanged_prefix_keeps_unused_debug_label(self):
        original = _mmac_operand_reuse_program().replace("  s_waitcnt lgkmcnt(1)", ".Ldebug:\n  s_waitcnt lgkmcnt(1)")
        changed, report = _mmac_operand_reuse_reuse.schedule_mmac_operand_reuse(original, "kernel")
        self.assertTrue(report["changed"])
        self.assertIn(".Ldebug:", changed)
        self.assertEqual(_mmac_operand_reuse_accumulator_updates(changed),
                         _mmac_operand_reuse_accumulator_updates(original))

    def test_incoming_prefix_edge_keeps_original(self):
        original = _mmac_operand_reuse_program().replace("kernel:\n", "kernel:\n  s_cbranch_scc1 .Ljoin\n")
        original = original.replace("  s_waitcnt lgkmcnt(1)", ".Ljoin:\n  s_waitcnt lgkmcnt(1)")
        changed, report = _mmac_operand_reuse_reuse.schedule_mmac_operand_reuse(original, "kernel")
        self.assertFalse(report["changed"])
        self.assertEqual(changed, original)

    def test_reorders_both_dtypes_without_changing_each_sum(self):
        for dtype in ("bf16", "f16"):
            for prefix in (False, True):
                with self.subTest(dtype=dtype, prefix=prefix):
                    original = _mmac_operand_reuse_program(dtype, prefix)
                    changed, report = _mmac_operand_reuse_reuse.schedule_mmac_operand_reuse(original, "kernel")
                    self.assertTrue(report["accepted"])
                    self.assertTrue(report["changed"])
                    self.assertNotEqual(changed, original)
                    self.assertEqual(_mmac_operand_reuse_accumulator_updates(changed),
                                     _mmac_operand_reuse_accumulator_updates(original))
                    self.assertTrue(_mmac_operand_reuse_reuse.check_native_readiness(changed, "kernel")["accepted"])
                    for op in ("s_setprio 1", "s_setprio 0", "s_barrier"):
                        self.assertEqual(original.count(op), changed.count(op))

    def test_memory_and_side_effects_end_a_cohort(self):
        for inst in ("ds_read_b32 v80, v100", "s_barrier", "s_add_u32 s0, s0, 1",
                     "buffer_store_dword v128, v100, s[8:11], 0 offen"):
            with self.subTest(inst=inst):
                original = _mmac_operand_reuse_program(extra=(inst, ))
                changed, report = _mmac_operand_reuse_reuse.schedule_mmac_operand_reuse(original, "kernel")
                self.assertFalse(report["changed"])
                self.assertEqual(changed, original)

    def test_accumulator_used_as_operand_keeps_original(self):
        original = _mmac_operand_reuse_program().replace(_mmac_operand_reuse_dot(8, 40, 128),
                                                         _mmac_operand_reuse_dot(8, 40, 4))
        changed, report = _mmac_operand_reuse_reuse.schedule_mmac_operand_reuse(original, "kernel")
        self.assertFalse(report["changed"])
        self.assertEqual(changed, original)

    def test_partially_overlapping_accumulators_keep_original(self):
        original = _mmac_operand_reuse_program().replace(_mmac_operand_reuse_dot(8, 40, 128),
                                                         _mmac_operand_reuse_dot(2, 40, 128))
        changed, report = _mmac_operand_reuse_reuse.schedule_mmac_operand_reuse(original, "kernel")
        self.assertFalse(report["changed"])
        self.assertEqual(changed, original)

    def test_internal_branch_target_keeps_original(self):
        original = _mmac_operand_reuse_program().replace("  s_setprio 1\n",
                                                         "  s_cbranch_scc1 .Linside\n  s_setprio 1\n")
        original = original.replace("  " + _mmac_operand_reuse_dot(4, 36, 128),
                                    ".Linside:\n  " + _mmac_operand_reuse_dot(4, 36, 128))
        changed, report = _mmac_operand_reuse_reuse.schedule_mmac_operand_reuse(original, "kernel")
        self.assertFalse(report["changed"])
        self.assertEqual(changed, original)

    def test_incomplete_original_input_is_not_repaired_silently(self):
        original = _mmac_operand_reuse_program().replace("s_waitcnt lgkmcnt(1)", "s_waitcnt lgkmcnt(2)")
        original = original.replace(_mmac_operand_reuse_dot(4, 36, 128), _mmac_operand_reuse_dot(4, 40, 128))
        changed, report = _mmac_operand_reuse_reuse.schedule_mmac_operand_reuse(original, "kernel")
        self.assertFalse(report["accepted"])
        self.assertEqual(changed, original)

    def test_missing_cohort_is_a_noop(self):
        original = "kernel:\n  s_endpgm\n.Lfunc_end0:\n"
        changed, report = _mmac_operand_reuse_reuse.schedule_mmac_operand_reuse(original, "kernel")
        self.assertTrue(report["accepted"])
        self.assertFalse(report["changed"])
        self.assertEqual(changed, original)

    def test_unknown_instruction_and_raw_code_fall_back(self):
        for inst in ("v_unknown_u32 v1, v2", ".long 0"):
            original = _mmac_operand_reuse_program(extra=(inst, ))
            changed, report = _mmac_operand_reuse_reuse.schedule_mmac_operand_reuse(original, "kernel")
            self.assertFalse(report["changed"])
            self.assertEqual(changed, original)


if __name__ == "__main__":
    unittest.main()

# ready lds issue

_ready_lds_issue_backend = Path(__file__).resolve().parents[6] / "third_party/hcu/backend"
_ready_lds_issue_package = "hcu_ready_lds_issue_test_backend"
if _ready_lds_issue_package not in sys.modules:
    module = ModuleType(_ready_lds_issue_package)
    module.__path__ = [str(_ready_lds_issue_backend)]
    sys.modules[_ready_lds_issue_package] = module
_ready_lds_issue_ready = importlib.import_module(_ready_lds_issue_package + ".ready_lds_issue")

_ready_lds_issue_LOADS = [
    "buffer_load_dwordx4 v[0:3], v[8:9], s[4:7], 0 idxen offen",
    "buffer_load_dwordx4 v[4:7], v[10:11], s[4:7], 0 idxen offen",
    "ds_read_b128 v[32:35], v100",
    "ds_read_b128 v[36:39], v100 offset:16",
]


def _ready_lds_issue_kernel(ops):
    instructions = [
        *ops, "s_waitcnt vmcnt(0) lgkmcnt(0)", "v_mmac_f32_16x16x16_bf16 v[40:43], v[32:33], v[0:1], v[40:43]",
        "s_endpgm"
    ]
    return "kernel:\n" + "".join(f"  {inst}\n" for inst in instructions) + ".Lfunc_end0:\n"


class TestReadyLdsIssue(unittest.TestCase):

    def test_independent_reads_move_and_are_idempotent(self):
        original = _ready_lds_issue_kernel(_ready_lds_issue_LOADS)
        changed, info = _ready_lds_issue_ready.schedule_ready_lds_issue(original, "kernel")
        self.assertEqual(len(info["sites"]), 1)
        self.assertEqual(changed, _ready_lds_issue_kernel(_ready_lds_issue_LOADS[2:] + _ready_lds_issue_LOADS[:2]))
        self.assertEqual(_ready_lds_issue_ready.schedule_ready_lds_issue(changed, "kernel")[0], changed)

    def test_control_and_dependency_boundaries_keep_original(self):
        negatives = [
            _ready_lds_issue_kernel(
                [*_ready_lds_issue_LOADS[:3], _ready_lds_issue_LOADS[3].replace("v[36:39]", "v[0:3]")]),
            _ready_lds_issue_kernel(
                [*_ready_lds_issue_LOADS[:1], _ready_lds_issue_LOADS[1] + " glc", *_ready_lds_issue_LOADS[2:]]),
            _ready_lds_issue_kernel(
                [*_ready_lds_issue_LOADS[:2], _ready_lds_issue_LOADS[2] + " gds", _ready_lds_issue_LOADS[3]]),
            _ready_lds_issue_kernel([*_ready_lds_issue_LOADS[:2], "s_waitcnt vmcnt(0)", *_ready_lds_issue_LOADS[2:]]),
            _ready_lds_issue_kernel([*_ready_lds_issue_LOADS[:2], "s_barrier", *_ready_lds_issue_LOADS[2:]]),
            _ready_lds_issue_kernel(
                ["s_cbranch_scc0 .Ltarget", *_ready_lds_issue_LOADS[:2], ".Ltarget:", *_ready_lds_issue_LOADS[2:]]),
            _ready_lds_issue_kernel(_ready_lds_issue_LOADS).replace("s_waitcnt vmcnt(0) lgkmcnt(0)",
                                                                    "s_waitcnt vmcnt(0)"),
        ]
        for original in negatives:
            with self.subTest(assembly=original):
                changed, info = _ready_lds_issue_ready.schedule_ready_lds_issue(original, "kernel")
                self.assertEqual(changed, original)
                self.assertFalse(info["sites"])

    def test_model_version_mismatch_keeps_original(self):
        original = _ready_lds_issue_kernel(_ready_lds_issue_LOADS)
        before = _ready_lds_issue_ready.isa.MODEL_API_VERSION
        try:
            _ready_lds_issue_ready.isa.MODEL_API_VERSION = -1
            changed, info = _ready_lds_issue_ready.schedule_ready_lds_issue(original, "kernel")
        finally:
            _ready_lds_issue_ready.isa.MODEL_API_VERSION = before
        self.assertEqual(changed, original)
        self.assertFalse(info["sites"])


# finite shared read plan
import os
import shlex
import shutil
import subprocess

import pytest


def test_finite_shared_read_plan(tmp_path):
    llvm_config = os.environ.get("LLVM_CONFIG") or shutil.which("llvm-config")
    compiler = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
    if not llvm_config or not compiler:
        pytest.skip("LLVM development libraries and a C++17 compiler required")
    root = Path(__file__).resolve().parents[6]
    source = tmp_path / "finite.cpp"
    source.write_text(r'''
#include "FiniteSharedReadPlan.h"
#include <cassert>
#include <vector>
using mlir::triton::HCU::planFiniteSharedReads;
int main() {
  // Independent hardware gather fixtures, including all lanes and four waves.
  std::vector<unsigned> normal, alt;
  for (unsigned r=0;r<8;++r) for (unsigned w=0;w<4;++w)
    for (unsigned l=0;l<64;++l) {
      normal.push_back(w*4096+r*512+(l&56)*8+(l&7));
      alt.push_back(w*4096+(r&3)*512+(r>>2)+(l&~3u)*8+2*(l&3));
    }
  auto n=planFiniteSharedReads(8,4,16384,normal,true,true,true);
  auto a=planFiniteSharedReads(8,4,16384,alt,true,true,true);
  assert(n && n->matrix && !n->matrixAlt && n->width==8);
  assert(a && a->matrix && a->matrixAlt && a->width==8);
  // A bank-only search used to split the two halves of each MMAC word.
  // This independently constructed row-XOR chart admits both packed and
  // unpacked read groups; the selected groups must retain the packed pairs.
  std::vector<unsigned> rowXor;
  for (unsigned r=0;r<32;++r) for (unsigned w=0;w<4;++w)
    for (unsigned l=0;l<64;++l) {
      unsigned k=(r&3)+((l>>4)&3)*4;
      unsigned n=((r>>2)&1)*16+((r>>3)&1)*32+
                 ((r>>4)&1)*128+(l&15)+(w&1)*64;
      rowXor.push_back(k*256+(n^(8*k)));
    }
  auto packed=planFiniteSharedReads(32,4,4096,rowXor,true,true,true);
  assert(packed && packed->matrix);
  for(unsigned anchor:packed->groups) for(unsigned j=0;j<8;j+=2) {
    unsigned lo=packed->repartition?packed->matrixByAnchor[anchor][j]:anchor+j;
    unsigned hi=packed->repartition?packed->matrixByAnchor[anchor][j+1]:anchor+j+1;
    assert(!(lo&1) && hi==lo+1);
  }
  // A single nonbasis point must defeat an otherwise valid matrix gather.
  auto corrupt=normal;corrupt[(7*4+3)*64+63]^=1;
  auto c=planFiniteSharedReads(8,4,16384,corrupt,true,true,true);
  assert(c && !c->matrix);
  auto outOfBounds=normal;outOfBounds.back()=16384;
  assert(!planFiniteSharedReads(8,4,16384,outOfBounds,true,true,true));
  assert(!planFiniteSharedReads(8,4,16384,
      llvm::ArrayRef<unsigned>(normal).drop_back(),true,true,true));
  auto noMatrix=planFiniteSharedReads(8,4,16384,normal,false,true,true);
  assert(noMatrix && !noMatrix->matrix);
  // Search exhaustion is explicit, cannot be used as proof of a matrix read.
  auto budget=planFiniteSharedReads(8,4,16384,normal,true,true,true,1);
  assert(budget && budget->optimizationBudgetExceeded && !budget->matrix);
  assert(budget->width==1 && budget->groups.size()==8);
}
''')

    def flags(*args):
        return shlex.split(subprocess.check_output([llvm_config, *args], text=True))

    exe = tmp_path / "finite"
    subprocess.run([
        compiler, *flags("--cxxflags"), "-std=c++17", "-O2", "-UNDEBUG", "-I",
        str(root / "third_party/hcu/lib/TritonHCUGPUToLLVM"),
        str(source), "-o",
        str(exe), *flags("--ldflags", "--libs", "support", "--system-libs")
    ], check=True, capture_output=True)
    subprocess.run([str(exe)], check=True, capture_output=True)
