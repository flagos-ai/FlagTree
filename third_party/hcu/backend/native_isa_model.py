"""Versioned gfx936 assembly effects used by native wait transformations.

This module parses kernel text and classifies events. It does not establish
register readiness or cross-wave memory ordering; the native checker remains
required after scheduling. Unknown opcodes keep their conservative class.

Bump MODEL_API_VERSION for an incompatible consumer contract. Opcode/effect
changes require corresponding semantic fixtures and affected native checks.
"""
import re
from contextlib import contextmanager
from contextvars import ContextVar

# One bounded analysis workspace per compiler invocation. Nested calls reuse it;
# exit (including exceptions) releases every assembly/model reference. There is
# no process-wide native-code cache or synchronization with tuning caches.
_analysis_models = ContextVar("hcu_native_models", default=None)


@contextmanager
def analysis_scope():
    if _analysis_models.get() is not None:
        yield
        return
    token = _analysis_models.set({})
    try:
        yield
    finally:
        _analysis_models.reset(token)


MODEL_API_VERSION = 1

_SCALAR = ('s_add', 's_sub', 's_and', 's_or', 's_xor', 's_mov', 's_mul', 's_lsh', 's_ash', 's_bfe', 's_cselect',
           's_min', 's_max', 's_not', 's_brev', 's_cmp')
_VECTOR = ('v_add', 'v_sub', 'v_mul', 'v_mad', 'v_fma', 'v_mmac', 'v_mov', 'v_readfirstlane', 'v_readlane_b32',
           'v_writelane_b32', 'v_cmp', 'v_cndmask', 'v_cvt', 'v_lsh', 'v_ash', 'v_and', 'v_or', 'v_xor', 'v_bfe',
           'v_bfrev', 'v_rcp', 'v_trunc', 'v_min', 'v_max', 'v_pack', 'v_align', 'v_not')
BRANCH_OPCODES = {
    's_branch', 's_cbranch_scc0', 's_cbranch_scc1', 's_cbranch_vccz', 's_cbranch_vccnz', 's_cbranch_execz',
    's_cbranch_execnz'
}
_DS_READ = re.compile(r'ds_read(?:2(?:st64)?)?_b(?:32|64|96|128)$|ds_read_m32x16_b16(?:_alt)?$')


def parse_instruction(line):
    text = line.strip()
    if not text or text.startswith(('.', ';', '#')) or re.match(r'[\w.$]+:', text):
        return None
    match = re.fullmatch(r'([a-z][\w.]*)\s*(.*)', text)
    return (match[1], match[2].split(';')[0].strip()) if match else None


def parse_kernel(asm, name):
    models = _analysis_models.get()
    if (models is None or not isinstance(asm, str) or not isinstance(name, str) or len(asm) > 512 * 1024):
        return _decode_kernel(asm, name)
    key = (asm, name)
    if key not in models:
        decoded = _decode_kernel(asm, name)
        # Rewrites create different text. Keep only a few successive versions;
        # readiness checks reuse exactly the text that was parsed, never a hash.
        # Large kernels are decoded without retention; the text budget bounds
        # additional memory when several tuning compilations run concurrently.
        while models and (len(models) >= 4 or sum(len(k[0]) for k in models) + len(asm) > 1024 * 1024):
            del models[next(iter(models))]
        models[key] = decoded
    return models[key]


def _decode_kernel(asm, name):
    if not isinstance(asm, str) or not isinstance(name, str) or not name:
        raise ValueError('assembly and exact kernel name required')
    lines = asm.splitlines(keepends=True)
    entries = [n for n, line in enumerate(lines) if re.match(re.escape(name) + r':(?:\s|$)', line.strip())]
    if len(entries) != 1:
        raise ValueError('missing or duplicate exact kernel entry')
    begin = entries[0]
    end = next((n for n in range(begin + 1, len(lines)) if lines[n].strip().startswith('.Lfunc_end')), len(lines))
    # LLVM may place the non-executable descriptor before .Lfunc_end. Parse
    # its envelope linearly instead of accepting arbitrary directives or using
    # a backtracking multiline pattern over untrusted imported assembly.
    descriptor = None
    labels, label_lines, instructions = {}, {}, []
    for n in range(begin, end):
        line = lines[n]
        directive = line.strip()
        if not directive or directive.startswith((';', '#')):
            continue
        if descriptor:
            if descriptor == 'header':
                if re.fullmatch(r'\.p2align\s+\d+(?:,\s*(?:0[xX][0-9a-fA-F]+|[0-9]+))?', directive):
                    continue
                if re.fullmatch(r'\.amdhsa_kernel\s+' + re.escape(name), directive):
                    descriptor = 'fields'
                    continue
            elif descriptor == 'fields':
                if directive == '.end_amdhsa_kernel':
                    descriptor = 'text'
                    continue
                if re.fullmatch(r'\.amdhsa_\w+\s+-?(?:0[xX][0-9a-fA-F]+|[0-9]+)', directive):
                    continue
            elif directive == '.text':
                descriptor = None
                continue
            raise ValueError('unsupported kernel descriptor envelope')
        if re.fullmatch(r'\.section\s+\.rodata,\s*"a",\s*@progbits', directive):
            descriptor = 'header'
            continue
        match = re.match(r'([\w.$]+):', directive)
        if match:
            tail = line.strip()[match.end():].strip()
            if tail and not tail.startswith((';', '#')):
                raise ValueError('inline label/instruction unsupported')
            if match[1] in labels:
                raise ValueError('duplicate label')
            labels[match[1]] = len(instructions)
            label_lines[match[1]] = n
            continue
        if re.match(
                r'\s*\.(?:byte|short|word|long|quad|[248]byte|inst|insn|fill|zero|incbin|ascii|asciz|string)(?:\s|$)',
                line):
            raise ValueError('raw code/data directive in kernel')
        directive = line.strip()
        if directive.startswith('.') and not re.match(r'\.(?:loc|file|cfi_[A-Za-z0-9_]+)(?:\s|$)', directive):
            raise ValueError('unsupported in-kernel assembler directive')
        op = parse_instruction(line)
        if not op and line.strip() and not line.strip().startswith(('.', ';', '#')):
            raise ValueError('unknown assembly syntax')
        if op:
            instructions.append(dict(line=n, op=op[0], args=op[1]))
    if descriptor:
        raise ValueError('incomplete kernel descriptor envelope')
    if not instructions or not any(x['op'] == 's_endpgm' for x in instructions):
        raise ValueError('incomplete kernel body')
    targets = set()
    for inst in instructions:
        op, args = inst['op'], inst['args']
        if op in BRANCH_OPCODES:
            if not re.fullmatch(r'[A-Za-z_.$][\w.$]*', args):
                raise ValueError('numeric or non-symbolic branch target')
            if args not in labels or labels[args] >= len(instructions):
                raise ValueError('unresolved direct branch target')
            targets.add(args)
        elif op in ('s_join', 's_fork') or ('branch' in op or op.startswith(
            ('s_setpc', 's_swappc', 's_call', 's_rfe', 's_trap', 's_return'))):
            raise ValueError('indirect or unsupported control flow')
    # Function entry plus true incoming CFG edges, not local debug/address
    # labels. With indirect/unresolved flow rejected, unused local symbols
    # do not introduce an instruction predecessor or a new basic block.
    return dict(lines=lines, begin=begin, end=end, instructions=instructions, labels=labels, label_lines=label_lines,
                targets=targets, leaders=targets | {name})


def parse_wait(args):
    fields = []
    for part in args.split():
        match = re.fullmatch(r'(vmcnt|lgkmcnt)\((\d+)\)', part)
        if not match:
            return None
        fields.append((match[1], int(match[2])))
    if not fields or len(set(k for k, v in fields)) != len(fields):
        return None
    if any(v < 0 or (k == 'lgkmcnt' and v > 15) or (k == 'vmcnt' and v > 63) for k, v in fields):
        return None
    return dict(fields)


def classify_instruction(inst):
    op, args = inst['op'], inst['args']
    if op in BRANCH_OPCODES or op == 's_endpgm':
        return 'CONTROL'
    if op == 's_waitcnt':
        return 'WAIT' if parse_wait(args) is not None else 'UNKNOWN'
    if op == 's_barrier':
        return 'CTA' if not args else 'UNKNOWN'
    if _DS_READ.fullmatch(op):
        return 'DS_READ'
    if op.startswith(('s_load_', 's_buffer_load_')):
        return 'SMEM'
    if re.fullmatch(r'buffer_load_dword(?:x[24])?', op) and re.search(r'\blds\b', args):
        return 'G2L'
    if op.startswith(('buffer_load_', 'buffer_store_', 'global_load_', 'global_store_', 'flat_load_', 'flat_store_')):
        return 'OTHER_VM'
    # Register permutation does not issue a memory event. Use its exact opcode,
    # leaving unknown variants conservative rather than accepting a prefix.
    if re.fullmatch(r's_bitset[01]_b(?:32|64)', op):
        return 'PURE'
    if op == 'v_perm_b32':
        return 'PURE'
    if op == 's_nop' or op.startswith(_SCALAR + _VECTOR):
        return 'PURE'
    return 'UNKNOWN'


def _registers(text):
    result = set()
    for match in re.finditer(r"\b([vs])(?:\[(\d+):(\d+)\]|(\d+))(?!\w)", text):
        kind, begin, end, single = match.groups()
        result.update((kind, index)
                      for index in ([int(single)] if single is not None else range(int(begin),
                                                                                   int(end) + 1)))
    return result


def _xad_u32_effects(operands):
    # GFX9 VOP3 XOR-add has one 32-bit VGPR result and three explicit
    # 32-bit sources. It has no asynchronous result or implicit data-register
    # writes (EXEC is read implicitly). Keep modifiers, ranges and special
    # registers outside this narrow model instead of guessing their effects.
    if len(operands) != 4 or not re.fullmatch(r"v[0-9]+", operands[0]):
        return None
    for source in operands[1:]:
        if re.fullmatch(r"[vs][0-9]+", source):
            continue
        if not re.fullmatch(r"-?(?:0[xX][0-9a-fA-F]{1,8}|[0-9]{1,10})", source):
            return None
        value = int(source, 16 if "x" in source.lower() else 10)
        if not -(1 << 31) <= value < (1 << 32):
            return None
    return _registers(",".join(operands[1:])), _registers(operands[0])


def _scalar_source_registers(source):
    if re.fullmatch(r"s[0-9]+", source):
        return _registers(source)
    if not re.fullmatch(r"-?(?:0[xX][0-9a-fA-F]{1,8}|[0-9]{1,10})", source):
        return None
    value = int(source, 16 if "x" in source.lower() else 10)
    return set() if -(1 << 31) <= value < (1 << 32) else None


def _sext_i32_i16_effects(operands):
    # Vega SOP1 S_SEXT_I32_I16: one synchronous 32-bit SGPR result;
    # D={16{S0[15]},S0[15:0]}, SCC is not written. No queue completion.
    if len(operands) != 2 or not re.fullmatch(r"s[0-9]+", operands[0]):
        return None
    reads = _scalar_source_registers(operands[1])
    return None if reads is None else (reads, _registers(operands[0]))


def _bitset_effects(op, operands):
    # SOP1 bit set/reset preserves all other destination bits: the old
    # destination is an input, including BOTH SGPRs of a 64-bit pair.
    # The bit index is a 32-bit scalar source (low5/low6 bits are used).
    # No memory event or async completion is introduced. Special registers
    # and malformed operands retain the conservative native fallback.
    match = re.fullmatch(r"s_bitset[01]_b(32|64)", op)
    if match is None or len(operands) != 2:
        return None
    if match[1] == "32":
        if not re.fullmatch(r"s[0-9]+", operands[0]):
            return None
    else:
        pair = re.fullmatch(r"s\[(\d+):(\d+)\]", operands[0])
        if pair is None or int(pair[1]) % 2 or int(pair[2]) != int(pair[1]) + 1:
            return None
    reads = _scalar_source_registers(operands[1])
    dest = _registers(operands[0])
    return None if reads is None else (reads | dest, dest)


def instruction_effects(op, args):
    """Register/queue effects for the bounded completion model.

    Event classification is intentionally coarser: e.g. a DS write ends a
    wait-coalescing segment while its precise queue effect is known here.
    Unknown operations and unsupported operand forms never grant readiness.
    """
    if op.startswith(("s_movrel", "v_movrel")):
        raise ValueError("unsupported implicit register indexing")
    operands = [part.strip() for part in args.split(",")]
    all_regs = _registers(args)
    dest = _registers(operands[0])
    reads, writes = all_regs - dest, dest
    queue, event = None, None
    if op.startswith("ds_read_") or op.startswith("ds_read2"):
        queue, event = 0, "DS"
    elif op.startswith("s_load_"):
        queue, event = 0, "SMEM"
    elif op.startswith(("buffer_load_", "global_load_", "flat_load_")):
        queue, event = 1, "VM"
        if re.search(r"\blds\b", args):
            reads, writes = all_regs, set()
    elif op.startswith(("buffer_store_", "global_store_", "flat_store_")):
        reads, writes, queue, event = all_regs, set(), 1, "VM"
    elif op.startswith("ds_write_"):
        reads, writes, queue, event = all_regs, set(), 0, "DS"
    elif op.startswith("s_"):
        if op == "s_sext_i32_i16":
            effects = _sext_i32_i16_effects(operands)
            if effects is None:
                raise ValueError("unsupported s_sext_i32_i16 operands")
            reads, writes = effects
        elif re.fullmatch(r"s_bitset[01]_b(?:32|64)", op):
            effects = _bitset_effects(op, operands)
            if effects is None:
                raise ValueError("unsupported scalar bitset operands")
            reads, writes = effects
        elif op == "s_setprio":
            # The generated priority controls neither define data registers
            # nor complete an async response. Preserve both pending queues.
            if operands not in (["0"], ["1"]):
                raise ValueError("unsupported s_setprio operands")
            reads, writes = set(), set()
        elif op in ("s_waitcnt", "s_barrier", "s_nop", "s_endpgm", "s_branch") or op.startswith(("s_cbranch", "s_cmp")):
            reads, writes = all_regs, set()
        elif not op.startswith(_SCALAR):
            raise ValueError("unsupported scalar opcode")
    elif op.startswith("v_"):
        if op in ("v_xad_u32", "v_perm_b32"):
            # V_PERM_B32 is a synchronous byte permutation: one explicit
            # 32-bit VGPR result, three explicit 32-bit sources. Reuse
            # the strict operand-effects grammar, never arithmetic semantics.
            effects = _xad_u32_effects(operands)
            if effects is None:
                raise ValueError("unsupported " + op + " operands")
            reads, writes = effects
        if op.startswith("v_cmp") and not dest:
            reads, writes = all_regs, set()
        # Opcode families below have only register/immediate side effects.
        if op not in ("v_xad_u32", "v_perm_b32") and not op.startswith(_VECTOR):
            raise ValueError("unsupported vector opcode")
        if op == "v_writelane_b32":
            # Writelane replaces one lane and preserves the others. Its
            # old VGPR value is a read as well as a destination; both that
            # value and the scalar source must have completed. Readlane
            # has an explicit SGPR destination and VGPR/index sources.
            reads |= dest
        if op.startswith(("v_add_co", "v_sub_co", "v_subrev_co", "v_subb_co", "v_mad_u64")) and len(operands) > 1:
            second_dest = _registers(operands[1])
            writes |= second_dest
            reads = _registers(",".join(operands[2:]))
    else:
        raise ValueError("unsupported opcode")
    wait = parse_wait(args) if op == "s_waitcnt" else None
    if op == "s_waitcnt" and wait is None:
        raise ValueError("unsupported wait operands")
    return reads, writes, queue, event, wait
