"""Versioned gfx936 assembly effects used by native wait transformations.

This module parses kernel text and classifies events. It does not establish
register readiness or cross-wave memory ordering; the native checker remains
required after scheduling. Unknown opcodes keep their conservative class.

Bump MODEL_API_VERSION for an incompatible consumer contract. Opcode/effect
changes require corresponding semantic fixtures and affected native checks.
"""
import re

MODEL_API_VERSION = 1

_SCALAR = ('s_add', 's_sub', 's_and', 's_or', 's_xor', 's_mov', 's_mul',
           's_lsh', 's_ash', 's_bfe', 's_cselect', 's_min', 's_max', 's_not',
           's_brev', 's_cmp')
_VECTOR = ('v_add', 'v_sub', 'v_mul', 'v_mad', 'v_fma', 'v_mmac', 'v_mov',
           'v_readfirstlane', 'v_readlane_b32', 'v_writelane_b32', 'v_cmp',
           'v_cndmask', 'v_cvt', 'v_lsh', 'v_ash', 'v_and', 'v_or', 'v_xor',
           'v_bfe', 'v_bfrev', 'v_rcp', 'v_trunc', 'v_min', 'v_max', 'v_pack',
           'v_align', 'v_not')
BRANCH_OPCODES = {'s_branch', 's_cbranch_scc0', 's_cbranch_scc1', 's_cbranch_vccz',
             's_cbranch_vccnz', 's_cbranch_execz', 's_cbranch_execnz'}
_DS_READ = re.compile(r'ds_read(?:2(?:st64)?)?_b(?:32|64|96|128)$|ds_read_m32x16_b16(?:_alt)?$')


def parse_instruction(line):
    text = line.strip()
    if not text or text.startswith(('.', ';', '#')) or re.match(r'[\w.$]+:', text):
        return None
    match = re.fullmatch(r'([a-z][\w.]*)\s*(.*)', text)
    return (match[1], match[2].split(';')[0].strip()) if match else None


def parse_kernel(asm, name):
    if not isinstance(asm, str) or not isinstance(name, str) or not name:
        raise ValueError('assembly and exact kernel name required')
    lines = asm.splitlines(keepends=True)
    entries = [n for n, line in enumerate(lines)
               if re.match(re.escape(name) + r':(?:\s|$)', line.strip())]
    if len(entries) != 1:
        raise ValueError('missing or duplicate exact kernel entry')
    begin = entries[0]
    end = next((n for n in range(begin + 1, len(lines))
                if lines[n].strip().startswith('.Lfunc_end')), len(lines))
    labels, label_lines, instructions = {}, {}, []
    for n in range(begin, end):
        match = re.match(r'([\w.$]+):', lines[n].strip())
        if match:
            tail = lines[n].strip()[match.end():].strip()
            if tail and not tail.startswith((';', '#')):
                raise ValueError('inline label/instruction unsupported')
            if match[1] in labels:
                raise ValueError('duplicate label')
            labels[match[1]] = len(instructions)
            label_lines[match[1]] = n
            continue
        if re.match(r'\s*\.(?:byte|short|word|long|quad|[248]byte|inst|insn|fill|zero|incbin|ascii|asciz|string)(?:\s|$)', lines[n]):
            raise ValueError('raw code/data directive in kernel')
        op = parse_instruction(lines[n])
        if not op and lines[n].strip() and not lines[n].strip().startswith(('.', ';', '#')):
            raise ValueError('unknown assembly syntax')
        if op:
            instructions.append(dict(line=n, op=op[0], args=op[1]))
    if not instructions or not any(x['op'] == 's_endpgm' for x in instructions):
        raise ValueError('incomplete kernel body')
    targets = set()
    for inst in instructions:
        op, args = inst['op'], inst['args']
        if op in BRANCH_OPCODES:
            if args not in labels or labels[args] >= len(instructions):
                raise ValueError('unresolved direct branch target')
            targets.add(args)
        elif ('branch' in op or op.startswith(('s_setpc', 's_swappc', 's_call',
                                              's_rfe', 's_trap', 's_return'))):
            raise ValueError('indirect or unsupported control flow')
    # Function entry plus true incoming CFG edges, not local debug/address
    # labels. With indirect/unresolved flow rejected, unused local symbols
    # do not introduce an instruction predecessor or a new basic block.
    return dict(lines=lines, begin=begin, end=end, instructions=instructions,
                labels=labels, label_lines=label_lines, targets=targets,
                leaders=targets | {name})


def parse_wait(args):
    fields = []
    for part in args.split():
        match = re.fullmatch(r'(vmcnt|lgkmcnt)\((\d+)\)', part)
        if not match:
            return None
        fields.append((match[1], int(match[2])))
    if not fields or len(set(k for k, v in fields)) != len(fields):
        return None
    if any(v < 0 or (k == 'lgkmcnt' and v > 15) or
           (k == 'vmcnt' and v > 63) for k, v in fields):
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
    if op.startswith(('buffer_load_', 'buffer_store_', 'global_load_',
                      'global_store_', 'flat_load_', 'flat_store_')):
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
