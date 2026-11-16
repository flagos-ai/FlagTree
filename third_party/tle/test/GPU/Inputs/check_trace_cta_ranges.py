# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Interpret actual emitted membership instructions against the declared set."""
import random
import re
import sys
from pathlib import Path

U64 = (1 << 64) - 1


def assemblies(text):
    for match in re.finditer(r'llvm.inline_asm[^\n]*?"([^"\n]*)", "([^"\n]*)"', text):
        code = re.sub(r'\\([0-9A-Fa-f]{2})', lambda m: chr(int(m[1], 16)), match[1])
        if 'mov.u64 timestamp, %globaltimer;' in code:
            yield code, match[2]


def program(code):
    start = code.index('.reg .pred selected, match;')
    end = code.index('@!selected bra TRACE_DONE;', start)
    lines = code[start:end].splitlines()[1:]
    instructions = []
    for line in lines:
        match = re.fullmatch(r'(setp\.(?:eq|lt)\.u(?:32|64)|sub\.u64|or\.pred) (\w+), (\w+), (\w+);', line)
        assert match, f'unknown selection instruction: {line}'
        opcode, dest, lhs, rhs = match.groups()
        assert dest in ('selected', 'match', 'temp'), f'CTA or unrelated register overwritten: {dest}'
        instructions.append((opcode, dest, lhs, rhs))
    assert instructions, 'empty selection program'
    return instructions


def selected(instructions, cta):
    registers = {'cta': cta}

    def read(value):
        if value.isdecimal():
            return int(value)
        assert value in registers, f'uninitialized selection register {value}'
        return registers[value]

    for opcode, dest, lhs, rhs in instructions:
        a, b = read(lhs), read(rhs)
        if opcode == 'sub.u64':
            assert type(a) is int and type(b) is int
            value = (a - b) & U64
        elif opcode == 'or.pred':
            assert type(a) is bool and type(b) is bool
            value = a or b
        else:
            assert type(a) is int and type(b) is int
            bits = int(opcode.rsplit('u', 1)[1])
            mask = (1 << bits) - 1
            assert 0 <= a <= mask and 0 <= b <= mask
            value = a == b if '.eq.' in opcode else a < b
        registers[dest] = value
    assert type(registers.get('selected')) is bool
    return registers['selected']


def main(fixture_path, lowered_path):
    fixture, lowered = Path(fixture_path).read_text(), Path(lowered_path).read_text()
    functions = {}
    for chunk in re.split(r'(?=^\s*llvm.func)', lowered, flags=re.MULTILINE):
        match = re.match(r'\s*llvm.func[^\n]*@select_(\w+)\(', chunk)
        if match:
            assert match[1] not in functions
            functions[match[1]] = list(assemblies(chunk))
    checks = records = 0
    costs = []
    cases = set()
    for module in fixture.split('// -----'):
        name = re.search(r'@select_(\w+)\(', module)
        if not name:
            continue
        name = name[1]
        cases.add(name)
        declaration = re.search(r'"tle.ws_trace.typed_ctas" = array<i32: ([^>]*)>', module)
        values = None if declaration is None else {int(v.strip()) for v in declaration[1].split(',')}
        probes = functions[name]
        assert len(probes) == 2, f'{name}: typed observation count changed'
        observed_kinds = []
        for code, constraints in probes:
            records += 1
            assert constraints == 'l,r,r,r,r,r,~{memory}'
            assert code.index('mov.u64 timestamp, %globaltimer;') < code.index('activemask.b32 mask;')
            assert code.index('@!leader bra TRACE_DONE;') < code.index('mul.wide.u32 cta, cz, ny;')
            assert 'st.global.v4.u32 [record+48], {$2, $3, $4, $5};' in code
            observed_kinds.append(int(re.search(r'mov.u32 event_value, (\d+);', code)[1]))
            if values is None:
                assert 'selected' not in code, 'absent selection now filters runtime recording'
                continue
            instructions = program(code)
            assert code.index('cvt.u64.u32 temp, cx;') < code.index('.reg .pred selected, match;')
            assert code.index('@!selected bra TRACE_DONE;') < code.index('mul.lo.u64 stream, cta,')
            assert code.index('mul.lo.u64 stream, cta,') < code.index('ld.global.u32 en, [stream+36];')
            candidates = {0, 1, 127, 128, (1 << 31) - 1, 1 << 31, (1 << 32) - 1,
                          1 << 32, (1 << 63) - 1, 1 << 63, U64 - 1, U64}
            for value in values:
                candidates.update(v for v in (value - 1, value, value + 1) if 0 <= v <= U64)
            # These small fixtures allow complete checking of all low gaps, too.
            candidates.update(range(256))
            rng = random.Random(734)
            candidates.update(rng.getrandbits(64) for _ in range(256))
            for cta in sorted(candidates):
                assert selected(instructions, cta) == (cta in values), f'{name}: wrong membership for {cta}'
                checks += 1
            if name == 'sparse':
                for cta in values:
                    assert ('setp.eq.u64', 'match', 'cta', str(cta)) in instructions
            if len(values) == 1:
                assert any(op == 'setp.eq.u64' for op, *_ in instructions), 'singleton lost equality'
            if name == 'full128':
                costs.append(len(instructions))
        assert observed_kinds == [272, 273], f'{name}: event order changed'
    assert set(functions) == cases and len(cases) == 11
    # Untyped WS entry/wait/exit probes must never gain the typed selector.
    for code, _ in assemblies(lowered):
        if 'st.global.v4.u32 [record+48], {$2, $3, $4, $5};' not in code:
            assert 'selected' not in code
    assert costs == [1, 1], f'full128 membership operations {costs} exceed exact single-range budget1'
    print(f'{len(cases)} cases, {records} typed probes, {checks} u64 truth checks: full128 cost1, guards/order preserved')


if __name__ == '__main__':
    main(*sys.argv[1:])
