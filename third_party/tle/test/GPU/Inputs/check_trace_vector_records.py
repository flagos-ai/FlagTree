# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Check emitted PTX against the independent 64-byte recorder wire layout."""
import re
import struct
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
assemblies = []
for match in re.finditer(r'llvm.inline_asm[^\n]*?"([^"\n]*)", "([^"\n]*)"', text):
    code = re.sub(r'\\([0-9A-Fa-f]{2})', lambda m: chr(int(m[1], 16)), match[1])
    if 'mov.u64 timestamp, %globaltimer;' in code:
        assemblies.append((code, match[2]))
assert assemblies, 'no recorder assembly emitted'
assert 'ttg.profile_scratch_memory_alignment = 64' in text
seen_typed = seen_runtime = seen_entry = seen_exit = False
for code, constraints in assemblies:
    assert '~{memory}' in constraints
    timer = code.index('mov.u64 timestamp, %globaltimer;')
    assert timer < code.index('activemask.b32 mask;') < code.index('@!leader bra TRACE_DONE;')
    assert timer < code.index('mul.lo.u64 stream, cta,')
    stride = int(re.search(r'mul.lo.u64 stream, stream, (\d+);', code)[1])
    assert stride % 64 == 0
    assert 'mul.wide.u32 record, count, 64;' in code
    assert 'add.u64 record, record, 64;' in code
    enabled = code.index('@!enabled bra TRACE_DONE;')
    attempted = code.index('st.global.u32 [stream+16], next;')
    room = code.index('@!room bra TRACE_OVERFLOW;')
    stores = list(re.finditer(r'st.global\.v(\d+)\.u(\d+) \[record\+(\d+)\], \{([^}]+)\};', code))
    assert len(stores) == 4, 'record must have four vector stores'
    assert not re.search(r'st.global\.u\d+ \[record\+', code), 'scalar record store remained'
    assert [(int(m[1]), int(m[2]), int(m[3])) for m in stores] == [(2, 64, 0), (4, 32, 16), (4, 32, 32), (4, 32, 48)]
    assert enabled < attempted < room < stores[0].start()
    assert room < code.index('mov.u32 event_value,') < stores[1].start()
    # Keep the original coordinate definitions for stream indexing, but resolve
    # record-only aliases from the same immutable special registers at use.
    coordinate_aliases = {}
    for axis in ('x', 'y'):
        original, alias = f'c{axis}', f'record_c{axis}'
        definition = f'mov.u32 {original}, %ctaid.{axis};'
        reread = f'mov.u32 {alias}, %ctaid.{axis};'
        assert code.count(definition) == code.count(reread) == 1
        assert code.index(definition) < code.index(f'cvt.u64.u32 temp, {original};') < enabled
        assert stores[0].end() < code.index(reread) < stores[1].start()
        assert len(re.findall(rf'\b{alias}\b', code)) == 3, 'alias must only be declared, read and recorded'
        coordinate_aliases[alias] = original
    assert re.search(r'mov.u32 record_cx, %ctaid.x;\s*mov.u32 record_cy, %ctaid.y;\s*$',
                     code[:stores[1].start()]), 'coordinate rereads must immediately precede the second vector'
    assert [v.strip() for v in stores[1][4].split(',')] == ['event_value', '$1', 'record_cx', 'record_cy']
    publication = code.index('st.global.u32 [stream+20], next;')
    assert stores[-1].end() < publication < code.index('TRACE_OVERFLOW:')
    assert 'ld.global.u32 dropped, [stream+24];' in code
    assert 'add.u32 dropped, dropped, 1;' in code
    assert 'st.global.u32 [stream+24], dropped;' in code
    kind = int(re.search(r'mov.u32 event_value, (\d+);', code)[1])
    typed = '$2' in stores[-1][4]
    seen_typed |= typed
    seen_runtime |= not typed
    seen_entry |= kind == 1
    seen_exit |= kind == 19
    if not typed:
        assert room < code.index('mov.u32 zero, 0;') < stores[-1].start()
    if kind == 1:
        assert 'st.global.u32 [stream+4], 2;' in code
        assert 'st.global.u32 [stream+28], 0;' in code
    if kind == 19:
        assert code.index('TRACE_FINISH:') < code.index('st.global.u32 [stream+28], 1;') < code.index('TRACE_DONE:')

    # Full-width timestamp/ordinal, signed/high-bit i32 values and zeros detect
    # accidental truncation, field swaps, sign extension or constant corruption.
    values = dict(timestamp=0xFEDCBA9876543210, ordinal=0x0123456789ABCDEF,
                  event_value=kind, zero=0, cx=0x89ABCDEF, cy=0xFFFFFFFF,
                  cz=0x80000000, warp=11, sx=0x12345678, mask=0x80000001)
    values.update({'$1': -0x80000000, '$2': 0x7FFF0011, '$3': -1,
                   '$4': 0x80001234, '$5': -123456789})
    values.update({alias: values[original] for alias, original in coordinate_aliases.items()})
    expected = struct.pack('<QQ12I', values['timestamp'], values['ordinal'],
                           kind, values['$1'] & 0xFFFFFFFF,
                           values['cx'], values['cy'], values['cz'], values['warp'],
                           values['sx'], values['mask'],
                           *([values[f'${i}'] & 0xFFFFFFFF for i in range(2, 6)] if typed else [0] * 4))
    wire = bytearray(64)
    for store in stores:
        lanes, bits, offset = map(int, store.group(1, 2, 3))
        names = [x.strip() for x in store[4].split(',')]
        assert len(names) == lanes and offset % 16 == 0
        data = b''.join((values[name] & ((1 << bits) - 1)).to_bytes(bits // 8, 'little') for name in names)
        assert len(data) == 16
        wire[offset:offset + 16] = data
    assert bytes(wire) == expected, 'vector fields changed the wire ABI'
assert seen_typed and seen_runtime and seen_entry and seen_exit
print(f'{len(assemblies)} typed/runtime records: vector shape, wire bytes and publication order verified')
