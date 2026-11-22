"""Reject artifacts that violate the signed H001 transport realization.

This is a static gate, not a substitute for device lineage/NUM or paired NCU.
The report records real compiler artifacts; no generated source is patched.
"""
from __future__ import annotations
import hashlib
import re
from pathlib import Path


def audit(kernel, destination: Path, tag: str):
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for kind in ('ttir', 'ttgir', 'llir', 'ptx', 'cubin'):
        data = kernel.asm[kind]
        data = data if isinstance(data, bytes) else data.encode()
        p = destination / f'{tag}.{kind}'
        p.write_bytes(data)
        files[kind] = dict(path=str(p.resolve()), sha256=hashlib.sha256(data).hexdigest())
    ttgir, llir, ptx = (kernel.asm[k] for k in ('ttgir', 'llir', 'ptx'))
    e32 = 'sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]'
    # Inspect every checked collective's operand/result layout, not just the
    # presence of one loader encoding somewhere in the module.
    aliases = re.findall(r'(#\w+) = #ttg.blocked<\{([^}]+)\}>', ttgir)
    owners = [name for name, body in aliases if e32 in body]
    collective = [line for line in ttgir.splitlines() if 'tle.indexed32' in line and 'elementwise_inline_asm' in line]
    shuffles = [line.strip() for line in ptx.splitlines() if re.search(r'\bshfl\.', line)]
    asm_calls = [line for line in llir.splitlines() if 'asm sideeffect' in line and 'tle.indexed32' in line]
    attribute_sets = dict(re.findall(r'attributes #(\d+) = \{([^}]+)\}', llir))
    checks = dict(
        checked_collectives_present=bool(collective),
        collective_E32=bool(owners) and all('tensor<4x32xi32, '+owners[0]+'>' in line for line in collective),
        no_shared_or_barrier=not re.search(r'\b(?:ld|st)\.shared|\bbar\.(?:sync|arrive)|\bbarrier\.', ptx),
        pinned_idx=bool(shuffles) and all('shfl.sync.idx.b32' in line and
            re.search(r',\s*0x1f,\s*(?:0xffffffff|-1);', line) for line in shuffles),
        unpredicated_shuffle=all(not line.startswith('@') for line in shuffles),
        convergence=bool(asm_calls) and all(
            any(word in attribute_sets.get(re.search(r'#(\d+)', line)[1], '') for word in ('convergent',))
            and 'noduplicate' in attribute_sets.get(re.search(r'#(\d+)', line)[1], '')
            for line in asm_calls),
        no_local_traffic=not re.search(r'\b(?:ld|st)\.local', ptx),
    )
    return dict(files=files, checks=checks, passed=all(checks.values()),
                static_collectives=len(collective), static_shuffles=len(shuffles),
                scope='Artifact static gate; full PC lineage and dynamic counters require independent review')
