"""Opt-in native packet reuse diagnostics; no compilation or hardware guarantee.

Both APIs return (assembly, report). Rejection returns the EXACT original text.
Packet reuse requires a fresh LLVM operand-model proof, then the existing
native readiness guard on the assembled result. No imported/stale proof is valid.
Standalone redundant-wait cleanup has an independent no-new-event upper-bound
proof and does not require a packet/LLVM operand-model proof.
"""
import collections
import hashlib
import re

from .native_isa_model import (
    BRANCH_OPCODES as _BRANCHES,
    classify_instruction as _kind,
    parse_instruction as _operation,
    parse_kernel as _parse,
    parse_wait as _wait,
)


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _base_report(asm, kind):
    return dict(accepted=False, changed=False, kind=kind, diagnostic_only=True,
                cross_wave_WAR='UNPROVEN_DIAGNOSTIC_ONLY',
                requires_fresh_LLVM_operand_model=True,
                requires_native_readiness_after_assembly=True,
                input_sha256=_sha(asm), output_sha256=_sha(asm),
                native_loops=0, native_total_groups=0, phase_changed_groups=0, deleted_count=0)


def prepare_packet_reuse(asm, name, mode, expected_groups):
    """Return a diagnostic rewrite of every complete native steady loop.

    expected_groups is the fresh LLVM SOURCE loop's group count. EVERY native
    clone must contain exactly that many CTA/read/copy groups; it is NOT the
    sum across cloned loops. Report native_total_groups separately. No shape,
    tile, wave count, SSA name, MMAC count or fixed clone count is a selector.

    precopy: add LGKM(actual group DS count) before first G2L.
    fenced: add that LGKM, move old VM/CTA there, VM credit -= group G2L count.
    fenced intentionally changes low-loop MMAC/CTA phase where applicable.
    """
    report = _base_report(asm, 'packet_reuse')
    report.update(mode=mode, expected_groups_per_native_loop=expected_groups,
                  loops=[], native_total_groups=0)
    try:
        if mode not in ('precopy', 'fenced'):
            raise ValueError('unsupported mode')
        if isinstance(expected_groups, bool) or not isinstance(expected_groups, int) or expected_groups <= 0:
            raise ValueError('positive fresh per-source-loop expected_groups required')
        parsed = _parse(asm, name)
        lines, insts = parsed['lines'], parsed['instructions']
        if any(_kind(x) == 'UNKNOWN' for x in insts):
            bad = next(x for x in insts if _kind(x) == 'UNKNOWN')
            raise ValueError('unknown instruction/event at line ' + str(bad['line'] + 1))
        backedges = []
        for end, inst in enumerate(insts):
            if inst['op'] not in _BRANCHES:
                continue
            start = parsed['labels'][inst['args']]
            if start <= end:
                backedges.append((start, end, inst['args']))
        if not backedges:
            raise ValueError('no steady-state backedge')
        ordered = sorted(backedges)
        for left, right in zip(ordered, ordered[1:]):
            if right[0] <= left[1]:
                raise ValueError('nested, overlapping or multiple-latch loops')
        insertions, removals, changed_vm, new_lgkm = {}, set(), [], []
        for start, end, header in ordered:
            body = insts[start:end + 1]
            # Fail whole-function for any unmodeled backedge, not only a
            # convenient matching subset of clones.
            if not any(x['op'].startswith('v_mmac') for x in body) or not any(_kind(x) == 'G2L' for x in body):
                raise ValueError('backedge without complete MMA/G2L steady skeleton')
            if body[-1]['op'] == 's_branch':
                raise ValueError('steady-state loop has no bounded conditional exit')
            if any(x['op'] in _BRANCHES or x['op'] == 's_endpgm' for x in body[:-1]):
                raise ValueError('internal loop branch/exit')
            for x in body:
                if _kind(x) in ('SMEM', 'OTHER_VM'):
                    raise ValueError('non-packet memory in steady-state loop')
                if ('saveexec' in x['op'] or x['op'].startswith('v_cmpx') or
                    re.match(r'exec(?:_lo|_hi)?(?:,|$)', x['args'])):
                    raise ValueError('conditional EXEC within DMA loop')
            # No predecessor from outside may enter the loop interior. A
            # header may have its normal preheader and the sole latch.
            for source in insts:
                if source['op'] in _BRANCHES:
                    target = parsed['labels'][source['args']]
                    if start < target <= end:
                        raise ValueError('loop interior side entry')
            ctas = [i for i, x in enumerate(body) if _kind(x) == 'CTA']
            if len(ctas) != expected_groups:
                raise ValueError('fresh source/native group coverage mismatch')
            for x in body[ctas[-1] + 1:-1]:
                if _kind(x) in ('DS_READ', 'G2L', 'CTA') or (_kind(x) == 'WAIT' and set(_wait(x['args'])) != {'lgkmcnt'}):
                    raise ValueError('uncovered memory/publication after final group')
            groups = []
            for q, cta_index in enumerate(ctas):
                segment = body[(ctas[q - 1] + 1 if q else 0):cta_index + 1]
                ds = [x for x in segment if _kind(x) == 'DS_READ']
                copies = [x for x in segment if _kind(x) == 'G2L']
                vm = [x for x in segment if _kind(x) == 'WAIT' and 'vmcnt' in _wait(x['args'])]
                if not ds or not copies or len(vm) != 1:
                    raise ValueError('incomplete DS/copy/publication group')
                first, cta = copies[0]['line'], body[cta_index]['line']
                if not all(x['line'] < first for x in ds):
                    raise ValueError('read group is not before first copy')
                fields = _wait(vm[0]['args'])
                if set(fields) != {'vmcnt'} or not first < vm[0]['line'] < cta:
                    raise ValueError('publication is not copy -> VM-only wait -> CTA')
                credit, copy_count, old_vm = len(ds), len(copies), fields['vmcnt']
                if not 0 < credit < 15 or old_vm < copy_count:
                    raise ValueError('unsupported or incomplete counter budget')
                if first in insertions:
                    raise ValueError('group overlap')
                tag = '; PACKET_REUSE_' + mode + '_' + header + '_' + str(q)
                lgkm = '\ts_waitcnt lgkmcnt(' + str(credit) + ') ' + tag + '\n'
                new_lgkm.append(lgkm)
                moved = []
                if mode == 'precopy':
                    insertions[first] = lgkm
                else:
                    new_vm = '\ts_waitcnt vmcnt(' + str(old_vm - copy_count) + ') ' + tag + '\n'
                    insertions[first] = lgkm + new_vm + lines[cta]
                    removals.update((vm[0]['line'], cta))
                    changed_vm.append((lines[vm[0]['line']], new_vm))
                    moved = [x['line'] + 1 for x in segment
                             if first <= x['line'] < cta and x['op'].startswith('v_mmac')]
                groups.append(dict(group=q, DS_count=credit, G2L_count=copy_count,
                    first_G2L_line=first + 1, CTA_line=cta + 1,
                    old_VM_credit=old_vm,
                    new_VM_credit=old_vm - copy_count if mode == 'fenced' else old_vm,
                    DS_lines=[x['line'] + 1 for x in ds],
                    G2L_lines=[x['line'] + 1 for x in copies],
                    original_VM_line=vm[0]['line'] + 1,
                    phase_changed=bool(moved), MMA_moved_across_CTA_lines=moved))
            # Homogeneous packet groups are required for a meaningful common
            # queue budget; no shape/width special case supplies the value.
            if len(set(g['DS_count'] for g in groups)) != 1 or len(set(g['G2L_count'] for g in groups)) != 1:
                raise ValueError('inconsistent group event counts')
            report['loops'].append(dict(header=header, begin_line=body[0]['line'] + 1,
                end_line=body[-1]['line'] + 1, groups=groups,
                MMA_count=sum(x['op'].startswith('v_mmac') for x in body)))
        # Apply only AFTER every loop/group passed; no partial rewrite on fail.
        result = ''.join(insertions.get(n, '') + ('' if n in removals else line)
                         for n, line in enumerate(lines))
        projected = result
        for new in insertions.values():
            if projected.count(new) != 1:
                raise ValueError('nonunique inserted sequence')
            projected = projected.replace(new, '', 1)
        if projected != ''.join(line for n, line in enumerate(lines) if n not in removals):
            raise ValueError('ordered byte projection mismatch')
        oldops = [_operation(x) for x in lines if _operation(x)]
        newops = [_operation(x) for x in result.splitlines() if _operation(x)]
        expected = collections.Counter(oldops)
        for old, new in changed_vm:
            expected.subtract([_operation(old)])
            expected.update([_operation(new)])
        expected.update(_operation(x) for x in new_lgkm)
        if +expected != collections.Counter(newops):
            raise ValueError('instruction multiset mismatch')
        total = sum(len(loop['groups']) for loop in report['loops'])
        report.update(accepted=True, changed=result != asm, reason='all native steady loops/groups covered',
            native_total_groups=total, native_loops=len(report['loops']), loop_count=len(report['loops']), added_waits=total,
            phase_changed_groups=sum(g['phase_changed'] for l in report['loops'] for g in l['groups']),
            phase_changed=any(g['phase_changed'] for l in report['loops'] for g in l['groups']),
            ordered_byte_projection_exact=True, instruction_multiset_exact=True,
            output_sha256=_sha(result))
        return result, report
    except Exception as error:
        report.update(reason=str(error), loops=[], native_total_groups=0,
                      output_sha256=_sha(asm), changed=False)
        return asm, report


def cleanup_redundant_lgkm(asm, name):
    """Single-BB no-new-LGKM-event upper-bound cleanup; FIFO is not required.

    DS/SMEM/unknown operations clear the bound. Only canonical LGKM-only wait
    N with known bound<=N is removed. All unsupported control flow returns the
    exact input. Non-CFG local labels do not manufacture an incoming edge.
    """
    report = _base_report(asm, 'redundant_lgkm')
    report['requires_fresh_LLVM_operand_model'] = False
    try:
        parsed = _parse(asm, name)
        lines = parsed['lines'];bound = witness = None
        between, deleted, ignored = [], [], []
        for n in range(parsed['begin'], parsed['end']):
            label = re.match(r'([\w.$]+):', lines[n].strip())
            if label:
                if label[1] in parsed['leaders']:
                    bound = witness = None;between = []
                else:
                    ignored.append(dict(line=n + 1, label=label[1], reason='no incoming direct CFG edge'))
                continue
            item = _operation(lines[n])
            if item is None:
                continue
            inst = dict(line=n, op=item[0], args=item[1]);kind = _kind(inst)
            if kind == 'WAIT':
                fields = _wait(inst['args']);lgkm = fields.get('lgkmcnt')
                if lgkm is None:
                    between.append(dict(line=n + 1, op=inst['op'], args=inst['args']));continue
                if set(fields) == {'lgkmcnt'} and bound is not None and bound <= lgkm:
                    deleted.append(dict(line=n + 1, text=lines[n], wait_credit=lgkm,
                        upper_bound=bound, witness=witness,
                        intervening_instructions=list(between)))
                    continue
                if lgkm < 15 and (bound is None or lgkm < bound):
                    bound = lgkm;witness = dict(line=n + 1, text=lines[n], credit=lgkm);between = []
                else:
                    between.append(dict(line=n + 1, op=inst['op'], args=inst['args']))
            elif kind in ('PURE', 'G2L', 'CTA'):
                between.append(dict(line=n + 1, op=inst['op'], args=inst['args']))
            else:
                bound = witness = None;between = []
        removed = {x['line'] - 1 for x in deleted}
        result = ''.join(line for n, line in enumerate(lines) if n not in removed)
        it = iter(result.splitlines(keepends=True))
        restored = ''.join(lines[n] if n in removed else next(it) for n in range(len(lines)))
        if next(it, None) is not None or restored != asm:
            raise ValueError('deleted-line reinsertion mismatch')
        report.update(accepted=True, changed=result != asm, reason='single-BB explicit LGKM upper-bound cleanup',
            removed_waits=len(deleted), deleted_count=len(deleted), deleted=deleted, ignored_non_CFG_labels=ignored,
            ordered_byte_projection_exact=True, output_sha256=_sha(result),
            rule_uses_FIFO=False)
        return result, report
    except Exception as error:
        report.update(reason=str(error), removed_waits=0, changed=False, output_sha256=_sha(asm))
        return asm, report
