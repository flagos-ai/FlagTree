"""Standalone LGKM waits: strengthen first/min, remove later waits in one BB.

Only the shared native classifier's PURE instructions may separate the waits.
Memory, CTA, CFG, unknown and non-standalone waits end a segment. No assumption
about completion cycles/FIFO is used. A later assembler/readiness check remains
mandatory; this module never assembles, dispatches, or claims hardware validity.
"""
import hashlib
import importlib
from pathlib import Path
import re

_MODEL_API_VERSION = 1


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _model():
    model = importlib.import_module('.native_isa_model', __package__)
    if model.MODEL_API_VERSION != _MODEL_API_VERSION:
        raise ValueError('unsupported HCU native ISA model API version')
    return model



def coalesce_lgkm_waits(assembly, kernel_name):
    """Return (assembly, report); rejection always returns the exact input.

    sites is a list of whole maximal PURE/standalone-LGKM segments containing
    at least two waits. Line numbers are one-based in the ORIGINAL input.
    This bounded counter proof does not install or replace a native checker.
    """
    report = dict(accepted=False, changed=False, reason='', sites=[],
                  site_count=0, deleted_count=0, strengthened_count=0,
                  diagnostic_only=True, requires_native_readiness_after_assembly=True,
                  hardware_validated=False, input_sha256=_sha(assembly) if isinstance(assembly, str) else None,
                  output_sha256=_sha(assembly) if isinstance(assembly, str) else None,
                  classifier_api_version=_MODEL_API_VERSION, classifier_sha256=None,
                  no_new_events_assumed=True, rule_uses_FIFO=False,
                  completion_cycle_assumption=False)
    try:
        model = _model()
        report["classifier_sha256"] = hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest()
        parsed = model.parse_kernel(assembly, kernel_name)
        lines, insts = parsed['lines'], parsed['instructions']
        # Do not let ignored assembler macros/directives hide events or edges.
        # Only zero-code debug/CFI directives are transparent here.
        for n in range(parsed['begin'], insts[-1]['line'] + 1):
            text = lines[n].strip()
            if re.match(r'[\w.$]+:', text):
                continue
            if text.startswith('.') and not re.match(r'\.(?:loc|file|cfi_[A-Za-z0-9_]+)(?:\s|$)', text):
                raise ValueError('unsupported in-kernel assembler directive at line ' + str(n + 1))
        if any(x['op'] in model.BRANCH_OPCODES and not re.fullmatch(r'[A-Za-z_.$][\w.$]*', x['args']) for x in insts):
            raise ValueError('numeric or non-symbolic branch target')
        if any(x['op'] in ('s_join', 's_fork') for x in insts):
            raise ValueError('unsupported implicit control flow')
        leaders = {0} | {parsed['labels'][name] for name in parsed['leaders']}
        for i, inst in enumerate(insts[:-1]):
            if model.classify_instruction(inst) == 'CONTROL':
                leaders.add(i + 1)
        modifications, deletions, sites, barriers = {}, set(), [], []
        segment, block_start = [], 0
        def flush(why):
            if sum(x['op'] == 's_waitcnt' for x in segment) >= 2:
                waits = [x for x in segment if x['op'] == 's_waitcnt']
                credits = [model.parse_wait(x['args'])['lgkmcnt'] for x in waits]
                credit = min(credits)
                first = waits[0]
                if credits[0] != credit:
                    changed, count = re.subn(r'lgkmcnt\(\d+\)', 'lgkmcnt(' + str(credit) + ')', lines[first['line']], count=1)
                    if count != 1:
                        raise ValueError('cannot preserve first wait line')
                    modifications[first['line']] = changed
                deletions.update(x['line'] for x in waits[1:])
                sites.append(dict(first_line=first['line'] + 1,
                    last_line=waits[-1]['line'] + 1,
                    basic_block_first_instruction_line=insts[block_start]['line'] + 1,
                    old_credits=credits, new_first_credit=credit,
                    strengthened=credit != credits[0],
                    first_original_text=lines[first['line']],
                    first_new_text=modifications.get(first['line'], lines[first['line']]),
                    deleted=[dict(line=x['line'] + 1, credit=model.parse_wait(x['args'])['lgkmcnt'], text=lines[x['line']]) for x in waits[1:]],
                    intervening_pure=[dict(line=x['line'] + 1, op=x['op'], args=x['args']) for x in segment if x['op'] != 's_waitcnt' and first['line'] < x['line'] < waits[-1]['line']],
                    segment_end_reason=why,
                    no_new_LGKM_VM_or_other_event_between_waits=True,
                    proof='first wait guarantees LGKM<=min(all credits); no new event in this BB segment can increase it'))
            segment.clear()
        for i, inst in enumerate(insts):
            if i in leaders:
                flush('basic block leader')
                block_start = i
            kind = model.classify_instruction(inst)
            fields = model.parse_wait(inst['args']) if kind == 'WAIT' else None
            if kind == 'PURE' or (kind == 'WAIT' and set(fields) == {'lgkmcnt'}):
                segment.append(inst)
            else:
                flush(kind + ' segment barrier')
                barriers.append(dict(line=inst['line'] + 1, op=inst['op'], kind=kind))
        flush('kernel end')
        result = ''.join('' if n in deletions else modifications.get(n, line) for n, line in enumerate(lines))
        # Every byte outside deleted wait lines and the first credit replacement
        # must remain untouched, including labels, metadata, MMAC and memory.
        restored, output = [], iter(result.splitlines(keepends=True))
        for n, line in enumerate(lines):
            if n in deletions:
                restored.append(line)
            else:
                observed = next(output)
                if observed != modifications.get(n, line):
                    raise ValueError('ordered non-wait byte projection mismatch')
                restored.append(line)
        if next(output, None) is not None or ''.join(restored) != assembly:
            raise ValueError('exact original reconstruction failed')
        reparsed = model.parse_kernel(result, kernel_name)
        projection = lambda xs: [(x['op'], x['args']) for x in xs
            if not (x['op'] == 's_waitcnt' and model.parse_wait(x['args']) is not None and set(model.parse_wait(x['args'])) == {'lgkmcnt'})]
        if projection(insts) != projection(reparsed['instructions']):
            raise ValueError('memory/MMAC/control/non-LGKM ordered projection changed')
        report.update(accepted=True, changed=result != assembly,
            reason='first=min standalone LGKM waits in PURE single-BB segments' if sites else 'no eligible multi-wait PURE segment',
            sites=sites, site_count=len(sites), deleted_count=len(deletions),
            strengthened_count=len(modifications), barriers=barriers,
            basic_block_leader_lines=sorted(insts[i]['line'] + 1 for i in leaders),
            ignored_non_CFG_labels=[dict(label=name, line=parsed['label_lines'][name] + 1) for name in parsed['labels'] if name not in parsed['leaders']],
            exact_other_lines_preserved=True,
            ordered_non_LGKM_instruction_projection_exact=True,
            output_sha256=_sha(result))
        return result, report
    except Exception as error:
        report.update(reason=str(error), sites=[], site_count=0, deleted_count=0,
                      strengthened_count=0, changed=False)
        return assembly, report
