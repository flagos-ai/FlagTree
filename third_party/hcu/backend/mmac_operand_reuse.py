"""Schedule independent MMAC updates in local-read completion order.

A priority cohort must contain only MMACs and completion waits. Memory issues,
priority boundaries and CTA synchronization stay in place. Each accumulator's
floating-point update order is preserved. The complete native CFG is checked
before and after scheduling; unsupported input always keeps its original text.
"""
import re

from . import native_isa_model as isa
from .mma_phase import _registers, check_native_readiness


def _mmac(inst):
    if inst["op"] not in ("v_mmac_f32_16x16x16_bf16", "v_mmac_f32_16x16x16_f16"):
        return None
    args = [item.strip() for item in inst["args"].split(",")]
    if len(args) != 4 or args[0] != args[3] or not all(
            re.fullmatch(r"v(?:[0-9]+|\[[0-9]+:[0-9]+\])", arg) for arg in args):
        return None
    return _registers(args[0]), _registers(",".join(args[1:3]))


def _transparent(lines, labels, first, last):
    for n in range(first, last + 1):
        text = lines[n].lstrip()
        if n in labels:
            return False
        if re.match(r"[\w.$]+:", text):
            continue
        if text.startswith(".") and not re.match(r"\.(loc|file|cfi_\w+)(?:\s|$)", text):
            return False
    return True


def _schedule(assembly, name):
    model = isa.parse_kernel(assembly, name)
    ops, lines = model["instructions"], model["lines"]
    labels = set(model["label_lines"].values())
    leaders = {model["label_lines"][name] for name in model["leaders"]}
    replacements, sites = {}, []
    for start, inst in enumerate(ops):
        if inst["op"] != "s_setprio" or inst["args"] != "1":
            continue
        stop = next((i for i in range(start + 1, len(ops)) if ops[i]["op"] == "s_setprio"), None)
        if stop is None or ops[stop]["args"] != "0":
            continue
        body = ops[start + 1:stop]
        if not body or any(not _mmac(x) and x["op"] != "s_waitcnt" for x in body):
            continue
        first, last = body[0]["line"], body[-1]["line"]
        if not _transparent(lines, labels, first, last):
            continue

        # There may be zero or several already-issued MMACs before the
        # priority boundary. Only pure MMAC/wait prefixes can separate it
        # from a contiguous LDS read group; no event may change queue ages.
        before = start - 1
        prefix = []
        while before >= 0 and (_mmac(ops[before]) or ops[before]["op"] == "s_waitcnt"):
            prefix.append(ops[before])
            before -= 1
        reads = []
        while before >= 0 and isa.classify_instruction(ops[before]) == "DS_READ":
            reads.append(_registers(ops[before]["args"].split(",")[0]))
            before -= 1
        if not reads or not all(reads):
            continue
        read_start = ops[before + 1]["line"]
        # Prefix labels are preserved verbatim. Only incoming CFG edges,
        # rather than unused debug/address labels, invalidate its queue age.
        if not _transparent(lines, leaders, read_start, inst["line"]):
            continue
        prior_waits = [isa.parse_wait(x["args"]) for x in prefix if x["op"] == "s_waitcnt"]
        waits = [isa.parse_wait(x["args"]) for x in body if x["op"] == "s_waitcnt"]
        if not all(prior_waits + waits):
            continue
        credit = min((w["lgkmcnt"] for w in prior_waits if "lgkmcnt" in w), default=len(reads))
        if credit == 0:
            continue
        age = {}
        for value, registers in enumerate(reads):
            if age.keys() & registers:
                break
            age.update({reg: value for reg in registers})
        else:
            dots = [x for x in body if _mmac(x)]
            if not dots:
                continue
            operands = [_mmac(x) for x in dots]
            destinations = set().union(*(dst for dst, _ in operands))
            if destinations & age.keys() or any(src & destinations for _, src in operands):
                continue
            owners = {}
            for dest, _ in operands:
                if any(reg in owners and owners[reg] != dest for reg in dest):
                    break
                owners.update({reg: dest for reg in dest})
            else:
                if any(not src & age.keys() for _, src in operands):
                    continue
                credits = [min(age[reg] for reg in src & age.keys()) for _, src in operands]
                pending, order, complete = list(range(len(dots))), [], set()
                previous, dependencies = {}, []
                for i, (dest, _) in enumerate(operands):
                    dependencies.append(previous.get(frozenset(dest)))
                    previous[frozenset(dest)] = i
                while pending:
                    ready = [i for i in pending if dependencies[i] is None or dependencies[i] in complete]
                    chosen = max(ready, key=lambda i: (credits[i], -i))
                    order.append(chosen)
                    complete.add(chosen)
                    pending.remove(chosen)
                if order == list(range(len(dots))):
                    continue
                vm = [w["vmcnt"] for w in waits if "vmcnt" in w]
                changed = [f"\ts_waitcnt vmcnt({min(vm)})\n"] if vm else []
                for i in order:
                    if credits[i] < credit:
                        credit = credits[i]
                        changed.append(f"\ts_waitcnt lgkmcnt({credit})\n")
                    changed.append(lines[dots[i]["line"]])
                final_credit = min((w.get("lgkmcnt", 15) for w in waits), default=15)
                if final_credit < credit:
                    changed.append(f"\ts_waitcnt lgkmcnt({final_credit})\n")
                replacements[first] = "".join(changed)
                replacements.update({i: "" for i in range(first + 1, last + 1)})
                sites.append(
                    dict(first_line=first + 1, last_line=last + 1, mmac=len(dots),
                         reordered=sum(i != j for i, j in enumerate(order))))
    return "".join(replacements.get(i, line) for i, line in enumerate(lines)), sites


def schedule_mmac_operand_reuse(assembly, name):
    """Return original code on every rejection, including analysis limits."""
    report = dict(accepted=False, changed=False, reason="", sites=[])
    try:
        if isa.MODEL_API_VERSION != 1:
            raise ValueError("unsupported native ISA model API")
        if not isinstance(assembly, str) or "s_setprio" not in assembly:
            report.update(accepted=True, reason="no priority cohort")
            return assembly, report
        original = check_native_readiness(assembly, name)
        if not original["accepted"]:
            report["reason"] = "original completion: " + original["reason"]
            return assembly, report
        candidate, sites = _schedule(assembly, name)
        if not sites:
            report.update(accepted=True, reason="no eligible register-only MMAC cohort")
            return assembly, report
        completion = check_native_readiness(candidate, name)
        report["completion"] = completion
        if not completion["accepted"]:
            report["reason"] = "candidate completion: " + completion["reason"]
            return assembly, report
        report.update(accepted=True, changed=candidate != assembly, sites=sites,
                      reason="independent MMACs consume completed LDS operands first")
        return candidate, report
    except (IndexError, KeyError, TypeError, ValueError) as error:
        report["reason"] = "unsupported native program: " + str(error)
        return assembly, report
