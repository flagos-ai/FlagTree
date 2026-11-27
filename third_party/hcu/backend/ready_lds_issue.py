"""Issue independent LDS operand reads before adjacent global register loads."""

import re
from itertools import takewhile

from . import native_isa_model as isa
from .native_isa_model import _registers
from .mma_phase import check_native_readiness

_VM_LOAD = re.compile(r"buffer_load_dword(?:x[24])?$")
_DS_LOAD = re.compile(r"ds_read(?:_b(?:32|64|96|128)|2(?:st64)?_b(?:32|64)|_m32x16_b16(?:_alt)?)$")
_VM_FLAGS = re.compile(r"^(?:\s+(?:offen|idxen|offset:\d+))*$")
_DS_FLAGS = re.compile(r"^(?:\s+offset(?:0|1)?:\d+)*$")


def _vm_read(inst):
    if not _VM_LOAD.fullmatch(inst["op"]):
        return False
    parts = inst["args"].rsplit(",", 1)
    if len(parts) != 2:
        return False
    offset = re.fullmatch(r"\s*(?:[0-9]+|s[0-9]+)(.*)", parts[-1])
    return bool(offset and _VM_FLAGS.fullmatch(offset[1]))


def _lds_read(inst):
    if not _DS_LOAD.fullmatch(inst["op"]):
        return False
    parts = inst["args"].rsplit(",", 1)
    if len(parts) != 2:
        return False
    address = re.fullmatch(r"\s*v[0-9]+(.*)", parts[-1])
    return bool(address and _DS_FLAGS.fullmatch(address[1]))


def _move_groups(assembly, name):
    model = isa.parse_kernel(assembly, name)
    ops, lines = model["instructions"], model["lines"]
    label_lines = set(model["label_lines"].values())
    target_lines = {model["label_lines"][label] for label in model["leaders"]}
    edits, sites = {}, []
    for index, inst in enumerate(ops[1:], 1):
        if not _vm_read(ops[index - 1]) or not _lds_read(inst):
            continue
        vm = list(takewhile(_vm_read, reversed(ops[:index])))[::-1]
        lds = list(takewhile(_lds_read, ops[index:]))
        if len(vm) < 2 or len(lds) < 2:
            continue
        start, end = vm[0]["line"], lds[-1]["line"]
        block = range(start, end + 1)
        if target_lines.intersection(block) or any(line not in label_lines and lines[line].lstrip().startswith(".")
                                                   and not lines[line].lstrip().startswith((".loc", ".file"))
                                                   for line in block):
            continue
        vm_registers = set().union(*(_registers(i["args"]) for i in vm))
        lds_registers = set().union(*(_registers(i["args"]) for i in lds))
        if vm_registers & lds_registers:
            continue
        edits[start] = "".join(lines[i["line"]] for i in lds) + lines[start]
        edits.update({i["line"]: "" for i in lds})
        sites.append({"vm_count": len(vm), "lds_count": len(lds), "line": start + 1})
    return "".join(edits.get(index, line) for index, line in enumerate(lines)), sites


def schedule_ready_lds_issue(assembly, name):
    """Keep the original on an unsupported instruction or completion path."""
    try:
        if isa.MODEL_API_VERSION != 1:
            raise ValueError("unsupported native ISA model")
        updated, sites = _move_groups(assembly, name)
        if not sites:
            return assembly, {"sites": [], "result": "no independent operand groups"}
        for candidate in (assembly, updated):
            readiness = check_native_readiness(candidate, name)
            if not readiness["accepted"]:
                return assembly, {"sites": [], "result": readiness["reason"]}
        return updated, {
            "sites": sites,
            "result": "independent LDS operands issued first",
        }
    except (IndexError, KeyError, TypeError, ValueError) as error:
        return assembly, {"sites": [], "result": str(error)}
