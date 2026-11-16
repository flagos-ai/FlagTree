"""Conservative TTIR argument effects for the full-buffer runtime bridge.

This tracks pointer provenance, not access bounds or parallel safety. Unknown
pointer operations make materialization unavailable instead of guessing outputs
from Python parameter names. Uses the existing libtriton IR inspection API.
"""

ACCESS_VERSION = 1


def argument_access(module):
    entry = module.get_function(module.get_entry_func_name())
    arguments = [entry.args(i) for i in range(entry.get_num_args())]
    operations = []
    module.walk(operations.append)
    owners = {}
    blocks = {}
    reasons = set()
    provenance = {}

    def pointer(value):
        return "!tt.ptr<" in str(value.get_type())

    def operands(op):
        return [op.get_operand(i) for i in range(op.get_num_operands())]

    def results(op):
        return [op.get_result(i) for i in range(op.get_num_results())]

    def merge(value, roots):
        if not pointer(value):
            return False
        old = provenance.setdefault(value.id(), set())
        size = len(old)
        old.update(roots)
        return len(old) != size

    for i, value in enumerate(arguments):
        if pointer(value):
            provenance[value.id()] = {i}
    for op in operations:
        for i in range(op.get_num_regions()):
            region = op.get_region(i)
            owners[region.id()] = op
            if region.size() > 1:
                reasons.add("multi-block control flow")
        block = op.get_block()
        if block is not None:
            blocks[block.get_parent().id()] = block

    forward = {
        "tt.addptr",
        "tt.splat",
        "tt.broadcast",
        "tt.expand_dims",
        "tt.trans",
        "tt.reshape",
        "tt.join",
        "tt.split",
        "tt.bitcast",
        "arith.select",
        "tt.gather",
        "tt.cat",
    }
    # Cyclic loop-carried pointers are solved to a fixed point. Each edge only
    # adds argument roots, and there are no more edges in a path than IR values.
    for _ in range(len(operations) + 2):
        changed = False
        for op in operations:
            name, ins, outs = op.get_name(), operands(op), results(op)
            if name in forward:
                roots = set().union(*(provenance.get(v.id(), set()) for v in ins if pointer(v)))
                for value in outs:
                    changed = merge(value, roots) or changed
            elif name == "scf.for":
                block = blocks.get(op.get_region(0).id())
                if block is None:
                    reasons.add("missing loop body")
                    continue
                for i, value in enumerate(ins[3:]):
                    roots = provenance.get(value.id(), set())
                    changed = merge(block.get_argument(i + 1), roots) or changed
                    changed = merge(outs[i], roots) or changed
            elif name == "scf.yield":
                parent = owners.get(op.get_block().get_parent().id())
                if parent is None:
                    reasons.add("unknown yield owner")
                    continue
                if parent.get_name() in ("scf.for", "scf.if"):
                    for i, value in enumerate(ins):
                        roots = provenance.get(value.id(), set())
                        changed = merge(parent.get_result(i), roots) or changed
                        if parent.get_name() == "scf.for":
                            block = blocks[parent.get_region(0).id()]
                            changed = merge(block.get_argument(i + 1), roots) or changed
        if not changed:
            break
    else:
        reasons.add("pointer provenance did not converge")

    effects = [set() for _ in arguments]
    for op in operations:
        name, ins, outs = op.get_name(), operands(op), results(op)
        has_pointer = any(pointer(v) for v in ins + outs)
        if name in ("tt.load", "tt.store"):
            roots = provenance.get(ins[0].id(), set())
            if not roots:
                reasons.add("unresolved memory pointer")
            for i in roots:
                effects[i].add("r" if name == "tt.load" else "w")
            if (name == "tt.load" and any(pointer(v) for v in outs)) or (name == "tt.store" and pointer(ins[1])):
                reasons.add("pointer-valued memory")
        elif name in ("tt.call", "func.call", "tt.elementwise_inline_asm"):
            reasons.add(name)
        elif name.startswith("tt.atomic"):
            reasons.add("atomic memory operation")
        elif name == "tt.extern_elementwise" and not op.get_bool_attr("pure"):
            reasons.add("external operation without a pure contract")
        elif has_pointer and name not in forward | {"scf.for", "scf.if", "scf.yield"}:
            reasons.add(name)
    roles = ["rw" if effect == {"r", "w"} else next(iter(effect), "none") for effect in effects]
    return {"version": ACCESS_VERSION, "supported": not reasons, "roles": roles, "reasons": sorted(reasons)}
