"""Conservative gfx936 async operand check for experimental MMA motion.

This is a completion check, not a timing model. Unknown instructions, mixed
partial LGKM completion, or an analysis budget limit reject the candidate.
The caller must compile the unchanged LLVM when this check rejects it.
"""
import collections
from . import native_isa_model as isa

# Consumers can require this operand model without relying on package versions.
# Acceptance of an individual kernel still requires the complete CFG audit.
NATIVE_READINESS_CAPABILITIES = frozenset({
    "gfx936.v_xad_u32.single_dst_3src",
    "gfx936.v_perm_b32.single_dst_3src",
    "gfx936.s_setprio.literal_0_1",
    "gfx936.scalar_bitset.sgpr_read_modify_write",
})


def check_native_readiness(assembly, name, max_states=100000):
    """Return a bounded, all-CFG-paths VM/LGKM completion audit."""

    def reject(reason, **details):
        return dict(accepted=False, reason=reason, **details)

    try:
        program = isa.parse_kernel(assembly, name)
    except (TypeError, ValueError) as error:
        return reject(str(error))
    instructions, labels = program["instructions"], program["labels"]
    todo = collections.deque([(0, (), (), ())])
    seen, visited = set(), set()
    maximum = [0, 0]
    while todo:
        pc, lgkm, vm, pending = todo.popleft()
        state = (pc, lgkm, vm, pending)
        if state in seen:
            continue
        seen.add(state)
        if len(seen) > max_states:
            return reject("CFG state budget")
        if pc >= len(instructions):
            return reject("fallthrough outside kernel")
        instruction = instructions[pc]
        line_number, op, args = instruction["line"] + 1, instruction["op"], instruction["args"]
        visited.add(pc)
        try:
            reads, writes, queue, event, wait = isa.instruction_effects(op, args)
        except ValueError as error:
            return reject(str(error), native_line=line_number, op=op)
        pending_map = dict(pending)
        bad = (reads | writes) & pending_map.keys()
        if bad:
            return reject("async register used before explicit completion", native_line=line_number, op=op,
                          registers=sorted(bad))
        queues = [list(lgkm), list(vm)]
        if op == "s_waitcnt":
            for which, field, unconstrained in [(0, "lgkmcnt", 15), (1, "vmcnt", 63)]:
                if field not in wait or wait[field] == unconstrained:
                    continue
                credit = wait[field]
                if credit and which == 0 and len(set(queues[which])) > 1:
                    return reject("mixed DS/SMEM partial completion", native_line=line_number)
                popped = max(0, len(queues[which]) - credit)
                queues[which] = queues[which][popped:]
                pending_map = {
                    reg: (q, index - popped if q == which else index)
                    for reg, (q, index) in pending_map.items()
                    if q != which or index >= popped
                }
        elif queue is not None:
            index = len(queues[queue])
            # Completion depends on FIFO age and event kind, never on the
            # issuing instruction's PC. Keep register->queue-index bindings,
            # but merge otherwise identical histories at CFG joins. Retaining
            # event PCs makes masked output stores grow exponentially despite
            # having identical effects on every subsequent transfer/check.
            queues[queue].append(event)
            pending_map.update({reg: (queue, index) for reg in writes})
        for which in range(2):
            maximum[which] = max(maximum[which], len(queues[which]))
            if len(queues[which]) > 256:
                return reject("queue budget", native_line=line_number)
        if op in isa.BRANCH_OPCODES:
            if args not in labels:
                return reject("unresolved branch", native_line=line_number)
            successors = [labels[args]] if op == "s_branch" else [pc + 1, labels[args]]
        elif op == "s_endpgm":
            successors = []
        else:
            successors = [pc + 1]
        # Only events newer than (or equal to) the oldest live VM result can
        # affect a future result's FIFO completion rank. Earlier stores/G2L
        # events have no register result and can be projected out exactly.
        # Keep all newer events, including stores, since they consume vmcnt
        # credit. LGKM is deliberately untouched: mixed SMEM/DS completion has
        # additional ordering restrictions. This check does not model LDS DMA
        # aliasing; MMA motion leaves the memory operation order unchanged.
        oldest_vm_result = min((index for q, index in pending_map.values() if q == 1), default=len(queues[1]))
        if oldest_vm_result:
            queues[1] = queues[1][oldest_vm_result:]
            pending_map = {
                reg: (q, index - oldest_vm_result if q == 1 else index)
                for reg, (q, index) in pending_map.items()
            }
        normalized = tuple(sorted(pending_map.items()))
        for successor in successors:
            todo.append((successor, tuple(queues[0]), tuple(queues[1]), normalized))
    return dict(accepted=True, reason="all modeled native CFG paths explicitly complete operands", states=len(seen),
                visited_instructions=len(visited), max_pending_lgkm=maximum[0], max_pending_vm=maximum[1])
