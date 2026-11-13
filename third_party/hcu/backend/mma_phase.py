"""Conservative gfx936 async operand check for experimental MMA motion.

This is a completion check, not a timing model. Unknown instructions, mixed
partial LGKM completion, or an analysis budget limit reject the candidate.
The caller must compile the unchanged LLVM when this check rejects it.
"""
import collections
import re

# Consumers can require this operand model without relying on package versions.
# Acceptance of an individual kernel still requires the complete CFG audit.
NATIVE_READINESS_CAPABILITIES = frozenset({
    "gfx936.v_xad_u32.single_dst_3src",
    "gfx936.v_perm_b32.single_dst_3src",
    "gfx936.s_setprio.literal_0_1",
    "gfx936.scalar_bitset.sgpr_read_modify_write",
})


def _registers(text):
    result = set()
    for match in re.finditer(r"\b([vs])(?:\[(\d+):(\d+)\]|(\d+))(?!\w)", text):
        kind, begin, end, single = match.groups()
        result.update((kind, index) for index in
                      ([int(single)] if single is not None else
                       range(int(begin), int(end) + 1)))
    return result


def _xad_u32_effects(operands):
    # GFX9 VOP3 XOR-add has one 32-bit VGPR result and three explicit
    # 32-bit sources. It has no asynchronous result or implicit data-register
    # writes (EXEC is read implicitly). Keep modifiers, ranges and special
    # registers outside this narrow model instead of guessing their effects.
    if len(operands) != 4 or not re.fullmatch(r"v[0-9]+", operands[0]):
        return None
    for source in operands[1:]:
        if re.fullmatch(r"[vs][0-9]+", source):
            continue
        if not re.fullmatch(r"-?(?:0[xX][0-9a-fA-F]{1,8}|[0-9]{1,10})", source):
            return None
        value = int(source, 16 if "x" in source.lower() else 10)
        if not -(1 << 31) <= value < (1 << 32):
            return None
    return _registers(",".join(operands[1:])), _registers(operands[0])


def _scalar_source_registers(source):
    if re.fullmatch(r"s[0-9]+", source):
        return _registers(source)
    if not re.fullmatch(r"-?(?:0[xX][0-9a-fA-F]{1,8}|[0-9]{1,10})", source):
        return None
    value = int(source, 16 if "x" in source.lower() else 10)
    return set() if -(1 << 31) <= value < (1 << 32) else None


def _sext_i32_i16_effects(operands):
    # Vega SOP1 S_SEXT_I32_I16: one synchronous 32-bit SGPR result;
    # D={16{S0[15]},S0[15:0]}, SCC is not written. No queue completion.
    if len(operands) != 2 or not re.fullmatch(r"s[0-9]+", operands[0]):
        return None
    reads = _scalar_source_registers(operands[1])
    return None if reads is None else (reads, _registers(operands[0]))


def _bitset_effects(op, operands):
    # SOP1 bit set/reset preserves all other destination bits: the old
    # destination is an input, including BOTH SGPRs of a 64-bit pair.
    # The bit index is a 32-bit scalar source (low5/low6 bits are used).
    # No memory event or async completion is introduced. Special registers
    # and malformed operands retain the conservative native fallback.
    match = re.fullmatch(r"s_bitset[01]_b(32|64)", op)
    if match is None or len(operands) != 2:
        return None
    if match[1] == "32":
        if not re.fullmatch(r"s[0-9]+", operands[0]):
            return None
    else:
        pair = re.fullmatch(r"s\[(\d+):(\d+)\]", operands[0])
        if pair is None or int(pair[1]) % 2 or int(pair[2]) != int(pair[1]) + 1:
            return None
    reads = _scalar_source_registers(operands[1])
    dest = _registers(operands[0])
    return None if reads is None else (reads | dest, dest)


def check_native_readiness(assembly, name, max_states=100000):
    """Return a bounded, all-CFG-paths VM/LGKM completion audit."""
    instructions, labels = [], {}
    inside = False
    for line_number, raw in enumerate(assembly.splitlines(), 1):
        line = raw.strip()
        if line == name + ":" or line.startswith(name + ": "):
            inside = True
        if not inside:
            continue
        if line.startswith(".Lfunc_end"):
            break
        label = re.match(r"([\w.$]+):", line)
        if label:
            labels[label[1]] = len(instructions)
            continue
        if not line or line.startswith((".", ";", "#")):
            continue
        operation = re.match(r"([a-z][\w.]*)\s*(.*)", line)
        if operation:
            instructions.append((line_number, operation[1], operation[2].split(";")[0].strip()))
    def reject(reason, **details):
        return dict(accepted=False, reason=reason, **details)
    if not instructions or name not in labels:
        return reject("missing native kernel body")
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
        line_number, op, args = instructions[pc]
        visited.add(pc)
        if op.startswith(("s_movrel", "v_movrel")):
            return reject("unsupported implicit register indexing", native_line=line_number, op=op)
        operands = [part.strip() for part in args.split(",")]
        all_regs = _registers(args)
        dest = _registers(operands[0])
        reads, writes = all_regs - dest, dest
        queue, event = None, None
        if op.startswith("ds_read_") or op.startswith("ds_read2"):
            queue, event = 0, "DS"
        elif op.startswith("s_load_"):
            queue, event = 0, "SMEM"
        elif op.startswith(("buffer_load_", "global_load_", "flat_load_")):
            queue, event = 1, "VM"
            if re.search(r"\blds\b", args):
                reads, writes = all_regs, set()
        elif op.startswith(("buffer_store_", "global_store_", "flat_store_")):
            reads, writes, queue, event = all_regs, set(), 1, "VM"
        elif op.startswith("ds_write_"):
            reads, writes, queue, event = all_regs, set(), 0, "DS"
        elif op.startswith("s_"):
            if op == "s_sext_i32_i16":
                effects = _sext_i32_i16_effects(operands)
                if effects is None:
                    return reject("unsupported s_sext_i32_i16 operands", native_line=line_number, op=op)
                reads, writes = effects
            elif re.fullmatch(r"s_bitset[01]_b(?:32|64)", op):
                effects = _bitset_effects(op, operands)
                if effects is None:
                    return reject("unsupported scalar bitset operands", native_line=line_number, op=op)
                reads, writes = effects
            elif op == "s_setprio":
                # The generated priority controls neither define data registers
                # nor complete an async response. Preserve both pending queues.
                if operands not in (["0"], ["1"]):
                    return reject("unsupported s_setprio operands", native_line=line_number, op=op)
                reads, writes = set(), set()
            elif op in ("s_waitcnt", "s_barrier", "s_nop", "s_endpgm", "s_branch") or op.startswith(("s_cbranch", "s_cmp")):
                reads, writes = all_regs, set()
            elif not op.startswith(("s_add", "s_sub", "s_and", "s_or", "s_xor", "s_mov", "s_mul", "s_lsh", "s_ash", "s_bfe", "s_cselect", "s_min", "s_max", "s_not", "s_brev")):
                return reject("unsupported scalar opcode", native_line=line_number, op=op)
        elif op.startswith("v_"):
            if op in ("v_xad_u32", "v_perm_b32"):
                # V_PERM_B32 is a synchronous byte permutation: one explicit
                # 32-bit VGPR result, three explicit 32-bit sources. Reuse
                # the strict operand-effects grammar, never arithmetic semantics.
                effects = _xad_u32_effects(operands)
                if effects is None:
                    return reject("unsupported " + op + " operands", native_line=line_number, op=op)
                reads, writes = effects
            if op.startswith("v_cmp") and not dest:
                reads, writes = all_regs, set()
            # Opcode families below have only register/immediate side effects.
            if op not in ("v_xad_u32", "v_perm_b32") and not op.startswith(("v_add", "v_sub", "v_mul", "v_mad", "v_fma", "v_mmac", "v_mov", "v_readfirstlane", "v_readlane_b32", "v_writelane_b32", "v_cmp", "v_cndmask", "v_cvt", "v_lsh", "v_ash", "v_and", "v_or", "v_xor", "v_bfe", "v_bfrev", "v_rcp", "v_trunc", "v_min", "v_max", "v_pack", "v_align", "v_not")):
                return reject("unsupported vector opcode", native_line=line_number, op=op)
            if op == "v_writelane_b32":
                # Writelane replaces one lane and preserves the others. Its
                # old VGPR value is a read as well as a destination; both that
                # value and the scalar source must have completed. Readlane
                # has an explicit SGPR destination and VGPR/index sources.
                reads |= dest
            if op.startswith(("v_add_co", "v_sub_co", "v_subrev_co", "v_subb_co", "v_mad_u64")) and len(operands) > 1:
                second_dest = _registers(operands[1])
                writes |= second_dest
                reads = _registers(",".join(operands[2:]))
        else:
            return reject("unsupported opcode", native_line=line_number, op=op)
        pending_map = dict(pending)
        bad = (reads | writes) & pending_map.keys()
        if bad:
            return reject("async register used before explicit completion", native_line=line_number,
                          op=op, registers=sorted(bad))
        queues = [list(lgkm), list(vm)]
        if op == "s_waitcnt":
            for which, field, unconstrained in [(0, "lgkmcnt", 15), (1, "vmcnt", 63)]:
                match = re.search(field + r"\((\d+)\)", args)
                if not match or int(match[1]) == unconstrained:
                    continue
                credit = int(match[1])
                if credit and which == 0 and len(set(queues[which])) > 1:
                    return reject("mixed DS/SMEM partial completion", native_line=line_number)
                popped = max(0, len(queues[which]) - credit)
                queues[which] = queues[which][popped:]
                pending_map = {reg: (q, index - popped if q == which else index)
                               for reg, (q, index) in pending_map.items()
                               if q != which or index >= popped}
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
        if op == "s_branch" or op.startswith("s_cbranch"):
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
        oldest_vm_result = min((index for q, index in pending_map.values() if q == 1),
                               default=len(queues[1]))
        if oldest_vm_result:
            queues[1] = queues[1][oldest_vm_result:]
            pending_map = {reg: (q, index - oldest_vm_result if q == 1 else index)
                           for reg, (q, index) in pending_map.items()}
        normalized = tuple(sorted(pending_map.items()))
        for successor in successors:
            todo.append((successor, tuple(queues[0]), tuple(queues[1]), normalized))
    return dict(accepted=True, reason="all modeled native CFG paths explicitly complete operands",
                states=len(seen), visited_instructions=len(visited), max_pending_lgkm=maximum[0],
                max_pending_vm=maximum[1])
