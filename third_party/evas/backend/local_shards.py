"""Bounded local-window planning from compiler-exported address expressions.

Only integer shape/PID/address expressions are evaluated on the host. Tensor
values never enter the planner. The original full-buffer bridge normalizes
incompatible incoming layouts; die-0 borrows that storage and die-1 uses a local
window. Exact written intervals are scattered back, preserving holes and aliases.
"""

import math
import operator
import sys

import numpy as np

from .buffer_bridge import AutoBufferPlan, BoundPointer

LOCAL_ACCESS_VERSION = 1
MAX_PROGRAMS = 4096
MAX_VISITS = 2_000_000
MAX_SCATTER_SEGMENTS = 256


class UnsupportedLocalAccess(ValueError):
    """A proof limit or unsupported layout selects the existing shared path."""


def local_signature(signature):
    """Append (die1 pointer, origin0, origin1) before the launcher adds grid."""
    result = dict(signature)
    next_index = max(signature, default=-1) + 1
    for ty in signature.values():
        if ty.startswith("*"):
            for extra in (ty, "i64", "i64"):
                result[next_index] = extra
                next_index += 1
    return result


def _segments(indices):
    result = []
    for index in sorted(indices):
        if result and result[-1][1] == index:
            result[-1][1] += 1
        else:
            result.append([index, index + 1])
    return result


class _Evaluator:

    def __init__(self, graph, arguments, grid):
        if graph.get("version") != LOCAL_ACCESS_VERSION or not graph.get("supported"):
            raise UnsupportedLocalAccess(graph.get("reason", "missing access graph"))
        self.nodes = graph["nodes"]
        self.arguments = arguments
        self.grid = grid
        self.visits = 0

    def charge(self, count):
        self.visits += count
        if self.visits > MAX_VISITS:
            raise UnsupportedLocalAccess("address analysis budget exceeded")

    def evaluate(self, index, pid, induction, cache):
        if index in cache:
            return cache[index]
        if not 0 <= index < len(self.nodes):
            raise UnsupportedLocalAccess("invalid access expression")
        node = self.nodes[index]
        op, shape, bits = node["op"], tuple(node["shape"]), node["bits"]
        self.charge(max(1, math.prod(shape)))
        args = [self.evaluate(i, pid, induction, cache) for i in node.get("args", ())]
        if op == "arg":
            try:
                value = operator.index(self.arguments[node["index"]])
            except TypeError as error:
                raise UnsupportedLocalAccess("non-integer scalar dependency") from error
        elif op == "const":
            value = np.full(shape, node["value"], dtype=object)
        elif op == "pid":
            value = pid[node["axis"]]
        elif op == "grid":
            value = self.grid[node["axis"]]
        elif op == "iv":
            value = induction[node["index"]]
        elif op == "tt.make_range":
            value = np.arange(node["start"], node["end"], dtype=object)
        elif op in ("tt.splat", "tt.broadcast"):
            value = np.broadcast_to(args[0], shape)
        elif op == "tt.expand_dims":
            value = np.expand_dims(args[0], node["axis"])
        elif op == "tt.reshape":
            value = args[0].reshape(shape)
        elif op == "tt.trans":
            value = args[0].transpose(node["order"])
        elif op == "arith.addi":
            value = args[0] + args[1]
        elif op == "arith.subi":
            value = args[0] - args[1]
        elif op == "arith.muli":
            value = args[0] * args[1]
        elif op in ("arith.divsi", "arith.remsi"):
            if np.any(args[1] == 0):
                raise UnsupportedLocalAccess("zero divisor in address expression")
            # Signed MLIR division truncates toward zero, unlike Python //.
            quotient = np.where((args[0] < 0) != (args[1] < 0), -1, 1)
            quotient = quotient * (np.abs(args[0]) // np.abs(args[1]))
            value = quotient if op == "arith.divsi" else args[0] - quotient * args[1]
        elif op in ("arith.minsi", "arith.maxsi"):
            value = (np.minimum if op == "arith.minsi" else np.maximum)(*args)
        elif op in ("arith.andi", "arith.ori"):
            value = (np.bitwise_and if op == "arith.andi" else np.bitwise_or)(*args)
        elif op == "arith.cmpi":
            predicate = node["predicate"]
            if predicate.startswith("u") and any(np.any(v < 0) for v in args):
                raise UnsupportedLocalAccess("unsigned comparison of negative values")
            comparisons = {
                "eq": np.equal, "ne": np.not_equal, "slt": np.less, "sle": np.less_equal, "sgt": np.greater, "sge":
                np.greater_equal, "ult": np.less, "ule": np.less_equal, "ugt": np.greater, "uge": np.greater_equal
            }
            value = comparisons[predicate](*args)
        elif op == "arith.select":
            value = np.where(args[0].astype(bool), args[1], args[2])
        elif op in ("arith.extsi", "arith.extui", "arith.trunci"):
            if op == "arith.extui" and np.any(args[0] < 0):
                raise UnsupportedLocalAccess("unsigned extension of negative values")
            value = args[0]
            if op == "arith.extsi" and self.nodes[node["args"][0]]["bits"] == 1:
                value = np.where(args[0].astype(bool), -1, 0)
        else:
            raise UnsupportedLocalAccess("unsupported access expression: " + op)
        value = np.asarray(value, dtype=object)
        if value.shape != shape:
            raise UnsupportedLocalAccess("address expression shape mismatch")
        low, high = (0, 1) if bits == 1 else (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
        # Reject wraparound instead of making an unsound range assumption.
        if value.size and (np.min(value) < low or np.max(value) > high):
            raise UnsupportedLocalAccess("integer overflow in address expression")
        cache[index] = value
        return value


def access_footprints(graph, arguments, grid):
    total = math.prod(grid)
    if total > MAX_PROGRAMS:
        raise UnsupportedLocalAccess("program count exceeds the local analysis limit")
    evaluator = _Evaluator(graph, arguments, grid)
    result = [[{"read": set(), "write": set()} for _ in arguments] for _ in range(2)]
    split = (total + 1) // 2
    for linear in range(total):
        die = int(linear >= split)
        pid = (linear % grid[0], (linear // grid[0]) % grid[1], linear // (grid[0] * grid[1]))
        for access in graph["accesses"]:

            def visit(depth, induction):
                cache = {}

                def evaluate(index):
                    return evaluator.evaluate(index, pid, induction, cache)

                contexts = access["contexts"]
                if depth < len(contexts):
                    context = contexts[depth]
                    if context["kind"] == "if":
                        condition = evaluate(context["condition"])
                        if condition.shape:
                            raise UnsupportedLocalAccess("non-scalar branch condition")
                        if bool(condition.item()) == context["then"]:
                            visit(depth + 1, induction)
                    else:
                        bounds = [evaluate(context[k]) for k in ("lower", "upper", "step")]
                        if any(value.shape for value in bounds):
                            raise UnsupportedLocalAccess("non-scalar loop bound")
                        begin, end, step = (int(value.item()) for value in bounds)
                        if step <= 0:
                            raise UnsupportedLocalAccess("non-positive loop step")
                        for i in range(begin, end, step):
                            evaluator.charge(1)
                            visit(depth + 1, {**induction, context["id"]: i})
                    return
                offsets = evaluate(access["offset"])
                if access["mask"] >= 0:
                    mask = np.broadcast_to(evaluate(access["mask"]), offsets.shape).astype(bool)
                    offsets = offsets[mask]
                evaluator.charge(offsets.size)
                result[die][access["argument"]]["write" if access["write"] else "read"].update(
                    int(v) for v in offsets.flat)

            visit(0, {})
    return result


_quarantined = []


def recover_local_temporaries():
    for plan in list(_quarantined):
        plan.normal.runtime.synchronize(plan.normal.device)
        plan.local_temporaries.clear()
        for window in plan.windows:
            window.pop("temporary", None)
        plan.normal._release()
        plan.bindings = list(plan.arguments)
        _quarantined.remove(plan)


class LocalBufferPlan:
    """Normalize once, preserve aliases, then bind proven die-local windows."""

    def __init__(self, signature, arguments, contract, graph, grid, runtime=None):
        if graph.get("version") != LOCAL_ACCESS_VERSION:
            raise ValueError("EVAS incompatible local access metadata; recompile the kernel")
        self.signature = signature
        self.arguments = tuple(arguments)
        self.normal = AutoBufferPlan(signature, arguments, contract, runtime=runtime)
        self.local_temporaries = []
        self.windows = []
        self.bindings = list(arguments)
        self.local_report = {"mode": "shared", "reason": "", "windows": [], "die1_bytes": 0, "borrowed_die1_bytes": 0}
        total = math.prod(grid)
        self.local_report["program_ranges"] = [[0, (total + 1) // 2], [(total + 1) // 2, total]]
        try:
            self._plan(graph, grid)
        except UnsupportedLocalAccess as error:
            self.windows.clear()
            self.local_report["reason"] = str(error)

    @property
    def temporaries(self):
        # Keep the existing bridge diagnostic separate from local allocations.
        return self.normal.temporaries

    def _plan(self, graph, grid):
        if not self.normal.descriptors:
            raise UnsupportedLocalAccess("local windows require tensor descriptors")
        for (_, ty), value in zip(self.signature.items(), self.arguments):
            if ty.startswith("*") and (value is None or isinstance(value, int)):
                raise UnsupportedLocalAccess("raw or null pointer lacks a local allocation descriptor")
        footprints = access_footprints(graph, self.arguments, grid)
        groups = list(self.normal.groups.values())
        for i, group in enumerate(groups):
            descriptor = group["descriptor"]
            root = descriptor.root
            if (not root.is_contiguous() or root.storage_offset() != 0
                    or root.numel() * root.element_size() != descriptor.root_bytes):
                raise UnsupportedLocalAccess("local windows need a dense zero-offset logical root")
            for other in groups[:i]:
                if any(
                        max(a, c) < min(b, d)
                        for a, b in descriptor.physical_ranges
                        for c, d in other["descriptor"].physical_ranges):
                    raise UnsupportedLocalAccess("local windows require unambiguous alias groups")
            merged = [{"read": set(), "write": set()} for _ in range(2)]
            for position, view in group["positions"]:
                if graph["element_bytes"][position] != view.itemsize:
                    raise UnsupportedLocalAccess("pointer element size disagrees with tensor dtype")
                if view.dtype != root.dtype or view.itemsize != root.element_size():
                    raise UnsupportedLocalAccess("mixed-dtype aliases need a byte-level access proof")
                if view.offset_bytes % view.itemsize:
                    raise UnsupportedLocalAccess("unaligned view offset")
                base = view.offset_bytes // view.itemsize
                for die in range(2):
                    for role in ("read", "write"):
                        merged[die][role].update(v + base for v in footprints[die][position][role])
            for die in range(2):
                used = merged[die]["read"] | merged[die]["write"]
                if used and (min(used) < 0 or max(used) >= root.numel()):
                    raise ValueError("EVAS access lies outside the logical root")
                if merged[die]["write"] & (merged[1 - die]["read"] | merged[1 - die]["write"]):
                    raise ValueError("EVAS local plan has a cross-die read/write dependence")
            used = merged[1]["read"] | merged[1]["write"]
            if not used:
                continue
            segments = _segments(merged[1]["write"])
            if len(segments) > MAX_SCATTER_SEGMENTS:
                raise UnsupportedLocalAccess("writeback fragmentation exceeds the local limit")
            self.windows.append(dict(group=group, begin=min(used), end=max(used) + 1, segments=segments))
        self.local_report["mode"] = "local"

    def _bind_shared(self):
        self.bindings = list(self.normal.bindings)
        for (_, ty), value in zip(self.signature.items(), self.normal.bindings):
            if ty.startswith("*"):
                self.bindings.extend((value, 0, 0))

    def __enter__(self):
        self.normal.__enter__()
        self._bind_shared()
        if self.local_report["mode"] != "local" or not self.windows:
            return self
        runtime = self.normal.runtime
        try:
            runtime.synchronize(self.normal.device)
            local_by_position = {}
            for window in self.windows:
                group, begin, end = window["group"], window["begin"], window["end"]
                descriptor = group["descriptor"]
                source = group.get("temporary", descriptor.root)
                temporary = runtime.borrow_local(descriptor, begin, end, 1)
                borrowed = temporary is not None
                if not borrowed:
                    temporary = runtime.allocate_local(descriptor, end - begin, 1)
                    self.local_temporaries.append(temporary)
                # Retain storage before the first potentially asynchronous copy,
                # including its failure path, until completion is established.
                window.update(source=source, temporary=temporary)
                if not borrowed:
                    runtime.copy(temporary, runtime.flat_view(source, begin, end - begin))
                for position, view in group["positions"]:
                    origin = begin - view.offset_bytes // view.itemsize
                    local_by_position[position] = (BoundPointer(view, int(temporary.data_ptr())), origin)
                self.local_report["borrowed_die1_bytes" if borrowed else "die1_bytes"] += (end -
                                                                                           begin) * descriptor.itemsize
                self.local_report["windows"].append(
                    dict(arguments=[p for p, _ in group["positions"]], begin=begin, end=end,
                         write_segments=window["segments"], affinity=1, borrowed=borrowed))
            extra = len(self.arguments)
            for position, (_, ty) in enumerate(self.signature.items()):
                if not ty.startswith("*"):
                    continue
                if position in local_by_position:
                    pointer, origin = local_by_position[position]
                    self.bindings[extra:extra + 3] = [pointer, 0, origin]
                extra += 3
            runtime.synchronize(self.normal.device)
            return self
        except BaseException:
            self._finish(sys.exc_info())
            raise

    def _finish(self, error):
        if any("temporary" in window for window in self.windows):
            try:
                self.normal.runtime.synchronize(self.normal.device)
            except BaseException:
                if self not in _quarantined:
                    _quarantined.append(self)
                raise
            self.local_temporaries.clear()
            for window in self.windows:
                window.pop("temporary", None)
        try:
            return self.normal.__exit__(*error)
        finally:
            self.bindings = list(self.arguments)

    def __exit__(self, exc_type, exc_value, traceback):
        error = (exc_type, exc_value, traceback)
        try:
            if exc_type is None:
                runtime = self.normal.runtime
                if any("temporary" in window for window in self.windows):
                    runtime.synchronize(self.normal.device)
                for window in self.windows:
                    if "temporary" not in window:
                        continue
                    for begin, end in window["segments"]:
                        destination = runtime.flat_view(window["source"], begin, end - begin)
                        source = runtime.flat_view(window["temporary"], begin - window["begin"], end - begin)
                        runtime.copy(destination, source)
        except BaseException:
            self._finish(sys.exc_info())
            raise
        return self._finish(error)


def make_buffer_plan(signature, arguments, contract, local_access=None, grid=None, runtime=None):
    if local_access is not None and local_access.get("abi"):
        return LocalBufferPlan(signature, arguments, contract, local_access, grid, runtime)
    return AutoBufferPlan(signature, arguments, contract, runtime=runtime)
