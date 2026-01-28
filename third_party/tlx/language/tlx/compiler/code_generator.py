# third_party/tlx/codegen/async.py

import ast
from typing import List
import triton.language.extra.tlx as tlx  # Make sure async_task(s) are exposed via tlx.__init__.py
from contextlib import contextmanager

# TLX allows users to specify the replicate number when definine
# a non-default partition region. We use a stack to keep track of
# replica_id of the region being compiled.
region_replica_id_stack: List[int] = []
sub_region_has_exception = False


@contextmanager
def tlx_enter_sub_region():
    global region_replica_id_stack
    global sub_region_has_exception
    replica_id_stack_backup = region_replica_id_stack.copy()
    try:
        yield
    except Exception as e:
        sub_region_has_exception = True
        raise e
    finally:
        if not sub_region_has_exception:
            assert region_replica_id_stack == replica_id_stack_backup, "region_replica_id_stack is not restored"


def _is_async_task(self, node) -> bool:
    if isinstance(node, ast.With):
        context = node.items[0].context_expr
        if isinstance(context, ast.Call):
            withitemClass = self.visit(context.func)
            if withitemClass == tlx.async_task:
                return True
    return False


def _get_async_task(self, node):
    context = node.items[0].context_expr
    # Parse positional args (e.g., [0])
    args = [self.visit(arg) for arg in context.args]
    # Extract keyword arguments as (key, value AST nodes)
    kwargs = {kw.arg: self.visit(kw.value) for kw in context.keywords}
    with tlx.async_task(*args, _builder=self.builder, **kwargs) as task:
        return task


def visit_withAsyncTask(self, node):
    # Visit the body of the `with` region
    self.visit_compound_statement(node.body)


def _validate_warp_group_start_ids(
    start_ids: List[int],
    num_warps: List[int],
    task_replicates: List[int],
    default_num_warps: int,
) -> None:
    """Validate that warp group start IDs are valid and non-overlapping across different tasks.

    Args:
        start_ids: List of warp group start IDs for each task (before replica expansion).
        num_warps: List of number of warps for each task (before replica expansion).
        task_replicates: List of replica counts for each task.
        default_num_warps: Number of warps used by the default region (starts at warp 0).

    Raises:
        AssertionError: If validation fails.
    """
    assert len(start_ids) == len(num_warps) == len(task_replicates), (
        f"start_ids length ({len(start_ids)}), num_warps length ({len(num_warps)}), "
        f"and task_replicates length ({len(task_replicates)}) must all match")

    # Check that all start IDs are non-negative
    for i, start_id in enumerate(start_ids):
        assert start_id >= 0, f"warp_group_start_id[{i}] = {start_id} must be non-negative"

    # Check for overlapping warp ranges between different tasks
    # Build list of (start, end) ranges for each task, considering replicas
    # Each task uses num_warps * replicate warps starting at start_id
    ranges = [(start_ids[i], start_ids[i] + num_warps[i] * task_replicates[i]) for i in range(len(start_ids))]

    # Default region uses warps [0, default_num_warps)
    default_range = (0, default_num_warps)

    # Check that no non-default task overlaps with the default region
    for i, (start_i, end_i) in enumerate(ranges):
        # Two ranges [a, b) and [c, d) overlap if a < d and c < b
        if start_i < default_range[1] and default_range[0] < end_i:
            assert False, (f"Overlapping warp ranges: task {i} uses warps [{start_i}, {end_i}) "
                           f"which overlaps with default region warps [{default_range[0]}, {default_range[1]})")

    # Check all pairs of non-default tasks for overlap
    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            start_i, end_i = ranges[i]
            start_j, end_j = ranges[j]
            # Two ranges [a, b) and [c, d) overlap if a < d and c < b
            if start_i < end_j and start_j < end_i:
                assert False, (f"Overlapping warp ranges: task {i} uses warps [{start_i}, {end_i}) "
                               f"and task {j} uses warps [{start_j}, {end_j})")


@tlx_enter_sub_region()
def visit_withAsyncTasks(self, node):
    from triton.compiler.code_generator import enter_sub_region, _is_list_like, _is_constexpr

    with enter_sub_region(self) as sr:
        liveins, _ = sr
        ip, last_loc = self._get_insertion_point_and_loc()

        def _flatten_value_handles(val):
            handles = []
            # Prefer the generic flatten hook to support multi-result values (e.g. tensor descriptors)
            if hasattr(val, "_flatten_ir"):
                val._flatten_ir(handles)
            else:
                handles.append(val.handle)
            return handles

        stmts = node.body
        # Ensure that stmts is iterable
        if not _is_list_like(stmts):
            stmts = [stmts]

        # dry visit async task body to count the number of sub tasks
        with tlx_enter_sub_region():
            block = self.builder.create_block()
            self.builder.set_insertion_point_to_start(block)
            taskNumWarps = []
            taskNumRegs = []
            taskReplica = []
            taskWarpGroupStartIds = []

            # Per-task data for validation (before replica expansion)
            perTaskNumWarps = []
            perTaskStartIds = []
            perTaskReplicates = []

            global region_replica_id_stack
            region_replica_id_stack.append(-1)  # dummy placeholder

            num_default = 0
            for stmt in stmts:
                assert _is_async_task(self, stmt)
                task = _get_async_task(self, stmt)
                assert task.is_explict
                assert task.replicate is not None, "Replicate must be non-None task"
                if task.is_default:
                    num_default += 1
                    if task.replicate > 1:
                        taskReplica.append(task.replicate - 1)
                        taskNumWarps.extend([self.builder.options.num_warps] * (task.replicate - 1))
                        if task.num_regs:
                            taskNumRegs.extend([task.num_regs] * (task.replicate - 1))
                        if task.warp_group_start_id is not None:
                            taskWarpGroupStartIds.extend([task.warp_group_start_id] * (task.replicate - 1))
                else:
                    taskReplica.append(task.replicate)
                    taskNumWarps.extend([task.num_warps] * task.replicate)
                    if task.num_regs:
                        taskNumRegs.extend([task.num_regs] * task.replicate)
                    if task.warp_group_start_id is not None:
                        # Each replica gets its own start ID, incrementing by num_warps
                        for r in range(task.replicate):
                            taskWarpGroupStartIds.append(task.warp_group_start_id + r * task.num_warps)
                        # Collect per-task data for validation
                        perTaskNumWarps.append(task.num_warps)
                        perTaskStartIds.append(task.warp_group_start_id)
                        perTaskReplicates.append(task.replicate)

            region_replica_id_stack.pop()  # revert adding dummy placeholder

        assert num_default == 1, "Default task must be one and only one"
        block.erase()

        assert len(taskNumRegs) in [0, len(taskNumWarps)
                                    ], ("Registers are set for either ALL or NONE of non-default tasks")
        assert len(taskWarpGroupStartIds) in [
            0, len(taskNumWarps)
        ], ("warp_group_start_id must be set for either ALL or NONE of non-default tasks")

        # Validate warp_group_start_ids
        if len(perTaskStartIds) > 0:
            _validate_warp_group_start_ids(
                perTaskStartIds,
                perTaskNumWarps,
                perTaskReplicates,
                self.builder.options.num_warps,
            )

        # Create tasks body block
        self._set_insertion_point_and_loc(ip, last_loc)
        ws_op = self.builder.create_warp_specialize_op(
            taskNumWarps,
            taskNumRegs if len(taskNumRegs) > 0 else None,
            sum(taskReplica),
            taskWarpGroupStartIds if len(taskWarpGroupStartIds) > 0 else None,
        )

        # dry visit async task body to calculate captures
        index = 0
        for stmt in stmts:
            assert _is_async_task(self, stmt)
            task = _get_async_task(self, stmt)
            assert task.is_explict
            task_replicate = (task.replicate - 1) if task.is_default else task.replicate
            if task_replicate > 0:
                task_body = ws_op.get_partition_region(index)
                block = self.builder.create_block_with_parent(task_body, [])
                # Only need to calculate captures for the first replica.
                region_replica_id_stack.append(0)
                self.builder.set_insertion_point_to_start(block)
                with enter_sub_region(self):
                    self.visit(stmt)
                region_replica_id_stack.pop()
                index += task_replicate
                block.erase()

        # Add captures
        captures = sorted(v for v in (liveins.keys() & self.used_vars) if not _is_constexpr(liveins[v]))
        for name in captures:
            val = liveins[name]
            if getattr(val, "__triton_aggregate__", False):
                for field in val.type.fields:
                    v = getattr(val, field[0])
                    for h in _flatten_value_handles(v):
                        ws_op.append_operand(h)
            else:
                for h in _flatten_value_handles(val):
                    ws_op.append_operand(h)

        # real codegen
        index = 0
        for stmt in stmts:
            assert _is_async_task(self, stmt)
            task = _get_async_task(self, stmt)
            if task.is_default:
                region_replica_id_stack.append(0)
                task_body = ws_op.get_default_region()

                block = self.builder.create_block_with_parent(task_body, [])
                self.builder.set_insertion_point_to_start(block)
                with enter_sub_region(self):
                    self.visit(stmt)

                self.builder.create_warp_yield_op()
                region_replica_id_stack.pop()

            replicate_start = 1 if task.is_default else 0

            for i in range(replicate_start, task.replicate):
                region_replica_id_stack.append(i)

                task_body = ws_op.get_partition_region(index)
                index += 1

                block = self.builder.create_block_with_parent(task_body, [])
                self.builder.set_insertion_point_to_start(block)
                with enter_sub_region(self):
                    self.visit(stmt)

                for name in captures:
                    val = liveins[name]
                    if getattr(val, "__triton_aggregate__", False):
                        for field in val.type.fields:
                            v = getattr(val, field[0])
                            for h in _flatten_value_handles(v):
                                arg = task_body.add_argument(h.get_type())
                                block.replace_use_in_block_with(h, arg)
                    else:
                        for h in _flatten_value_handles(val):
                            arg = task_body.add_argument(h.get_type())
                            block.replace_use_in_block_with(h, arg)

                self.builder.create_warp_return_op()
                region_replica_id_stack.pop()
