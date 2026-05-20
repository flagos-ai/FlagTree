"""
MLIR built-in passes with typed Python API.

This module provides type-safe wrapper functions for commonly used
MLIR framework passes (debugging, timing, optimization, etc.).
"""

__all__ = [
    'add_print_ir_after_all',
    'add_print_ir_before',
    'add_disable_threading',
    'add_print_ir_module_scope',
    'add_timing',
    'add_timing_display',
    'add_canonicalize',
    'add_cse',
    'add_loop_invariant_code_motion',
    'add_print_op_generic',
]


def add_print_ir_after_all(pipeline):
    """
    Print IR after each pass for debugging.

    Enables verbose output showing how the IR changes after every pass.
    Useful for debugging pass interactions.
    """
    pipeline.add_pass('mlir-print-ir-after-all')


def add_print_ir_before(pipeline, pass_name: str):
    """
    Print IR before a specific pass.

    Args:
        pass_name: The name of the pass to print IR before.
                  Example: 'tle-to-triton-gcu'
    """
    pipeline.add_pass('mlir-print-ir-before', pass_name)


def add_disable_threading(pipeline):
    """
    Disable multi-threaded pass execution.

    Forces single-threaded execution of passes, which can make debugging
    easier by ensuring deterministic ordering of diagnostics.
    """
    pipeline.add_pass('mlir-disable-threading')


def add_print_ir_module_scope(pipeline):
    """
    Print full module IR instead of per-operation IR.

    When combined with print-ir-after-all, shows the entire module
    instead of individual operations for better context.
    """
    pipeline.add_pass('mlir-print-ir-module-scope')


def add_timing(pipeline):
    """
    Enable timing statistics for passes.

    Collects and reports execution time for each pass in the pipeline.
    """
    pipeline.add_pass('mlir-timing')


def add_timing_display(pipeline, display_mode: str = 'list'):
    """
    Configure timing statistics display format.

    Args:
        display_mode: Display format, either 'list' or 'tree'.
                     'list' shows flat timing info, 'tree' shows hierarchical.
    """
    if display_mode not in ('list', 'tree'):
        raise ValueError(f"display_mode must be 'list' or 'tree', got '{display_mode}'")
    pipeline.add_pass('mlir-timing-display', display_mode)


def add_canonicalize(pipeline):
    """
    Canonicalize operations.

    Applies canonicalization patterns to simplify and normalize IR.
    This includes constant folding, operation simplification, and
    pattern-based rewrites that put operations into canonical forms.
    """
    pipeline.add_pass('canonicalize')


def add_cse(pipeline):
    """
    Common Subexpression Elimination (CSE).

    Eliminates redundant computations by identifying and removing
    duplicate operations that compute the same value.
    """
    pipeline.add_pass('cse')


def add_loop_invariant_code_motion(pipeline):
    """
    Loop-Invariant Code Motion (LICM).

    Moves loop-invariant operations outside of loops to reduce
    redundant computation. Operations that don't depend on loop
    variables are hoisted out of the loop body.
    """
    pipeline.add_pass('loop-invariant-code-motion')


def add_print_op_generic(pipeline):
    """
    Print operations in generic format.

    Outputs IR in a generic textual format that shows all operation
    details explicitly. Useful for debugging and understanding the
    exact structure of operations.
    """
    pipeline.add_pass('mlir-print-op-generic')
