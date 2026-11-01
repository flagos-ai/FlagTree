# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Bind per-entry launch resources to the verified source and TIR request."""

from collections.abc import Mapping
from pathlib import PurePath

from triton.flagmega.codegen.triton.kernel_programs import (
    KERNEL_PROGRAM_DAG_SCHEMA, KERNEL_PROGRAM_SCHEMA, KERNEL_PROGRAM_DAG_COMPLETION_SCHEMA,
    KERNEL_PROGRAM_MESH_SCHEMA,
    KERNEL_PROGRAM_MESH_DAG_SCHEMA,
)
from triton.flagmega.codegen.triton.package_plan import plain_package_value
from triton.flagmega.codegen.triton.function_programs import KERNEL_PROGRAM_FUNCTION_SCHEMA
from triton.flagmega.errors import ArtifactError


def _verify_source_program_literals(codegen, raw, kernels):
    from triton.flagmega.artifacts.program_sources import program_source_records
    from triton.flagmega.codegen.triton.program_sources import PROGRAM_SOURCE_LAYOUT

    records = program_source_records(codegen)
    scoped = codegen.get("program_source_layout") == PROGRAM_SOURCE_LAYOUT
    exports = {name: record for record in records for name in record["symbols"]}
    aliases = codegen.get("program_entry_aliases", {})

    def record_for(symbol):
        while symbol in aliases:
            symbol = aliases[symbol]
        return exports[symbol]

    by_source = {}
    if scoped:
        for spec in raw:
            path = record_for(spec["symbol"])["source"]
            by_source.setdefault(path, []).append(spec)

    def matches(filename, relative):
        parts = PurePath(relative).parts
        return bool(parts) and PurePath(filename).parts[-len(parts):] == parts

    for spec, kernel in zip(raw, kernels, strict=True):
        function = getattr(kernel, "fn", kernel)
        expected = raw
        if scoped:
            filename = getattr(getattr(function, "__code__", None), "co_filename", "")
            if not matches(filename, codegen["source"]):
                record = record_for(spec["symbol"])
                if not matches(filename, record["source"]):
                    raise ArtifactError("Kernel program does not resolve its declared source file.")
                expected = by_source[record["source"]]
        literal = getattr(function, "__globals__", {}).get("FLAGMEGA_KERNEL_PROGRAMS")
        if plain_package_value(literal) != plain_package_value(expected):
            raise ArtifactError("Kernel program launch resources differ from the verified source.")


def verified_kernel_programs(codegen, kernels, module):
    from triton.flagmega.codegen.triton.structured_programs import KERNEL_PROGRAM_STRUCTURED_SCHEMA
    if codegen.get('kernel_program_schema') == KERNEL_PROGRAM_STRUCTURED_SCHEMA:
        return _verified_structured_programs(codegen, kernels, module)
    mesh_dag = codegen.get('kernel_program_schema') == KERNEL_PROGRAM_MESH_DAG_SCHEMA
    mesh_programs = mesh_dag or codegen.get('kernel_program_schema') == KERNEL_PROGRAM_MESH_SCHEMA
    shared = codegen.get("kernel_program_schema") == KERNEL_PROGRAM_FUNCTION_SCHEMA
    completion = shared or mesh_dag or codegen.get("kernel_program_schema") == KERNEL_PROGRAM_DAG_COMPLETION_SCHEMA
    dag = completion or codegen.get("kernel_program_schema") == KERNEL_PROGRAM_DAG_SCHEMA
    if codegen.get("kernel_program_schema") not in {KERNEL_PROGRAM_SCHEMA, KERNEL_PROGRAM_DAG_SCHEMA, KERNEL_PROGRAM_DAG_COMPLETION_SCHEMA, KERNEL_PROGRAM_FUNCTION_SCHEMA, KERNEL_PROGRAM_MESH_SCHEMA, KERNEL_PROGRAM_MESH_DAG_SCHEMA}:
        raise ArtifactError("Unsupported kernel program resource schema.")
    raw = codegen.get("kernels")
    requests = module.metadata.get("kernel_program_partition")
    if (
        not isinstance(raw, (tuple, list))
        or not raw
        or len(raw) != len(kernels)
        or not isinstance(requests, (tuple, list))
        or len(requests) != len(raw)
    ):
        raise ArtifactError("Kernel program entries differ from the source or TIR partition.")
    _verify_source_program_literals(codegen, raw, kernels)
    result = []
    if mesh_programs:
        from triton.flagmega.codegen.triton.tir_package import describe_tir_package
        expected = describe_tir_package(module)
        if (expected['kernel_program_schema'] != codegen['kernel_program_schema']
                or plain_package_value(expected['kernels']) != plain_package_value(raw)):
            raise ArtifactError('Execution mesh launch resources differ from the typed program plan.')
    from triton.flagmega.ir.program_order import program_stream_indices
    streams = program_stream_indices(requests)
    if dag != (streams is not None):
        raise ArtifactError("Kernel program dependency schema differs from its TIR request.")
    if shared:
        # Rebuild from the typed physical call ABI and descriptor resources.
        # A matching manifest/source literal alone cannot authorize a forged
        # pointer frame, offset record, scalar or result binding.
        from triton.flagmega.codegen.triton.tir_package import describe_tir_package
        expected = describe_tir_package(module)
        if (expected["kernel_program_schema"] != KERNEL_PROGRAM_FUNCTION_SCHEMA
                or plain_package_value(expected["kernels"]) != plain_package_value(raw)):
            raise ArtifactError("Shared GPU entry bindings differ from the typed physical call ABI.")
        entries = {}
        for spec, kernel in zip(raw, kernels, strict=True):
            entry = spec["entry_symbol"]
            if getattr(getattr(kernel, "fn", kernel), "__name__", None) != entry:
                raise ArtifactError("Shared GPU entry does not resolve its verified definition.")
            if entries.setdefault(entry, kernel) is not kernel:
                raise ArtifactError("Repeated GPU calls must resolve the same shared entry definition.")
    if dag:
        from triton.flagmega.codegen.triton.program_dependencies import verify_program_dependencies
        external_writes, device_completion = verify_program_dependencies(
            module, requests, codegen["runtime_binding"], with_device_completion=True)
    for spec, request in zip(raw, requests, strict=True):
        if (
            not isinstance(spec, Mapping)
            or set(spec)
            != ({"symbol", "grid", "num_warps", "dynamic_argument_indices", "requires_grid_wide_sync", "calls"}
                | ({"depends_on", "stream_index", "external_writes"} if dag else set())
                | ({"device_completion"} if completion else set())
                | ({"entry_function", "entry_symbol", "argument_bindings"} if shared else set())
                | ({"task_owner_partition"} if request.get("tasks_per_owner", 1) > 1 else set())
                | ({'execution_mesh', 'program_kind', 'data_placement'} if mesh_programs else set())
                | ({'launch_recipe'} if mesh_programs and 'launch_recipe' in request else set())
                | ({'maxnreg'} if 'maxnreg' in request else set()))
            or spec.get("symbol") != request.get("symbol")
            or spec.get("num_warps") != request.get("num_warps")
            or spec.get('maxnreg') != request.get('maxnreg')
            or plain_package_value(spec.get("calls")) != plain_package_value(request.get("calls"))
        ):
            raise ArtifactError("Kernel program resources differ from the typed TIR request.")
        if request.get("tasks_per_owner", 1) > 1:
            from triton.flagmega.codegen.triton.tir_package import describe_tir_package
            if not shared:
                expected = describe_tir_package(module)
            rebuilt = next(v for v in expected["kernels"] if v["symbol"] == spec["symbol"])
            if (plain_package_value(spec["task_owner_partition"]) != plain_package_value(rebuilt["task_owner_partition"])
                    or plain_package_value(spec["grid"]) != plain_package_value(rebuilt["grid"])):
                raise ArtifactError("Kernel program task ownership differs from its typed callee write partition.")
        if dag and (
            plain_package_value(spec.get("depends_on")) != plain_package_value(request["depends_on"])
            or type(spec.get("stream_index")) is not int
            or spec["stream_index"] != streams[spec["symbol"]]
            or not isinstance(spec.get("external_writes"), (tuple, list))
            or any(type(v) is not int or v < 0 or v >= len(codegen["runtime_binding"]["arguments"])
                   for v in spec["external_writes"])
            or tuple(spec["external_writes"]) != external_writes[spec["symbol"]]
        ):
            raise ArtifactError("Kernel program dependencies differ from the typed TIR request.")
        if completion and (type(spec.get("device_completion")) is not bool
                           or spec["device_completion"] != device_completion[spec["symbol"]]
                           or spec["device_completion"] and spec["requires_grid_wide_sync"] is not True):
            raise ArtifactError("Kernel program device completion differs from its typed peer effects.")
        warps = spec["num_warps"]
        grid = spec["grid"]
        dynamic = spec["dynamic_argument_indices"]
        if (
            type(warps) is not int
            or warps not in (1, 2, 4, 8, 16, 32)
            or not isinstance(grid, (tuple, list))
            or len(grid) != 3
            or any(type(v) is not int or v <= 0 for v in grid)
            or not isinstance(dynamic, (tuple, list))
            or any(type(v) is not int or v < 0 for v in dynamic)
            or not shared and tuple(dynamic) != tuple(codegen["dynamic_argument_indices"])
            or type(spec["requires_grid_wide_sync"]) is not bool
        ):
            raise ArtifactError("Invalid kernel program physical launch resources.")
        result.append({**spec, "grid": tuple(grid), "dynamic_argument_indices": tuple(dynamic)})
    return tuple(result)


def _verified_structured_programs(codegen, kernels, module):
    """Rebuild invocations and every physical binding from authoritative TIR."""
    from triton.flagmega.codegen.triton.tir_package import describe_tir_package
    from triton.flagmega.codegen.triton.structured_programs import KERNEL_PROGRAM_STRUCTURED_SCHEMA
    expected = describe_tir_package(module)
    raw = codegen.get('kernels')
    execution = codegen.get('program_execution')
    if (expected.get('kernel_program_schema') != KERNEL_PROGRAM_STRUCTURED_SCHEMA
            or plain_package_value(expected['kernels']) != plain_package_value(raw)
            or plain_package_value(expected['program_execution']) != plain_package_value(execution)
            or not isinstance(raw, (tuple, list)) or len(raw) != len(kernels)):
        raise ArtifactError('Structured invocations or physical bindings differ from the typed TIR plan.')
    _verify_source_program_literals(codegen, raw, kernels)
    entries = {}
    for spec, kernel in zip(raw, kernels, strict=True):
        function = getattr(kernel, 'fn', kernel)
        namespace = getattr(function, '__globals__', {})
        if plain_package_value(namespace.get('FLAGMEGA_PROGRAM_EXECUTION')) != plain_package_value(execution):
            raise ArtifactError('Structured invocation resources differ from the verified source.')
        entry = spec['entry_symbol']
        if getattr(function, '__name__', None) != entry or entries.setdefault(entry, kernel) is not kernel:
            raise ArtifactError('Structured calls must resolve their shared physical entry definition.')
    return tuple({**spec, 'grid': tuple(spec['grid']),
                  'dynamic_argument_indices': tuple(spec['dynamic_argument_indices'])} for spec in raw)
