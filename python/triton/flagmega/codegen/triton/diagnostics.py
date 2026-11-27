# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Schedule and generated-source diagnostic producers."""

from __future__ import annotations

import json

from triton.flagmega.diagnostics import DumpFlags, Dumper
from triton.flagmega.ir import IRModule, kernel_dispatch_of
from triton.flagmega.codegen.triton.function_schedule import (
    describe_function_schedule,
)
from triton.flagmega.codegen.triton.runtime_binding import (
    describe_function_runtime_binding,
)
from triton.flagmega.codegen.triton.package_plan import plain_package_value


def dump_schedules(module: IRModule, dumper: Dumper | None) -> None:
    if dumper is None or not dumper.is_enabled(DumpFlags.SCHEDULE):
        return
    root = dumper.create_sub_dumper("Schedule")
    for function in module.functions:
        with root.open_artifact(
            f"{function.name}.call_schedule.json",
            category=DumpFlags.SCHEDULE,
            kind="tir.function-schedule/v2",
            producer="triton-function-scheduler",
            source_semantic_hash=module.semantic_hash,
            encoding="utf-8",
        ) as stream:
            json.dump(
                plain_package_value(describe_function_schedule(
                    module,
                    function_name=function.name,
                )),
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
        with root.open_artifact(
            f"{function.name}.runtime_binding.json",
            category=DumpFlags.SCHEDULE,
            kind="tir.runtime-binding/v1",
            producer="triton-runtime-binder",
            source_semantic_hash=module.semantic_hash,
            encoding="utf-8",
        ) as stream:
            json.dump(
                plain_package_value(describe_function_runtime_binding(
                    module,
                    function_name=function.name,
                )),
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
    for function in (*module.prim_functions, *module.kernel_definitions):
        dispatch = kernel_dispatch_of(function)
        if dispatch is None:
            continue
        with root.open_artifact(
            f"{function.name}.schedule.json",
            category=DumpFlags.SCHEDULE,
            kind="tir.schedule/v1",
            producer="triton-package-renderer",
            source_semantic_hash=module.semantic_hash,
            encoding="utf-8",
        ) as stream:
            dispatch_data = dispatch.to_data()
            # ``tir.schedule/v1`` predates first-class semantic/microkernel
            # separation. Keep its resolved convenience fields for external
            # artifact readers while the nested canonical fields preserve the
            # editable IR distinction.
            encoded_microkernel = dispatch_data.get("microkernel") or {}
            resolved_parameters = {
                **dict(encoded_microkernel.get("parameters", {})),
                **dict(dispatch_data.get("semantic_parameters", {})),
            }
            resolved_facts = {
                **dict(encoded_microkernel.get("facts", {})),
                **dict(dispatch_data.get("semantic_facts", {})),
            }
            if encoded_microkernel.get("requires"):
                resolved_facts["requires"] = encoded_microkernel["requires"]
            dispatch_data.update({
                "candidate": dispatch.candidate,
                "parameters": resolved_parameters,
                "facts": resolved_facts,
            })
            json.dump(plain_package_value({
                "schema": "flagmega.tir-schedule/v1",
                "function": function.name,
                "parameters": [value.to_data() for value in function.parameters],
                "return_type": function.runtime_return_type.to_data(),
                "dispatch": dispatch_data,
            }), stream, indent=2, sort_keys=True)
            stream.write("\n")


def dump_codegen(
    module: IRModule,
    descriptor: dict[str, object],
    source: str,
    dumper: Dumper | None,
) -> None:
    if dumper is None or not dumper.is_enabled(DumpFlags.CODEGEN):
        return
    root = dumper.create_sub_dumper("CodeGen")
    with root.open_artifact(
        "package.json",
        category=DumpFlags.CODEGEN,
        kind="triton.package-descriptor/v1",
        producer="triton-package-renderer",
        source_semantic_hash=module.semantic_hash,
        encoding="utf-8",
    ) as stream:
        json.dump(
            plain_package_value(descriptor),
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    with root.open_artifact(
        "generated_kernels.py",
        category=DumpFlags.CODEGEN,
        kind="triton.python-source/v1",
        producer="triton-package-renderer",
        source_semantic_hash=module.semantic_hash,
        encoding="utf-8",
    ) as stream:
        stream.write(source)


__all__ = ["dump_codegen", "dump_schedules"]
