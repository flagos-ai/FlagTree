# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Owner-contiguous readonly K walks for proven BI-V150 packed weights."""

from triton.flagmega.codegen.readonly_layout import (
    READONLY_DEVICE_LAYOUT_SCHEMA, readonly_roots, owner_major_weight_records,
)
from triton.flagmega.codegen.triton.physical_access import emit_buffer_pointer
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import verify_buffer_plan


_weight_layout_records = owner_major_weight_records


def select_readonly_layouts(module, calls):
    """Prove all readers before changing any byte range or call address.

    Flat SME proves complete rows, packing, alignment and matched origins.
    Each K group must also contain consecutive N owners, without a K split.
    GLU gate/up share one K stride, so both weights require the device layout.
    """
    plan = verify_buffer_plan(module)
    layouts, selected = {}, []
    for call in calls:
        if "corex_flat_sme_down" in call:
            sme = call["corex_flat_sme_down"]
            roles = (("rhs", "weight"),)
        elif "corex_flat_sme_glu" in call:
            sme = call["corex_flat_sme_glu"]
            roles = (("gate_weight", "gate_weight"), ("up_weight", "up_weight"))
        else:
            continue
        n, k = int(call["local_n_capacity"]), int(call["local_k_capacity"])
        candidates = []
        for formal, field in roles:
            argument = next((item for item in call["inputs"] if item["formal"] == formal), None)
            if argument is None or len(argument.get("buffers", ())) != 1:
                break
            binding = argument["buffers"][0]
            records = _weight_layout_records(plan, binding, n, k, int(sme["weight_k_group_stride"]))
            if records is None:
                break
            candidates.append((binding, field, records))
        if len(candidates) != len(roles):
            continue
        # Both GLU operands are proved before selecting either byte range.
        for binding, field, records in candidates:
            for record in records:
                prior = layouts.setdefault(record["offset"], record)
                if prior != record:
                    raise CodegenError("Conflicting COREX readonly layouts share one byte range.")
            selected.append((call, binding, sme, field, n * 16, n * k))
    selected_bindings = {(id(call), id(binding)) for call, binding, *_ in selected}
    for call in calls:
        for argument in call.get("inputs", ()):
            for binding in argument.get("buffers", ()):
                roots = readonly_roots(plan, plan.buffer_map[binding["actual"]])
                if any(root.offset in layouts for root in roots) and (
                        (id(call), id(binding)) not in selected_bindings):
                    raise CodegenError("A COREX readonly layout has a consumer without its device-address contract.")
    # Validation must finish before modifying any call's address expressions.
    for call, binding, sme, field, group, owner_size in selected:
        abi = binding["abi"]
        pointer = emit_buffer_pointer(abi, binding["runtime_argument"])
        sme[field] = f"({pointer} + ({abi['owner_index_expression']}) * {owner_size})"
        sme["weight_k_group_stride"] = group
    return {"schema": READONLY_DEVICE_LAYOUT_SCHEMA,
            "buffers": sorted(layouts.values(), key=lambda item: item["offset"])}
