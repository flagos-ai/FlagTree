# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Partition independent rows/copy tiles of shared broadcast buffers by CTA."""
import hashlib
import json
from pathlib import Path

from jinja2 import ChoiceLoader, DictLoader
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry

original_prepare = TritonTemplateRegistry.prepare


def _shared_broadcast(call):
    if call["participation_active"] != "True":
        return False
    for group in ("inputs", "outputs"):
        for argument in call[group]:
            for buffer in argument["buffers"]:
                abi = buffer["abi"]
                if (abi.get("memory_sharing_scope") != "chip"
                    or abi.get("coordinate_space") != "canonical_global"):
                    return False
                for key in ("distributed_type", "distributed_backing_type"):
                    distributed = abi.get(key)
                    if distributed is not None and (
                        distributed.get("partial") is not None
                        or distributed.get("exclusive") is not None
                        or any(p.get("kind") != "broadcast" for p in distributed.get("axis_policies", ()))
                    ):
                        return False
    return True


def prepare(self, name, context):
    if name != "module.py.jinja":
        return original_prepare(self, name, context)
    selected = []
    for call in context["render_calls"]:
        eligible = call["family"] == "reduce_sum" or (
            call["family"] == "distributed_boxing" and call["variant"] == "tensor_load"
            and all(not leaf.get("source_is_scalar", False) and not leaf["reduction"]
                    and "remote_source" not in leaf for leaf in call["leaves"])
        )
        call["shared_work_partition"] = eligible and _shared_broadcast(call)
        if call["shared_work_partition"]:
            selected.append(call["call"])
    # Existing overrides prepare their templates first. Read their
    # resulting sources so the small-partial reduction layout remains intact.
    original_prepare(self, name, context)
    sources = {}
    key = "kernels/distributed_boxing/_function.py.jinja"
    source = self.environment.loader.get_source(self.environment, key)[0]
    marker = "    for boxing_start in tl.range(0, {{ leaf.capacity }}, {{ leaf.tile }}):"
    assert source.count(marker) == 1
    source = source.replace(marker, '''{% if call.shared_work_partition %}
    for boxing_start in tl.range(tl.program_id(0) * {{ leaf.tile }}, {{ leaf.capacity }}, tl.num_programs(0) * {{ leaf.tile }}):
{% else %}
''' + marker + '\n{% endif %}')
    source = source.replace('{{ leaf.writer_active }}', '{{ "True" if call.shared_work_partition else leaf.writer_active }}')
    sources[key] = source
    key = "kernels/reduce_sum/local.py.jinja"
    source = self.environment.loader.get_source(self.environment, key)[0]
    marker = "    for _fm_row in tl.range(0, {{ call.rows }}):"
    assert source.count(marker) == 1
    source = source.replace(marker, '''{% if call.shared_work_partition %}
    for _fm_row in tl.range(tl.program_id(0), {{ call.rows }}, tl.num_programs(0)):
{% else %}
''' + marker + '\n{% endif %}')
    source = source.replace('{{ call.writer_active }}', '{{ "True" if call.shared_work_partition else call.writer_active }}')
    sources[key] = source
    self.environment.loader = ChoiceLoader([DictLoader(sources), self.environment.loader])
    print(json.dumps({"trial": "shared broadcast copy tiles and reduction rows assigned once across CTAs",
                      "calls": selected, "override_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}), flush=True)


TritonTemplateRegistry.prepare = prepare
