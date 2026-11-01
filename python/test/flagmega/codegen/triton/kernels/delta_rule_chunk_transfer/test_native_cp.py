# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""CP state transfer preserves native chunk composition and zero-spill launch."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.passes.functions import function_nodes
from triton.flagmega.runtime import load


@pytest.mark.parametrize("tokens", [32, 128, 1024, 2048])
def test_chunk_transfers_and_output_match_native(tmp_path, tokens):
    torch = pytest.importorskip("torch")
    native_python = os.environ.get("FLAGMEGA_NATIVE_CP_PYTHON")
    if not native_python or not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90 and FLAGMEGA_NATIVE_CP_PYTHON with the pinned FlashInfer native CP implementation")
    reference = tmp_path / "native.pt"
    subprocess.run([native_python, str(Path(__file__).with_name("native_reference.py")), str(tokens), str(reference)], check=True)
    native = torch.load(reference, weights_only=True)
    mesh = fm.Placement((2, 2), "yx", "bb")
    b, head = fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 1)
    state_type = fm.RefType("State", (("matrix", fm.tensor_type("float32", (2, 128, 128))),))
    names = ("query", "key", "value", "coefficients", "alpha")

    class Graph(fm.Module):
        def forward(self):
            values = {name: self.input(name, fm.tensor_type(str(native[name].dtype).removeprefix("torch."), native[name].shape), id=name)
                      for name in names}
            state = self.input("state", state_type, id="state")
            params = (*values.values(), state)
            values = {name: fm.F.distributed.force_boxing(value, fm.DistributedType(value.type, (b, head, *(b for _ in range(value.type.rank - 2))), mesh))
                      for name, value in values.items()}
            prefix = fm.F.nn.delta_rule_log_prefix(values["alpha"], name="prefix")
            args = (values["key"], values["value"], values["coefficients"], prefix)
            transfer = fm.F.nn.delta_rule_chunk_transfer(*args, matrix_kind="transfer", name="transfer")
            local = fm.F.nn.delta_rule_chunk_transfer(*args, matrix_kind="state", name="local_state")
            fixed = fm.F.nn.delta_rule_chunk_fixup(transfer, local, state, name="fixed_state")
            update = fm.F.nn.delta_rule_chunked_block_update(values["query"], *args, state, fixed, name="update")
            output = fm.F.tensors.get_item(update, 0)
            results = tuple(fm.F.distributed.force_boxing(value, value.type.tensor) for value in (transfer, local, fixed, output))
            self.function("main", params, (*results, fm.F.tensors.get_item(update, 1)))

    source = Graph(dialect="distributed", stage="frozen_constants", entry="main",
                   metadata={"auto_distribution": {"placement": mesh.to_data()}}).build()
    compiler = Compiler()
    selected = compiler.compile(source, stop_after="select-microkernels").module
    calls = [n.id for n in function_nodes(selected, selected.function_map[selected.entry]) if n.op in {"tir.call", "tir.kernel"}]
    programs = tuple(dict(symbol=f"stage_{i}", calls=(call,), num_warps=8,
                          depends_on=(f"stage_{i-1}",) if i else ()) for i, call in enumerate(calls))
    selected = fm.verify_module(replace(selected, metadata={**selected.metadata, "kernel_program_partition": programs}))
    compiled = compiler.compile(selected).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    buffers = {}
    for arg in runtime.external_arguments:
        buffer = runtime.buffer_plan.buffer_map[arg["buffer"]]
        value = native.get(arg["value"]) if arg["value"] in names else None
        if value is not None and arg["role"] == "parameter":
            buffers[buffer.id] = value.cuda()
        else:
            buffers[buffer.id] = torch.zeros(buffer.shape, dtype=getattr(torch, buffer.dtype.value), device="cuda")
    arguments = tuple(buffers[arg["buffer"]] for arg in runtime.external_arguments)
    try:
        runtime.prepare(*arguments)
        (tmp_path / 'resources.json').write_text(json.dumps(runtime.resource_report, indent=2))
        runtime.run_into(*arguments)
        torch.cuda.synchronize()
        outputs = runtime.buffer_plan.function_map["main"].outputs
        for (_, ids), name in zip(outputs[:4], ("transfer", "local_state", "fixed_state", "output")):
            torch.testing.assert_close(buffers[ids[0]], native[name].cuda(), rtol=0, atol=0)
        state_ids = dict(runtime.buffer_plan.function_map["main"].parameters)["state"]
        torch.testing.assert_close(buffers[state_ids[0]], native["state"].cuda(), rtol=0, atol=0)
        assert all(item["spill_bytes"] == 0 for item in runtime.resource_report)
    finally:
        runtime.close()
