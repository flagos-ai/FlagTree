# Copyright 2026- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Entry-owned participation predicates need coordinates in the entry scope."""
import importlib.util

import pytest

from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry


@pytest.mark.parametrize("distributed", [False, True])
def test_entry_participation_predicate_has_local_coordinates(tmp_path, distributed):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    context = {"symbol": "run", "signature": "Output", "distributed_entry": distributed,
               "mesh_hierarchy": (2, 2), "mesh_axis_names": ("block_y", "block_x"),
               "entry_events": [{"kind": "kernel_call", "call": "write", "family": "elementwise",
                                 "variant": "add", "execution_kind": "local_shard", "symbol": "write",
                                 "arguments": "Output", "barrier_before": False,
                                 "participation_active": "(shard_y == 0) & (shard_x == 0)"}]}
    source = TritonTemplateRegistry().render("entrypoints/call_graph.py.jinja", context)
    path = tmp_path / "entry_participation.py"
    path.write_text('''import triton
import triton.language as tl
import triton.experimental.tle.language as tle
FLAGMEGA_GRID_MESH = tl.constexpr(tle.device_mesh({"block": [("block_y", 2), ("block_x", 2)]}))
@triton.jit
def write(Output):
    tl.atomic_add(Output, 1)
''' + source)
    spec = importlib.util.spec_from_file_location("entry_participation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = torch.zeros((1,), device="cuda", dtype=torch.int32)
    module.run[(4 if distributed else 1,)](output)
    # Exactly one CTA participates, including when the grid has four CTAs.
    assert output.item() == 1
