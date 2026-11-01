"""CPU scope regressions for the native RMS policy on real selected TIR."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_optimizations.native_precision import apply_native_norm_precision
from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler


class NormGraph(fm.Module):
    def __init__(self, dtype, use_mean):
        self.dtype = dtype
        self.use_mean = use_mean
        placement = fm.Placement((4, 8, 16), "cyx", "cbb")
        super().__init__(dialect="distributed", stage="frozen_constants", entry="main",
            metadata={"auto_distribution": {"placement": placement.to_data()}})

    def forward(self):
        kind = fm.tensor_type(self.dtype, (1, 4096))
        x = self.input("x", kind)
        placement = fm.Placement((4, 8, 16), "cyx", "cbb")
        owner = fm.DistributedType(kind, (fm.SBP.broadcast(), fm.SBP.broadcast()), placement)
        value = fm.F.distributed.sharded_view(x, owner)
        stats = fm.F.nn.norm_stats(value, axis=1, use_mean=self.use_mean, name="stats")
        self.function("main", (x,), (stats,))


@pytest.mark.parametrize("dtype,use_mean", [
    ("float32", False), ("bfloat16", False), ("float32", True),
])
def test_native_RMS_order_follows_type_and_reduction_contract(dtype, use_mean):
    before = Compiler().compile(NormGraph(dtype, use_mean).build()).module
    after = apply_native_norm_precision(before)
    assert before.nodes == after.nodes
    assert before.functions == after.functions
    assert before.metadata == after.metadata
    definitions = [d for d in after.kernel_definitions if d.dispatch.microkernel.family == "norm_stats"]
    assert definitions
    for definition in definitions:
        parameters = definition.dispatch.microkernel.parameters
        if dtype == "float32" and not use_mean:
            assert parameters["reduction_algorithm"] == "fma_1024"
        elif dtype == "bfloat16":
            assert parameters["block_size"] == 2048
            assert parameters["reduction_layout"] == "contiguous_8"
        else:
            original = next(d for d in before.kernel_definitions if d.name == definition.name)
            assert definition == original

