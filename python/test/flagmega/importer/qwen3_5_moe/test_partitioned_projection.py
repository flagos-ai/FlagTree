"""Native small-M GDN projections retain their FP32 K partials."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.importer.numerics.partitioned_projection import (
    ordered_projection_parts, qwen35_native_projection_parts,
)
from triton.flagmega.rules.neutral._utility import make_node


def projection_graph(rows=32, columns=2048, reduction=4096, *, readonly=False):
    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type("bfloat16", (rows, reduction)), id="x")
            dtype = fm.tensor_type("bfloat16", (columns, reduction))
            w = (self.weight("w", dtype, id="w", source="memory", key="w") if readonly
                 else self.input("w", dtype, id="w"))
            def emit(op, name, operands=(), attrs=None):
                node = make_node(op, name, operands, attrs or {}, {})
                return self._builder.call(op, operands, node.type, id=name, attrs=node.attrs)
            y = ordered_projection_parts(x, w, emit, name="projection", parts=2)
            self.function("main", (x,) if readonly else (x, w), (y,))
    return Graph(dialect="high_level", stage="imported", entry="main").build()


@pytest.mark.parametrize("rows,parts", [(1, 2), (32, 2), (128, 1), (1024, 1), (2048, 1)])
def test_pinned_native_partition_contract(rows, parts):
    x = fm.Node("x", "builtin.var", (), fm.tensor_type("bfloat16", (rows, 4096)))
    w = fm.Node("w", "builtin.var", (), fm.tensor_type("bfloat16", (2048, 4096)))
    assert qwen35_native_projection_parts(x, w) == parts


def test_partitioned_projection_declares_one_final_rounding_boundary():
    module = projection_graph(1, 8, 64)
    calls = [n for n in module.nodes if n.op == "math.matmul"]
    assert len(calls) == 2
    assert all(n.attrs["output_data_type"] == "float32" and n.attrs["reduction_order"] == "mma_k16" for n in calls)
    output = module.node_map["projection"]
    assert output.op == "tensors.cast" and output.attrs["rounding_contract"] == "preserve"


def test_partitioned_gdn_projection_matches_native_cuda(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.runtime import load
    from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
    torch.manual_seed(1)
    x = torch.randn((32, 4096), device="cuda", dtype=torch.bfloat16) * .125
    weight = torch.randn((2048, 4096), device="cuda", dtype=torch.bfloat16) * .03125
    source = projection_graph(readonly=True)
    compiled = Compiler().compile(source).module
    checkpoint = MemoryCheckpoint({}, {"w": TensorInfo("w", fm.DType.BFLOAT16, (2048, 4096), "memory")},
                                  {"w": weight.cpu()})
    runtime = load(write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90",
                                 checkpoint=checkpoint, emit_executable=True), device="cuda:0")
    expected = torch.nn.functional.linear(x, weight)
    actual = torch.empty_like(expected)
    try:
        runtime.prepare(x, output=actual)
        runtime.run_into(actual, x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        reports = runtime.resource_report
        reports = reports if isinstance(reports, tuple) else (reports,)
        assert all(r["spill_bytes"] == 0 for r in reports)
    finally:
        runtime.close()


def test_partitioned_projection_after_vectorized_normalization_cuda(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.runtime import load
    from triton.flagmega.importer import MemoryCheckpoint, TensorInfo

    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type("bfloat16", (32, 32, 128)), id="x")
            z = self.input("z", fm.tensor_type("float32", (32, 32, 128)), id="z")
            w = self.weight("w", fm.tensor_type("bfloat16", (2048, 4096)), id="w", source="memory", key="w")
            def emit(op, name, operands=(), attrs=None):
                node = make_node(op, name, operands, attrs or {}, {})
                return self._builder.call(op, operands, node.type, id=name, attrs=node.attrs)
            gamma = emit("builtin.splat_const", "gamma", attrs={"result_type": fm.tensor_type("float32", (128,)), "value": 1.})
            bias = emit("builtin.splat_const", "bias", attrs={"result_type": fm.tensor_type("float32", (128,)), "value": 0.})
            stats = emit("nn.norm_stats", "stats", (x,), {"axis": -1, "use_mean": False})
            norm = emit("nn.norm_apply", "norm", (x, stats, gamma, bias),
                        {"axis": -1, "epsilon": 1.e-6, "use_mean": False, "round_before_scale": False,
                         "input_dtype": "float32", "output_dtype": "float32"})
            product = emit("math.mul", "product", (norm, z))
            rounded = emit("tensors.cast", "rounded", (product,), {"dtype": "bfloat16", "rounding_contract": "preserve"})
            lhs = emit("tensors.reshape", "lhs", (rounded,), {"shape": (32, 4096)})
            y = ordered_projection_parts(lhs, w, emit, name="projection", parts=2)
            self.function("main", (x, z), (y,))

    torch.manual_seed(1)
    x = torch.randn((32, 32, 128), device="cuda", dtype=torch.bfloat16)
    z = torch.randn_like(x, dtype=torch.float32) * .125
    weight = torch.randn((2048, 4096), device="cuda", dtype=torch.bfloat16) * .03125
    source = Graph(dialect="high_level", stage="imported", entry="main").build()
    compiled = Compiler().compile(source).module
    checkpoint = MemoryCheckpoint({}, {"w": TensorInfo("w", fm.DType.BFLOAT16, (2048, 4096), "memory")}, {"w": weight.cpu()})
    runtime = load(write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", checkpoint=checkpoint,
                                 emit_executable=True), device="cuda:0")
    expected_lhs = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1.e-6) * z).bfloat16().reshape(32, 4096)
    expected = torch.nn.functional.linear(expected_lhs, weight)
    actual = torch.empty_like(expected)
    try:
        runtime.prepare(x, z, output=actual)
        runtime.run_into(actual, x, z)
        # Norm reduction may differ at BF16 midpoints. Missing owner shards
        # instead leave almost all output columns uncomputed.
        torch.testing.assert_close(actual, expected, rtol=.02, atol=.001)
        reports = runtime.resource_report
        reports = reports if isinstance(reports, tuple) else (reports,)
        assert all(r["spill_bytes"] == 0 for r in reports)
    finally:
        runtime.close()
