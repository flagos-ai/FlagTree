# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Native GGUF GEMVs must store every scalar of a lane-packed result."""

from math import ceil, prod

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule

torch = pytest.importorskip("torch")


def _executable_target():
    if not torch.cuda.is_available():
        return None
    if torch.version.hip:
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        return {"gfx1201": "amd-gfx1201"}.get(arch)
    return {(9, 0): "nvidia-sm90", (12, 0): "nvidia-sm120"}.get(tuple(torch.cuda.get_device_capability()))


N, K = 64, 512
BLOCK_BYTES = {"IQ4_XS": 136, "Q4_K": 144}


def _module(encoding, output_lanes):

    class Graph(fm.Module):

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (1, K)), id="value")
            weight = self.input("weight", fm.tensor_type("uint8", (N, K // 256 * BLOCK_BYTES[encoding])), id="weight")
            scale = self.input("weight_scale", fm.tensor_type("float32", (ceil(N / 128), ceil(K / 128))),
                               id="weight_scale")
            output = fm.F.math.block_scaled_matmul(value, weight, scale, weight_block_n=128, weight_block_k=128,
                                                   weight_encoding=encoding, output_lanes=output_lanes,
                                                   name="projection")
            self.function("main", (value, weight, scale), (output, ))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


def _run(tmp_path, name, module, target, values, shape):
    options = CompileOptions(target=target)
    lowered = Compiler(options).compile(module).module
    path = write_artifact(lowered, tmp_path / name, target=target, emit_executable=True)
    assert "_linear_local_tile(" in (path / "generated_kernels.py").read_text()
    runtime = load(path, device="cuda:0")
    output = torch.full(shape, float("nan"), dtype=torch.bfloat16, device="cuda")
    arguments = tuple(output if spec["role"] == "result" else values[spec["value"]]
                      for spec in runtime.external_arguments)
    GeneratedTirCallGraphModule.prepare(runtime, *arguments)
    GeneratedTirCallGraphModule.run_into(runtime, *arguments)
    torch.cuda.synchronize()
    return output.reshape(1, N).float().cpu()


@pytest.mark.parametrize("encoding", ["IQ4_XS", "Q4_K"])
@pytest.mark.parametrize("lanes", [(8, ), (2, 4)])
def test_native_quantized_packed_output_stores_every_row(tmp_path, encoding, lanes):
    target = _executable_target()
    if target is None:
        pytest.skip("requires a GPU whose FlagMega target offers the native GGUF block-FP8 kernel")
    from python.test.flagmega.codegen.triton.test_gguf_native_quant import _packed_fixture

    packed, dequantized = _packed_fixture(encoding, rows=N, k=K)
    generator = torch.Generator().manual_seed(29)
    value = torch.randn(1, K, generator=generator).bfloat16()
    values = {"value": value.cuda(), "weight": torch.from_numpy(packed).cuda(),
              "weight_scale": torch.ones(ceil(N / 128), ceil(K / 128), device="cuda")}
    expected = value.float() @ torch.from_numpy(dequantized).T
    scalar = _run(tmp_path, "scalar", _module(encoding, ()), target, values, (1, N))
    packed_result = _run(tmp_path, "packed", _module(encoding, lanes), target, values, (1, N // prod(lanes), *lanes))
    assert torch.isfinite(packed_result).all(), "every packed output scalar must be written"
    torch.testing.assert_close(packed_result, scalar, rtol=0, atol=0)
    torch.testing.assert_close(scalar, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("encoding", ["IQ4_XS", "Q4_K"])
def test_native_quantized_template_masks_in_the_scalar_output_domain(encoding):
    from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry

    # Five 8-lane output vectors per owner: the loop walks 40 scalar rows, so
    # the mask must be the renderer's scalar-domain predicate, not the vector
    # extent (which would keep only the first five rows).
    call = {"family": "block_fp8", "variant": "simt", "symbol": "linear", "signature": "value, weight, result",
            "weight_encoding": encoding, "local_n_capacity": 40, "tile_n": 16, "active_n": "(5)",
            "n_mask": "(((local_n_offsets) // 8) < (5))", "value": "value", "weight": "weight", "result": "result",
            "weight_row_offset": "local_n_offsets * 136", "destination_offset": "local_n_offsets", "k": 256}
    source = TritonTemplateRegistry().render(
        "kernels/block_fp8/_function.py.jinja", {
            "render_calls": [call], "block_fp8_variant": "simt", "distributed_entry": False, "mesh_hierarchy": (8, 16),
            "mesh_axis_names": ("block_y", "block_x")
        })
    compile(source, "native_block_fp8.py", "exec")
    assert "n_mask = (((local_n_offsets) // 8) < (5))" in source
    assert "local_n_offsets < ((5))" not in source


def _render_native(tmp_path, monkeypatch, target, encoding, lanes):
    from triton.flagmega.codegen.triton import kernel_call_renderers as renderers

    captured = []
    render = renderers._FAMILY_ENCODERS["block_fp8"]
    monkeypatch.setitem(renderers._FAMILY_ENCODERS, "block_fp8", lambda raw: captured.append(raw) or render(raw))
    lowered = Compiler(CompileOptions(target=target)).compile(_module(encoding, lanes)).module
    path = write_artifact(lowered, tmp_path / "artifact", target=target, emit_executable=True)
    return (path / "generated_kernels.py").read_text(), captured


@pytest.mark.parametrize("target", ["amd-gfx1201", "nvidia-sm120"])
@pytest.mark.parametrize("encoding", ["IQ4_XS", "Q4_K"])
@pytest.mark.parametrize("lanes", [(), (8, )])
def test_native_quantized_weight_rows_are_owner_local(tmp_path, monkeypatch, target, encoding, lanes):
    # The UINT8 weight is sharded with the output rows, so each owner addresses
    # its own packed rows; a global row index reads another owner's weights.
    source, _ = _render_native(tmp_path, monkeypatch, target, encoding, lanes)
    row_bytes = K // 256 * BLOCK_BYTES[encoding]
    assert f"(local_n_offsets) * {row_bytes}" in source
    assert f"({K} // 256) * {BLOCK_BYTES[encoding]}" not in source


def test_native_quantized_gemv_rejects_a_k_sharded_activation(tmp_path, monkeypatch):
    import copy

    from triton.flagmega.codegen.triton.kernel_call_renderers import _FAMILY_ENCODERS, _buffer
    from triton.flagmega.errors import CodegenError

    _, captured = _render_native(tmp_path, monkeypatch, "amd-gfx1201", "IQ4_XS", (8, ))
    raw = copy.deepcopy(captured[0])
    value = _buffer(raw, "inputs", "value")["abi"]
    value["local_capacity_shape"] = [*value["local_capacity_shape"][:-1], K // 2]
    # The packed kernel walks whole GGUF blocks from K = 0 on every owner.
    with pytest.raises(CodegenError, match="unsharded K axis"):
        _FAMILY_ENCODERS["block_fp8"](raw)
