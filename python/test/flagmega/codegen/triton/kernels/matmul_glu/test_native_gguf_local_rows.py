# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Native GGUF MatMulGlu must address owner-local gate/up weight rows."""

import copy
from math import ceil

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions

torch = pytest.importorskip("torch")

N, K = 64, 512
BLOCK_BYTES = {"IQ4_XS": 136, "Q4_K": 144}


def _module(encoding):

    class Graph(fm.Module):

        def forward(self):
            row_bytes = K // 256 * BLOCK_BYTES[encoding]
            value = self.input("value", fm.tensor_type("bfloat16", (1, K)), id="value")
            gate = self.input("gate", fm.tensor_type("uint8", (N, row_bytes)), id="gate")
            up = self.input("up", fm.tensor_type("uint8", (N, row_bytes)), id="up")
            scale_shape = (ceil(N / 128), ceil(K / 128))
            gate_scale = self.input("gate_scale", fm.tensor_type("float32", scale_shape), id="gate_scale")
            up_scale = self.input("up_scale", fm.tensor_type("float32", scale_shape), id="up_scale")
            output = fm.F.nn.matmul_glu(value, gate, up, gate_scale, up_scale, activation="silu", weight_block_n=128,
                                        weight_block_k=128, gate_weight_encoding=encoding,
                                        up_weight_encoding=encoding, name="glu")
            self.function("main", (value, gate, up, gate_scale, up_scale), (output, ))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


def _render(tmp_path, monkeypatch, target, encoding):
    from triton.flagmega.codegen.triton import kernel_call_renderers as renderers

    captured = []
    render = renderers._FAMILY_ENCODERS["matmul_glu"]
    monkeypatch.setitem(renderers._FAMILY_ENCODERS, "matmul_glu", lambda raw: captured.append(raw) or render(raw))
    lowered = Compiler(CompileOptions(target=target)).compile(_module(encoding)).module
    path = write_artifact(lowered, tmp_path / "artifact", target=target, emit_executable=True)
    return (path / "generated_kernels.py").read_text(), captured


@pytest.mark.parametrize("target", ["amd-gfx1201", "nvidia-sm120"])
@pytest.mark.parametrize("encoding", ["IQ4_XS", "Q4_K"])
def test_native_glu_weight_rows_are_owner_local(tmp_path, monkeypatch, target, encoding):
    # 64 rows over the 128-CTA mesh: the solver stores each owner's gate/up rows
    # in owner-local buffers, so a global row index reads past the local rows.
    source, _ = _render(tmp_path, monkeypatch, target, encoding)
    row_bytes = K // 256 * BLOCK_BYTES[encoding]
    assert f"weight_rows = (local_n_offsets) * {row_bytes}" in source
    assert f"up_weight_rows = (local_n_offsets) * {row_bytes}" in source
    assert f"({K} // 256) * {BLOCK_BYTES[encoding]}" not in source


@pytest.mark.parametrize("encoding", ["IQ4_XS", "Q4_K"])
def test_native_glu_template_reads_up_through_its_own_rows(encoding):
    from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry

    # Gate and up may live in different owner-local layouts; the up dot product
    # must use the up offsets rather than reusing the gate rows.
    call = {"family": "matmul_glu", "variant": "simt", "symbol": "glu", "signature": "value, gate, up, result",
            "weight_encoding": encoding, "local_n_capacity": 32, "tile_n": 16, "active_n": "(32)",
            "value": "value", "gate_weight": "gate", "up_weight": "up", "result": "result",
            "weight_row_offset": "(local_n_offsets) * 1000", "up_weight_row_offset": "(local_n_offsets) * 2000",
            "destination_offset": "local_n_offsets", "k": 512}
    source = TritonTemplateRegistry().render(
        "kernels/matmul_glu/_function.py.jinja", {
            "render_calls": [call], "matmul_glu_variant": "simt", "distributed_entry": False,
            "mesh_hierarchy": (8, 16), "mesh_axis_names": ("block_y", "block_x")
        })
    compile(source, "native_glu.py", "exec")
    dot = "_flagmega_iq4_xs_dot" if encoding == "IQ4_XS" else "_flagmega_q4_k_dot"
    assert "up_weight_rows = (local_n_offsets) * 2000" in source
    assert f"{dot}(value, up, up_weight_rows, n_mask, 512)" in source
    assert f"{dot}(value, gate, weight_rows, n_mask, 512)" in source


def test_native_glu_rejects_a_k_sharded_activation(tmp_path, monkeypatch):
    from triton.flagmega.codegen.triton.kernel_call_renderers import _FAMILY_ENCODERS, _buffer
    from triton.flagmega.errors import CodegenError

    _, captured = _render(tmp_path, monkeypatch, "amd-gfx1201", "IQ4_XS")
    raw = copy.deepcopy(captured[0])
    _FAMILY_ENCODERS["matmul_glu"](copy.deepcopy(raw))  # control: the unsharded K axis renders
    value = _buffer(raw, "inputs", "value")["abi"]
    value["local_capacity_shape"] = [*value["local_capacity_shape"][:-1], K // 2]
    # The packed kernel walks whole GGUF blocks from K = 0 on every owner.
    with pytest.raises(CodegenError, match="unsharded K axis"):
        _FAMILY_ENCODERS["matmul_glu"](raw)


def _executable_target():
    if not torch.cuda.is_available():
        return None
    if torch.version.hip:
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        return {"gfx1201": "amd-gfx1201"}.get(arch)
    return {(9, 0): "nvidia-sm90", (12, 0): "nvidia-sm120"}.get(tuple(torch.cuda.get_device_capability()))


@pytest.mark.parametrize("encoding", ["IQ4_XS", "Q4_K"])
def test_native_glu_matches_the_dequantized_reference(tmp_path, encoding):
    target = _executable_target()
    if target is None:
        pytest.skip("requires a GPU whose FlagMega target offers the native GGUF MatMulGlu kernel")
    from python.test.flagmega.codegen.triton.test_gguf_native_quant import _packed_fixture
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule

    packed, dequantized = _packed_fixture(encoding, rows=N, k=K)
    gate_packed, gate = torch.from_numpy(packed), torch.from_numpy(dequantized)
    # Distinct gate/up rows, so a row mix-up cannot cancel out.
    up_packed, up = gate_packed.flip(0).contiguous(), gate.flip(0).contiguous()
    value = torch.randn(1, K, generator=torch.Generator().manual_seed(31)).bfloat16()
    scale = torch.ones(ceil(N / 128), ceil(K / 128), device="cuda")
    values = {"value": value.cuda(), "gate": gate_packed.cuda(), "up": up_packed.cuda(),
              "gate_scale": scale, "up_scale": scale}

    lowered = Compiler(CompileOptions(target=target)).compile(_module(encoding)).module
    path = write_artifact(lowered, tmp_path / "artifact", target=target, emit_executable=True)
    runtime = load(path, device="cuda:0")
    output = torch.full((1, N), float("nan"), dtype=torch.bfloat16, device="cuda")
    arguments = tuple(output if spec["role"] == "result" else values[spec["value"]]
                      for spec in runtime.external_arguments)
    GeneratedTirCallGraphModule.prepare(runtime, *arguments)
    GeneratedTirCallGraphModule.run_into(runtime, *arguments)
    torch.cuda.synchronize()

    expected = torch.nn.functional.silu(value.float() @ gate.T) * (value.float() @ up.T)
    assert torch.isfinite(output).all(), "every GLU output row must be written"
    torch.testing.assert_close(output.float().cpu(), expected, rtol=2e-2, atol=2e-2)
