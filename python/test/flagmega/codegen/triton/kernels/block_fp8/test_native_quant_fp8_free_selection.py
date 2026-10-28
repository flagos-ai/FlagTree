# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Native GGUF GEMVs never quantize activations, so they need no fp8 support."""

from math import ceil

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.options import CompileOptions

N, K = 64, 512
BLOCK_BYTES = {"IQ4_XS": 136, "Q4_K": 144}


def _packed(builder, name, encoding):
    return builder.input(name, fm.tensor_type("uint8", (N, K // 256 * BLOCK_BYTES[encoding])), id=name)


def _scale(builder, name):
    return builder.input(name, fm.tensor_type("float32", (ceil(N / 128), ceil(K / 128))), id=name)


def _native_projection(encoding):

    class Graph(fm.Module):

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (1, K)), id="value")
            weight, scale = _packed(self, "weight", encoding), _scale(self, "weight_scale")
            output = fm.F.math.block_scaled_matmul(value, weight, scale, weight_block_n=128, weight_block_k=128,
                                                   weight_encoding=encoding, name="projection")
            self.function("main", (value, weight, scale), (output, ))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


def _native_glu(encoding):

    class Graph(fm.Module):

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (1, K)), id="value")
            gate, up = _packed(self, "gate", encoding), _packed(self, "up", encoding)
            gate_scale, up_scale = _scale(self, "gate_scale"), _scale(self, "up_scale")
            output = fm.F.nn.matmul_glu(value, gate, up, gate_scale, up_scale, activation="silu",
                                        weight_block_n=128, weight_block_k=128, gate_weight_encoding=encoding,
                                        up_weight_encoding=encoding, name="glu")
            self.function("main", (value, gate, up, gate_scale, up_scale), (output, ))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


def _block_fp8_projection():

    class Graph(fm.Module):

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (1, K)), id="value")
            weight = self.input("weight", fm.tensor_type("float8_e4m3fn", (N, K)), id="weight")
            scale = _scale(self, "weight_scale")
            output = fm.F.math.block_scaled_matmul(value, weight, scale, weight_block_n=128, weight_block_k=128,
                                                   name="projection")
            self.function("main", (value, weight, scale), (output, ))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


def _selected(module, target, family):
    lowered = Compiler(CompileOptions(target=target)).compile(module, stop_after="lower-tir").module
    return {record.candidate_id for record in lowered.selections if record.candidate_id.startswith(f"tir.{family}.")}


@pytest.mark.parametrize("encoding", ("IQ4_XS", "Q4_K"))
@pytest.mark.parametrize("build,family", ((_native_projection, "block_fp8"), (_native_glu, "matmul_glu")))
def test_native_gguf_gemv_selects_the_fp8_free_simt_without_fp8(build, family, encoding):
    assert _selected(build(encoding), "amd-gfx1100", family) == {f"tir.{family}.native_simt"}


@pytest.mark.parametrize("target", ("amd-gfx1201", "nvidia-sm120", "nvidia-sm90"))
@pytest.mark.parametrize("build,family", ((_native_projection, "block_fp8"), (_native_glu, "matmul_glu")))
def test_fp8_targets_keep_the_existing_native_gguf_selection(build, family, target):
    assert _selected(build("IQ4_XS"), target, family) == {f"tir.{family}.simt"}


def test_block_fp8_activation_quantization_still_requires_fp8():
    with pytest.raises(IRVerificationError, match="no capability-supported candidate"):
        _selected(_block_fp8_projection(), "amd-gfx1100", "block_fp8")
    assert _selected(_block_fp8_projection(), "amd-gfx1201", "block_fp8") <= {"tir.block_fp8.simt",
                                                                              "tir.block_fp8.mma"}
