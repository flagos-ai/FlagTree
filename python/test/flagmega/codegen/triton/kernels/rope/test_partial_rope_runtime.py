# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Native partial RoPE: scalar/vector storage and two-dimensional ownership."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("table_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("head,rotary", [(24, 16), (256, 64)])
def test_native_partial_rope_matches_prefix_math_and_copies_tail_bits(tmp_path, dtype, table_dtype, vector, split, head, rotary):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    lane = 8 if vector else 1
    value_type = fm.tensor_type(fm.vector_type(dtype, (lane, )) if vector else dtype, (2, 4, head // lane))
    table_type = fm.tensor_type(
        fm.vector_type(table_dtype, (2, lane)) if vector else table_dtype,
        (2, 1, rotary // (2 * lane) if vector else rotary))
    metadata = {}
    if split:
        placement = fm.Placement((2, 2), "yx", "bb")
        b = fm.SBP.broadcast()
        value_type = fm.DistributedType(value_type, (b, fm.SBP.split_contiguous((0, 1)), b), placement)
        table_type = fm.DistributedType(table_type, (b, b, b), placement)
        metadata = {"auto_distribution": {"placement": placement.to_data()}}

    class Graph(fm.Module):

        def forward(self):
            value = self.input("value", value_type)
            cosine = self.input("cosine", table_type)
            sine = self.input("sine", table_type)
            op = fm.F.ntt.vectorized_rope if vector else fm.F.nn.rope
            output = op(value, cosine, sine, rotary_dim=rotary, name="output")
            self.function("main", (value, cosine, sine), (output, ))

    module = Graph(dialect="ntt", stage="frozen_constants", entry="main", metadata=metadata).build()
    compiled = Compiler().compile(module).module
    runtime = load(write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True),
                   device="cuda:0")
    generator = torch.Generator().manual_seed(92)
    value = torch.randn((2, 4, head), generator=generator).to(device="cuda", dtype=getattr(torch, dtype))
    cosine = torch.randn((2, 1, rotary), generator=generator).to(device="cuda", dtype=getattr(torch, table_dtype))
    sine = torch.randn((2, 1, rotary), generator=generator).to(device="cuda", dtype=getattr(torch, table_dtype))
    # Include signed zero and NaN payloads. The unrotated region is a copy,
    # not x * 1 + 0, nor a round trip through another floating-point dtype.
    bits_type = torch.int16 if dtype == "bfloat16" else torch.int32
    payload = 0x7FC5 if dtype == "bfloat16" else 0x7FC00005
    value.view(bits_type)[..., rotary] = payload
    value[..., rotary + 1] = -0.0
    value[..., rotary + 2] = float("inf")
    prefix = value[..., :rotary].float()
    partner = torch.cat((-prefix[..., rotary // 2:], prefix[..., :rotary // 2]), dim=-1)
    expected = (prefix * cosine.float() + partner * sine.float()).to(value.dtype)
    packed_value = value.reshape(2, 4, head // lane, lane) if vector else value
    packed_cosine = cosine.reshape(2, 1, rotary // (2 * lane), 2, lane) if vector else cosine
    packed_sine = sine.reshape_as(packed_cosine)
    output = torch.empty_like(packed_value)
    runtime.prepare(packed_value, packed_cosine, packed_sine, output=output)
    runtime.run_into(output, packed_value, packed_cosine, packed_sine)
    torch.cuda.synchronize()
    output = output.reshape_as(value)
    torch.testing.assert_close(output[..., :rotary], expected, rtol=0, atol=0)
    assert torch.equal(output.view(bits_type)[..., rotary:], value.view(bits_type)[..., rotary:])
