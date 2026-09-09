# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load


@pytest.mark.parametrize("op,input_name", [
    ("softmax", "row"),
    ("reduce_sum", "row"),
    ("top_k", "candidates"),
    ("concat", "value"),
    ("broadcast_to", "value"),
])
def test_local_primitive_temporaries_do_not_shadow_editable_ir_arguments(tmp_path, op, input_name):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    class Graph(fm.Module):

        def forward(self):
            value = self.input(input_name, fm.tensor_type("float32", (3, 7)), id=input_name)
            if op == "softmax":
                result = fm.F.nn.softmax(value, name="probabilities")
            elif op == "reduce_sum":
                result = fm.F.math.reduce_sum(value)
            elif op == "top_k":
                result = fm.F.tensors.get_item(fm.F.tensors.top_k(value, k=3), 0)
            elif op == "concat":
                result = fm.F.tensors.concat(value, value, axis=-1)
            else:
                result = fm.F.tensors.broadcast_to(value, shape=(2, 3, 7))
            self.function("main", (value, ), (result, ))

    compiled = Compiler().compile(Graph(dialect="high_level", stage="frozen_constants", entry="main").build()).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    value = torch.arange(21, device="cuda", dtype=torch.float32).reshape(3, 7)
    runtime.prepare(value)
    output = runtime.run(value)
    torch.cuda.synchronize()
    expected = {
        "softmax": lambda: value.softmax(-1), "reduce_sum": lambda: value.sum(-1, keepdim=True), "top_k":
        lambda: value.flip(-1)[:, :3], "concat": lambda: torch.cat(
            (value, value), -1), "broadcast_to": lambda: value.expand(2, 3, 7)
    }[op]()
    torch.testing.assert_close(output, expected, rtol=2e-6, atol=1e-7)
