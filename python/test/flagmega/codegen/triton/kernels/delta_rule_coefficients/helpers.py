# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load


def execute_coefficients(tmp_path, key, beta, *, block_size=64, split_axes=(), local_inputs=False):
    placement = fm.Placement((2, 2), "yx", "bb")
    b = fm.SBP.broadcast()
    head = fm.SBP.split_contiguous(split_axes) if split_axes else b
    key_tensor, beta_tensor = fm.tensor_type("bfloat16", key.shape), fm.tensor_type("float32", beta.shape)
    key_type = fm.DistributedType(key_tensor, (b, head, b), placement)
    beta_type = fm.DistributedType(beta_tensor, (b, head), placement)

    class Graph(fm.Module):

        def forward(self):
            k = self.input("key", key_tensor if local_inputs else key_type, id="key")
            beta = self.input("beta", beta_tensor if local_inputs else beta_type, id="beta")
            params = (k, beta)
            if local_inputs:
                k = fm.F.distributed.force_boxing(k, key_type)
                beta = fm.F.distributed.force_boxing(beta, beta_type)
            value = fm.F.nn.delta_rule_coefficients(k, beta, block_size=block_size)
            output = fm.F.distributed.force_boxing(value, value.type.tensor)
            self.function("main", params, (output, ))

    module = Graph(dialect="distributed", stage="frozen_constants", entry="main",
                   metadata={"auto_distribution": {"placement": placement.to_data()}}).build()
    namespace = {}
    exec(fm.module_source(module), namespace)
    compiled = Compiler().compile(namespace["MODULE"]).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    args = key.cuda(), beta.cuda()
    runtime.prepare(*args)
    return runtime.run(*args).cpu()
