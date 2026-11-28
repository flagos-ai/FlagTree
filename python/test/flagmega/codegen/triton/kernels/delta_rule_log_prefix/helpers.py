# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load


def execute_log_prefix(tmp_path, alpha, *, head_policy=None, local_input=False, **attrs):
    placement = fm.Placement((2, 2), "yx", "bb")
    broadcast = fm.SBP.broadcast()
    head = broadcast if head_policy is None else head_policy
    tensor = fm.tensor_type("float32", alpha.shape)
    distributed = fm.DistributedType(tensor, (broadcast, head), placement)

    class Graph(fm.Module):

        def forward(self):
            source = self.input("alpha", tensor if local_input else distributed)
            value = fm.F.distributed.force_boxing(source, distributed) if local_input else source
            prefix = fm.F.nn.delta_rule_log_prefix(value, **attrs)
            output = fm.F.distributed.force_boxing(prefix, prefix.type.tensor)
            self.function("main", (source, ), (output, ))

    module = Graph(dialect="distributed", stage="frozen_constants", entry="main",
                   metadata={"auto_distribution": {"placement": placement.to_data()}}).build()
    namespace = {}
    exec(fm.module_source(module), namespace)
    compiled = Compiler().compile(namespace["MODULE"]).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    source = alpha.cuda()
    runtime.prepare(source)
    return runtime.run(source).cpu()
