# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load


def execute_l2(tmp_path, value, *, policies=None, local_input=False, local_result=False, **attrs):
    placement = fm.Placement((2, 2), "xy", "bb")
    tensor = fm.tensor_type(str(value.dtype).removeprefix("torch."), value.shape)
    policies = (fm.SBP.broadcast(),) * len(value.shape) if policies is None else policies
    distributed = fm.DistributedType(tensor, policies, placement)

    class Graph(fm.Module):
        def forward(self):
            source = self.input("value", tensor if local_input else distributed)
            data = fm.F.distributed.force_boxing(source, distributed) if local_input else source
            normalized = fm.F.nn.l2_normalization(data, **attrs)
            output = fm.F.distributed.force_boxing(normalized, tensor) if local_result else normalized
            self.function("main", (source,), (output,))

    module = Graph(dialect="distributed", stage="frozen_constants", entry="main",
                   metadata={"auto_distribution": {"placement": placement.to_data()}}).build()
    namespace = {}
    exec(fm.module_source(module), namespace)
    assert namespace["MODULE"].semantic_hash == module.semantic_hash
    compiled = Compiler().compile(namespace["MODULE"]).module
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    source = value.cuda()
    runtime.prepare(source)
    output = runtime.run(source).cpu()
    assert source.cpu().equal(value)
    return output
