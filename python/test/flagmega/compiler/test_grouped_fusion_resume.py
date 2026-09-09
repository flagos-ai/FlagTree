# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.stages import get_stage


@pytest.mark.parametrize("old_stage",
                         ["frozen_constants", "gather_reduce_add_norm_apply_fused", "gather_reduce_norm_apply_fused"])
def test_old_fusion_checkpoints_resume_into_one_group(old_stage):

    class Graph(fm.Module):

        def forward(self):
            x = self.input("x", fm.tensor_type("float32", (8, )))
            self.function("main", (x, ), (x, ))

    module = Graph(dialect="ntt", stage=old_stage, entry="main").build()
    result = Compiler().compile(module, stop_after="fuse-gather-reduce-qkv-rope-with-cache")
    assert result.module.stage == "gather_reduce_qkv_fused"
    assert [p.name for p in result.reports[0].pass_executions] == ["FuseDistributedOps"]
    assert get_stage("fuse-gather-reduce-norm-apply").name == "fuse-distributed-ops"


def test_old_normalization_checkpoint_resumes_grouped_patterns():
    from python.test.flagmega.passes.target_independent.test_form_qkv_rope_with_cache import QKVRoPERegion
    source = QKVRoPERegion(rotary_dim=32).build()
    result = Compiler().compile(source, stop_after="form-qkv-rope-with-cache")
    assert result.module.stage == "decomposed"
    assert sum(n.op == "nn.qkv_rope_with_cache" for n in result.module.nodes) == 1
    assert [p.name for p in result.reports[0].pass_executions] == ["DecomposeComplexOps"]
