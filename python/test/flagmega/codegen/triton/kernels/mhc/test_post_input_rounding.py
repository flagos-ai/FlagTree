# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Consumer-owned residual rounding preserves all bits without a Cast call."""

import pytest
import torch
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime

from python.test.flagmega.ir.ops.nn.test_mhc_post_input_rounding import PostRounding


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype,hidden,mode", (("bfloat16", 4096, "value_fma_first"),
    ("bfloat16", 259, "value_fma_first"), ("float16", 259, "value_fma_first"),
    ("bfloat16", 8, "residual_then_value")))
def test_residual_rounding_kernel_matches_explicit_conversion(tmp_path, dtype, hidden, mode):
    runtimes = []
    generator = torch.Generator(device="cuda").manual_seed(931)
    try:
        for explicit in (True, False):
            module = Compiler().compile(PostRounding(explicit=explicit, dtype=dtype, hidden=hidden, mode=mode).build()).module
            casts = [d for d in module.kernel_definitions if d.dispatch.microkernel.family == "elementwise"
                     and d.dispatch.microkernel.variant == "cast"]
            assert bool(casts) == explicit
            runtimes.append(load(write_artifact(module, tmp_path / str(explicit), target="nvidia-sm90", emit_executable=True),
                                 device="cuda:0"))
        for mutation in range(3):
            residual = torch.randn((2, 4, hidden), generator=generator, device="cuda")
            residual[0, 0, :4] = torch.tensor([1.00390625, -1.00390625, 1.003905, 2.0078125], device="cuda")
            inputs = {"value": torch.randn((2, hidden), generator=generator, device="cuda").bfloat16(),
                      "residual": residual, "post": torch.randn((2, 4), generator=generator, device="cuda"),
                      "comb": torch.randn((2, 4, 4), generator=generator, device="cuda")}
            if mutation == 2:
                inputs["value"].zero_()
                inputs["post"].zero_()
                inputs["comb"].copy_(torch.tensor([2.**24, 1., -2.**24, 2.], device="cuda").reshape(1, 4, 1))
                inputs["residual"].fill_(1.00390625)
            outputs = []
            for runtime in runtimes:
                storage = torch.full((2 * 4 * hidden + 64,), -123., device="cuda")
                result = storage[32:-32].reshape(2, 4, hidden)
                result.fill_(float("nan"))
                arguments = tuple(result if spec["role"] == "result" else inputs[spec["value"]]
                                  for spec in runtime.external_arguments)
                Runtime.prepare(runtime, *arguments)
                Runtime.run_into(runtime, *arguments)
                torch.cuda.synchronize()
                assert bool((storage[:32] == -123).all() & (storage[-32:] == -123).all())
                assert runtime.resource_report["spill_bytes"] == 0
                outputs.append(result)
            assert torch.equal(outputs[0].view(torch.uint8), outputs[1].view(torch.uint8))
    finally:
        for runtime in runtimes:
            runtime.close()
