# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Ordered MMA prefetch is an explicit, serializable implementation choice."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import CodegenError
from triton.flagmega.options import CompileOptions
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule
from triton.flagmega.selection import apply_plan, override_plan
from triton.flagmega.targets import NvidiaSm90Target

from python.test.flagmega.codegen.triton.kernels.dense_matmul.test_small_bf16_schedule import _graph


@pytest.fixture(scope="module")
def policy():
    path = (
        Path(__file__).resolve().parents[6]
        / "tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800/local_optimizations/small_mma_prefetch_schedule.py"
    )
    name = "_flagmega_small_ordered_mma_prefetch_policy"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        sys.modules.pop(name)


def _compile(source, mesh, policy, selected=True, warps=8, force_prefetch=False):
    class Target(NvidiaSm90Target):
        def propose_tir(self, module):
            proposal = super().propose_tir(module)
            if force_prefetch:
                return apply_plan(
                    proposal,
                    override_plan(
                        proposal,
                        [("tir.projection", policy.IMPLEMENTATION)],
                        policy="test.ordered_mma_prefetch",
                        rationale="Exercise the complete implementation domain outside the small-row policy.",
                    ),
                )
            return policy.select(proposal)

    base = policy.implementation_model(NvidiaSm90Target().triton_implementation_model)
    mma = "tir.dense_matmul.packed_k_major_mma"
    catalog = replace(
        base,
        implementations=tuple(
            replace(i, parameters={**i.parameters, "compute_num_warps": warps})
            if i.id in (mma, policy.IMPLEMENTATION)
            else i
            for i in base.implementations
        ),
        preferences={
            **base.preferences,
            "dense_matmul": tuple(dict.fromkeys((mma, *base.preferences["dense_matmul"]))),
        },
    )
    compiler = Compiler(CompileOptions(placement=mesh))
    compiler.target = (Target if selected else NvidiaSm90Target)(triton_implementation_model=catalog)
    return compiler.compile(source).module


@pytest.mark.parametrize(
    "m,k,n,dtype,selected",
    [
        (1, 4096, 16, "bfloat16", True),
        (1, 1152, 8, "bfloat16", True),
        (1, 1088, 8, "bfloat16", True),
        (1, 1552, 16, "bfloat16", True),
        (1, 512, 8, "bfloat16", False),
        (1, 4096, 24, "bfloat16", False),
        (2, 4096, 16, "bfloat16", False),
        (1, 4096, 16, "float32", False),
    ],
)
def test_prefetch_selection_roundtrip(tmp_path, policy, m, k, n, dtype, selected):
    mesh = fm.Placement((1, 1), "yx", "bb")
    module = _compile(_graph(m, k, n, mesh, dtype), mesh, policy)
    record = module.selection_map["tir.projection"]
    (definition,) = (d for d in module.kernel_definitions if d.dispatch.microkernel.family == "dense_matmul")
    micro = definition.dispatch.microkernel
    assert (record.candidate_id == policy.IMPLEMENTATION) is selected
    assert record.candidate_id == micro.implementation
    if selected:
        assert record.policy == policy.POLICY and micro.parameters["block_k"] == 32
        assert micro.parameters["prefetch_elements"] == 16384
        assert not micro.shared_workspaces and micro.transfer_pipeline is None
    path = fm.emit_module(module, tmp_path / "selected.py")
    restored = fm.load_module(path)
    assert restored.semantic_hash == module.semantic_hash
    assert fm.verify_buffer_plan(restored) == fm.verify_buffer_plan(module)
    # A valid selected implementation must render through the ordinary artifact API.
    write_artifact(restored, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)


@pytest.mark.parametrize(
    "m,k,n,axes,warps,force_prefetch",
    [
        (1, 4096, 128, (1,), 8, False),
        (1, 4096, 2048, (0, 1), 8, False),
        (1, 1536, 128, (1,), 8, False),
        (1, 1552, 2048, (0, 1), 8, False),
        (1, 1152, 128, (1,), 16, False),
        (2, 80, 24, (), 8, True),
    ],
)
def test_prefetch_keeps_mma_bits_tail_and_visibility(tmp_path, policy, m, k, n, axes, warps, force_prefetch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    mesh = fm.Placement((8, 16), "yx", "bb")
    source = _graph(m, k, n, mesh, axes=axes)
    modules = [_compile(source, mesh, policy, selected, warps, force_prefetch) for selected in (False, True)]
    paths = [
        write_artifact(m, tmp_path / str(i), target="nvidia-sm90", emit_executable=True) for i, m in enumerate(modules)
    ]
    runtimes = [load(p, device="cuda:0") for p in paths]
    x = torch.empty((m, k), dtype=torch.bfloat16, device="cuda")
    w = torch.empty((k, n), dtype=torch.bfloat16, device="cuda")
    packed = torch.empty((k // 16, n // 8, 8, 2, 8), dtype=torch.bfloat16, device="cuda")
    sentinel = torch.arange(128, dtype=torch.float32, device="cuda")
    outputs, guards = [], []
    for rt in runtimes:
        a = torch.full((m * n + 32,), 123.0, dtype=torch.bfloat16, device="cuda")
        b = torch.full((160,), 123.0, device="cuda")
        names = [s["value"] for s in rt.external_arguments if s["role"] == "result"]
        outputs.append(dict(zip(names, (a[: m * n].reshape(m, n // 8, 8), b[:128]), strict=True)))
        guards.extend((a[m * n :], b[128:]))
    rng = torch.Generator(device="cuda").manual_seed(712)
    try:
        for mutation in range(4):
            if mutation == 0:
                x.copy_((((torch.arange(m * k, device="cuda") % 13) - 6) / 128).reshape(m, k))
                w.copy_((((torch.arange(k * n, device="cuda") % 17) - 8) / 128).reshape(k, n))
            elif mutation == 1:
                x.zero_()
                w.zero_()
                x[:, 0] = 129 / 128
                x[:, 1] = 1.0
                w[0].fill_(129 / 128)
                w[1].fill_(-65 / 64)
            else:
                x.copy_(torch.randn((m, k), device="cuda", generator=rng))
                w.copy_(torch.randn((k, n), device="cuda", generator=rng) * 0.01)
            packed.copy_(w.reshape(k // 16, 2, 8, n // 8, 8).permute(0, 3, 4, 1, 2))
            for rt, out in zip(runtimes, outputs, strict=True):
                values = dict(x=x, w=packed, sentinel=sentinel, **out)
                args = tuple(values[s["value"]] for s in rt.external_arguments)
                GeneratedTirCallGraphModule.prepare(rt, *args)
                GeneratedTirCallGraphModule.run_into(rt, *args)
                assert rt.resource_report["grid"] == [128, 1, 1] and rt.resource_report["spill_bytes"] == 0
            torch.cuda.synchronize()
            a, b = (next(iter(o.values())).reshape(m, n) for o in outputs)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            if mutation < 2:
                expected = (x.cpu().double() @ w.cpu().double()).bfloat16().cuda()
                torch.testing.assert_close(b, expected, rtol=0, atol=0)
                if mutation == 1:
                    assert torch.all(b == 2**-14)
            for out in outputs:
                torch.testing.assert_close(tuple(out.values())[1], sentinel * 2, rtol=0, atol=0)
            assert all(torch.all(g == 123.0) for g in guards)
    finally:
        for rt in runtimes:
            rt.close()


@pytest.mark.parametrize(
    "change",
    [
        {"prefetch_elements": 0},
        {"prefetch_elements": -1},
        {"prefetch_elements": 3},
        {"prefetch_elements": True},
        {"prefetch_elements": 128},
        {"prefetch_max_k": 16},
        {"prefetch_max_k": 3},
    ],
)
def test_prefetch_rejects_invalid_private_shared_budget(change):
    from triton.flagmega.codegen.triton.dense_matmul_renderer import mma_prefetch_context

    parameters = {"prefetch_elements": 16384, "prefetch_max_k": 2048, **change}
    call = {"block_k": 32, "tile_n": 16, "local_k_capacity": 4096, "local_n_capacity": 8}
    with pytest.raises(CodegenError, match="prefetch"):
        mma_prefetch_context(parameters, call)


def test_prefetch_requires_explicit_target_registration(policy):
    base = NvidiaSm90Target().triton_implementation_model
    assert base.implementation(policy.IMPLEMENTATION) is None
    configured = policy.implementation_model(base)
    assert policy.implementation_model(configured) == configured
    assert configured.implementation(policy.IMPLEMENTATION).requires == ("mma_v3",)
