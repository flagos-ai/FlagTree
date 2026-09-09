# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Scalar and lane-wise cached gates preserve the original device formula."""

import importlib.util

import pytest

from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry

_WRAPPER = '''
@triton.jit
def gates_kernel(B, A, HEAD, ACTIVE, LOG, BIAS, BETA, DECAY,
                 N: tl.constexpr, SCALAR: tl.constexpr, ROUND: tl.constexpr,
                 REFERENCE: tl.constexpr):
    if SCALAR:
        offsets = tl.program_id(0)
    else:
        offsets = tl.program_id(0) * 32 + tl.arange(0, 32)
    active = tl.load(ACTIVE + offsets, mask=offsets < N, other=0).to(tl.int1)
    head = tl.load(HEAD + offsets, mask=offsets < N, other=-1)
    b = tl.load(B + offsets, mask=offsets < N, other=0.)
    a = tl.load(A + offsets, mask=offsets < N, other=0.)
    if REFERENCE:
        # Original per-value-lane core formula, before cache extraction.
        beta = tl.sigmoid(b)
        if ROUND:
            beta = beta.to(tl.bfloat16).to(tl.float32)
        decay_input = a + tl.load(BIAS + head, mask=active, other=0.).to(tl.float32)
        softplus = tl.where(decay_input > 20., decay_input,
                            libdevice.log1p(libdevice.exp(decay_input)))
        decay_log = -libdevice.exp(
            tl.load(LOG + head, mask=active, other=0.).to(tl.float32)) * softplus
        decay = libdevice.exp(decay_log)
    else:
        beta, decay = _flagmega_gdn_head_gates(b, a, head, active, LOG, BIAS, ROUND)
    tl.store(BETA + offsets, beta, mask=(offsets < N) & active)
    tl.store(DECAY + offsets, decay, mask=(offsets < N) & active)
'''


@pytest.fixture
def gates_kernel(tmp_path):
    source = TritonTemplateRegistry().render_kernel(
        KernelTemplateSpec("gdn_recurrent", "persistent", "nvidia", "sm90"),
        {"recurrent_value_tile": 8, "head_block": 128, "query_scale_repr": "0.125"},
    ).source
    path = tmp_path / "head_gates.py"
    path.write_text("import triton\nimport triton.language as tl\n"
                    "from triton.language.extra.cuda import libdevice\n" + source + _WRAPPER)
    spec = importlib.util.spec_from_file_location("flagmega_head_gates_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.gates_kernel


@pytest.mark.parametrize("scalar", [True, False])
@pytest.mark.parametrize("round_beta", [True, False])
@pytest.mark.parametrize("warps", [4, 8])
def test_cached_head_gates_match_original_formula_bitwise(gates_kernel, scalar, round_beta, warps):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    import triton

    # Covers softplus's threshold, underflow/overflow of its unselected arm,
    # BF16 sigmoid rounding, mixed/repeated heads, tails and an inactive CTA.
    b = torch.tensor([-100., -20., -1., -.00390625, -0., 0., .00390625, 1., 20., 100.] * 4, dtype=torch.float32,
                     device="cuda")[:37]
    a = torch.tensor([-100., -20., -1., 0., 19.75, 20., 20.25, 100.] * 5, dtype=torch.float32, device="cuda")[:37]
    log = torch.tensor([-90., -20., -1., 0., .5, 1., 20., 88.], device="cuda")
    bias = torch.tensor([0., 0., -.25, .25, -.5, .5, 0., 0.], device="cuda")
    heads = (torch.arange(b.numel(), device="cuda") * 3 // 2) % log.numel()
    active = torch.arange(b.numel(), device="cuda") % 7 != 0
    heads[~active] = -1000000  # No inactive lane may read the readonly tables.
    expected = [torch.full_like(b, -123.) for _ in range(2)]
    actual = [torch.full_like(b, -123.) for _ in range(2)]
    grid = (b.numel() + 1 if scalar else triton.cdiv(b.numel(), 32) + 1, )
    for reference, outputs in ((True, expected), (False, actual)):
        gates_kernel[grid](b, a, heads, active, log, bias, *outputs, b.numel(), scalar, round_beta, reference,
                           num_warps=warps, enable_fp_fusion=False)
    for observed, original in zip(actual, expected):
        torch.testing.assert_close(observed.view(torch.int32), original.view(torch.int32), rtol=0, atol=0)
