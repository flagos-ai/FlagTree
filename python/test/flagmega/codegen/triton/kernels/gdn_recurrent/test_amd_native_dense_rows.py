# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""AMD RDNA GDN gate projections and the IQ4_XS embedding use the layout-aware
native GGUF lookup helpers; generic and NVIDIA sources are unchanged."""

import importlib.util

import pytest

from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry

torch = pytest.importorskip("torch")

BLOCK_BYTES = {"IQ4_XS": 136, "Q4_K": 144}
GENERIC_CODEBOOK = "tl.where(code == 0, -127.0"
CONTEXT = {"mesh_size": 1}


@pytest.mark.parametrize("kernel,variant", (("gdn_recurrent", "persistent"), ("embedding", "decode")))
@pytest.mark.parametrize("architecture", ("gfx1100", "gfx1201"))
def test_amd_targets_resolve_the_amd_lookup_template(kernel, variant, architecture):
    registry = TritonTemplateRegistry()
    # gfx1201 GDN wraps the AMD template with its nontemporal state store.
    amd = f"kernels/{kernel}/platforms/amd/{variant}.py.jinja"
    if (kernel, architecture) == ("gdn_recurrent", "gfx1201"):
        amd = f"kernels/{kernel}/platforms/amd/gfx1201/{variant}.py.jinja"
    assert registry.resolve(KernelTemplateSpec(kernel, variant, "amd", architecture)) == amd
    assert registry.resolve(KernelTemplateSpec(kernel, variant, "nvidia", "sm120")) == \
        f"kernels/{kernel}/{variant}.py.jinja"


@pytest.mark.parametrize("kernel,variant", (("gdn_recurrent", "persistent"), ("embedding", "decode")))
def test_amd_source_uses_the_lookup_and_generic_source_keeps_the_codebook(kernel, variant):
    registry = TritonTemplateRegistry()
    amd = registry.render_kernel(KernelTemplateSpec(kernel, variant, "amd", "gfx1201"), CONTEXT).source
    generic = registry.render_kernel(KernelTemplateSpec(kernel, variant, "nvidia", "sm120"), CONTEXT).source
    assert f"# flagmega-kernel: {kernel}/{variant} platform=amd" in amd and "v_perm_b32" in amd
    assert f"# flagmega-kernel: {kernel}/{variant}\n" in generic
    assert "_flagmega_amd_" not in generic and "v_perm_b32" not in generic
    if kernel == "gdn_recurrent":
        # Scalar (uniform) rows take the AMD row dot; vector rows keep the generic path.
        assert "        if REDUCE_AXIS == 0:\n            row_value = _flagmega_amd_gdn_native_row(" in amd
        assert GENERIC_CODEBOOK in amd and GENERIC_CODEBOOK in generic
    else:
        embedding = amd.split("def _flagmega_iq4_xs_embedding(", 1)[1].split("@triton.jit", 1)[0]
        assert GENERIC_CODEBOOK not in embedding and "_flagmega_amd_iq4_nl_lookup(" in embedding
        assert GENERIC_CODEBOOK in generic


def _amd_gpu():
    return torch.cuda.is_available() and bool(torch.version.hip) and \
        torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] in ("gfx1100", "gfx1201")


def _module(tmp_path, name, platform, architecture, kernels):
    registry = TritonTemplateRegistry()
    source = "import triton\nimport triton.language as tl\n"
    for kernel, variant in kernels:
        source += registry.render_kernel(KernelTemplateSpec(kernel, variant, platform, architecture), CONTEXT).source
    source += '''
@triton.jit
def scalar_rows_kernel(source, weight, output, K: tl.constexpr, ENCODING: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    tl.store(output + row, _flagmega_dense_rows(source, weight, row, row >= 0, K, 128, False, ENCODING))


@triton.jit
def uniform_rows_kernel(source, weight, output, K: tl.constexpr, ENCODING: tl.constexpr):
    row = tl.program_id(0)
    lanes = tl.arange(0, 16)
    rows = tl.zeros((16, ), tl.int64) + row
    active = lanes < 9
    result = _flagmega_dense_rows(source, weight, rows, active, K, 128, True, ENCODING)
    tl.store(output + row * 16 + lanes, result)


@triton.jit
def embedding_kernel(weight, token, output, H: tl.constexpr):
    feature = tl.arange(0, H)
    tl.store(output + feature, _flagmega_iq4_xs_embedding(weight, token, feature, feature < H, H))
'''
    path = tmp_path / f"{name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# K=5120 is the model's GDN input (160 IQ4_XS groups: a full and a masked
# 128-group step), K=768 a short masked row.
@pytest.mark.parametrize("encoding", ("IQ4_XS", "Q4_K"))
@pytest.mark.parametrize("k", (5120, 768))
def test_amd_gdn_gate_rows_match_fp64(tmp_path, encoding, k):
    if not _amd_gpu():
        pytest.skip("requires an AMD RDNA GPU (gfx1100 / gfx1201)")
    from python.test.flagmega.codegen.triton.test_gguf_native_quant import _packed_fixture

    rows = 6
    packed, dequantized = _packed_fixture(encoding, rows=rows, k=k)
    value = torch.randn(k, generator=torch.Generator().manual_seed(3)).bfloat16()
    expected = (torch.from_numpy(dequantized).double() @ value.double()).bfloat16().double()
    module = _module(tmp_path, f"amd_gdn_{encoding}_{k}", "amd", "gfx1201", (("gdn_recurrent", "persistent"), ))
    weight, source = torch.from_numpy(packed).cuda(), value.cuda()

    scalar = torch.full((rows, ), float("nan"), device="cuda")
    module.scalar_rows_kernel[(rows, )](source, weight, scalar, k, encoding, num_warps=4)
    uniform = torch.full((rows, 16), float("nan"), device="cuda")
    module.uniform_rows_kernel[(rows, )](source, weight, uniform, k, encoding, num_warps=4)
    torch.cuda.synchronize()

    tolerance = expected.abs() * 2.0**-7 + expected.abs().max() * 1e-4
    assert ((scalar.double().cpu() - expected).abs() <= tolerance).all()
    uniform = uniform.double().cpu()
    assert torch.equal(uniform[:, :9], scalar.double().cpu()[:, None].expand(rows, 9))
    assert (uniform[:, 9:] == 0).all(), "inactive lanes return zero"


def test_amd_iq4_xs_embedding_is_bitwise_the_generic_one(tmp_path):
    if not _amd_gpu():
        pytest.skip("requires an AMD RDNA GPU (gfx1100 / gfx1201)")
    from python.test.flagmega.codegen.triton.test_gguf_native_quant import _packed_fixture

    hidden = 1024
    packed, dequantized = _packed_fixture("IQ4_XS", rows=5, k=hidden)
    weight = torch.from_numpy(packed).cuda()
    amd = _module(tmp_path, "amd_embedding", "amd", "gfx1201", (("embedding", "decode"), ))
    generic = _module(tmp_path, "generic_embedding", "nvidia", "sm120", (("embedding", "decode"), ))
    for token in range(5):
        outputs = []
        for module in (amd, generic):
            output = torch.empty(hidden, dtype=torch.bfloat16, device="cuda")
            module.embedding_kernel[(1, )](weight, token, output, hidden, num_warps=4)
            outputs.append(output.cpu())
        assert torch.equal(outputs[0], outputs[1])
        assert torch.equal(outputs[0], torch.from_numpy(dequantized[token]).bfloat16())
