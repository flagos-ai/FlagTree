# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Independent attention heads retain the original ordered partial merge."""
import importlib.util

import pytest

from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry


def _call(capacity=512, active=None):
    return dict(family="paged_attention_combine", variant="decode", symbol="run",
                signature="M, S, A, Y", local_capacity=capacity, tile=128,
                active=f"attention_output_offsets < {capacity if active is None else active}",
                writer_active="True", head="attention_output_offsets//128",
                dimension="attention_output_offsets%128", context_shards=4,
                partial_owner="attention_part", partial_max="M", partial_sum="S",
                partial_accumulator="A", partial_max_owner_stride=capacity // 128,
                partial_sum_owner_stride=capacity // 128, partial_accumulator_owner_stride=capacity,
                partial_max_offset="attention_head", partial_sum_offset="attention_head",
                partial_accumulator_offset="attention_head*128+attention_dimension",
                result="Y", result_offset="attention_output_offsets")


def _render(call, corex=True, warps=16):
    return TritonTemplateRegistry().render_kernel(
        KernelTemplateSpec("paged_attention_combine", "decode", "corex" if corex else None,
                           "bi_v150" if corex else None),
        dict(render_calls=[call], num_warps=warps, distributed_entry=False, mesh_hierarchy=(1, 1)),
    ).source


@pytest.mark.parametrize("capacity,width", [(128, 128), (256, 256), (384, 256), (512, 512), (768, 512)])
def test_batched_combine_selects_independent_output_width(capacity, width):
    source = _render(_call(capacity))
    assert f"tl.range(0, {capacity}, {width})" in source
    compile(source, "combine.py", "exec")


@pytest.mark.parametrize("corex,warps", [(False, 16), (True, 32)])
def test_other_combine_templates_keep_original_tile(corex, warps):
    source = _render(_call(), corex, warps)
    assert "tl.range(0, 512, 128)" in source
    annotation = source.split("def run(", 1)[0].rstrip().splitlines()[-1]
    assert annotation == "@triton.jit(noinline=True)"


@pytest.mark.parametrize("capacity,active", [(256, 251), (384, 381), (512, 509), (768, 765)])
def test_batched_combine_preserves_partial_order_and_output_masks(tmp_path, capacity, active):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("requires an Iluvatar COREX GPU")
    heads = capacity // 128
    torch.manual_seed(842)
    maximum = torch.randn((4, heads), device="cuda", dtype=torch.float32)
    total = torch.rand((4, heads), device="cuda", dtype=torch.float32) + .5
    accumulator = torch.randn((4, capacity), device="cuda", dtype=torch.float32)
    # Equal maxima distinguish a sequential four-part merge from reassociation.
    maximum[:, 0] = 0
    total[:, 0] = 1
    accumulator[:, :128] = torch.tensor([33554432., 1., -33554432., 1.], device="cuda")[:, None]
    results = []
    for corex in (False, True):
        path = tmp_path / f"combine_{corex}.py"
        path.write_text("import triton\nimport triton.language as tl\nimport triton.experimental.tle.language as tle\n" + _render(_call(capacity, active), corex))
        spec = importlib.util.spec_from_file_location(f"combine_{corex}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        output = torch.full((capacity + 17,), -7., device="cuda", dtype=torch.bfloat16)
        module.run[(1,)](maximum, total, accumulator, output, num_warps=16, enable_fp_fusion=False)
        torch.cuda.synchronize()
        results.append(output)
        assert torch.equal(output[active:], torch.full_like(output[active:], -7.))
        assert torch.equal(output[:128], torch.full_like(output[:128], .25))
    assert torch.equal(*results)
