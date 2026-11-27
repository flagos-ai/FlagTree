# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""C++ RoPE rounding and paged-cache ownership through normal compilation."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.cpp import render_cpp_source_package, toolchain_available
from triton.flagmega.codegen.cpp.build import build_cpp_package
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import PagedAttentionStateConfig, create_paged_attention_state
from triton.flagmega.options import CompileOptions
from triton.flagmega.runtime import load

TARGET = 'nvidia-sm120'


def rope_module(dtype='bfloat16'):
    class Graph(fm.Module):
        def forward(self):
            value = self.input('value', fm.tensor_type(dtype, (2, 2, 64)))
            cosine = self.input('cosine', fm.tensor_type('float32', (2, 1, 32)))
            sine = self.input('sine', cosine.type)
            result = fm.F.nn.rope(value, cosine, sine, rotary_dim=32, name='rotated')
            self.function('main', (value, cosine, sine), (result,))
    return Graph(dialect='high_level', stage='imported', entry='main').build()


def cache_module(packed, advance):
    from python.test.flagmega.codegen.triton.kernels.update_paged_attention_kv_cache.test_packed_slots_runtime import cache_update_module
    config = PagedAttentionStateConfig(3, 4, 32, block_size=16, num_blocks=4)
    module = cache_update_module(config, packed, ('seq', 'head', 'dim'), True, advance,
                                 tokens=3, placement=fm.Placement((2, 2), 'yx', 'bb'))
    return config, module


@pytest.mark.parametrize('kind', ['rope', 'cache'])
def test_cpp_renders_normal_rope_and_cache_modules(tmp_path, kind):
    module = rope_module() if kind == 'rope' else cache_module(True, True)[1]
    compiled = Compiler(CompileOptions(target=TARGET)).compile(module).module
    package = render_cpp_source_package(compiled, tmp_path)
    assert package['source'] == 'generated_kernels.cpp'
    assert (tmp_path / package['source']).stat().st_size > 0
    if toolchain_available(TARGET):
        library = build_cpp_package(tmp_path / package['source'], tmp_path / 'kernels.so', target=TARGET)
        assert library.stat().st_size > 0


def sm120_torch():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip('An NVIDIA SM120 device is required')
    return torch


@pytest.mark.parametrize('dtype', ['bfloat16', 'float32'])
def test_cpp_rope_partial_rotation_preserves_fp32_products(tmp_path, dtype):
    torch = sm120_torch()
    compiled = Compiler(CompileOptions(target=TARGET)).compile(rope_module(dtype)).module
    artifact = write_artifact(compiled, tmp_path / 'rope', target=TARGET,
                              emit_executable=True, codegen_backend='cpp')
    runtime = load(artifact, device='cuda:0')
    generator = torch.Generator().manual_seed(75)
    value = torch.randn((2, 2, 64), generator=generator).to(device='cuda', dtype=getattr(torch, dtype))
    cosine = torch.randn((2, 1, 32), generator=generator).cuda()
    sine = torch.randn((2, 1, 32), generator=generator).cuda()
    rotated = torch.cat((-value[..., 16:32], value[..., :16]), dim=-1)
    expected = value.clone()
    expected[..., :32] = (value[..., :32].float() * cosine[..., :32]
                          + rotated.float() * sine[..., :32]).to(value.dtype)
    output = torch.empty_like(value)
    try:
        runtime.prepare(value, cosine, sine, output=output)
        runtime.run_into(output, value, cosine, sine)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    finally:
        runtime.close()


@pytest.mark.parametrize('packed,advance', [(False, False), (True, False), (True, True)])
def test_cpp_cache_chunk_crosses_pages_without_neighbor_writes(tmp_path, packed, advance):
    torch = sm120_torch()
    config, module = cache_module(packed, advance)
    compiled = Compiler(CompileOptions(target=TARGET)).compile(module).module
    artifact = write_artifact(compiled, tmp_path / 'cache', target=TARGET,
                              emit_executable=True, codegen_backend='cpp')
    runtime = load(artifact, device='cuda:0')
    state = create_paged_attention_state(config, device='cuda')
    state.kv_caches.fill_(-9)
    state.block_table.copy_(torch.tensor([[2, 0, 3, 1]], device='cuda', dtype=torch.int32))
    semantic = torch.arange(3 * 128, device='cuda').reshape(3, 4, 32).bfloat16()
    slots = semantic.reshape(3, 4, 4, 8) if packed else semantic
    binding = runtime.buffer_plan.function_map[compiled.entry]
    state_buffers = (state.kv_caches, state.query_start_loc, state.seq_lens,
                     state.slot_mapping, state.block_table)
    buffers = {}
    for value, names in binding.parameters:
        if isinstance(compiled.node_map[value].type, fm.RefType):
            buffers.update(zip(names, state_buffers, strict=True))
        else:
            buffers[names[0]] = slots
    for _, names in binding.outputs:
        buffers.update(zip(names, state_buffers, strict=True))
    arguments = [buffers[str(arg['buffer'])] for arg in runtime.external_arguments]
    try:
        runtime.prepare(*arguments)
        expected = state.kv_caches.clone()
        for position in (13, 31):
            state.seq_lens.fill_(position)
            runtime.run_into(*arguments)
            for row in range(3):
                page = (2, 0, 3, 1)[(position + row) // 16]
                expected[page, 1, 1, (position + row) % 16].copy_(semantic[row].reshape(4, 4, 8))
            torch.testing.assert_close(state.kv_caches, expected, rtol=0, atol=0)
            assert state.slot_mapping.item() == position
            assert state.seq_lens.item() == position + 3 * advance
            assert state.query_start_loc.tolist() == [0, 3]
    finally:
        runtime.close()
