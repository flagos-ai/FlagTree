# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""K32 transport retains both ordered K64 streams and the distant-half fold."""
import importlib.util
import pytest
from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry


@pytest.mark.parametrize('n,k,output,pattern', [
    (520, 128, 'tl.float32', 'random'), (768, 192, 'tl.bfloat16', 'random'),
    (792, 256, 'tl.float32', 'random'), (9496, 2048, 'tl.float32', 'random'),
    (520, 128, 'tl.float32', 'register_cancel'),
    (768, 256, 'tl.float32', 'half_cancel'), (792, 256, 'tl.float32', 'step_cancel')])
def test_wide_gemv_retains_canonical_stream_and_half_order(tmp_path, n, k, output, pattern):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available() or not getattr(torch, 'corex', False):
        pytest.skip('requires an Iluvatar COREX GPU')
    call = dict(family='dense_matmul', variant='packed_k_major_gemv', symbol='run', signature='X,W,Y',
                local_m_capacity=1, local_n_capacity=n, local_k_capacity=k, tile_n=128, block_k=64,
                source='X', weight='W', result='Y', source_active='True', result_active='True', output_type=output,
                global_n='dense_local_n_offsets', global_k='dense_local_k_offsets', source_offset='dense_local_k_offsets',
                result_offset='dense_local_n_offsets', weight_offset=f'(dense_global_k[None,:]//16)*{n * 16}+dense_global_n[:,None]*16+dense_global_k[None,:]%16')
    generator = torch.Generator().manual_seed(n + k)
    x = torch.randn(k, generator=generator).bfloat16().cuda()
    dense = (torch.randn((n, k), generator=generator) / 32).bfloat16()
    if pattern != 'random':
        x.fill_(1)
        dense.zero_()
        positions = {'register_cancel': [0, 1, 2, 3], 'half_cancel': [0, 16, 32, 48], 'step_cancel': [0, 64, 128, 192]}[pattern]
        dense[:, positions] = torch.tensor([33554432., 1., -33554432., 1.]).bfloat16()
    weight = dense.reshape(n // 8, 8, k // 16, 16).permute(2, 0, 1, 3).contiguous().cuda()
    dtype = torch.bfloat16 if output == 'tl.bfloat16' else torch.float32
    results = []
    for corex, warps in [(False, 32), (True, 16)]:
        if corex:
            call['corex_flat_sme_gemv'] = dict(
                source='X', weight='W', result='Y', weight_k_group_stride=n * 16)
        source = TritonTemplateRegistry().render_kernel(
            KernelTemplateSpec('dense_matmul', 'packed_k_major_gemv',
                               'corex' if corex else None, 'bi_v150' if corex else None),
            dict(render_calls=[call], num_warps=warps,
                 distributed_entry=False, mesh_hierarchy=(1, 1)),
        ).source
        path = tmp_path / f'wide_{warps}.py'
        path.write_text('import triton\nimport triton.language as tl\nimport triton.experimental.tle.language as tle\n' + source)
        spec = importlib.util.spec_from_file_location(f'wide_{warps}', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        y = torch.full((n + 17,), -7., device='cuda', dtype=dtype)
        module.run[(1,)](x, weight, y, num_warps=warps, enable_fp_fusion=False)
        with torch.cuda.graph(graph := torch.cuda.CUDAGraph()):
            module.run[(1,)](x, weight, y, num_warps=warps, enable_fp_fusion=False)
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        results.append(y)
        assert torch.equal(y[-17:], torch.full_like(y[-17:], -7.))
    assert torch.equal(*results)
    if pattern != 'random':
        assert torch.equal(results[0][:n], torch.full_like(
            results[0][:n], 2. if pattern == 'half_cancel' else 1.))
