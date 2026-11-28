# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Factor only the nonnegative BF16 packed-owner coordinates."""
from triton.flagmega.codegen.triton import describe_tir_package, render_tir_package


def test_factored_coordinates_preserve_every_slot_element():
    seen = set()
    for k in range(1024):
        for n in range(16):
            payload = (n % 8) * 16 + k % 16
            old = (k // 16, n // 8, payload // 64, payload % 64)
            new = (k // 16, n // 8, (n % 8) // 4, (n % 4) * 16 + k % 16)
            assert old == new
            seen.add(new)
    assert len(seen) == 16384


def check_render(module):
    package = describe_tir_package(module)
    call, = package['render_calls']
    call.update(accumulation_mode='ordered_mma', mma_block_k=128)
    before = render_tir_package(package, 'factored-direct-test')
    call['mma_rhs_load'] = 'direct_dot'
    direct = render_tir_package(package, 'factored-direct-test')
    assert '(dense_stage_n[None, :] % 8) // 4' in direct
    assert '(dense_stage_n[None, :] % 4) * 16 + dense_stage_k[:, None] % 16' in direct
    assert 'dense_payload // 64' not in direct
    assert 'dense_payload // 64' in before
    del call['mma_rhs_load']
    assert render_tir_package(package, 'factored-direct-test') == before


def test_dense_descriptor_factored_render(packed_descriptor_pipeline_module):
    check_render(packed_descriptor_pipeline_module)


def test_norm_stats_factored_render():
    from .test_dense_lhs_staging import direct_norm_stats_staged_module
    check_render(direct_norm_stats_staged_module())
