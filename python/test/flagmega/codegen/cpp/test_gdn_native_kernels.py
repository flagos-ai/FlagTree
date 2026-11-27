# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Host execution regressions for state ownership and GDN rounding contracts."""
import shutil

import numpy as np
import pytest

from python.test.flagmega.codegen.cpp.test_native_quantized_kernels import (
    bf16, f32, library, packed_rows, render, run,
)


pytestmark = pytest.mark.skipif(shutil.which("c++") is None, reason="A host C++ compiler is required")


@pytest.mark.parametrize('history', [0, 3])
@pytest.mark.parametrize('round_products,current_first,round_activation', [(True, True, True), (False, False, False), (True, False, False)])
def test_convolution_updates_canonical_history_once_per_token(tmp_path, history, round_products, current_first, round_activation):
    channels, tokens = 16, 3
    types = {'qkv': 'ftt::bfloat16_t', 'state': 'ftt::bfloat16_t', 'weight': 'ftt::bfloat16_t', 'result': 'ftt::bfloat16_t'}
    call = dict(family='gdn_convolution', variant='persistent', qkv='qkv', state='state', weight='weight', result='result',
                source_offset='token_index * 16 + shard_coord1 * 8 + local_channels',
                result_offset='token_index * 16 + shard_coord1 * 8 + local_channels',
                weight_offset=f'(shard_coord1 * 8 + local_channels) * {history+1}', weight_kernel_stride=1,
                round_products=round_products, round_before_activation=round_activation, current_first=current_first,
                tokens=tokens, local_capacity=9, active_channels='8', global_channel='shard_coord1 * 8 + local_channels',
                history=history, writer_active='shard_coord0 == 0')
    lib = library(tmp_path, render(call, types, mesh=(2,2)), types, mesh=(2,2))
    rng = np.random.default_rng(516)
    qkv = bf16(rng.normal(0, 0.4, (tokens, channels)))
    weight = bf16(rng.normal(0, 0.3, (channels, history+1)))
    old = bf16(rng.normal(0, 0.1, (2, history, 8)))
    state = old.copy()
    actual = np.full((tokens, channels), 0x7fc0, np.uint16)
    run(lib, qkv, state, weight, actual)
    reference_state = f32(old).transpose(0,2,1).reshape(channels, history).copy()
    expected = []
    for token in f32(qkv):
        current = token * f32(weight)[:, history]
        if round_products: current = f32(bf16(current))
        acc = current.copy() if current_first else np.zeros(channels, np.float32)
        for index in range(history):
            product = reference_state[:, index] * f32(weight)[:, index]
            if round_products: product = f32(bf16(product))
            acc += product
        if not current_first: acc += current
        if round_activation: acc = f32(bf16(acc))
        expected.append(bf16(acc / (1 + np.exp(-acc))))
        if history:
            reference_state[:, :-1] = reference_state[:, 1:]
            reference_state[:, -1] = token
    np.testing.assert_allclose(f32(actual), f32(np.stack(expected)), rtol=0.008, atol=1e-5)
    expected_state = bf16(reference_state.reshape(2,8,history).transpose(0,2,1))
    np.testing.assert_array_equal(state, expected_state)


@pytest.mark.parametrize('encoding', [None, 'IQ4_XS', 'Q4_K'])
@pytest.mark.parametrize('round_qk,round_beta,round_core,add_norm,tiled', [(False,True,False,False,False), (True,False,True,True,True)])
def test_recurrent_global_state_and_cross_owner_norm_over_repeated_tokens(tmp_path, encoding, round_qk, round_beta, round_core, add_norm, tiled):
    key_heads, value_heads, key_dim, value_dim, projection_dim = 2, 4, 8, 8, 256
    values = value_heads * value_dim
    types = {name: typ for name, typ in [
        ('qkv','ftt::bfloat16_t'), ('z','ftt::bfloat16_t'), ('projection_input','ftt::bfloat16_t'),
        ('b_weight','unsigned char' if encoding else 'ftt::bfloat16_t'), ('a_weight','unsigned char' if encoding else 'ftt::bfloat16_t'),
        ('a_log','float'), ('dt_bias','ftt::bfloat16_t'), ('norm_weight','ftt::bfloat16_t'), ('state','float'), ('scratch','float'), ('result','ftt::bfloat16_t')
    ]}
    call = dict(family='gdn_recurrent', variant='persistent', **{name:name for name in types if name != 'scratch'}, core_scratch='scratch',
                z_offset='shard_coord1 * 4 + local_values', result_offset='shard_coord1 * 4 + local_values',
                local_capacity=5, active_values='4', global_value='shard_coord1 * 4 + local_values',
                value_heads=value_heads, key_heads=key_heads, key_dim=key_dim, value_dim=value_dim,
                projection_dim=projection_dim, repeats=2, epsilon='1e-5', qk_norm_epsilon='1e-12',
                query_scale=repr(key_dim**-0.5), qk_norm_add=add_norm, round_normalized_qk=round_qk,
                round_beta=round_beta, round_core=round_core, gate_math_mode='accurate', accurate_norm_silu=True,
                writer_active='shard_coord0 == 0', b_weight_encoding=encoding, a_weight_encoding=encoding,
                tiled_value_heads=tiled)
    # Each head spans two owners; every replica exists but only one writes.
    mesh = (2, 8)
    lib = library(tmp_path, render(call, types, mesh), types, mesh, phases=('kernel__core','kernel__norm'))
    rng = np.random.default_rng(973)
    if encoding:
        bw, bd = packed_rows(encoding, rows=value_heads, blocks=1)
        aw, ad = packed_rows(encoding, rows=value_heads, blocks=1)
        aw = aw[::-1].copy(); ad = ad[::-1].copy()
    else:
        bw = bf16(rng.normal(0, 0.03, (value_heads, projection_dim))); bd = f32(bw)
        aw = bf16(rng.normal(0, 0.03, (value_heads, projection_dim))); ad = f32(aw)
    a_log = np.array([-1, -0.5, 0, 0.5], np.float32)
    bias = bf16([-1, 0, 1, 21])  # exercise softplus threshold branch
    norm = bf16(rng.normal(1, 0.1, value_dim))
    state = rng.normal(0, 0.03, (values, key_dim)).astype(np.float32)
    reference_state = state.copy()
    scratch = np.full(values, np.nan, np.float32)
    actual = np.full(values, 0x7fc0, np.uint16)
    for epoch in range(3):
        qkv = bf16(rng.normal(0, 0.2, 2*key_heads*key_dim + values))
        if epoch == 0: qkv[:key_dim] = 0  # clamp/add zero-vector norm must stay finite
        z = bf16(rng.normal(0, 0.3, values))
        source = bf16(rng.normal(0, 0.1, projection_dim))
        run(lib, qkv, z, source, bw, aw, a_log, bias, norm, state, scratch, actual)
        beta_proj = f32(bf16(bd @ f32(source)))
        a_proj = f32(bf16(ad @ f32(source)))
        beta = 1 / (1 + np.exp(-beta_proj))
        if round_beta: beta = f32(bf16(beta))
        di = a_proj + f32(bias)
        softplus = np.where(di > 20, di, np.log1p(np.exp(di)))
        decay = np.exp(-np.exp(a_log) * softplus)
        q = f32(qkv[:key_heads*key_dim]).reshape(key_heads,key_dim)
        k = f32(qkv[key_heads*key_dim:2*key_heads*key_dim]).reshape(key_heads,key_dim)
        def normalized(x):
            squared = np.sum(x*x, axis=1, dtype=np.float32)
            divisor = np.sqrt(squared + 1e-12) if add_norm else np.maximum(np.sqrt(squared), 1e-12)
            result = x / divisor[:,None]
            return f32(bf16(result)) if round_qk else result
        q, k = normalized(q), normalized(k)
        mapping = np.arange(value_heads) % key_heads if tiled else np.arange(value_heads) // 2
        q = np.repeat(q[mapping], value_dim, axis=0) * np.float32(key_dim**-0.5)
        k = np.repeat(k[mapping], value_dim, axis=0)
        dec = np.repeat(decay, value_dim)
        recalled = np.sum((reference_state * dec[:,None]) * k, axis=1, dtype=np.float32)
        target = f32(qkv[2*key_heads*key_dim:])
        delta = (target - recalled) * np.repeat(beta, value_dim)
        reference_state = reference_state * dec[:,None] + k * delta[:,None]
        core = np.sum(reference_state * q, axis=1, dtype=np.float32)
        if round_core: core = f32(bf16(core))
        inverse = 1/np.sqrt(np.mean(core.reshape(value_heads,value_dim)**2, axis=1) + np.float32(1e-5))
        gate = f32(z)/(1+np.exp(-f32(z)))
        expected = ((core * np.repeat(inverse,value_dim)) * np.tile(f32(norm),value_heads)) * gate
        np.testing.assert_allclose(state, reference_state, rtol=4e-5, atol=2e-6)
        np.testing.assert_allclose(scratch, core, rtol=4e-5, atol=2e-6)
        np.testing.assert_array_equal(actual, bf16(expected))
