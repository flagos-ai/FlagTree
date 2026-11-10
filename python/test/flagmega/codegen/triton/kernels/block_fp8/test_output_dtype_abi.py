# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""An explicit projection result dtype is independent of activation storage."""

from math import prod

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.kernel_call_renderers import prepare_kernel_calls
from triton.flagmega.errors import CodegenError
from python.test.flagmega.codegen.triton.kernels.block_fp8.test_semantic_scale_geometry import _descriptor, _parameter


def raw_projection(encoding, lanes=()):
    dtype = fm.vector_type('float32', lanes) if lanes else fm.DType.FLOAT32
    width = {'IQ4_XS': 136, 'Q4_K': 144, None: 256}[encoding]
    weight_dtype = fm.DType.UINT8 if encoding else fm.DType.FLOAT8_E4M3FN
    attrs = {'weight_block_n': 128, 'weight_block_k': 128,
             'output_data_type': 'float32', 'output_lanes': lanes}
    if encoding:
        attrs.update(weight_encoding=encoding)
    return {'call': 'logits', 'family': 'block_fp8', 'variant': 'simt',
            'execution_kind': 'local_shard', 'parameters': {'tile_n': 16},
            'semantic_attrs': attrs,
            'inputs': (_parameter('value', _descriptor('value', fm.DType.BFLOAT16, (1, 256))),
                       _parameter('weight', _descriptor('weight', weight_dtype, (16, width))),
                       _parameter('weight_scale', _descriptor('scale', fm.DType.FLOAT32, (1, 2)))),
            'outputs': (_parameter('result', _descriptor('result', dtype, (1, 16 // prod(lanes)))),),
            'workspaces': ()}


@pytest.mark.parametrize('encoding', [None, 'IQ4_XS', 'Q4_K'])
@pytest.mark.parametrize('lanes', [(), (4,)])
def test_explicit_fp32_result_with_bf16_activation(encoding, lanes):
    call, = prepare_kernel_calls((raw_projection(encoding, lanes),), function_name='main')
    assert call['local_n_capacity'] == 16


@pytest.mark.parametrize('mismatch', ['dtype', 'lanes'])
def test_result_abi_still_rejects_wrong_declared_dtype_or_lanes(mismatch):
    raw = raw_projection('IQ4_XS', (4,))
    raw['semantic_attrs']['output_data_type' if mismatch == 'dtype' else 'output_lanes'] = (
        'bfloat16' if mismatch == 'dtype' else ())
    with pytest.raises(CodegenError, match='output ABI'):
        prepare_kernel_calls((raw,), function_name='main')
