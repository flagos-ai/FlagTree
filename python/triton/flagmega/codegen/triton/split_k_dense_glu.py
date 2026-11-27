# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Physical global-buffer and descriptor bindings for the normal GU island."""
from math import prod
from triton.flagmega.codegen.triton.cooperative_access import require_chip_shared
from triton.flagmega.errors import CodegenError


def _canonical(raw, section, name, dtype, shape):
    from triton.flagmega.codegen.triton.kernel_call_renderers import _buffer, _pointer
    binding = _buffer(raw, section, name)
    require_chip_shared(binding, name, owner='SplitKDenseGlu')
    abi = binding['abi']
    if (abi.get('coordinate_space') != 'canonical_global' or abi['scalar_dtype'] != dtype
            or int(abi.get('scalar_lane_count', 1)) != 1 or tuple(abi.get('logical_shape', ())) != shape
            or tuple(abi.get('scalar_storage_strides', ())) != tuple(prod(shape[i + 1:]) for i in range(len(shape)))):
        raise CodegenError(f'SplitKDenseGlu requires canonical contiguous {dtype} {name}{shape}.')
    return binding, _pointer(binding)


def split_k_dense_glu_projection_call(raw):
    a, source = _canonical(raw, 'inputs', 'value', 'bfloat16', (1, 2048))
    w, weight = _canonical(raw, 'inputs', 'weight', 'bfloat16', (2048, 12288))
    _, result = _canonical(raw, 'outputs', 'result', 'float32', (2, 12288))
    requests = []
    for name, binding, shape, box in (('weight', w, (2048, 12288), (64, 256)), ('value', a, (1, 2048), (1, 64))):
        requests.append({
            'parameter': name + '_descriptor', 'source': binding['runtime_argument'], 'kind': 'single', 'storage':
            'device', 'offset_bytes': int(binding['abi'].get('pool_byte_offset', 0)), 'dtype': 'bfloat16', 'shape':
            shape, 'strides': (shape[1], 1), 'block_shape': box, 'box_shape': box, 'swizzle_mode': 0,
            'source_shape_axes': ((), ()), 'padding': 'zero'
        })
    return {'source': source, 'weight': weight, 'result': result, 'host_tensor_descriptor_requests': tuple(requests)}


def split_k_dense_glu_epilogue_call(raw):
    _, partial = _canonical(raw, 'inputs', 'partial', 'float32', (2, 12288))
    _, result = _canonical(raw, 'outputs', 'result', 'bfloat16', (1, 6144))
    return {'partial': partial, 'result': result}


def split_k_dense_glu_call(raw):
    return (split_k_dense_glu_projection_call(raw)
            if raw['variant'] == 'projection' else split_k_dense_glu_epilogue_call(raw))
