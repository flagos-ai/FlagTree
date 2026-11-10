# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Exact byte transport and loader selection; CPU tests do not qualify CUDA."""
from contextlib import nullcontext
import copy
from math import prod
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import ArtifactError
from triton.flagmega.runtime import source_rdata as loader


def geometry(shape=(2, 3, 4), *, axis=0, source=0, src=13, dst=7):
    n = prod(shape)
    program = dict(buffer='virtual', shape=list(shape), concat_shape=list(shape),
                   axis=axis, transforms=[], ranges=[[dst, n]],
                   operands=[dict(source=source, offset=src, extent=shape[axis], nbytes=n)])
    return program, [(dst, list(shape), [])], [dict(nbytes=src+n+5)] * (source+1), dst+n+11


@pytest.mark.parametrize('shape,axis', [((4,), 0), ((2, 3, 4), 0), ((2, 3, 4), 1),
                                       ((2, 3, 4), 2), ((3, 8, 2), 1)])
def test_direct_plan_matches_byte_oracle(shape, axis):
    p, layouts, sources, size = geometry(shape, axis=axis)
    assert loader._contiguous_source_copy(p, layouts, sources, size) == (0, 13, 7, prod(shape))
    child, address, valid = loader.byte_addresses(p, shape, [], 0, prod(shape))
    assert np.all(child == 0) and np.all(valid)
    np.testing.assert_array_equal(address, np.arange(prod(shape)))


def test_direct_plan_keeps_wide_integer_offsets():
    p, layouts, sources, size = geometry((7,), source=1, src=2**34+17, dst=2**35+9)
    assert loader._contiguous_source_copy(p, layouts, sources, size) == (1, 2**34+17, 2**35+9, 7)


@pytest.mark.parametrize('field', ['concat', 'component'])
def test_boolean_extent_does_not_alias_integer_one(field):
    p, layouts, sources, size = geometry((1, 3, 4))
    if field == 'concat': p['concat_shape'][0] = True
    else: layouts[0][1][0] = True
    assert loader._contiguous_source_copy(p, layouts, sources, size) is None


@pytest.mark.parametrize('change', [
    'operands', 'transpose', 'identity_transform', 'split', 'cyclic', 'active_tail',
    'components', 'component_shape', 'concat_shape', 'range_offset', 'range_length',
    'source_index', 'source_negative', 'source_bool', 'source_end', 'negative_offset',
    'float_offset', 'destination_end', 'destination_negative', 'nbytes', 'extent',
    'axis', 'axis_bool', 'dimension_bool', 'empty_shape',
])
def test_noncontiguous_or_invalid_geometry_never_gets_a_direct_copy(change):
    p, layouts, sources, size = geometry()
    if change == 'operands': p['operands'].append(copy.deepcopy(p['operands'][0]))
    elif change == 'transpose': p['transforms'] = [dict(shape=[2, 3, 4], source_strides=[4, 8, 1])]
    elif change == 'identity_transform': p['transforms'] = [dict(shape=[2, 3, 4], source_strides=[12, 4, 1])]
    elif change == 'split': layouts[0][2].append((2, [(0, 0, 2, 2)]))
    elif change == 'cyclic': layouts[0][2].append((2, [(0, 1, 2, 2)]))
    elif change == 'active_tail': layouts[0][2].append((1, []))
    elif change == 'components': layouts.append(copy.deepcopy(layouts[0]))
    elif change == 'component_shape': layouts[0][1][:] = [3, 2, 4]
    elif change == 'concat_shape': p['concat_shape'] = [3, 2, 4]
    elif change == 'range_offset': p['ranges'][0][0] += 1
    elif change == 'range_length': p['ranges'][0][1] -= 1
    elif change == 'source_index': p['operands'][0]['source'] = 1
    elif change == 'source_negative': p['operands'][0]['source'] = -1
    elif change == 'source_bool': p['operands'][0]['source'] = False
    elif change == 'source_end': sources[0]['nbytes'] = 13+23
    elif change == 'negative_offset': p['operands'][0]['offset'] = -1
    elif change == 'float_offset': p['operands'][0]['offset'] = 13.0
    elif change == 'destination_end': size = 7+23
    elif change == 'destination_negative': layouts[0] = (-1, layouts[0][1], [])
    elif change == 'nbytes': p['operands'][0]['nbytes'] -= 1
    elif change == 'extent': p['operands'][0]['extent'] -= 1
    elif change == 'axis': p['axis'] = 3
    elif change == 'axis_bool': p['axis'] = False
    elif change == 'dimension_bool': p['shape'][0] = True
    elif change == 'empty_shape': p['shape'] = []
    assert loader._contiguous_source_copy(p, layouts, sources, size) is None


def test_real_file_copy_exact_tail_and_sentinels(tmp_path, monkeypatch):
    path = tmp_path/'source.bin'; raw = bytes(range(96)); path.write_bytes(raw)
    host = torch.from_file(str(path), shared=False, size=96, dtype=torch.uint8)
    actual = torch.full((80,), 229, dtype=torch.uint8)
    counts = []
    class Destination:
        def narrow(self, dim, begin, count):
            counts.append((begin, count))
            target = actual.narrow(dim, begin, count)
            class Slice:
                def copy_(self, source, *, non_blocking):
                    assert non_blocking is False
                    return target.copy_(source)
            return Slice()
    monkeypatch.setattr(loader, 'SOURCE_COPY_CHUNK_BYTES', 8)
    loader._copy_source_bytes(Destination(), host, 13, 7, 29)
    assert counts == [(7, 8), (15, 8), (23, 8), (31, 5)]
    assert actual[7:36].tolist() == list(raw[13:42])
    assert torch.all(actual[:7] == 229) and torch.all(actual[36:] == 229)
    assert path.read_bytes() == raw


class _Device:
    type = 'cuda'
    def __eq__(self, other):
        return isinstance(other, _Device)


class _CPUPool:
    """CPU bytes behind the CUDA branch's Tensor protocol, never a CUDA tensor."""
    device = _Device()
    def __init__(self, value): self.value = value
    @property
    def dtype(self): return self.value.dtype
    @property
    def shape(self): return self.value.shape
    def is_contiguous(self): return self.value.is_contiguous()
    def narrow(self, dim, start, count): return _CPUPool(self.value.narrow(dim, start, count))
    def copy_(self, other, *, non_blocking=False):
        assert non_blocking is False
        self.value.copy_(other)
        return self
    def zero_(self): self.value.zero_(); return self


@pytest.fixture
def simulated_cuda_loader(tmp_path, monkeypatch):
    """Exercise real loader dispatch with CPU files/tensors; no driver import."""
    raw = bytes(range(96)); (tmp_path/'source.bin').write_bytes(raw)
    (tmp_path/'second.bin').write_bytes(bytes(reversed(range(96))))
    physical = bytes([211] * 80); (tmp_path/'physical.bin').write_bytes(physical)
    direct, layout, _, _ = geometry((29,), src=13, dst=7)
    direct['buffer'] = 'a'
    second = copy.deepcopy(direct); second.update(buffer='b', ranges=[[40, 29]])
    index = dict(segment_nbytes=80, image='physical.bin', nbytes=80,
        entries=[dict(buffer=x) for x in ['a', 'b', 'physical']],
        sources=[dict(path='source.bin', nbytes=96), dict(path='second.bin', nbytes=96)],
        programs=[direct, second], segment_chunks=[[dict(offset=0, nbytes=7), dict(offset=70, nbytes=4, zero=True)]])
    layouts = dict(a=layout, b=[(40, [29], [])], physical=[(0, [4], [])])
    monkeypatch.setattr(fm, 'verify_buffer_plan', lambda _: SimpleNamespace(buffer_map={k:k for k in layouts}))
    monkeypatch.setattr(loader, 'component_layouts', lambda name, _: iter(layouts[name]))
    opened = []
    def from_file(path, **kwargs):
        opened.append(Path(path).name)
        assert kwargs['shared'] is False and kwargs['dtype'] == torch.uint8
        return torch.from_file(path, **kwargs)
    class TorchFacade(SimpleNamespace):
        def __getattr__(self, name): return getattr(torch, name)
    fake = TorchFacade(device=lambda _: _Device(), Tensor=_CPUPool, uint8=torch.uint8,
        int64=torch.int64, uint64=torch.uint64, from_file=from_file,
        zeros=lambda n, **kwargs: _CPUPool(torch.zeros(n, dtype=kwargs['dtype'])),
        tensor=lambda data, **kwargs: torch.tensor(data, dtype=kwargs['dtype']),
        cuda=SimpleNamespace(device=lambda _: nullcontext(), synchronize=lambda _: None))
    monkeypatch.setitem(sys.modules, 'torch', fake)
    monkeypatch.setitem(sys.modules, 'triton.flagmega.runtime.source_rdata_kernel',
                        SimpleNamespace(gather_source_bytes=object()))
    def unavailable(*args, **kwargs):
        raise ArtifactError('GPU cannot register readonly source pages: driver status 801.')
    monkeypatch.setattr(loader, '_RegisteredFile', unavailable)
    return tmp_path, index, layouts, opened


def test_loader_direct_does_not_register(simulated_cuda_loader):
    root, index, _, opened = simulated_cuda_loader
    actual = loader.load_source_rdata(root, index, object(), chip=0, device='cuda').value
    assert actual[:7].tolist() == [211]*7
    assert actual[7:36].tolist() == list(range(13, 42))
    assert actual[40:69].tolist() == list(range(13, 42))
    assert torch.all(actual[36:40] == 0) and torch.all(actual[69:] == 0)
    assert opened.count('source.bin') == 1  # One file mapping, repeated output legal.


@pytest.mark.parametrize('selected', [None, ['b'], ['physical'], ['b', 'physical'], []])
def test_selected_shared_pool_preserves_unselected_and_padding(simulated_cuda_loader, selected):
    root, index, _, opened = simulated_cuda_loader
    pool = _CPUPool(torch.full((80,), 229, dtype=torch.uint8))
    result = loader.load_source_rdata(root, index, object(), chip=0, device='cuda', pool=pool, buffer_ids=selected)
    expected = torch.full((80,), 229, dtype=torch.uint8)
    wanted = set(['a', 'b', 'physical'] if selected is None else selected)
    if 'a' in wanted: expected[7:36] = torch.arange(13, 42, dtype=torch.uint8)
    if 'b' in wanted: expected[40:69] = torch.arange(13, 42, dtype=torch.uint8)
    if 'physical' in wanted: expected[:4] = 211
    assert result is pool and torch.equal(result.value, expected)
    assert 'source.bin' not in opened if not wanted & {'a', 'b'} else opened.count('source.bin') == 1


def test_selected_fresh_pool_and_distinct_source(simulated_cuda_loader):
    root, index, _, opened = simulated_cuda_loader
    index['programs'][1]['operands'][0]['source'] = 1
    result = loader.load_source_rdata(root, index, object(), chip=0, device='cuda', buffer_ids=['b']).value
    expected = torch.zeros(80, dtype=torch.uint8); expected[40:69] = torch.arange(82, 53, -1, dtype=torch.uint8)
    assert torch.equal(result, expected) and 'source.bin' not in opened and opened.count('second.bin') == 1


def test_generic_registration_error_is_not_caught(simulated_cuda_loader):
    root, index, _, opened = simulated_cuda_loader
    index['programs'][1]['transforms'] = [dict(shape=[29], source_strides=[1])]
    with pytest.raises(ArtifactError, match='driver status 801'):
        loader.load_source_rdata(root, index, object(), chip=0, device='cuda')
    assert opened.count('source.bin') == 1  # First direct output preceded generic failure.


def test_direct_copy_error_propagates(simulated_cuda_loader, monkeypatch):
    root, index, _, _ = simulated_cuda_loader
    def fail(*args, **kwargs): raise RuntimeError('copy failed')
    monkeypatch.setattr(loader, '_copy_source_bytes', fail)
    with pytest.raises(RuntimeError, match='copy failed'):
        loader.load_source_rdata(root, index, object(), chip=0, device='cuda')
