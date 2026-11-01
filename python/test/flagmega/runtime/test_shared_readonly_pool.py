# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Independent phases share only identical audited byte-address layouts."""

import numpy as np
import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import pack_rdata, verify_rdata
from triton.flagmega.artifacts.readonly_layout import relocate_readonly
from triton.flagmega.errors import ArtifactError
from triton.flagmega.runtime.shared_readonly import SharedReadonlyPool
from python.test.flagmega.artifacts.test_source_backed_rdata import _fixture


def assets(tmp_path, module, checkpoint):
    root = tmp_path / "assets"
    index = pack_rdata(module, checkpoint, root)
    verify_rdata(root, module=module, segment=0)
    return root, index


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_two_identical_phases_bind_one_exact_readonly_allocation(tmp_path, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    module, checkpoint, _ = _fixture(tmp_path)
    root, index = assets(tmp_path, module, checkpoint)
    size = index["segment_nbytes"]
    pool = SharedReadonlyPool(size, device=device)
    first = pool._bind_verified(root, index, module, chip=0, device=device, expected_nbytes=size)
    expected = first.clone()
    second = pool._bind_verified(root, index, module, chip=0, device=device, expected_nbytes=size)
    assert first.data_ptr() == second.data_ptr()
    torch.testing.assert_close(second, expected, rtol=0, atol=0)


def test_same_checkpoint_with_different_owner_layout_cannot_alias(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    first, checkpoint, _ = _fixture(a)
    second, checkpoint2, _ = _fixture(b, cyclic=True)
    size = max(fm.verify_buffer_plan(first).rdata_bytes, fm.verify_buffer_plan(second).rdata_bytes)
    first = relocate_readonly(first, {v.id: v.offset for v in fm.verify_buffer_plan(first).buffers
                                     if v.storage == "rdata"}, nbytes=size)
    second = relocate_readonly(second, {v.id: v.offset for v in fm.verify_buffer_plan(second).buffers
                                       if v.storage == "rdata"}, nbytes=size)
    root, index = assets(a, first, checkpoint)
    root2, index2 = assets(b, second, checkpoint2)
    assert [s["sha256"] for s in index["sources"]] == [s["sha256"] for s in index2["sources"]]
    size = index["segment_nbytes"]
    assert size == index2["segment_nbytes"]
    pool = SharedReadonlyPool(size, device="cpu")
    value = pool._bind_verified(root, index, first, chip=0, device="cpu", expected_nbytes=size)
    expected = value.clone()
    with pytest.raises(ArtifactError, match="byte proofs"):
        pool._bind_verified(root2, index2, second, chip=0, device="cpu", expected_nbytes=size)
    torch.testing.assert_close(value, expected, rtol=0, atol=0)


def test_relocated_new_phase_preserves_prior_bytes_and_views(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    first, checkpoint, _ = _fixture(a)
    second, checkpoint2, _ = _fixture(b, cyclic=True)
    plan = fm.verify_buffer_plan(first)
    initial_size = plan.rdata_bytes
    size = initial_size + fm.verify_buffer_plan(second).rdata_bytes
    owners = tuple(v for v in plan.buffers if v.storage == "rdata")
    first = relocate_readonly(first, {v.id: v.offset for v in owners}, nbytes=size)
    second = relocate_readonly(second, {v.id: v.offset + initial_size for v in owners}, nbytes=size)
    root, index = assets(a, first, checkpoint)
    root2, index2 = assets(b, second, checkpoint2)
    pool = SharedReadonlyPool(size, device="cpu")
    value = pool._bind_verified(root, index, first, chip=0, device="cpu", expected_nbytes=size)
    prefix = value[:initial_size].clone()
    pool._bind_verified(root2, index2, second, chip=0, device="cpu", expected_nbytes=size)
    torch.testing.assert_close(value[:initial_size], prefix, rtol=0, atol=0)
    from triton.flagmega.runtime.source_rdata import load_source_rdata
    expected = load_source_rdata(root2, index2, second, chip=0, device="cpu")
    torch.testing.assert_close(value[initial_size:], expected[initial_size:], rtol=0, atol=0)
    assert np.count_nonzero(value[initial_size:].numpy())
