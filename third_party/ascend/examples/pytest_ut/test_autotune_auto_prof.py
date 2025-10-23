# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import os
import shutil
import itertools

import pytest
import triton
import triton.language as tl
import torch
import torch_npu


_AUTO_PROF_DIR1 = "./TEST_AUTO_PROF1"
_AUTO_PROF_DIR2 = "./TEST_AUTO_PROF2"


def get_autotune_config():
    configs = []
    block_size_list = [1024, 2048]
    
    multibuffer_list = [False]
    for combo in itertools.product(
        block_size_list,
        multibuffer_list,
    ):
        (
            block_size,
            multibuffer,
        ) = combo

        configs.append(
            triton.Config(
                {
                    "BLOCK_SIZE": block_size,
                },
                multibuffer=multibuffer,
            )
        )

    return configs


@triton.autotune(
    configs=get_autotune_config(),
    key=["n_elements"],
    auto_profile_dir=_AUTO_PROF_DIR1,  # auto profile the best configuration and store the result
)
@triton.jit
def add_kernel_1(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
    # NOTE: `constexpr` so it can be used as a shape value.
):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


# ASCEND-affinity-aware autotune
@triton.autotune(
    configs=[],
    key={"x": "n_elements"},
    split_params={"x": "BLOCK_SIZE"},
    tiling_params={},
    low_dims=["x"],
    persistent_reduction=False,
    dual_reduction=False,
    auto_profile_dir=_AUTO_PROF_DIR2,
)
@triton.jit
def add_kernel_2(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
    # NOTE: `constexpr` so it can be used as a shape value.
):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


@pytest.fixture(autouse=True, scope="session")
def cleanup_prof_dirs():
    # setup: ensure directories don't exist before test
    if os.path.exists(_AUTO_PROF_DIR1):
        shutil.rmtree(_AUTO_PROF_DIR1)
    if os.path.exists(_AUTO_PROF_DIR2):
        shutil.rmtree(_AUTO_PROF_DIR2)

    yield

    # teardown: clean up directories after test
    if os.path.exists(_AUTO_PROF_DIR1):
        shutil.rmtree(_AUTO_PROF_DIR1)
    if os.path.exists(_AUTO_PROF_DIR2):
        shutil.rmtree(_AUTO_PROF_DIR2)


@pytest.mark.parametrize(
    "size, fn, prof_dir",
    [
        (98432, add_kernel_1, _AUTO_PROF_DIR1),
        (98432, add_kernel_2, _AUTO_PROF_DIR2),
    ],
)
def test(size, fn, prof_dir):
    x = torch.rand(size, device="npu")
    y = torch.rand(size, device="npu")

    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    fn[grid](x, y, output, n_elements)

    assert os.path.exists(prof_dir), f"Profiling directory {prof_dir} was not created!"

    prof_files = os.listdir(prof_dir)
    assert len(prof_files) > 0, f"No profiling files found in {prof_dir}"