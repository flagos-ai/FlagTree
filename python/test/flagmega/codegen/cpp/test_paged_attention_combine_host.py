# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Run the rendered C++ combine body on the host for every CTA and thread.

The C++ package executes only on an Iluvatar GPU, but a kernel body built from
the FTT scalar helpers compiles with a host C++ compiler. Replacing the two
arch primitives the body uses -- the mesh coordinates and the thread id -- with
test-controlled values lets one host loop play every CTA of the mesh.
"""

import ctypes
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from python.test.flagmega.codegen.triton.kernels.paged_attention_combine.test_vector_result_local_abi import (
    _dim_sharded_source_call, )
import triton.flagmega.codegen.cpp as cpp_codegen
from triton.flagmega.codegen.cpp.templates import CppTemplateRegistry

FTT_INCLUDE = Path(cpp_codegen.__file__).parent / "ftt/include"
BLOCK_THREADS = 1024

requires_host_compiler = pytest.mark.skipif(shutil.which("c++") is None, reason="A host C++ compiler is required")


def _render(gated):
    mesh, call = _dim_sharded_source_call(gated)
    family = "paged_attention_gated_combine" if gated else "paged_attention_combine"
    names = ["max_state", "sum_state", "acc_state", "result"] + (["gate"] if gated else [])
    call = dict(call, family=family, variant="decode", symbol="combine", arguments=names)
    context = dict(render_calls=[call], mesh_hierarchy=list(mesh), distributed_entry=True,
                   block_threads=BLOCK_THREADS,
                   argument_types={name: {"type": "unsigned char", "pointer": True}
                                   for name in names})
    return mesh, names, CppTemplateRegistry().render(f"kernels/{family}/decode.cpp.jinja", context)


def _host_library(tmp_path, mesh, names, body):
    body = re.sub(r"ftt::arch::mesh_shard_id<(\d+)>\([^)]*\)", r"ftt_test_shard[\1]", body)
    body = body.replace("ftt::arch::thread_id()", "ftt_test_thread")
    parameters = ", ".join(f"unsigned char *{name}" for name in names)
    source = f"""#include <cmath>
#include <cstdint>
#include <utility>
// The device toolchain provides these; the host pass only needs them to parse.
struct float4 {{ float x, y, z, w; }};
#include <ftt/ftt.h>
static ftt::dim_t ftt_test_shard[2];
static ftt::dim_t ftt_test_thread;
{body}
extern "C" void run({parameters}) {{
    for (int y = 0; y < {mesh[0]}; ++y)
        for (int x = 0; x < {mesh[1]}; ++x)
            for (int thread = 0; thread < {BLOCK_THREADS}; ++thread) {{
                ftt_test_shard[0] = y;
                ftt_test_shard[1] = x;
                ftt_test_thread = thread;
                combine({", ".join(names)});
            }}
}}
"""
    (tmp_path / "combine.cpp").write_text(source)
    library = tmp_path / "combine.so"
    subprocess.run([
        "c++", "-std=c++17", "-O1", "-shared", "-fPIC", "-w", "-DFTT_ARCH_ILUVATAR", f"-I{FTT_INCLUDE}",
        str(tmp_path / "combine.cpp"), "-o",
        str(library)
    ], check=True)
    return ctypes.CDLL(str(library))


@requires_host_compiler
@pytest.mark.parametrize("gated", [False, True])
def test_cpp_combine_reads_dim_sharded_states_from_their_own_owners(tmp_path, gated):
    # 2x2 mesh, partial axis 0, Dim split block-cyclically over axis 1: the
    # unit-Dim stats live on owner 2 * part, accumulator dim d on 2 * part + d % 2.
    mesh, names, body = _render(gated)
    library = _host_library(tmp_path, mesh, names, body)
    maximum = np.zeros(4, np.float32)
    total = np.ones(4, np.float32)
    accumulator = np.array([[0, 32, 64, 96], [16, 48, 80, 112], [2, 34, 66, 98], [18, 50, 82, 114]], np.float32)
    result = np.full(8, 0xC2C8, np.uint16)  # bf16 -100
    buffers = [maximum, total, accumulator, result]
    if gated:
        buffers.append(np.zeros(8, np.uint16))  # sigmoid(0) halves the output
    library.run(*(buffer.ctypes.data_as(ctypes.c_void_p) for buffer in buffers))
    expected = np.array([1, 17, 33, 49, 65, 81, 97, 113], np.float32) * (0.5 if gated else 1.0)
    np.testing.assert_array_equal((result.astype(np.uint32) << 16).view(np.float32), expected)
