"""Structured packet descriptors preserve their 14-bit byte-stride field."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_structured_stride_encoding_boundaries(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("A C++17 compiler is required for the header-only ABI test")
    include = Path(__file__).resolve().parents[4] / "third_party/hcu/include"
    source = tmp_path / "domain.cpp"
    source.write_text(r'''
#include "Dialect/TritonHCUGPU/IR/WholePacketLayout.h"
#include <cassert>
#include <limits>
using mlir::triton::hcugpu::wholepacket::StructuredDomain;
int main() {
  for (unsigned pitch : {256u, 272u, 512u, 768u, 1024u, 1280u, 2048u, 3072u, 4096u,
                         6144u, 7936u, 8176u}) {
    for (unsigned rows : {0u, 1u, 17u}) {
      auto domain = StructuredDomain::get(pitch, rows);
      assert(domain && domain->remainingRows == rows);
      unsigned mode = domain->mode();
      assert(mode >> 14 == 1u);
      assert((mode & 0x3fffu) == 2u * pitch);
      assert(((mode & 0x3fffu) / 2u) == pitch);
    }
  }
  for (unsigned pitch : {0u, 16u, 255u, 257u, 8191u, 8192u, 8193u,
                         std::numeric_limits<unsigned>::max()})
    assert(!StructuredDomain::get(pitch, 17));
}
''')
    executable = tmp_path / "domain"
    subprocess.run([compiler, "-std=c++17", "-O2", "-UNDEBUG", "-I", str(include),
                    str(source), "-o", str(executable)], check=True, capture_output=True)
    subprocess.run([str(executable)], check=True)
