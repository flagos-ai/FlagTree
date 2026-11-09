"""CPU proof checks for shared-memory instruction selection, without Packet IR."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


def test_finite_shared_read_plan(tmp_path):
    llvm_config = os.environ.get("LLVM_CONFIG") or shutil.which("llvm-config")
    compiler = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
    if not llvm_config or not compiler:
        pytest.skip("LLVM development libraries and a C++17 compiler required")
    root = Path(__file__).resolve().parents[4]
    source = tmp_path / "finite.cpp"
    source.write_text(r'''
#include "FiniteSharedReadPlan.h"
#include <cassert>
#include <vector>
using mlir::triton::HCU::planFiniteSharedReads;
int main() {
  // Independent hardware gather fixtures, including all lanes and four waves.
  std::vector<unsigned> normal, alt;
  for (unsigned r=0;r<8;++r) for (unsigned w=0;w<4;++w)
    for (unsigned l=0;l<64;++l) {
      normal.push_back(w*4096+r*512+(l&56)*8+(l&7));
      alt.push_back(w*4096+(r&3)*512+(r>>2)+(l&~3u)*8+2*(l&3));
    }
  auto n=planFiniteSharedReads(8,4,16384,normal,true,true,true);
  auto a=planFiniteSharedReads(8,4,16384,alt,true,true,true);
  assert(n && n->matrix && !n->matrixAlt && n->width==8);
  assert(a && a->matrix && a->matrixAlt && a->width==8);
  // A bank-only search used to split the two halves of each MMAC word.
  // This independently constructed row-XOR chart admits both packed and
  // unpacked read groups; the selected groups must retain the packed pairs.
  std::vector<unsigned> rowXor;
  for (unsigned r=0;r<32;++r) for (unsigned w=0;w<4;++w)
    for (unsigned l=0;l<64;++l) {
      unsigned k=(r&3)+((l>>4)&3)*4;
      unsigned n=((r>>2)&1)*16+((r>>3)&1)*32+
                 ((r>>4)&1)*128+(l&15)+(w&1)*64;
      rowXor.push_back(k*256+(n^(8*k)));
    }
  auto packed=planFiniteSharedReads(32,4,4096,rowXor,true,true,true);
  assert(packed && packed->matrix);
  for(unsigned anchor:packed->groups) for(unsigned j=0;j<8;j+=2) {
    unsigned lo=packed->repartition?packed->matrixByAnchor[anchor][j]:anchor+j;
    unsigned hi=packed->repartition?packed->matrixByAnchor[anchor][j+1]:anchor+j+1;
    assert(!(lo&1) && hi==lo+1);
  }
  // A single nonbasis point must defeat an otherwise valid matrix gather.
  auto corrupt=normal;corrupt[(7*4+3)*64+63]^=1;
  auto c=planFiniteSharedReads(8,4,16384,corrupt,true,true,true);
  assert(c && !c->matrix);
  auto outOfBounds=normal;outOfBounds.back()=16384;
  assert(!planFiniteSharedReads(8,4,16384,outOfBounds,true,true,true));
  assert(!planFiniteSharedReads(8,4,16384,
      llvm::ArrayRef<unsigned>(normal).drop_back(),true,true,true));
  auto noMatrix=planFiniteSharedReads(8,4,16384,normal,false,true,true);
  assert(noMatrix && !noMatrix->matrix);
  // Search exhaustion is explicit, cannot be used as proof of a matrix read.
  auto budget=planFiniteSharedReads(8,4,16384,normal,true,true,true,1);
  assert(budget && budget->optimizationBudgetExceeded && !budget->matrix);
  assert(budget->width==1 && budget->groups.size()==8);
}
''')
    def flags(*args):
        return shlex.split(subprocess.check_output([llvm_config, *args], text=True))
    exe = tmp_path / "finite"
    subprocess.run([compiler, *flags("--cxxflags"), "-std=c++17", "-O2",
                    "-UNDEBUG", "-I", str(root / "third_party/hcu/lib/TritonHCUGPUToLLVM"),
                    str(source), "-o", str(exe),
                    *flags("--ldflags", "--libs", "support", "--system-libs")],
                   check=True, capture_output=True)
    subprocess.run([str(exe)], check=True, capture_output=True)
