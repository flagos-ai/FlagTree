#pragma once

// Ascend, TLE, buffer and Gluon bindings are linked into the same libtriton.
// They must use one TritonOpBuilder definition: different member layouts cause
// ODR violations when a derived builder is constructed or destroyed.
// The Ascend CMake configuration defines __TLE__, which exposes the shared
// pybind11 builder accessor in this header.
#include "../../../../python/src/ir.h"

using namespace mlir;
using namespace triton;
