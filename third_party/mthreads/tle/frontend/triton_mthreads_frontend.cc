// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifdef __TLE__

#include "MUSATLE/Frontend/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Frontend marker adapters consume shared TLE markers emitted by Python
// frontend code, such as `tt.memory_space` and `tt.load.async`, before the
// mthreads/MUSA TTGIR pipeline reaches backend-local `musa_tle` dialect
// optimization. They are not `musa_tle` dialect passes.
void init_triton_musa_tle_frontend_passes_ttgpuir(py::module m) {
  ADD_PASS_WRAPPER_0("add_tle_early_assign_memory_space",
                     mlir::createTritonMUSAGPUTLEEarlyAssignMemorySpace);
  ADD_PASS_WRAPPER_0("add_tle_lower_async_load",
                     mlir::createTritonMUSAGPUTLELowerAsyncLoad);
}

#endif // __TLE__
