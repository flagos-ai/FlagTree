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
