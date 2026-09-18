// MThreads owns a backend-local TritonGPUTransforms target and generated pass
// headers, but the RLC algorithm is shared with the core implementation. Keep
// this translation-unit shim so the vendor build retains that ownership while
// compiling the portable phases against the MThreads dialect definitions.
#define __FLAGTREE_MTHREADS_RLC__
#include "../../../../../../lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp"
