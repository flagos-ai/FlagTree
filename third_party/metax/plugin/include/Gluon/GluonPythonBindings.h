#ifndef TRITON_METAX_GLUON_PYTHON_BINDINGS_H
#define TRITON_METAX_GLUON_PYTHON_BINDINGS_H

namespace pybind11 {
class module_;
}

namespace mlir::triton::gpu::metax::gluon {

void registerPassBindings(pybind11::module_ &module);
void registerCandidateBundleBinding(pybind11::module_ &module);

} // namespace mlir::triton::gpu::metax::gluon

#endif // TRITON_METAX_GLUON_PYTHON_BINDINGS_H
