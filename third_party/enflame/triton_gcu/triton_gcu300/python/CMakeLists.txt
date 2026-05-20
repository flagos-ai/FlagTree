# -----------------------------------------------------------------------------
# GCU300 Python extension
#
# The core library (lib_triton_gcu300_core.so) is built in cmake/triton_gcu.cmake.
# This file calls the centralized build_triton_python_bindings() function to
# build the per-Python-version pybind11 bindings.
# -----------------------------------------------------------------------------

build_triton_python_bindings(gcu300 ${CMAKE_CURRENT_SOURCE_DIR})
