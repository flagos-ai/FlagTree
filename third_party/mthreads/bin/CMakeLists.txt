# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

add_executable(triton-opt triton-opt.cpp)

target_compile_options(triton-opt PRIVATE -fno-rtti -fno-exceptions)
target_link_libraries(triton-opt PRIVATE
  ${triton_libs}
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
)

mlir_check_all_link_libraries(triton-opt)

add_executable(triton-reduce triton-reduce.cpp)
mlir_check_all_link_libraries(triton-reduce)
target_compile_options(triton-reduce PRIVATE -fno-rtti -fno-exceptions)

target_link_libraries(triton-reduce PRIVATE
  ${triton_libs}
  # MLIR core
  MLIRReduceLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
)

mlir_check_all_link_libraries(triton-reduce)


add_executable(triton-llvm-opt triton-llvm-opt.cpp)
add_dependencies(triton-llvm-opt intrinsics_gen)
target_compile_options(triton-llvm-opt PRIVATE -fno-rtti -fno-exceptions)
target_link_libraries(triton-llvm-opt PRIVATE
  TritonLLVMIR

  LLVMAnalysis
  LLVMCore
  LLVMSupport
  LLVMOption
  LLVMCodeGen
  )
export_executable_symbols_for_plugins(triton-llvm-opt)


add_executable(triton-tensor-layout triton-tensor-layout.cpp)
target_compile_options(triton-tensor-layout PRIVATE -fno-rtti -fno-exceptions)
target_link_libraries(triton-tensor-layout PRIVATE
  ${triton_libs}
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
  )
