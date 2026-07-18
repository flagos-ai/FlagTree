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

add_xpu_sdnn_object(TritonSDNNTransforms
  ${CMAKE_CURRENT_SOURCE_DIR}/TritonSDNNConversion.cpp.o
  ${CMAKE_CURRENT_SOURCE_DIR}/Bufferize.cpp.o
  ${CMAKE_CURRENT_SOURCE_DIR}/ConvertType.cpp.o
  ${CMAKE_CURRENT_SOURCE_DIR}/EWActTable.cpp.o
  ${CMAKE_CURRENT_SOURCE_DIR}/Legalize.cpp.o
  ${CMAKE_CURRENT_SOURCE_DIR}/Combine.cpp.o
  ${CMAKE_CURRENT_SOURCE_DIR}/CombinePatterns.cpp.o
  ${CMAKE_CURRENT_SOURCE_DIR}/CombineBefore.cpp.o
  ${CMAKE_CURRENT_SOURCE_DIR}/LoopGrid.cpp.o
  ${CMAKE_CURRENT_SOURCE_DIR}/Pipeline.cpp.o
  ${CMAKE_CURRENT_SOURCE_DIR}/MultipleBuffer.cpp.o
  ${CMAKE_CURRENT_SOURCE_DIR}/KLoopAccModification.cpp.o
  ${CMAKE_CURRENT_SOURCE_DIR}/Utility.cpp.o
  ${CMAKE_CURRENT_SOURCE_DIR}/SchedulePrintOp.cpp.o
  ${CMAKE_CURRENT_SOURCE_DIR}/RemoveDsOp.cpp.o
  ${CMAKE_CURRENT_SOURCE_DIR}/StripAllOp.cpp.o

  LINK_LIBS
  TritonIR
  TritonSDNNIR
  MLIRTransformUtils
  MLIRLinalgTransformOps
  MLIRLinalgTransforms
  MLIRLinalgDialect
  MLIRControlFlowDialect
  MLIRControlFlowTransforms
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRArithDialect
  MLIRArithTransforms
  MLIRArithValueBoundsOpInterfaceImpl
  MLIRAffineDialect
  MLIRAffineTransforms
  MLIRTensorDialect
  MLIRTensorTransforms
  MLIRTensorInferTypeOpInterfaceImpl
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRMemRefDialect
  MLIRMemRefTransforms
)
