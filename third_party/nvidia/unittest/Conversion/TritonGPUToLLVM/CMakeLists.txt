add_triton_ut(
  NAME TestPtxAsmFormat
  SRCS PTXAsmFormatTest.cpp
  LIBS
    TritonGPUToLLVM
    TritonNVIDIAGPUToLLVM
    NVGPUIR MLIRUBToLLVM
)

add_dependencies(TestPtxAsmFormat
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUCTAAttrIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen
)
