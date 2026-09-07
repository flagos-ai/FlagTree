add_triton_library(TritonInstrumentTransforms
  ConcurrencySanitizer.cpp

  DEPENDS
  TritonInstrumentTransformsIncGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUCTAAttrIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRTransforms
  MLIRTransformUtils
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
  TritonToTritonGPU
  TritonInstrumentIR
  MLIRTransformUtils
)
