add_triton_library(TritonAMDUtils
  CommonUtils.cpp

  DEPENDS
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUCTAAttrIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)
