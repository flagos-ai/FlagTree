add_triton_library(TritonInstrumentIR
  Dialect.cpp
  FunctionBuilder.cpp
  Ops.cpp
  Utility.cpp

  DEPENDS
    TritonInstrumentTableGen
    TritonGPUTableGen
    TritonGPUAttrDefsIncGen
    TritonGPUCTAAttrIncGen
    TritonGPUTypeInterfacesIncGen
    TritonGPUOpInterfacesIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    TritonIR
    TritonGPUIR
)
