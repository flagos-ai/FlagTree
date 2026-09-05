add_triton_library(TleDsaToCore
  DsaToCore.cpp

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransformUtils
    MLIRRewrite
    TleDsaIR
    TritonIR
)

target_include_directories(TleDsaToCore
  PUBLIC
    ${TLE_INCLUDE_SOURCE}
    ${TLE_INCLUDE_BINARY}
)
