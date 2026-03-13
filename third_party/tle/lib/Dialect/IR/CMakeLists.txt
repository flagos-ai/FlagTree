add_triton_library(TleDsaIR
  DsaDialect.cpp

  DEPENDS
    TleDsaDialectIncGen
    TleDsaTypesIncGen
    TleDsaOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    TritonIR
)

target_include_directories(TleDsaIR
  PUBLIC
    ${TLE_INCLUDE_SOURCE}
    ${TLE_INCLUDE_BINARY}
)
