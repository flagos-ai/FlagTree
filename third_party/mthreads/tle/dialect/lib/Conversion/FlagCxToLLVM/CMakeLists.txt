# MUSA FlagCX / distributed TLE→LLVM lowering only.
# Avoids the full shared TleToLLVM library (ExclusiveCumsum, NVGPU, NVIDIA TargetInfo).

set(_TLE_TO_LLVM_DIR
    ${CMAKE_SOURCE_DIR}/third_party/tle/dialect/lib/Conversion/TleToLLVM)

set(FLAGCX_TO_LLVM_SRC
    ${_TLE_TO_LLVM_DIR}/FlagCxOpToLLVM/FlagCxBarrierOpToLLVM.cpp
    ${_TLE_TO_LLVM_DIR}/FlagCxOpToLLVM/GetLocalRankOpToLLVM.cpp
    ${_TLE_TO_LLVM_DIR}/FlagCxOpToLLVM/FlagCxSignalOpToLLVM.cpp
    ${_TLE_TO_LLVM_DIR}/FlagCxOpToLLVM/GetWorldRankOpToLLVM.cpp
    ${_TLE_TO_LLVM_DIR}/FlagCxOpToLLVM/FlagCxOpToLLVM.cpp
    ${_TLE_TO_LLVM_DIR}/GetDeviceIdToFlagCX.cpp
    ${_TLE_TO_LLVM_DIR}/DistributedBarrierOpToLLVM.cpp
    ${_TLE_TO_LLVM_DIR}/LocalPointersOpToLLVM.cpp
    ${_TLE_TO_LLVM_DIR}/SignalOpToLLVM.cpp
    ${_TLE_TO_LLVM_DIR}/NodeTransferOpToLLVM.cpp
)

add_triton_library(FlagCxToLLVM
    ${FLAGCX_TO_LLVM_SRC}

    DEPENDS
    TleTableGen
    TritonGPUTableGen
    TritonGPUAttrDefsIncGen
    ${TLE_TRITON_GPU_ATTR_DEPENDS}
    TritonGPUTypeInterfacesIncGen
    TritonGPUOpInterfacesIncGen
    TleTools

    LINK_LIBS PUBLIC
    TleTools
)

target_compile_definitions(FlagCxToLLVM PRIVATE TRITON_MTHREADS_BACKEND=1)
# FlagcxUtils lives in TleTools; MUSA FlagCX bitcode uses generic (AS0) pointers.
if(TARGET TleTools)
  target_compile_definitions(TleTools PRIVATE TRITON_MTHREADS_BACKEND=1)
endif()
