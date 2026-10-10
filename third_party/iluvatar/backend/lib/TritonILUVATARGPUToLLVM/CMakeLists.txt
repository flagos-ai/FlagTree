if(FLAGTREE_ILUVATAR_TLE)
    set(_ILUVATAR_TLE_LIBS IluvatarTleToLLVM TleIR)
    set(_ILUVATAR_TLE_BARRIER_SOURCES DistributedBarrierOpToLLVM.cpp)
else()
    set(_ILUVATAR_TLE_LIBS "")
    set(_ILUVATAR_TLE_BARRIER_SOURCES "")
endif()

add_triton_library(TritonILUVATARGPUToLLVM
    AtomicRMWOpsEmitter.cpp
    BarrierOpToLLVM.cpp
    ${_ILUVATAR_TLE_BARRIER_SOURCES}
    ConvertWarpSpecializeToLLVM.cpp
    MaskedOpsToLLVM.cpp
    DotOpToLLVM/FMA.cpp
    DotOpToLLVM/TCU.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    MembarUtility.cpp
    TritonGPUToLLVM.cpp
    Utility.cpp
    TargetInfo.cpp
    TargetUtils.cpp
    SPMDOpToLLVM.cpp

    DEPENDS
    TritonILUVATARGPUConversionPassIncGen
    LLVMIRIncGen

    LINK_LIBS PUBLIC
    TritonGPUToLLVM
    TritonILUVATARGPUIR
    TritonILUVATARUtils
    ${_ILUVATAR_TLE_LIBS}
    MLIRUBToLLVM
    MLIRReconcileUnrealizedCasts
    LLVMCore
    LLVMPasses
    LLVMSupport
)
