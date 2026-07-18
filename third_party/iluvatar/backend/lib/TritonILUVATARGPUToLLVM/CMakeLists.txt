if(FLAGTREE_ILUVATAR_TLE)
    set(_ILUVATAR_TLE_LIBS IluvatarTleToLLVM)
else()
    set(_ILUVATAR_TLE_LIBS "")
endif()

add_triton_library(TritonILUVATARGPUToLLVM
    AtomicRMWOpsEmitter.cpp
    MaskedOpsToLLVM.cpp
    DotOpToLLVM/FMA.cpp
    DotOpToLLVM/TCU.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    TritonGPUToLLVM.cpp
    Utility.cpp
    TargetInfo.cpp
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
    LLVMCore
    LLVMPasses
    LLVMSupport
)
