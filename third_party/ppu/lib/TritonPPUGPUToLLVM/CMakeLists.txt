if(FLAGTREE_TLE)
    set(_TLE_LIBS
        TritonTLEAnalysis
        TleToLLVM
        TritonTLETransforms
    )
else()
    set(_TLE_LIBS "")
endif()

add_triton_library(TritonPPUGPUToLLVM
    ConvertLayoutOpToLLVM.cpp
    ConvertLayoutOpToLLVM/SharedToDotOperandPPUAIUV1.cpp
    ConvertLayoutOpToLLVM/SharedToDotOperandPPUAIUV2.cpp
    MemoryOpToLLVM.cpp
    DotOpToLLVM/PPUMMAv1.cpp
    DotOpToLLVM/PPUMMAv2.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    TritonGPUToLLVM.cpp
    SPMDOpToLLVM.cpp
    TensorPtrOpsToLLVM.cpp
    TIXAsmFormat.cpp
    Utility.cpp
    AIUUtility.cpp
    Fp4ToFpOpToLLVM.cpp
    TargetInfo.cpp
    Allocation.cpp
    ConvertLibdeviceFuncToPPU.cpp
    # Not gated on FLAGTREE_TLE: gluon creates ttg.warp_specialize without TLE,
    # BarrierUtility is used by the shared-memory barrier in TargetInfo.cpp, and
    # BarrierOpToLLVM.cpp also holds the plain LocalBarrierOp conversion.
    ConvertWarpSpecializeToLLVM.cpp
    BarrierUtility.cpp
    BarrierOpToLLVM.cpp

    DEPENDS
    TritonPPUGPUConversionPassIncGen
    PPUGPUAttrDefsIncGen

    LINK_LIBS PUBLIC
    TritonAnalysis
    TritonGPUToLLVM
    TritonInstrumentToLLVM
    TritonNvidiaGPUIR
    MLIRReconcileUnrealizedCasts
    PPUGPUIR
    MLIRUBToLLVM
    ${_TLE_LIBS}
)
