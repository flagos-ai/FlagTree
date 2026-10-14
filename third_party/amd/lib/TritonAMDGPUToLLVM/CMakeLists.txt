if(FLAGTREE_TLE)
    set(_TLE_LIBS
        TritonTLEAnalysis
        TritonNVIDIAGPUToLLVM
        TleToLLVM
        TritonTLETransforms
    )
else()
    set(_TLE_LIBS "")
endif()

add_triton_library(TritonAMDGPUToLLVM
    AsyncUtility.cpp
    AtomicRMWOpsEmitter.cpp
    AllocateSharedMemory.cpp
    BufferOpsEmitter.cpp
    TensorPtrOpsToLLVM.cpp
    ConvertLayoutOpToLLVM.cpp
    MemoryOpToLLVM.cpp
    MaskedOpsToLLVM.cpp
    DotOpToLLVM/FMA.cpp
    DotOpToLLVM/MFMA.cpp
    DotOpToLLVM/WMMA.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    GCNAsmFormat.cpp
    TritonGPUToLLVM.cpp
    BuiltinFuncToLLVM.cpp
    Utility.cpp
    TargetInfo.cpp
    TargetUtils.cpp
    OptimizeLDSUsage.cpp
    OptimizeLDSUtility.cpp
    SPMDOpToLLVM.cpp
    SchedInstructions.cpp
    UpcastMXFPToLLVM.cpp
    Fp4ToFpOpToLLVM.cpp
    MembarUtility.cpp
    ScalarizePackedFOps.cpp
    TDMUtility.cpp
    BarrierOpToLLVM.cpp
    ../../tle/DistributedBarrierOpToLLVM.cpp

    DEPENDS
    TritonAMDGPUConversionPassIncGen
    LLVMIRIncGen

    LINK_LIBS PUBLIC
    TritonGPUToLLVM
    TritonAMDGPUIR
    LLVMCore
    LLVMPasses
    LLVMSupport
    ${_TLE_LIBS}
)

# The TLE primitive lowering lives under third_party/amd/tle (kept out of the
# upstream backend sources) but compiles into this library under __TLE__; add
# this dir so its "PatternTritonGPUOpToLLVM.h" include resolves.
target_include_directories(TritonAMDGPUToLLVM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
