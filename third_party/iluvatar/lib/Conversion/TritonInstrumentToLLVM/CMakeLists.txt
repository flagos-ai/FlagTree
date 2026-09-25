flagtree_add_iluvatar_library(TritonInstrumentToLLVM
    InstrumentationToLLVM.cpp

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    TritonIR
    TritonGPUIR
    TritonInstrumentIR
    TritonNvidiaGPUIR
    NVGPUIR
)
