#===------------------------------------------------------------------------===#
#
# Copyright (c) Triton Project Contributors.
#
#===------------------------------------------------------------------------===#

add_triton_library(AddLLVMDebugInfo
  AddLLVMDebugInfoPass.cpp

  DEPENDS
  AddLLVMDebugInfoTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
