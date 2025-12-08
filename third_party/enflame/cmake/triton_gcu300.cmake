
set(triton_${arch}_objs
${third_party_triton_${arch}_fetch_bin}/lib/Conversion/TritonGPUToLLVM/CMakeFiles/TritonGPUToLLVM.dir/AllocateSharedMemory.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Conversion/TritonGPUToLLVM/CMakeFiles/TritonGPUToLLVM.dir/AllocateWarpGroups.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Conversion/TritonGPUToLLVM/CMakeFiles/TritonGPUToLLVM.dir/AssertOpToLLVM.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Conversion/TritonGPUToLLVM/CMakeFiles/TritonGPUToLLVM.dir/ControlFlowOpToLLVM.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Conversion/TritonGPUToLLVM/CMakeFiles/TritonGPUToLLVM.dir/ConvertLayoutOpToLLVM.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Conversion/TritonGPUToLLVM/CMakeFiles/TritonGPUToLLVM.dir/DecomposeUnsupportedConversions.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Conversion/TritonGPUToLLVM/CMakeFiles/TritonGPUToLLVM.dir/ElementwiseOpToLLVM.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Conversion/TritonGPUToLLVM/CMakeFiles/TritonGPUToLLVM.dir/FuncOpToLLVM.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Conversion/TritonGPUToLLVM/CMakeFiles/TritonGPUToLLVM.dir/GatherOpToLLVM.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Conversion/TritonGPUToLLVM/CMakeFiles/TritonGPUToLLVM.dir/GlobalScratchMemoryAllocation.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Conversion/TritonGPUToLLVM/CMakeFiles/TritonGPUToLLVM.dir/HistogramOpToLLVM.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Conversion/TritonGPUToLLVM/CMakeFiles/TritonGPUToLLVM.dir/MakeRangeOpToLLVM.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Conversion/TritonGPUToLLVM/CMakeFiles/TritonGPUToLLVM.dir/MemoryOpToLLVM.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Conversion/TritonGPUToLLVM/CMakeFiles/TritonGPUToLLVM.dir/PrintOpToLLVM.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Conversion/TritonGPUToLLVM/CMakeFiles/TritonGPUToLLVM.dir/ReduceOpToLLVM.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Conversion/TritonGPUToLLVM/CMakeFiles/TritonGPUToLLVM.dir/SPMDOpToLLVM.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Conversion/TritonGPUToLLVM/CMakeFiles/TritonGPUToLLVM.dir/ScanOpToLLVM.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Conversion/TritonGPUToLLVM/CMakeFiles/TritonGPUToLLVM.dir/TypeConverter.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Conversion/TritonGPUToLLVM/CMakeFiles/TritonGPUToLLVM.dir/Utility.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Conversion/TritonGPUToLLVM/CMakeFiles/TritonGPUToLLVM.dir/ViewOpToLLVM.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Conversion/TritonGPUToLLVM/CMakeFiles/TritonGPUToLLVM.dir/DotOpToLLVM/FMA.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Conversion/TritonGPUToLLVM/CMakeFiles/TritonGPUToLLVM.dir/DotOpToLLVM/FMADotUtility.cpp.o

${third_party_triton_${arch}_fetch_bin}/lib/Conversion/TritonToTritonGPU/CMakeFiles/TritonToTritonGPU.dir/TritonGPUConversion.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Conversion/TritonToTritonGPU/CMakeFiles/TritonToTritonGPU.dir/TritonToTritonGPUPass.cpp.o

${third_party_triton_${arch}_fetch_bin}/lib/Analysis/CMakeFiles/TritonAnalysis.dir/Allocation.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Analysis/CMakeFiles/TritonAnalysis.dir/AxisInfo.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Analysis/CMakeFiles/TritonAnalysis.dir/Alias.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Analysis/CMakeFiles/TritonAnalysis.dir/Membar.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Analysis/CMakeFiles/TritonAnalysis.dir/Utility.cpp.o

${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/AccelerateMatmul.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/Coalesce.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/CoalesceAsyncCopy.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/CombineTensorSelectAndIf.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/DecomposeScaledBlocked.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/F32DotTC.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/FuseNestedLoops.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/OptimizeAccumulatorInit.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/OptimizeDotOperands.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/OptimizeThreadLocality.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/PingPong.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/Prefetch.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/ReduceDataDuplication.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/RemoveLayoutConversions.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/ReorderInstructions.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/TaskIdPropagate.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/Utility.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/WSCanonicalization.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/WSCodePartition.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/WSDataPartition.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/WSLowering.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/WSTaskPartition.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/Pipeliner/AssignLatencies.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/Pipeliner/LowerLoops.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/Pipeliner/ModifiedAccMMAPipeline.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/Pipeliner/PipelineExpander.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/Pipeliner/PipeliningUtility.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/Pipeliner/Schedule.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/Pipeliner/ScheduleLoops.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/Pipeliner/SoftwarePipeliner.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/Pipeliner/TC05MMAPipeline.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/Pipeliner/TMAStoresPipeline.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/Pipeliner/TestPipelineAssignLatencies.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/Pipeliner/TestPipelineLowerLoop.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/Pipeliner/TestPipelineScheduleLoop.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/Pipeliner/WGMMAPipeline.cpp.o

${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/IR/CMakeFiles/TritonGPUIR.dir/Ops.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/IR/CMakeFiles/TritonGPUIR.dir/Dialect.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/IR/CMakeFiles/TritonGPUIR.dir/LinearLayoutConversions.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonGPU/IR/CMakeFiles/TritonGPUIR.dir/Types.cpp.o

${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonNvidiaGPU/Transforms/CMakeFiles/TritonNvidiaGPUTransforms.dir/FenceInsertion.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonNvidiaGPU/Transforms/CMakeFiles/TritonNvidiaGPUTransforms.dir/KeepAccInTMem.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonNvidiaGPU/Transforms/CMakeFiles/TritonNvidiaGPUTransforms.dir/MMALowering.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonNvidiaGPU/Transforms/CMakeFiles/TritonNvidiaGPUTransforms.dir/PlanCTA.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonNvidiaGPU/Transforms/CMakeFiles/TritonNvidiaGPUTransforms.dir/PromoteLHSToTMem.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonNvidiaGPU/Transforms/CMakeFiles/TritonNvidiaGPUTransforms.dir/TMALowering.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonNvidiaGPU/Transforms/CMakeFiles/TritonNvidiaGPUTransforms.dir/TensorMemoryAllocation.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonNvidiaGPU/Transforms/CMakeFiles/TritonNvidiaGPUTransforms.dir/Utility.cpp.o

${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonNvidiaGPU/IR/CMakeFiles/TritonNvidiaGPUIR.dir/Ops.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonNvidiaGPU/IR/CMakeFiles/TritonNvidiaGPUIR.dir/Dialect.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/TritonNvidiaGPU/IR/CMakeFiles/TritonNvidiaGPUIR.dir/Types.cpp.o

${third_party_triton_${arch}_fetch_bin}/lib/Dialect/Triton/Transforms/CMakeFiles/TritonTransforms.dir/LoopUnroll.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/Triton/Transforms/CMakeFiles/TritonTransforms.dir/ReorderBroadcast.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/Triton/Transforms/CMakeFiles/TritonTransforms.dir/RewriteTensorPointer.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/Triton/Transforms/CMakeFiles/TritonTransforms.dir/Combine.cpp.o

${third_party_triton_${arch}_fetch_bin}/lib/Dialect/Triton/IR/CMakeFiles/TritonIR.dir/Ops.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/Triton/IR/CMakeFiles/TritonIR.dir/Dialect.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/Triton/IR/CMakeFiles/TritonIR.dir/Types.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/Triton/IR/CMakeFiles/TritonIR.dir/Traits.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Dialect/Triton/IR/CMakeFiles/TritonIR.dir/OpInterfaces.cpp.o

${third_party_triton_${arch}_fetch_bin}/lib/Target/LLVMIR/CMakeFiles/TritonLLVMIR.dir/LLVMDIScope.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Target/LLVMIR/CMakeFiles/TritonLLVMIR.dir/LLVMIRBreakPhiStruct.cpp.o

${third_party_triton_${arch}_fetch_bin}/lib/Tools/CMakeFiles/TritonTools.dir/LinearLayout.cpp.o
${third_party_triton_${arch}_fetch_bin}/lib/Tools/CMakeFiles/TritonTools.dir/LayoutUtils.cpp.o

${third_party_triton_${arch}_fetch_bin}/third_party/f2reduce/CMakeFiles/f2reduce.dir/f2reduce.cpp.o
)
