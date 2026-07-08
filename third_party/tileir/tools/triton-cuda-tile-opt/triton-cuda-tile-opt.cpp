#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "RegisterTritonCudaTileDialects.h"
#include "cuda_tile/Dialect/CudaTile/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registerTritonCudaTileDialects(registry);
  mlir::cuda_tile::registerFuseFMAPass();
  mlir::cuda_tile::registerLoopSplitPass();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Triton-Cuda-Tile test driver\n", registry));
}
