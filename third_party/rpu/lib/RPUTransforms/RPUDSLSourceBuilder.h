#pragma once

#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <string>

namespace mlir {
namespace rpu {

std::string buildRPUDSLProgram(llvm::StringRef kernelName, llvm::StringRef args,
                               llvm::StringRef body);
std::string buildAddBody(int64_t n, int64_t logicalN, int64_t out, int64_t lhs,
                         int64_t rhs, bool masked);
std::string buildGemmBody(int64_t m, int64_t n, int64_t k, int64_t out,
                          int64_t lhs, int64_t rhs);
std::string buildSoftmaxBody(int64_t n, int64_t input, int64_t out);
std::string buildSqrtBody(int64_t n, int64_t input, int64_t out);
std::string buildReduceSumAllBody(int64_t n, int64_t input, int64_t out);
std::string buildConvKxKBody(int64_t kernelSize, int64_t m, int64_t inChannels,
                             int64_t outChannels, int64_t inputWidth,
                             int64_t input, int64_t weight, int64_t out);
std::string buildResNetBlockBody(int64_t m, int64_t channels, int64_t hidden,
                                 int64_t out, int64_t x, int64_t w1,
                                 int64_t w2);
std::string buildResNet50BottleneckBody(int64_t kernelSize, int64_t m,
                                        int64_t channels, int64_t bottleneck,
                                        int64_t inputWidth, int64_t out,
                                        int64_t input, int64_t w1, int64_t w2,
                                        int64_t w3);

} // namespace rpu
} // namespace mlir
