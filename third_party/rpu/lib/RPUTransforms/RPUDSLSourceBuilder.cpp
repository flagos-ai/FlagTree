#include "RPUDSLSourceBuilder.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdlib>

namespace mlir {
namespace rpu {

std::string buildRPUDSLProgram(llvm::StringRef kernelName, llvm::StringRef args,
                               llvm::StringRef body) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << "#include \"rpu_tile.h\"\n"
     << "using namespace rpu;\n"
     << "__rprog__ void " << kernelName << "(" << args << ") {\n"
     << "    Context ctx;\n"
     << body << "\n"
     << "}\n";
  return os.str();
}

std::string buildAddBody(int64_t n, int64_t logicalN, int64_t out, int64_t lhs,
                         int64_t rhs, bool masked) {
  auto ptrExpr = [](int64_t argIndex, int64_t vecOffset) {
    if (vecOffset == 0)
      return llvm::formatv("arg{0}", argIndex).str();
    return llvm::formatv("arg{0} + {1}", argIndex, vecOffset * 16).str();
  };

  auto emitAddChunk = [&](llvm::raw_ostream &os, int64_t chunkN,
                          int64_t vecOffset, llvm::StringRef indent) {
    int64_t chunkNVec = chunkN;
    if (const char *env = std::getenv("RPU_BOARD_LOAD_CONTIG_NVEC");
        env && env[0] == '1' && env[1] == '\0') {
      chunkNVec = (chunkN + 255) / 256;
      if (chunkNVec < 1)
        chunkNVec = 1;
    }
    std::string suffix;
    if (vecOffset != 0)
      suffix = llvm::formatv("_{0}", vecOffset).str();
    os << indent << "auto lhs" << suffix << " = ctx.load_contig<" << chunkNVec
       << ">(" << ptrExpr(lhs, vecOffset) << ");\n";
    os << indent << "auto rhs" << suffix << " = ctx.load_contig<" << chunkNVec
       << ">(" << ptrExpr(rhs, vecOffset) << ");\n";
    os << indent << "auto result" << suffix << " = lhs" << suffix << " + rhs"
       << suffix << ";\n";
    os << indent << "ctx.store_contig<" << chunkNVec << ">("
       << ptrExpr(out, vecOffset) << ", result" << suffix << ");\n";
  };

  std::string body;
  llvm::raw_string_ostream os(body);
  if (masked) {
    os << llvm::formatv("    auto lhs_tensor = rpu::make_tensor<half, 2, "
                        "rpu::MemScope::Local>(arg{0}, rpu::make_shape(1, "
                        "{1}), rpu::make_stride({2}, 1));\n"
                        "    auto rhs_tensor = rpu::make_tensor<half, 2, "
                        "rpu::MemScope::Local>(arg{3}, rpu::make_shape(1, "
                        "{1}), rpu::make_stride({2}, 1));\n"
                        "    auto out_tensor = rpu::make_tensor<half, 2, "
                        "rpu::MemScope::Local>(arg{4}, rpu::make_shape(1, "
                        "{1}), rpu::make_stride({2}, 1));\n"
                        "    auto lhs = static_cast<rpu::Tile<half, 16, "
                        "{2}>>(ctx.load<half, 16, {2}>(rpu::local_tile<16, "
                        "{2}>(lhs_tensor, rpu::make_coord(0, 0))));\n"
                        "    auto rhs = static_cast<rpu::Tile<half, 16, "
                        "{2}>>(ctx.load<half, 16, {2}>(rpu::local_tile<16, "
                        "{2}>(rhs_tensor, rpu::make_coord(0, 0))));\n"
                        "    auto result = lhs + rhs;\n"
                        "    ctx.store(rpu::local_tile<16, {2}>(out_tensor, "
                        "rpu::make_coord(0, 0)), result);",
                        lhs, logicalN, n, rhs, out);
    return os.str();
  }

  // Opt-in board ABI threshold; mirror of RPUExecutableEmitter::emitAddBody.
  int64_t contigThreshold = 128;
  if (const char *env = std::getenv("RPU_BOARD_LOAD_CONTIG_NVEC");
      env && env[0] == '1' && env[1] == '\0') {
    contigThreshold = 256;
  }
  if (n <= contigThreshold) {
    emitAddChunk(os, n, 0, "    ");
    return os.str();
  }

  int64_t totalRows = n * 16;
  os << llvm::formatv("    auto lhs_tensor = rpu::make_tensor<half, 2, "
                      "rpu::MemScope::Local>(arg{0}, rpu::make_shape({1}, 16), "
                      "rpu::make_stride(16, 1));\n"
                      "    auto rhs_tensor = rpu::make_tensor<half, 2, "
                      "rpu::MemScope::Local>(arg{2}, rpu::make_shape({1}, 16), "
                      "rpu::make_stride(16, 1));\n"
                      "    auto out_tensor = rpu::make_tensor<half, 2, "
                      "rpu::MemScope::Local>(arg{3}, rpu::make_shape({1}, 16), "
                      "rpu::make_stride(16, 1));\n",
                      lhs, totalRows, rhs, out);
  constexpr int64_t kMaxRowsPerFrame = 1024;
  for (int64_t rowOffset = 0; rowOffset < totalRows;
       rowOffset += kMaxRowsPerFrame) {
    int64_t chunkRows =
        std::min<int64_t>(kMaxRowsPerFrame, totalRows - rowOffset);
    os << "    {\n";
    os << "        auto frame = ctx.tile_frame();\n";
    os << llvm::formatv("        auto lhs = static_cast<rpu::Tile<half, {0}, "
                        "16>>(ctx.load<half, {0}, 16>(rpu::local_tile<{0}, "
                        "16>(lhs_tensor, rpu::make_coord({1}, 0))));\n"
                        "        auto rhs = static_cast<rpu::Tile<half, {0}, "
                        "16>>(ctx.load<half, {0}, 16>(rpu::local_tile<{0}, "
                        "16>(rhs_tensor, rpu::make_coord({1}, 0))));\n"
                        "        auto result = lhs + rhs;\n"
                        "        ctx.store(rpu::local_tile<{0}, "
                        "16>(out_tensor, rpu::make_coord({1}, 0)), result);\n",
                        chunkRows, rowOffset);
    os << "    }\n";
  }
  return os.str();
}

std::string buildGemmBody(int64_t m, int64_t n, int64_t k, int64_t out,
                          int64_t lhs, int64_t rhs) {
  std::string body;
  llvm::raw_string_ostream os(body);
  os << "    rpu::Array<half, 2> lhs{arg" << lhs << ", " << m << ", " << k
     << "};\n"
     << "    rpu::Array<half, 2> rhs{arg" << rhs << ", " << n << ", " << k
     << "};\n"
     << "    rpu::Array<half, 2> out{arg" << out << ", " << m << ", " << n
     << "};\n"
     << "    auto acc = ctx.zeros<half, " << m << ", " << n << ">();\n"
     << "    auto a = ctx.load<half, " << m << ", " << k
     << ">(lhs, rpu::IndexList{0, 0});\n"
     << "    auto b_storage = ctx.load<half, " << n << ", " << k
     << ">(rhs, rpu::IndexList{0, 0});\n"
     << "    rpu::LayoutTile<half, " << k << ", " << n
     << ", rpu::layout::physical_b> b{b_storage.offset, b_storage.top};\n"
     << "    ctx.mma<" << m << ", " << k << ", " << n << ">(a, b, acc);\n"
     << "    ctx.store<half, " << m << ", " << n
     << ">(out, rpu::IndexList{0, 0}, acc);";
  return os.str();
}

std::string buildSoftmaxBody(int64_t n, int64_t input, int64_t out) {
  int64_t nvec = n / 16;
  return llvm::formatv("    auto x = ctx.load_contig<{0}>(arg{1});\n"
                       "    auto m = ctx.reduce_max_all(x);\n"
                       "    auto shifted = x - m;\n"
                       "    auto e = rpu::exp(shifted);\n"
                       "    auto s = ctx.reduce_sum_all(e);\n"
                       "    auto inv_s = rpu::reciprocal(s);\n"
                       "    auto y = e * inv_s;\n"
                       "    ctx.store_contig<{0}>(arg{2}, y);",
                       nvec, input, out)
      .str();
}

std::string buildSqrtBody(int64_t n, int64_t input, int64_t out) {
  int64_t nvec = n / 16;
  return llvm::formatv("    auto x = ctx.load_contig<{0}>(arg{1});\n"
                       "    auto y = rpu::sqrt(x);\n"
                       "    ctx.store_contig<{0}>(arg{2}, y);",
                       nvec, input, out)
      .str();
}

std::string buildReduceSumAllBody(int64_t n, int64_t input, int64_t out) {
  int64_t nvec = n / 16;
  // Reduce to scalar, broadcast back via ctx.full<half, NVEC>(s) so we can
  // use store_contig; harness reads element[0] from the output region.
  return llvm::formatv("    auto x = ctx.load_contig<{0}>(arg{1});\n"
                       "    auto s = ctx.reduce_sum_all(x);\n"
                       "    auto y = ctx.full<half, {0}>(s);\n"
                       "    ctx.store_contig<{0}>(arg{2}, y);",
                       nvec, input, out)
      .str();
}

std::string buildConvKxKBody(int64_t kernelSize, int64_t m, int64_t inChannels,
                             int64_t outChannels, int64_t inputWidth,
                             int64_t input, int64_t weight, int64_t out) {
  int64_t inputRows = (kernelSize - 1) * inputWidth + (kernelSize - 1) + m;
  int64_t weightRows = kernelSize * kernelSize * inChannels;

  std::string body;
  llvm::raw_string_ostream os(body);
  os << "    rpu::Array<half, 2> x_arr{arg" << input << ", " << inputRows
     << ", " << inChannels << "};\n"
     << "    rpu::Array<half, 2> w_arr{arg" << weight << ", " << weightRows
     << ", " << outChannels << "};\n"
     << "    rpu::Array<half, 2> out_arr{arg" << out << ", " << m << ", "
     << outChannels << "};\n"
     << "    auto acc = ctx.zeros<half, " << m << ", " << outChannels
     << ">();\n";

  for (int64_t ky = 0; ky < kernelSize; ++ky) {
    for (int64_t kx = 0; kx < kernelSize; ++kx) {
      int64_t kernelIndex = ky * kernelSize + kx;
      int64_t xRow = ky * inputWidth + kx;
      int64_t wRow = kernelIndex * inChannels;
      os << "    auto x_" << ky << "_" << kx << " = ctx.load<half, " << m
         << ", " << inChannels << ">(x_arr, rpu::IndexList{" << xRow
         << ", 0});\n"
         << "    auto w_" << ky << "_" << kx << " = ctx.load<half, "
         << inChannels << ", " << outChannels << ">(w_arr, rpu::IndexList{"
         << wRow << ", 0});\n"
         << "    ctx.mma<" << m << ", " << inChannels << ", " << outChannels
         << ">(x_" << ky << "_" << kx << ", w_" << ky << "_" << kx
         << ", acc);\n";
    }
  }
  os << "    ctx.store<half, " << m << ", " << outChannels
     << ">(out_arr, rpu::IndexList{0, 0}, acc);";
  return os.str();
}

std::string buildResNetBlockBody(int64_t m, int64_t channels, int64_t hidden,
                                 int64_t out, int64_t x, int64_t w1,
                                 int64_t w2) {
  std::string body;
  llvm::raw_string_ostream os(body);
  os << "    rpu::Array<half, 2> x_arr{arg" << x << ", " << m << ", "
     << channels << "};\n"
     << "    rpu::Array<half, 2> w1_arr{arg" << w1 << ", " << channels << ", "
     << hidden << "};\n"
     << "    rpu::Array<half, 2> w2_arr{arg" << w2 << ", " << hidden << ", "
     << channels << "};\n"
     << "    rpu::Array<half, 2> out_arr{arg" << out << ", " << m << ", "
     << channels << "};\n"
     << "    auto x = ctx.load<half, " << m << ", " << channels
     << ">(x_arr, rpu::IndexList{0, 0});\n";

  if (hidden == 32) {
    os << "    auto w1_0 = ctx.load<half, " << channels
       << ", 16>(w1_arr, rpu::IndexList{0, 0});\n"
       << "    auto conv1_0 = ctx.zeros<half, " << m << ", 16>();\n"
       << "    ctx.mma<" << m << ", " << channels
       << ", 16>(x, w1_0, conv1_0);\n"
       << "    auto zero1_0 = ctx.zeros<half, " << m << ", 16>();\n"
       << "    auto relu1_0 = rpu::max_binop(conv1_0, zero1_0);\n"
       << "    auto w1_1 = ctx.load<half, " << channels
       << ", 16>(w1_arr, rpu::IndexList{0, 16});\n"
       << "    auto conv1_1 = ctx.zeros<half, " << m << ", 16>();\n"
       << "    ctx.mma<" << m << ", " << channels
       << ", 16>(x, w1_1, conv1_1);\n"
       << "    auto zero1_1 = ctx.zeros<half, " << m << ", 16>();\n"
       << "    auto relu1_1 = rpu::max_binop(conv1_1, zero1_1);\n"
       << "    auto w2_0 = ctx.load<half, 16, " << channels
       << ">(w2_arr, rpu::IndexList{0, 0});\n"
       << "    auto conv2 = ctx.zeros<half, " << m << ", " << channels
       << ">();\n"
       << "    ctx.mma<" << m << ", 16, " << channels
       << ">(relu1_0, w2_0, conv2);\n"
       << "    auto w2_1 = ctx.load<half, 16, " << channels
       << ">(w2_arr, rpu::IndexList{16, 0});\n"
       << "    ctx.mma<" << m << ", 16, " << channels
       << ">(relu1_1, w2_1, conv2);\n"
       << "    auto residual = conv2 + x;\n"
       << "    auto zero2 = ctx.zeros<half, " << m << ", " << channels
       << ">();\n"
       << "    auto out_tile = rpu::max_binop(residual, zero2);\n"
       << "    ctx.store<half, " << m << ", " << channels
       << ">(out_arr, rpu::IndexList{0, 0}, out_tile);";
    return os.str();
  }

  os << "    auto w1 = ctx.load<half, " << channels << ", " << hidden
     << ">(w1_arr, rpu::IndexList{0, 0});\n"
     << "    auto conv1 = ctx.zeros<half, " << m << ", " << hidden << ">();\n"
     << "    ctx.mma<" << m << ", " << channels << ", " << hidden
     << ">(x, w1, conv1);\n"
     << "    auto zero1 = ctx.zeros<half, " << m << ", " << hidden << ">();\n"
     << "    auto relu1 = rpu::max_binop(conv1, zero1);\n"
     << "    auto w2 = ctx.load<half, " << hidden << ", " << channels
     << ">(w2_arr, rpu::IndexList{0, 0});\n"
     << "    auto conv2 = ctx.zeros<half, " << m << ", " << channels << ">();\n"
     << "    ctx.mma<" << m << ", " << hidden << ", " << channels
     << ">(relu1, w2, conv2);\n"
     << "    auto residual = conv2 + x;\n"
     << "    auto zero2 = ctx.zeros<half, " << m << ", " << channels << ">();\n"
     << "    auto out_tile = rpu::max_binop(residual, zero2);\n"
     << "    ctx.store<half, " << m << ", " << channels
     << ">(out_arr, rpu::IndexList{0, 0}, out_tile);";
  return os.str();
}

std::string buildResNet50BottleneckBody(int64_t kernelSize, int64_t m,
                                        int64_t channels, int64_t bottleneck,
                                        int64_t inputWidth, int64_t out,
                                        int64_t input, int64_t w1, int64_t w2,
                                        int64_t w3) {
  int64_t inputRows = (kernelSize - 1) * inputWidth + (kernelSize - 1) + m;
  int64_t w2Rows = kernelSize * kernelSize * bottleneck;

  std::string body;
  llvm::raw_string_ostream os(body);
  os << "    rpu::Array<half, 2> x_arr{arg" << input << ", " << inputRows
     << ", " << channels << "};\n"
     << "    rpu::Array<half, 2> w1_arr{arg" << w1 << ", " << channels << ", "
     << bottleneck << "};\n"
     << "    rpu::Array<half, 2> w2_arr{arg" << w2 << ", " << w2Rows << ", "
     << bottleneck << "};\n"
     << "    rpu::Array<half, 2> w3_arr{arg" << w3 << ", " << bottleneck << ", "
     << channels << "};\n"
     << "    rpu::Array<half, 2> out_arr{arg" << out << ", " << m << ", "
     << channels << "};\n"
     << "    auto x_skip = ctx.load<half, " << m << ", " << channels
     << ">(x_arr, rpu::IndexList{0, 0});\n";

  if (bottleneck == 32) {
    os << "    auto w1_0 = ctx.load<half, " << channels
       << ", 16>(w1_arr, rpu::IndexList{0, 0});\n"
       << "    auto w1_1 = ctx.load<half, " << channels
       << ", 16>(w1_arr, rpu::IndexList{0, 16});\n"
       << "    auto conv2_acc_0 = ctx.zeros<half, " << m << ", 16>();\n"
       << "    auto conv2_acc_1 = ctx.zeros<half, " << m << ", 16>();\n";

    for (int64_t ky = 0; ky < kernelSize; ++ky) {
      for (int64_t kx = 0; kx < kernelSize; ++kx) {
        int64_t kernelIndex = ky * kernelSize + kx;
        int64_t xRow = ky * inputWidth + kx;
        int64_t w2Row = kernelIndex * bottleneck;
        os << "    auto x_" << ky << "_" << kx << " = ctx.load<half, " << m
           << ", " << channels << ">(x_arr, rpu::IndexList{" << xRow
           << ", 0});\n"
           << "    auto conv1_" << ky << "_" << kx << "_0 = ctx.zeros<half, "
           << m << ", 16>();\n"
           << "    ctx.mma<" << m << ", " << channels << ", 16>(x_" << ky << "_"
           << kx << ", w1_0, conv1_" << ky << "_" << kx << "_0);\n"
           << "    auto zero1_" << ky << "_" << kx << "_0 = ctx.zeros<half, "
           << m << ", 16>();\n"
           << "    auto relu1_" << ky << "_" << kx
           << "_0 = rpu::max_binop(conv1_" << ky << "_" << kx << "_0, zero1_"
           << ky << "_" << kx << "_0);\n"
           << "    auto conv1_" << ky << "_" << kx << "_1 = ctx.zeros<half, "
           << m << ", 16>();\n"
           << "    ctx.mma<" << m << ", " << channels << ", 16>(x_" << ky << "_"
           << kx << ", w1_1, conv1_" << ky << "_" << kx << "_1);\n"
           << "    auto zero1_" << ky << "_" << kx << "_1 = ctx.zeros<half, "
           << m << ", 16>();\n"
           << "    auto relu1_" << ky << "_" << kx
           << "_1 = rpu::max_binop(conv1_" << ky << "_" << kx << "_1, zero1_"
           << ky << "_" << kx << "_1);\n"
           << "    auto w2_" << ky << "_" << kx
           << "_00 = ctx.load<half, 16, 16>(w2_arr, rpu::IndexList{" << w2Row
           << ", 0});\n"
           << "    ctx.mma<" << m << ", 16, 16>(relu1_" << ky << "_" << kx
           << "_0, w2_" << ky << "_" << kx << "_00, conv2_acc_0);\n"
           << "    auto w2_" << ky << "_" << kx
           << "_10 = ctx.load<half, 16, 16>(w2_arr, rpu::IndexList{"
           << (w2Row + 16) << ", 0});\n"
           << "    ctx.mma<" << m << ", 16, 16>(relu1_" << ky << "_" << kx
           << "_1, w2_" << ky << "_" << kx << "_10, conv2_acc_0);\n"
           << "    auto w2_" << ky << "_" << kx
           << "_01 = ctx.load<half, 16, 16>(w2_arr, rpu::IndexList{" << w2Row
           << ", 16});\n"
           << "    ctx.mma<" << m << ", 16, 16>(relu1_" << ky << "_" << kx
           << "_0, w2_" << ky << "_" << kx << "_01, conv2_acc_1);\n"
           << "    auto w2_" << ky << "_" << kx
           << "_11 = ctx.load<half, 16, 16>(w2_arr, rpu::IndexList{"
           << (w2Row + 16) << ", 16});\n"
           << "    ctx.mma<" << m << ", 16, 16>(relu1_" << ky << "_" << kx
           << "_1, w2_" << ky << "_" << kx << "_11, conv2_acc_1);\n";
      }
    }

    os << "    auto zero2_0 = ctx.zeros<half, " << m << ", 16>();\n"
       << "    auto relu2_0 = rpu::max_binop(conv2_acc_0, zero2_0);\n"
       << "    auto zero2_1 = ctx.zeros<half, " << m << ", 16>();\n"
       << "    auto relu2_1 = rpu::max_binop(conv2_acc_1, zero2_1);\n"
       << "    auto w3_0 = ctx.load<half, 16, " << channels
       << ">(w3_arr, rpu::IndexList{0, 0});\n"
       << "    auto conv3 = ctx.zeros<half, " << m << ", " << channels
       << ">();\n"
       << "    ctx.mma<" << m << ", 16, " << channels
       << ">(relu2_0, w3_0, conv3);\n"
       << "    auto w3_1 = ctx.load<half, 16, " << channels
       << ">(w3_arr, rpu::IndexList{16, 0});\n"
       << "    ctx.mma<" << m << ", 16, " << channels
       << ">(relu2_1, w3_1, conv3);\n"
       << "    auto residual = conv3 + x_skip;\n"
       << "    auto zero3 = ctx.zeros<half, " << m << ", " << channels
       << ">();\n"
       << "    auto out_tile = rpu::max_binop(residual, zero3);\n"
       << "    ctx.store<half, " << m << ", " << channels
       << ">(out_arr, rpu::IndexList{0, 0}, out_tile);";
    return os.str();
  }

  os << "    auto w1 = ctx.load<half, " << channels << ", " << bottleneck
     << ">(w1_arr, rpu::IndexList{0, 0});\n"
     << "    auto conv2_acc = ctx.zeros<half, " << m << ", " << bottleneck
     << ">();\n";

  for (int64_t ky = 0; ky < kernelSize; ++ky) {
    for (int64_t kx = 0; kx < kernelSize; ++kx) {
      int64_t kernelIndex = ky * kernelSize + kx;
      int64_t xRow = ky * inputWidth + kx;
      int64_t w2Row = kernelIndex * bottleneck;
      os << "    auto x_" << ky << "_" << kx << " = ctx.load<half, " << m
         << ", " << channels << ">(x_arr, rpu::IndexList{" << xRow << ", 0});\n"
         << "    auto conv1_" << ky << "_" << kx << " = ctx.zeros<half, " << m
         << ", " << bottleneck << ">();\n"
         << "    ctx.mma<" << m << ", " << channels << ", " << bottleneck
         << ">(x_" << ky << "_" << kx << ", w1, conv1_" << ky << "_" << kx
         << ");\n"
         << "    auto zero1_" << ky << "_" << kx << " = ctx.zeros<half, " << m
         << ", " << bottleneck << ">();\n"
         << "    auto relu1_" << ky << "_" << kx << " = rpu::max_binop(conv1_"
         << ky << "_" << kx << ", zero1_" << ky << "_" << kx << ");\n"
         << "    auto w2_" << ky << "_" << kx << " = ctx.load<half, "
         << bottleneck << ", " << bottleneck << ">(w2_arr, rpu::IndexList{"
         << w2Row << ", 0});\n"
         << "    ctx.mma<" << m << ", " << bottleneck << ", " << bottleneck
         << ">(relu1_" << ky << "_" << kx << ", w2_" << ky << "_" << kx
         << ", conv2_acc);\n";
    }
  }

  os << "    auto zero2 = ctx.zeros<half, " << m << ", " << bottleneck
     << ">();\n"
     << "    auto relu2 = rpu::max_binop(conv2_acc, zero2);\n"
     << "    auto w3 = ctx.load<half, " << bottleneck << ", " << channels
     << ">(w3_arr, rpu::IndexList{0, 0});\n"
     << "    auto conv3 = ctx.zeros<half, " << m << ", " << channels << ">();\n"
     << "    ctx.mma<" << m << ", " << bottleneck << ", " << channels
     << ">(relu2, w3, conv3);\n"
     << "    auto residual = conv3 + x_skip;\n"
     << "    auto zero3 = ctx.zeros<half, " << m << ", " << channels << ">();\n"
     << "    auto out_tile = rpu::max_binop(residual, zero3);\n"
     << "    ctx.store<half, " << m << ", " << channels
     << ">(out_arr, rpu::IndexList{0, 0}, out_tile);";
  return os.str();
}

} // namespace rpu
} // namespace mlir
