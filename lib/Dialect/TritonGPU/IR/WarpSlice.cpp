/*
 * Copyright 2018-2020 Philippe Tillet
 * Copyright 2020-2022 OpenAI
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/Support/MathExtras.h"
#include <limits>
#include <map>

namespace mlir::triton::gpu {

std::optional<WarpSlice> inferWarpSlice(RankedTensorType src,
                                        RankedTensorType dst,
                                        ArrayRef<int64_t> offsets) {
  if (!src || !dst || src.getRank() == 0 || src.getRank() != dst.getRank() ||
      src.getRank() != offsets.size() ||
      src.getElementType() != dst.getElementType() ||
      !isa_and_nonnull<DistributedEncodingTrait>(src.getEncoding()) ||
      !isa_and_nonnull<DistributedEncodingTrait>(dst.getEncoding()))
    return std::nullopt;

  for (auto [srcSize, dstSize, offset] :
       llvm::zip(src.getShape(), dst.getShape(), offsets)) {
    if (srcSize <= 0 || dstSize <= 0 ||
        srcSize > std::numeric_limits<int32_t>::max() ||
        !llvm::isPowerOf2_64(srcSize) || !llvm::isPowerOf2_64(dstSize) ||
        dstSize > srcSize || offset < 0 || offset > srcSize - dstSize ||
        offset % dstSize != 0)
      return std::nullopt;
  }

  LinearLayout srcLayout = toLinearLayout(src);
  LinearLayout dstLayout = toLinearLayout(dst);
  auto *ctx = src.getContext();
  StringAttr reg = StringAttr::get(ctx, "register");
  StringAttr lane = StringAttr::get(ctx, "lane");
  StringAttr warp = StringAttr::get(ctx, "warp");
  StringAttr block = StringAttr::get(ctx, "block");
  if (srcLayout.getNumInDims() != 4 || dstLayout.getNumInDims() != 4 ||
      !llvm::equal(srcLayout.getOutDimNames(), dstLayout.getOutDimNames()))
    return std::nullopt;
  for (StringAttr dim : {reg, lane, warp, block})
    if (!srcLayout.hasInDim(dim) || !dstLayout.hasInDim(dim))
      return std::nullopt;

  // An aligned warp range changes only the high bits of the physical warp ID.
  // Compare the bases for every varying thread ID bit. This proves the mapping
  // for all lanes, local warps and blocks without enumerating their product.
  for (StringAttr dim : {lane, warp, block}) {
    unsigned srcBits = srcLayout.getInDimSizeLog2(dim);
    unsigned dstBits = dstLayout.getInDimSizeLog2(dim);
    if ((dim == warp && srcBits < dstBits) ||
        (dim != warp && srcBits != dstBits))
      return std::nullopt;
    for (unsigned bit = 0; bit < dstBits; ++bit)
      if (srcLayout.getBasis(dim, bit) != dstLayout.getBasis(dim, bit))
        return std::nullopt;
  }

  auto applyDim = [](const LinearLayout &layout, StringAttr dim,
                     unsigned index) {
    std::vector<int32_t> coords(layout.getNumOutDims(), 0);
    for (unsigned bit = 0; index; ++bit, index >>= 1) {
      if (!(index & 1))
        continue;
      for (auto [coord, basis] : llvm::zip(coords, layout.getBasis(dim, bit)))
        coord ^= basis;
    }
    return coords;
  };

  // Register replicas may share a coordinate: choosing one such register is
  // safe because they hold the same logical element in the same thread.
  std::map<std::vector<int32_t>, unsigned> srcRegisters;
  unsigned numSrcRegisters = srcLayout.getInDimSize(reg);
  unsigned numDstRegisters = dstLayout.getInDimSize(reg);
  for (unsigned i = 0; i < numSrcRegisters; ++i)
    srcRegisters.try_emplace(applyDim(srcLayout, reg, i), i);

  unsigned srcWarps = srcLayout.getInDimSize(warp);
  unsigned dstWarps = dstLayout.getInDimSize(warp);
  std::optional<WarpSlice> result;
  for (unsigned startWarp = 0; startWarp < srcWarps; startWarp += dstWarps) {
    auto warpOrigin = applyDim(srcLayout, warp, startWarp);
    SmallVector<unsigned> registers;
    for (unsigned i = 0; i < numDstRegisters; ++i) {
      auto coords = applyDim(dstLayout, reg, i);
      // Tile alignment makes offset + dstCoord equal offset XOR dstCoord.
      // The remaining lane/warp/block terms already match by the basis proof.
      for (auto [coord, offset, origin] :
           llvm::zip(coords, offsets, warpOrigin))
        coord ^= offset ^ origin;
      auto it = srcRegisters.find(coords);
      if (it == srcRegisters.end())
        break;
      registers.push_back(it->second);
    }
    if (registers.size() != numDstRegisters)
      continue;
    if (result)
      return std::nullopt;
    result = WarpSlice{startWarp, dstWarps, std::move(registers)};
  }
  return result;
}

std::optional<WarpSlice> composeWarpSlices(const WarpSlice &parent,
                                           const WarpSlice &child) {
  if (child.startWarp + child.numWarps > parent.numWarps)
    return std::nullopt;

  WarpSlice composed;
  composed.startWarp = parent.startWarp + child.startWarp;
  composed.numWarps = child.numWarps;
  composed.localRegisters = child.registers;
  composed.registers.reserve(child.registers.size());
  for (unsigned reg : child.registers) {
    if (reg >= parent.registers.size())
      return std::nullopt;
    composed.registers.push_back(parent.registers[reg]);
  }
  return composed;
}

} // namespace mlir::triton::gpu
