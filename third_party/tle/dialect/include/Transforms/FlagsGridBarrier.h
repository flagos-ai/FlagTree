#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
#include <optional>

namespace mlir::triton::tle {

constexpr uint32_t kFlagsGridAlignment = 128;

inline bool isFlagsGridBarrier(Operation *op) {
  auto protocol = op->getAttrOfType<StringAttr>("tle.grid_barrier_protocol");
  return protocol && protocol.getValue() == "flags32";
}

struct FlagsGridLayout {
  uint32_t participants, groups, groupBytes, bytes;
  SmallVector<int32_t> domain, grid;
};

inline FailureOr<FlagsGridLayout> getFlagsGridLayout(Operation *op) {
  auto domain = op->getAttrOfType<DenseI32ArrayAttr>("tle.flags_domain");
  auto grid = op->getAttrOfType<DenseI32ArrayAttr>("tle.flags_launch_grid");
  auto kind = op->getAttrOfType<StringAttr>("group_kind");
  if (!domain || domain.asArrayRef().empty() || !grid || grid.size() != 3 || !kind ||
      (kind.getValue() != "grid" && kind.getValue() != "grid_axis")) {
    op->emitError("flags32 requires static domain and three-dimensional launch grid");
    return failure();
  }
  constexpr uint64_t limit = std::numeric_limits<int32_t>::max();
  auto population = [&](DenseI32ArrayAttr values) -> FailureOr<uint64_t> {
    uint64_t n = 1;
    for (int32_t v : values.asArrayRef()) {
      if (v <= 0 || n > limit / uint64_t(v)) {
        op->emitError("flags32 dimension/product exceeds positive signed i32");
        return failure();
      }
      n *= v;
    }
    return n;
  };
  auto total = population(domain), launched = population(grid);
  if (failed(total) || failed(launched))
    return failure();
  if (*total != *launched) {
    op->emitError("flags32 physical launch and logical domain populations differ");
    return failure();
  }
  uint64_t participants = *total;
  if (kind.getValue() == "grid_axis") {
    auto axes = op->getAttrOfType<DenseI32ArrayAttr>("group_axes");
    auto shape = op->getAttrOfType<DenseI32ArrayAttr>("group_shape");
    auto groupDomain = op->getAttrOfType<DenseI32ArrayAttr>("group_domain_shape");
    if (!axes || axes.asArrayRef().empty() || !shape || axes.size() != shape.size() ||
        groupDomain != domain) {
      op->emitError("flags32 axis descriptor must match the explicit domain");
      return failure();
    }
    participants = 1;
    llvm::SmallSet<int32_t, 8> seen;
    for (auto [axis, extent] : llvm::zip(axes.asArrayRef(), shape.asArrayRef())) {
      if (axis < 0 || axis >= int32_t(domain.size()) || extent <= 0 ||
          !seen.insert(axis).second || domain.asArrayRef()[axis] % extent != 0 ||
          participants > limit / uint64_t(extent)) {
        op->emitError("flags32 has invalid or overflowing axis membership");
        return failure();
      }
      participants *= extent;
    }
  }
  uint64_t groupBytes = llvm::alignTo(participants * 4, uint64_t(kFlagsGridAlignment));
  uint64_t groups = *total / participants;
  if (groupBytes > limit || groups > limit / groupBytes ||
      groups * groupBytes > limit - (kFlagsGridAlignment - 1)) {
    op->emitError("flags32 scratch layout exceeds signed i32 byte offsets");
    return failure();
  }
  return FlagsGridLayout{uint32_t(participants), uint32_t(groups),
                         uint32_t(groupBytes), uint32_t(groups * groupBytes),
                         SmallVector<int32_t>(domain.asArrayRef()),
                         SmallVector<int32_t>(grid.asArrayRef())};
}

inline LogicalResult verifyFlagsGridModule(ModuleOp module) {
  std::optional<FlagsGridLayout> common;
  bool invalid = false;
  module.walk([&](Operation *op) {
    if (invalid || !isFlagsGridBarrier(op))
      return;
    auto layout = getFlagsGridLayout(op);
    if (failed(layout)) {
      invalid = true;
      return;
    }
    if (common && (layout->domain != common->domain || layout->grid != common->grid)) {
      op->emitError("flags32 sites/callees have conflicting launch domains");
      invalid = true;
      return;
    }
    common = *layout;
  });
  if (invalid)
    return failure();
  if (!common)
    return success();
  std::string value;
  llvm::raw_string_ostream out(value);
  out << "{\"schema\":\"tle.flags32/v1\",\"domain\":[";
  llvm::interleaveComma(common->domain, out);
  out << "],\"grid\":[";
  llvm::interleaveComma(common->grid, out);
  out << "]}";
  auto attr = StringAttr::get(module.getContext(), out.str());
  if (module->hasAttr("tle.flags_launch_contract") &&
      module->getAttr("tle.flags_launch_contract") != attr)
    return module.emitError("flags32 has conflicting preexisting launch contract");
  module->setAttr("tle.flags_launch_contract", attr);
  return success();
}

} // namespace mlir::triton::tle
