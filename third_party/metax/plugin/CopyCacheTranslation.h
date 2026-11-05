// Copyright 2025- FlagOS Contributors
// SPDX-License-Identifier: MIT

#ifndef TRITON_METAX_COPY_CACHE_TRANSLATION_H
#define TRITON_METAX_COPY_CACHE_TRANSLATION_H

#include "mlir/IR/DialectRegistry.h"
#include <string>
#include <utility>

namespace mlir::triton::METAX {
void registerCopyCacheTranslation(DialectRegistry &registry);
// Keep MACA operations on the SDK's existing translation path. Only the pure
// LLVM helper functions are exported with the extension metadata interface.
std::pair<std::string, std::string>
splitCopyCacheHelpers(const std::string &source);
} // namespace mlir::triton::METAX
#endif
