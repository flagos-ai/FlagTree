// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
#pragma once

#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::triton::tle {
// This resource has no user pointer operand. Its sole implementation resolves
// the configured hidden entry profile ABI or the verified forwarded argument.
// It cannot denote a model tensor, shared memdesc, or arbitrary user allocation.
struct ProfileScratchResource
    : SideEffects::Resource::Base<ProfileScratchResource> {
  StringRef getName() final { return "tle.profile_scratch"; }
};
} // namespace mlir::triton::tle
