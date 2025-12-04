//===------------------- MagicKernelFuncOps.td ----------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Common abstraction layer for non-instruction driven ML accelerator.
//
// The target glue layer that translates target independent kernel operations
// into NPU like APIs call (There are other jargons such as intrinsic or driver
// functions etc).
//
// The NPU APIs are categories by data type, like traditional compilers, integer
// and floating point function unit are separated, so for every MK(MagicKernel)
// op, it is lowered to 2 MKF(MagicKernelFunc) which are integer version and
// floating point version.
//
//===----------------------------------------------------------------------===//
