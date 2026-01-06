// ===------------------------ assert.c
// ------------------------------------===//

// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.

// ===---------------------------------------------------------------------===//

// Enable tx8 kernel assert support

#include "tx81.h"
#include <stdarg.h>
#include <stdio.h>

void __Assert(const char *message, ...) {
  INTRNISIC_RUN_SWITCH;
  va_list args;
  va_start(args, message);

  char *file = va_arg(args, char *);
  int line = va_arg(args, int);
  int col = va_arg(args, int);
  int pidX = va_arg(args, int);
  int pidY = va_arg(args, int);
  int pidZ = va_arg(args, int);

#ifdef USE_SIM_MODE
  printf("%s(line %d, col %d)::tile (%d, %d, %d): %s\n", file, line, col, pidX,
         pidY, pidZ, message);
  assert(0);
#else
  tsm_ep_log(__FILE__, __func__, __LINE__, KCORE_LOG_ERROR,
             "%s(line %d, col %d)::tile (%d, %d, %d): %s\n", file, line, col,
             pidX, pidY, pidZ, message);
  RT_ASSERT(0);
#endif
  va_end(args);
}