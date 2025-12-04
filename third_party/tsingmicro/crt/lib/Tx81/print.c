// ===------------------------ print.c ------------------------------------===//

// ===---------------------------------------------------------------------===//

// Enable tx8 kernel printf support

#include "lib_log.h"
#include "tx81.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

void __Print(const char *__restrict fmt, ...) {
  va_list args;
  va_start(args, fmt);

  // FIXME: va_list memory layout is specific to the platform.
#ifndef USE_SIM_MODE
  monitor_write_log(__FILE__, __func__, __LINE__, (char *)fmt, args);
#else
  vprintf(fmt, args);
#endif
  va_end(args);
}
