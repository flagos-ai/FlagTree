#ifndef CRT_TARGET_TX81_DEF_H
#define CRT_TARGET_TX81_DEF_H

#define CONFIG_NO_PLATFORM_HOOK_H
#include "instr_adapter.h"
#include "instr_adapter_plat.h"
#include "instr_def.h"
#include "instr_operator.h"
#include "common_func.h"

typedef enum {
  UNKNOWN = 0,
  SPM = 1,
  DDR = 2,
} MemorySpace;

// Neural engine activate mode
typedef enum {
  None = 0,
  ENRelu = 1,
  ENLeakRelu = 2,
} ActFuncMode;
typedef union {
  int32_t i;
  uint32_t u;
  float f;
} tmp_32suf;

// fw define in common_base.h
typedef union {
  int16_t i16;
  float fp32;
  uint8_t data[4]; // 按字节访问
} triton_hybrid_value;

#endif // CRT_TARGET_TX81_DEF_H
