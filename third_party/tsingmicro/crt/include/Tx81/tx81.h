//===----------------------- tx81.h ---------------------------*- C -*-----===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef CRT_TARGET_TX81_H
#define CRT_TARGET_TX81_H

#define CONFIG_NO_PLATFORM_HOOK_H
#include "instr_adapter.h"
#include "instr_def.h"
#include "instr_operator.h"
#include "instr_adapter_plat.h"
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include "lib_log.h"

#include "profiler.h"

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

typedef union {
  int16_t i16;
  float fp32;
  uint8_t data[4]; // 按字节访问
} hybrid_value;
#ifdef __cplusplus
extern "C" {
#endif

float set_value2float32(Data_Format fmt, int8_t *value);
hybrid_value set_float2value(Data_Format dtype, float value);

uint32_t get_dtype_size_new(Data_Format fmt);

uint32_t get_cx_align_base_new(uint32_t c, Data_Format fmt);

uint64_t next_power_of_two_64(uint64_t x);

bool is_contiguous(int *shape, int *strides, int elem_bytes);

bool no_reverse_memory_access(int *stride, int rank);

// Copy data byte by byte
void tx81_memcpy(char *srcPtr, char *dstPtr, int *src_shape, int *src_stride,
                 int *dst_shape, int *dst_stride, int rank,
                 uint32_t elem_bytes);

void legalizeMemoryOpAttribute(int *src_shape, int *src_stride, int *dst_shape,
                               int *dst_stride, int rank, uint32_t *elem_bytes,
                               uint32_t *fmt);

// Use in simulation mode, return the spm address mapping
int8_t *get_spm_memory_mapping(uint64_t offset);
// Hardware mode will use add the spmMappingOffset to get the real spm address
// Simulation mode will call get_spm_memory_mapping
int8_t *get_spm_memory_mapping_wrapper(uint64_t offset);

#ifdef USE_SIM_MODE
#else
void atomic_barrier_in();
void atomic_barrier_out();
void RT_ASSERT(bool value);
#endif

#ifdef __cplusplus
}
#endif

#ifdef NO_INTRNISIC_RUN
  #define INTRNISIC_RUN_SWITCH return
#else
  #define INTRNISIC_RUN_SWITCH
#endif

#endif // CRT_TARGET_TX81_H
