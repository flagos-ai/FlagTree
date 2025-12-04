//===----------------------- tx81.h ---------------------------*- C -*-----===//
//
//
//===----------------------------------------------------------------------===//
#ifndef CRT_TARGET_TX81_H
#define CRT_TARGET_TX81_H

#define CONFIG_NO_PLATFORM_HOOK_H
#include "instr_adapter.h"
#include "instr_def.h"
#include "instr_operator.h"
#include "lib_log.h"
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

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

#ifdef __cplusplus
extern "C" {
#endif

float set_value2float32(Data_Format fmt, int8_t *value);

bool is_contiguous(int *shape, int *strides, int elem_bytes);

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

#ifdef __cplusplus
}
#endif

#endif // CRT_TARGET_TX81_H
