//===----------------------- tx81.h ---------------------------*- C -*-----===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef CRT_TARGET_TX81_RUN_H
#define CRT_TARGET_TX81_RUN_H

#include "tx81_def.h"
#include "lib_log.h"
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

#include "lib_log.h"
#include "instr_adapter.h"
#include "instr_operator.h"
#include "riscv.h"


#ifdef __cplusplus
extern "C" {
#endif

#define ClientPtrHeaderBytes  24
#define ClientPtrMagic 0x54445841u

typedef enum {
  DMA_ACT_LOG = 0x1,
  DMA_ACT_CHK = 0x2,
  DMA_ACT_END = 0x4
} DMAAction;

bool is_dma_action_logging(uint32_t action);
bool is_dma_action_checking(uint32_t action);
uint64_t* get_header(const void *ddr_addr);
uint32_t crc32(const uint8_t *data, int len);

float set_value2float32(Data_Format fmt, int8_t *value);
triton_hybrid_value set_float2value(Data_Format dtype, float value);

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

#define TX81_MAX_TENSOR_RANK 8

// Inclusive min_addr and exclusive max_addr for a memory access envelope
// (may include holes when strides are non-contiguous).
typedef struct {
  uintptr_t min_addr;
  uintptr_t max_addr;
} Tx81MemAddrRange;

typedef Tx81MemAddrRange Tx81SrcAddrRange;

Tx81MemAddrRange compute_rdma_src_addr_range(const void *src, int *src_shape,
                                             int *src_stride, int *dst_shape,
                                             int *dst_stride, int rank,
                                             uint32_t elem_bytes, uint32_t fmt);

typedef Tx81MemAddrRange Tx81DstAddrRange;

Tx81MemAddrRange compute_wdma_dst_addr_range(const void *dst, int *src_shape,
                                             int *src_stride, int *dst_shape,
                                             int *dst_stride, int rank,
                                             uint32_t elem_bytes, uint32_t fmt);

uint32_t get_dma_oob_count(void);
uint32_t get_dma_bad_magic_count(void);
void reset_dma_oob_count(void);
void reset_dma_bad_magic_count(void);

bool get_dma_check_abort(void);

// Global counters — incremented by rdma.c/wdma.c on OOB detection
extern uint32_t dma_oob_count;
extern uint32_t dma_bad_magic_count;

// Use in simulation mode, return the spm address mapping
int8_t *get_spm_memory_mapping(uint64_t offset);
// Hardware mode will use add the spmMappingOffset to get the real spm address
// Simulation mode will call get_spm_memory_mapping
int8_t *get_spm_memory_mapping_wrapper(uint64_t offset);

// #ifdef USE_SIM_MODE
// #else
// void atomic_barrier_in();
// void atomic_barrier_out();
// void RT_ASSERT(bool value);
// #endif

void debug_dump_f32_data(uint64_t *in, uint32_t elem_count, char *dump_flag);
void debug_dump_i32_data(uint64_t *in, uint32_t elem_count, char *dump_flag);
void debug_dump_i64_data(uint64_t *in, uint32_t elem_count, char *dump_flag);
void debug_dump_bf16_data(uint64_t *in, uint32_t elem_count, char *dump_flag);
void debug_dump_f16_data(uint64_t *in, uint32_t elem_count, char *dump_flag);


#ifdef __cplusplus
}
#endif

#ifdef ENABLE_NO_INTRINSIC_RUN
#define INTRNISIC_RUN_SWITCH return
#else
#define INTRNISIC_RUN_SWITCH
#endif

#ifdef ENABLE_SYNCHRONOUS_INTRINSIC
#define SYNCHRONOUS_INTRINSIC_SWITCH RcsWaitfinish()
#else
#define SYNCHRONOUS_INTRINSIC_SWITCH
#endif

#ifdef ENABLE_DEBUG_DUMP_DATA
#define DEBUG_DUMP_F32_DATA(in, elem_count, dump_flag) debug_dump_f32_data(in, elem_count, dump_flag)
#define DEBUG_DUMP_F16_DATA(in, elem_count, dump_flag) debug_dump_f16_data(in, elem_count, dump_flag)
#define DEBUG_DUMP_BF16_DATA(in, elem_count, dump_flag) debug_dump_bf16_data(in, elem_count, dump_flag)
#define DEBUG_DUMP_I32_DATA(in, elem_count, dump_flag) debug_dump_i32_data(in, elem_count, dump_flag)
#define DEBUG_DUMP_I64_DATA(in, elem_count, dump_flag) debug_dump_i64_data(in, elem_count, dump_flag)
#else
#define DEBUG_DUMP_F32_DATA(in, elem_count, dump_flag)
#define DEBUG_DUMP_F16_DATA(in, elem_count, dump_flag)
#define DEBUG_DUMP_BF16_DATA(in, elem_count, dump_flag)
#define DEBUG_DUMP_I32_DATA(in, elem_count, dump_flag)
#define DEBUG_DUMP_I64_DATA(in, elem_count, dump_flag)
#endif

#endif // CRT_TARGET_TX81_RUN_H
