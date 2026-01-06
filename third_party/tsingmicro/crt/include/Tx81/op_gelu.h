#ifndef CRT_TARGET_GELU_H
#define CRT_TARGET_GELU_H

#include "tx81.h"


hybrid_value get_ptr_value_by_idx(void *addr, size_t idx, Data_Format dtype);

void set_ptr_value_by_idx(void *addr, hybrid_value value, uint64_t idx,
                          Data_Format dtype);
void get_erf_value(void *in_addr, void *out_addr, uint64_t count,
                   Data_Format dtype);
void get_tanh_value(uint64_t *in, uint64_t *imm, uint64_t *out,
                    uint32_t elem_count, uint16_t fmt);
void op_gelu_none(uint64_t *src, uint64_t *dst, uint32_t elem_count,
                  uint16_t fmt);
void op_gelu_tanh(uint64_t *src, uint64_t *imm, uint64_t *dst,
                  uint32_t elem_count, uint16_t fmt);

#endif // OP_GELU_H