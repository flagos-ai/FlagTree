#ifndef CRT_TARGET_REDUCE_MUL_H
#define CRT_TARGET_REDUCE_MUL_H

#define CONFIG_NO_PLATFORM_HOOK_H
#include "instr_adapter.h"

void op_reduce_mul_impl(void *in, void *out, Data_Shape shape,
                        uint32_t reduce_dim, Data_Format fmt);

#endif // CRT_TARGET_REDUCE_MUL_H
