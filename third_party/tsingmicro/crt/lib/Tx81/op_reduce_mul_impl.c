#include "op_reduce_mul_impl.h"
#include "tx81.h"
#include <assert.h>

void op_reduce_mul_impl(void *in, void *out, Data_Shape shape,
                        uint32_t reduce_dim, Data_Format fmt) {
  TsmArith *arith = TsmNewArith();
  TsmArithInstr arithIns = {I_CGRA, {0}, {0}};

  uint32_t n = shape.n;
  uint32_t h = shape.h;
  uint32_t w = shape.w;
  uint32_t c = shape.c;

  // reduce mul impl, only Cx is supported.

  // get_cx_align_base: 4, 8, 16, 32, 64 (> 32 except int_8) , 128 ( > 64 int_8)
  uint32_t align_val = (c > 4) ? get_cx_align_base_new(c, fmt) : c;
  // uint32_t one_align = get_cx_align_base(1, fmt);
  uint32_t one_align = 1;

  // In triton, the channel dimension is always a power of two.
  uint32_t cx_align = c;
  if (cx_align > align_val && cx_align % align_val != 0) {
    assert(cx_align % align_val == 0);
  } else if (cx_align < align_val) {
    assert(cx_align == next_power_of_two_64(cx_align) &&
           "cx_align should be power of two");
  }
  uint64_t cx_align_mem_size = (uint64_t)cx_align * get_dtype_size_new(fmt);
  uint64_t one_align_mem_size = (uint64_t)one_align * get_dtype_size_new(fmt);

  if (reduce_dim == 0) {

    TsmDataMoveInstr data_move_inst = {I_CGRA,
                                       {
                                           0,
                                       },
                                       {
                                           0,
                                       }};
    // reduce C in NHWC.
    int32_t nhw_cnt = n * h * w;

    if (cx_align == 1) {
      TsmDataMove *data_move = TsmNewDataMove();
      St_StrideIteration src_si = {1, 1, 1, 1, 1, 1};
      St_StrideIteration dst_si = {1, 1, 1, 1, 1, 1};

      data_move->GatherScatter(&data_move_inst, (uint64_t)in, (uint64_t)out,
                               nhw_cnt * cx_align_mem_size, &src_si, &dst_si);

      // Dispatch the command to accelerator
      TsmExecute(&data_move_inst);

      // Destroy the command buffer.
      TsmDeleteDataMove(data_move);
    }

    for (int32_t nhw_index = 0; nhw_index < nhw_cnt; nhw_index++) {
      uint64_t src_in_addr =
          (uint64_t)in + (uint64_t)nhw_index * cx_align_mem_size;
      uint64_t dst_out_addr =
          (uint64_t)out + (uint64_t)nhw_index * one_align_mem_size;
      // init src
      uint8_t bytes = get_dtype_size_new(fmt);
      hybrid_value init_v = set_float2value(fmt, 1.0);
      if (cx_align > c) {

        // op_reduce_micro_op_memset(src_in_addr + c * bytes, *(uint32_t
        // *)&init_v,
        //                           cx_align - c, fmt);
        uint32_t loop = cx_align / align_val;

        for (int32_t loop_index = 1; loop_index < loop; loop_index++) {
          uint64_t arith_src0_addr = src_in_addr;
          uint64_t arith_src1_addr =
              arith_src0_addr + (uint64_t)bytes * loop_index * align_val;
          uint64_t arith_dst_addr = arith_src0_addr;

          // TsmArith *arith = (TsmArith *)getTsmOpPointer()->arith_pointer;

          arith->MulVV(&arithIns, arith_src0_addr, arith_src1_addr,
                       arith_dst_addr, align_val, RND_NEAREST_EVEN, fmt);
          TsmExecute(&arithIns);
        }
      }

      uint32_t c_tail = (cx_align > align_val) ? align_val : cx_align;
      for (uint32_t stride = c_tail / 2; stride > 0; stride = stride / 2) {
        uint64_t arith_src0_addr = src_in_addr;
        uint64_t arith_src1_addr = arith_src0_addr + (uint64_t)bytes * stride;
        uint64_t arith_dst_addr =
            (stride == 1) ? dst_out_addr : arith_src0_addr;

        // TsmArith *arith = (TsmArith *)getTsmOpPointer()->arith_pointer;

        arith->MulVV(&arithIns, arith_src0_addr, arith_src1_addr,
                     arith_dst_addr, stride, RND_NEAREST_EVEN, fmt);
        TsmExecute(&arithIns);
      }
    }
  } else if (reduce_dim == 1) {
    TsmDataMove *data_move = TsmNewDataMove();

    TsmDataMoveInstr data_move_inst = {I_CGRA,
                                       {
                                           0,
                                       },
                                       {
                                           0,
                                       }};

    // reduce W in NHWC.
    int32_t nh_cnt = n * h;
    for (int32_t nh_index = 0; nh_index < nh_cnt; nh_index++) {
      uint64_t src_in_addr =
          (uint64_t)in + (uint64_t)nh_index * cx_align_mem_size;
      uint64_t dst_out_addr =
          (uint64_t)out + (uint64_t)nh_index * cx_align_mem_size;
      // TsmOperatorPointer *opp = getTsmOpPointer();
      // TsmDataMove *data_move = (TsmDataMove *)opp->datamove_pointer;

      // Create command buffer.

      St_StrideIteration src_si = {1, 1, 1, 1, 1, 1};
      St_StrideIteration dst_si = {1, 1, 1, 1, 1, 1};

      data_move->GatherScatter(&data_move_inst, (uint64_t)src_in_addr,
                               (uint64_t)dst_out_addr, cx_align_mem_size,
                               &src_si, &dst_si);

      // Dispatch the command to accelerator
      TsmExecute(&data_move_inst);

      for (int32_t w_index = 1; w_index < w; w_index++) {
        uint64_t arith_src0_addr = dst_out_addr;
        uint64_t arith_src1_addr =
            src_in_addr + (uint64_t)w_index * cx_align_mem_size;
        uint64_t arith_dst_addr = dst_out_addr;

        // TsmArith *arith = (TsmArith *)getTsmOpPointer()->arith_pointer;

        arith->MulVV(&arithIns, arith_src0_addr, arith_src1_addr,
                     arith_dst_addr, cx_align, RND_NEAREST_EVEN, fmt);
        TsmExecute(&arithIns);
      }
    }
    // Destroy the command buffer.
    TsmDeleteDataMove(data_move);
  } else {
    assert(0);
  }

  TsmDeleteArith(arith);
}
