//===------------------------ recv.c --------------------------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::Recv, see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

// #include "instr_adapter_plat.h"

#include "direct_dte_and_fsm.h"
#include "tx81.h"
#include "tx81_spm.h"
#include <stdint.h>
#include <stdio.h>

uint32_t __get_pid(uint32_t);

// Blockingly receive data from a source tile into a destination buffer.
// Returns the destination buffer address.
void __Recv(int64_t chip_x, int64_t chip_y, int64_t die_id, int64_t tile_id,
            void *dst, uint32_t elem_bytes, uint32_t data_size) {
  // TODO
  return;
}
