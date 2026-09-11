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
#include "rcs1_spm.h"
#include "tx81_run.h"
#include <stdint.h>
#include <stdio.h>

uint32_t __get_pid(uint32_t);

// Blockingly receive data from a source tile into a destination buffer.
// Sets up an FSM monitor for the indicated source tile and waits until the
// data transfer is complete.
void __Recv(int64_t chip_x, int64_t chip_y, int64_t die_id, int64_t tile_id,
            void *dst, uint32_t elem_bytes, uint32_t data_size) {
  uint32_t coreIndex = __get_pid(0);
  (void)chip_x;
  (void)chip_y;
  (void)die_id;

  int fsmId = DIRECT_DTE_FSM_ID_0;
  void *fringFsmHd = direct_fsm_monitor_init(fsmId, 0, data_size, 1);

  uint64_t srcTileBaseAddr = get_tile_spm_addr_base((uint32_t)tile_id, 4, 4);
  set_direct_fsm_monitor_dst_addr(fsmId, srcTileBaseAddr + (uint64_t)dst);

  // Blocking receive: wait for FSM to signal data arrival from source tile.
  direct_fsm_monitor_receive(coreIndex, (uint32_t)tile_id, fringFsmHd);

  direct_fsm_monitor_deinit(fringFsmHd);
}
