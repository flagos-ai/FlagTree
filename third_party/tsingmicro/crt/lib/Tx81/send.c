//===------------------------ send.c --------------------------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::Send, see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

// #include "instr_adapter_plat.h"
#include "direct_dte_and_fsm.h"
#include "rcs1_spm.h"
#include "tx81_run.h"
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

// Find the tile that precedes `tileId` in the given physical_ids ring.
static int32_t getPrevInTopology(uint32_t tileId, const uint32_t *physical_ids,
                                 uint32_t mesh_size, uint32_t ring_size) {
  if (mesh_size == 0 || !physical_ids)
    return -1;
  if (ring_size == 0 || ring_size > mesh_size)
    ring_size = mesh_size;
  for (uint32_t i = 0; i < mesh_size; i++) {
    if (physical_ids[i] == tileId) {
      uint32_t ring_start = (i / ring_size) * ring_size;
      uint32_t ring_end = ring_start + ring_size - 1;
      if (ring_end >= mesh_size)
        ring_end = mesh_size - 1;
      if (i == ring_start)
        return physical_ids[ring_end];
      return physical_ids[i - 1];
    }
  }
  return -1;
}

void tile_sync_by_spm_single_direction(int32_t tile_this, int32_t tile_a,
                                       int32_t tile_x, int32_t tile_y,
                                       uint32_t this_sync_spm_index,
                                       uint32_t other_sync_spm_index) {
  // this_debug_spm_ptr 只用于板端记录调试信息
  volatile uint32_t *this_debug_spm_ptr =
      (volatile uint32_t *)get_spm_memory_mapping(SINGLE_SPM_SYNC_DEBUG_ADDR);
  this_debug_spm_ptr[0] = tile_this;
  this_debug_spm_ptr[1] = tile_a;
  this_debug_spm_ptr[2] = tile_x;
  this_debug_spm_ptr[3] = tile_y;
  this_debug_spm_ptr[4] = this_sync_spm_index;
  this_debug_spm_ptr[5] = other_sync_spm_index;

  uint64_t tile_a_spm =
      get_tile_spm_addr_base(tile_a, tile_x, tile_y) + SINGLE_SPM_SYNC_ADDR;
  *(uint64_t *)(this_debug_spm_ptr + 6) = tile_a_spm;
  volatile uint32_t *this_spm_ptr =
      (volatile uint32_t *)get_spm_memory_mapping(SINGLE_SPM_SYNC_ADDR);
  *(uint64_t *)(this_debug_spm_ptr + 8) = (uint64_t)this_spm_ptr;
  volatile uint32_t *tile_a_spm_ptr = (volatile uint32_t *)(tile_a_spm);

  this_debug_spm_ptr[10] = 0; // 写对端开始
  // 等对端消费掉上一次 post
  while (tile_a_spm_ptr[other_sync_spm_index] != 0) {
  }
  tile_a_spm_ptr[other_sync_spm_index] = 1; // forward

  this_debug_spm_ptr[10] = 1; // 写对端结束

  while (!this_spm_ptr[this_sync_spm_index]) {
  }

  this_spm_ptr[this_sync_spm_index] = 0; // consume
}

#define SCFG_TILE_ID_ADDR 0x6A0058 // KUIPER_ADDR_MAP_REG_BASE 0x6A0000

uint32_t __get_pid(uint32_t);

int initTileId(uint32_t tileId, uint32_t rowLength) {
  // init 1D tile-id
  *(volatile uint32_t *)(get_spm_memory_mapping(TILE_ID_ADDR)) = tileId;
  // init 2D logic id
  *(volatile uint32_t *)(get_spm_memory_mapping(LOGIC_ID_ADDR)) =
      *(volatile uint32_t *)(SCFG_TILE_ID_ADDR);
  // init row-length
  *(volatile uint32_t *)(get_spm_memory_mapping(ROW_LENGTH_ADDR)) = rowLength;
  *(volatile uint32_t *)(get_spm_memory_mapping(INNER_CHIP_ERROR_CODE)) = 0;

  // __LOG__(KCORE_LOG_DEBUG, "logic_id:0x%x, tileId:%u, rowLength:%u\n",
  //         *(volatile uint32_t *)(get_spm_memory_mapping(LOGIC_ID_ADDR)),
  //         tileId, rowLength);
  return 0;
}

// Asynchronously send data to a destination tile.
// physical_ids and mesh_size carry the ring topology so that recv-source
// lookup works per-call without hardcoded globals.
void __Send(int64_t chipX, int64_t chipY, int64_t dieId, int64_t tileId,
            void *restrict dst, void *restrict src, uint32_t elem_bytes,
            uint64_t data_size, const uint32_t *physical_ids,
            uint32_t mesh_size, uint32_t ring_size) {
  uint32_t coreIndex = __get_pid(0);
  initTileId(coreIndex, 4);
  (void)chipX;
  (void)chipY;
  (void)dieId;
  int64_t nextTileId = tileId;
  int64_t preTileId =
      getPrevInTopology(coreIndex, physical_ids, mesh_size, ring_size);
  const RcsOperatorPointer *intrinsic = g_intrinsic();

  int fringFsmId = DIRECT_DTE_FSM_ID_0;
  int remottFringFsmId = DIRECT_DTE_FSM_ID_0;

  void *fdteNode = direct_dte_attach(0);
  void *fringFsmHd = direct_fsm_monitor_init(fringFsmId, 0, data_size, 1);

  RcsStream *stream = (RcsStream *)(intrinsic->stream_pointer);
  uint64_t nextTileBaseAddr = get_tile_spm_addr_base(nextTileId, 4, 4);

  stream->wait_finish();

  DirectDTESendInfo fdteInfo = {.src_addr = (uint64_t)src,
                                .dst_addr = nextTileBaseAddr + (uint64_t)dst,
                                .length = data_size,
                                .remote_fsm_id = remottFringFsmId,
                                .mode = 0,
                                .dst_tile = nextTileId,
                                .tile_this = coreIndex,
                                .dte_node = fdteNode};
  set_direct_fsm_monitor_dst_addr(fringFsmId, nextTileBaseAddr + (uint64_t)dst);
  tile_sync_by_spm_single_direction(coreIndex, preTileId, 4, 4, 0, 0);
  direct_dte_send_async(&fdteInfo);

  direct_fsm_monitor_receive(coreIndex, preTileId, fringFsmHd);
  direct_dte_wait_done(&fdteInfo);
  RcsWaitfinish();

  tile_sync_by_spm_single_direction(coreIndex, preTileId, 4, 4, 0, 0);
  direct_dte_release(fdteNode);
  direct_fsm_monitor_deinit(fringFsmHd);
}
