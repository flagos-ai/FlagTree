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
#include "tx81.h"
#include "tx81_spm.h"
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#define MAX_TILE_NUM 16

const uint32_t tile_physical_relation[MAX_TILE_NUM] = {
    0, 1, 2, 3, 7, 11, 15, 14, 13, 12, 8, 9, 10, 6, 5, 4};

// 获取物理连接最近的下一个tile id
int32_t getNextNearestTileId(uint32_t tileId) {
  for (int i = 0; i < MAX_TILE_NUM; i++) {
    if (tile_physical_relation[i] == tileId) {
      if (i != (MAX_TILE_NUM - 1)) {
        return tile_physical_relation[++i];
      } else {
        return tile_physical_relation[0];
      }
    }
  }

  return -1;
}

// 获取物理连接最近的上一个tile id
int32_t getPrevNearestTileId(uint32_t tileId) {
  for (int i = 0; i < MAX_TILE_NUM; i++) {
    if (tile_physical_relation[i] == tileId) {
      if (i != 0) {
        return tile_physical_relation[--i];
      } else {
        return tile_physical_relation[MAX_TILE_NUM - 1];
      }
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

  this_debug_spm_ptr[10] = 0;               // 写对端开始
  tile_a_spm_ptr[other_sync_spm_index] = 1; // forward

  this_debug_spm_ptr[10] = 1; // 写对端结束

  while (!this_spm_ptr[this_sync_spm_index]) {
  }

  this_spm_ptr[this_sync_spm_index] = 0; // forward
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
// The operation reads from the given source buffer and sends it to the remote
// tile. The operation is non-blocking and returns immediately.
void __Send(int64_t chipX, int64_t chipY, int64_t dieId, int64_t tileId,
            void *restrict dst, void *restrict src, uint32_t elem_bytes,
            uint64_t data_size) {
  uint32_t coreIndex = __get_pid(0); // 全局tile id
  initTileId(coreIndex, 4);
  // __EP_LOG__(0, "+++++++++ Send444 dst:%lx, src: %lx, cur_tileId:%d,
  // nextTileId: %d, data_size: %d\n", dst, src, coreIndex, tileId, data_size);
  (void)chipX;
  (void)chipY;
  (void)dieId;
  int64_t nextTileId = getNextNearestTileId(coreIndex);
  int64_t preTileId = getPrevNearestTileId(coreIndex);
  // tile_sync_by_spm_single_direction(coreIndex, preTileId, 4, 4, 0, 0);
  const TsmOperatorPointer *intrinsic = g_intrinsic();

  int fringFsmId = DIRECT_DTE_FSM_ID_0;
  int remottFringFsmId = DIRECT_DTE_FSM_ID_0;

  void *fdteNode = direct_dte_attach(0);
  void *fringFsmHd = direct_fsm_monitor_init(fringFsmId, 0, data_size, 1);

  TsmStream *stream = (TsmStream *)(intrinsic->stream_pointer);
  uint64_t nextTileBaseAddr = get_tile_spm_addr_base(nextTileId, 4, 4);

  stream->wait_finish();

  // __EP_LOG__(0, "fdte info base: %ld, dst: %ld, src: %ld, remote_fsm_id: %d,
  // data_size: %d, dst_tileId: %d, this_tileId: %d\n",
  //     nextTileBaseAddr, nextTileBaseAddr + (uint64_t)dst, (uint64_t)src,
  //     remottFringFsmId, data_size, tileId, coreIndex);
  DirectDTESendInfo fdteInfo = {.src_addr = (uint64_t)src,
                                .dst_addr = nextTileBaseAddr + (uint64_t)dst,
                                .length = data_size,
                                .remote_fsm_id = remottFringFsmId,
                                .mode = 0, // unicast
                                .dst_tile = nextTileId,
                                .tile_this = coreIndex,
                                .dte_node = fdteNode};
  set_direct_fsm_monitor_dst_addr(fringFsmId, nextTileBaseAddr + (uint64_t)dst);
  tile_sync_by_spm_single_direction(coreIndex, preTileId, 4, 4, 0, 0);
  direct_dte_send_async(&fdteInfo); // 把当前数据异步发送给下一个tile
  // __EP_LOG__(KCORE_LOG_DEBUG, "send data to next tile: %u, current
  // tile:%u.\n",
  //             getNextNearestTileId(coreIndex), coreIndex);

  direct_fsm_monitor_receive(coreIndex, preTileId,
                             fringFsmHd); // 阻塞接收前一个tile发送的数据
  // __EP_LOG__(KCORE_LOG_DEBUG,
  //             "receive data from coreIndex: %u, current tile:%u.\n",
  //             getPrevNearestTileId(coreIndex), coreIndex);
  direct_dte_wait_done(&fdteInfo); // 等待异步发送完成
  TsmWaitfinish();

  tile_sync_by_spm_single_direction(coreIndex, preTileId, 4, 4, 0, 0);
  direct_dte_release(fdteNode);
  direct_fsm_monitor_deinit(fringFsmHd);
  // __EP_LOG__(0, "-------- Send\n")
}
