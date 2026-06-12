// launch_kernel_runner.cc
//
// Minimal CLI that runs a flagtree-compiled kernel through the
// rhino-launch-kernel host runtime (Program/Kernel/Queue + LocalSPM_t). It
// loads a flagtree-emitted .ref ELF, stages one SPM/DDR buffer per kernel
// argument, dispatches the kernel on the device, and writes the outputs back.
//
// Usage:
//   sudo ./launch_kernel_runner <elf> <kernel_name> <arg_spec> [<arg_spec> ...]
//
// arg_spec format:
//   in:<size_bytes>:<input.bin>     -> alloc SPM+DDR, CopyToDevice from file
//   out:<size_bytes>:<output.bin>   -> alloc SPM+DDR, CopyFromDevice into file
//
// Each arg gets an independent LocalSPM_t+HostDDR_t pair (demo_single.cc
// pattern) and is bound to the kernel via the 3-arg set_regs overload
// (idx, rpu_addr, memory_stride) so the framework handles addressing. The
// reg_idx for arg i is 2*i, matching flagtree's .args_pos 0,2,4 convention.
// The kernel is dispatched on core 0 with block_dims = {1,1,1} (single warp).

#include "rhino_launch_buffer.h"
#include "rhino_launch_kernel.h"
#include "rhino_launch_program.h"
#include "rhino_launch_queue.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#if defined(__aarch64__)
static inline void flush_cache_range(void *addr, size_t size) {
  if (size == 0)
    return;
  constexpr size_t CACHE_LINE = 64;
  uintptr_t start = (uintptr_t)addr & ~(CACHE_LINE - 1);
  uintptr_t end = ((uintptr_t)addr + size + CACHE_LINE - 1) & ~(CACHE_LINE - 1);
  for (uintptr_t p = start; p < end; p += CACHE_LINE)
    __asm__ volatile("dc civac, %0" : : "r"(p) : "memory");
  __asm__ volatile("dsb ish" ::: "memory");
}
#else
static inline void flush_cache_range(void *, size_t) {}
#endif

using namespace rhino_lkn;

namespace {

struct ArgSpec {
  bool is_out = false;
  size_t size_bytes = 0;
  std::string path;
};

ArgSpec parse_arg(const std::string &spec) {
  ArgSpec a;
  size_t p1 = spec.find(':');
  size_t p2 = spec.find(':', p1 + 1);
  if (p1 == std::string::npos || p2 == std::string::npos) {
    std::cerr << "bad arg_spec: " << spec << " (expected role:size:path)"
              << std::endl;
    std::exit(2);
  }
  std::string role = spec.substr(0, p1);
  a.size_bytes = std::stoul(spec.substr(p1 + 1, p2 - p1 - 1));
  a.path = spec.substr(p2 + 1);
  if (role == "in") {
    a.is_out = false;
  } else if (role == "out") {
    a.is_out = true;
  } else {
    std::cerr << "bad role '" << role << "' (expect in/out)" << std::endl;
    std::exit(2);
  }
  return a;
}

std::vector<uint8_t> read_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::cerr << "read_file: cannot open " << path << std::endl;
    std::exit(3);
  }
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
}

void write_file(const std::string &path, const void *data, size_t size) {
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    std::cerr << "write_file: cannot open " << path << std::endl;
    std::exit(3);
  }
  f.write(reinterpret_cast<const char *>(data), size);
}

size_t round_up(size_t v, size_t a) { return ((v + a - 1) / a) * a; }

} // namespace

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <elf> <kernel_name> <arg_spec> [<arg_spec> ...]\n"
              << "  arg_spec: in:<size_bytes>:<input.bin>\n"
              << "         |  out:<size_bytes>:<output.bin>" << std::endl;
    return 1;
  }

  const std::string elf_path = argv[1];
  const std::string kernel_name = argv[2];

  std::vector<ArgSpec> args;
  for (int i = 3; i < argc; ++i) {
    args.push_back(parse_arg(argv[i]));
  }
  if (args.empty()) {
    std::cerr << "no args" << std::endl;
    return 2;
  }

  std::cerr << "[runner] start" << std::endl;
  // Path selection:
  //   default = Program::create_with_binary_file(.ref) per demo_single.cc —
  //   this is the supported hxcc-produced .ref + launch_kernel API flow.
  //   LK_USE_RPUBIN=1 = legacy raw Kernel_t(instr_data, size) ctor that takes
  //   a .rpubin file directly. Kept for cross-checks; no longer the default.
  std::unique_ptr<Program_t> prog;
  std::unique_ptr<Kernel_t> kernel_ptr;
  if (std::getenv("LK_USE_RPUBIN")) {
    auto instr_bytes = read_file(elf_path); // here elf_path actually = rpubin
    std::cerr << "[runner] direct ctor with " << instr_bytes.size()
              << " bytes of raw instr" << std::endl;
    kernel_ptr = std::make_unique<Kernel_t>(
        reinterpret_cast<const char *>(instr_bytes.data()), instr_bytes.size(),
        kernel_name.c_str());
  } else {
    prog = std::make_unique<Program_t>();
    if (prog->create_with_binary_file(elf_path, kernel_name.c_str()) != 0) {
      std::cerr << "Failed to load ELF " << elf_path
                << " kernel=" << kernel_name << std::endl;
      return 4;
    }
    kernel_ptr = std::make_unique<Kernel_t>(*prog, kernel_name.c_str());
  }
  Kernel_t &kernel = *kernel_ptr;
  kernel.reset_regs();

  // Per-arg HostDDR_t (staging) + LocalSPM_t (kernel-visible) pair.
  //
  // Cache coherency: HostDDR cpu_ptr is on a cached host mapping, so we MUST
  // call ddr.flush() both BEFORE CopyToDevice (clean: push CPU writes down
  // to DRAM so the DMA reads the latest data) AND AFTER CopyFromDevice
  // (invalidate: drop stale CPU cache lines so subsequent CPU reads pick up
  // what the DMA just wrote into DRAM). Both are the same underlying
  // HxilBufObjFlushCpuCacheRange call.
  //
  // Address binding: kernel reg slots 2*i / 2*i+1 hold the lo16/hi16 of the
  // SPM-local rpu_addr. We write them explicitly via the 16-bit set_regs
  // overload instead of the 3-arg overload which does an unwanted
  // `addr >> stride` shift.
  constexpr size_t kSpmBankSize = 256;
  std::vector<std::unique_ptr<LocalSPM_t>> spms;
  std::vector<std::unique_ptr<HostDDR_t>> ddrs;
  std::vector<size_t> allocs;
  spms.reserve(args.size());
  ddrs.reserve(args.size());
  allocs.reserve(args.size());

  for (size_t i = 0; i < args.size(); ++i) {
    size_t alloc = round_up(args[i].size_bytes, kSpmBankSize);
    if (alloc < kSpmBankSize)
      alloc = kSpmBankSize;
    auto spm = std::make_unique<LocalSPM_t>(alloc, read_write, kStride32B,
                                            /*align=*/2, /*core_id=*/0);
    auto ddr = std::make_unique<HostDDR_t>(alloc, read_write, kStride256B,
                                           /*align=*/4096);
    if (!ddr->get_cpu_ptr()) {
      std::cerr << "HostDDR cpu_ptr nullptr for arg " << i << std::endl;
      return 6;
    }
    // Stage input/output into the DDR cpu mapping.
    std::memset(ddr->get_cpu_ptr(), 0, alloc);
    if (!args[i].is_out) {
      auto bytes = read_file(args[i].path);
      if (bytes.size() < args[i].size_bytes) {
        std::cerr << "input " << args[i].path << " short" << std::endl;
        return 5;
      }
      std::memcpy(ddr->get_cpu_ptr(), bytes.data(), args[i].size_bytes);
    }
    // FLUSH #1 (clean): push CPU writes down so CopyToDevice DMA sees them.
    ddr->flush();
    if (CopyToDevice(*spm, *ddr) != 0) {
      std::cerr << "CopyToDevice failed for arg " << i << std::endl;
      return 7;
    }

    // Bind addr (16-bit lo/hi) to the kernel's param registers.
    uint32_t addr = static_cast<uint32_t>(spm->get_rpu_addr());
    uint32_t lo = kernel.set_regs(2 * i, static_cast<uint16_t>(addr & 0xFFFF));
    uint32_t hi = kernel.set_regs(2 * i + 1, static_cast<uint16_t>(addr >> 16));
    std::cerr << "[runner]   arg " << i
              << " role=" << (args[i].is_out ? "out" : "in ")
              << " size=" << args[i].size_bytes << " alloc=" << alloc
              << " rpu_addr=0x" << std::hex << addr << " set_regs(" << std::dec
              << (2 * i) << ",lo=0x" << std::hex << (addr & 0xFFFF) << ")/("
              << std::dec << (2 * i + 1) << ",hi=0x" << std::hex << (addr >> 16)
              << ") rc=" << std::dec << lo << "/" << hi << std::endl;
    if (lo != 0 || hi != 0)
      return 7;

    spms.push_back(std::move(spm));
    ddrs.push_back(std::move(ddr));
    allocs.push_back(alloc);
  }

  std::cerr << "[runner] kernel_info ----" << std::endl;
  kernel.print_kernel_info();
  std::cerr << "------------------------" << std::endl;
  std::cerr << "[runner] launching" << std::endl;
  Queue_t queue(8);
  // Single warp on core 0 is controlled by block_dims={1,1,1} + core_ids={0};
  // no explicit set_warp_num needed (per LK convention, grid drives warp
  // count). LK_WARP_NUM env left in only for forced ablation.
  if (const char *wn = std::getenv("LK_WARP_NUM")) {
    queue.set_warp_num(static_cast<uint8_t>(std::stoul(wn)));
  }

  uint32_t rc = queue.enqueu_kernel(kernel, {1, 1, 1}, {0});
  std::cerr << "[runner] kernel done rc=" << rc << std::endl;
  if (rc != 0) {
    std::cerr << "enqueu_kernel rc=" << rc << std::endl;
    return 7;
  }

  // Per-arg readback for outputs: DMA SPM→DDR, FLUSH (invalidate cpu cache),
  // then read DDR cpu_ptr.
  for (size_t i = 0; i < args.size(); ++i) {
    if (!args[i].is_out)
      continue;
    if (CopyFromDevice(*spms[i], *ddrs[i]) != 0) {
      std::cerr << "CopyFromDevice failed for arg " << i << std::endl;
      return 8;
    }
    // FLUSH #2 (invalidate): drop stale CPU cache so the read picks up DMA's
    // fresh DRAM contents.
    ddrs[i]->flush();
    write_file(args[i].path, ddrs[i]->get_cpu_ptr(), args[i].size_bytes);
  }

  return 0;
}
