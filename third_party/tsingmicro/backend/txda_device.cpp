#include <c10/core/Allocator.h>
#include <c10/core/impl/alloc_cpu.h>

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#include <torch/csrc/Device.h>
#include <torch/extension.h>

#include <ATen/EmptyTensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Resize.h>
#include <ATen/native/cpu/Loops.h>

#include <ATen/native/CPUFallback.h>
#include <ATen/ops/as_strided_cpu_dispatch.h>
#include <ATen/ops/set_cpu_dispatch.h>

#include "hrt_common.h"
#include "hrt_interface.h"

#include "profiler.h"

namespace at {
namespace detail {

C10_REGISTER_GUARD_IMPL(
    PrivateUse1, c10::impl::NoOpDeviceGuardImpl<DeviceType::PrivateUse1>);

}
} // namespace at

// Global device vector
std::vector<TsmDevice *> g_txda_devices;
static bool g_runtime_initialized = false;
static int device_id = 0;
static int stream_id = 0;
// std::string chip_out = "/tmp/chip_out/node0/";

bool init_device(std::string chip_out) {
  if (g_runtime_initialized) {
    return true;
  }

  uint32_t eventId = EVENT_INIT;
  PROFILE_CALL(addOrderPorfile, TIME_RUNTIME, TIME_INIT_START, &eventId);
  bool is_succ_init = false;
  auto guard = std::shared_ptr<void>(nullptr, [&is_succ_init](void *) {
    if (!is_succ_init) {
      for (auto *dev : g_txda_devices) {
        TsmNpuPowerOff(dev);
        TsmResetDevice(dev);
        delete dev;
      }
      g_txda_devices.clear();
      TsmDeInitRuntime();
      g_runtime_initialized = false;
    }
  });

  if (TsmInitRuntime(true) != RET_SUCCESS) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to initialize txda runtime");
    return false;
  }
  // Get device count
  uint32_t device_num = 0;
  if (TsmGetDeviceNum(device_num) != RET_SUCCESS || device_num == 0) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Failed to get txda device count or no devices found");
    TsmDeInitRuntime();
    return false;
  }

  device_id = 0;

  TsmDevice *dev = new TsmDevice();
  if (TsmSetDevice(&dev, 0, UINT32_MAX) != RET_SUCCESS) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to set txda devices");
    TsmDeInitRuntime();
    return false;
  }
  g_txda_devices.push_back(dev);

  // TSM_RETCODE TsmNpuPowerOn(TsmDevice *dev, std::vector<std::string>
  // kcore_file_list);
  std::vector<std::string> kcore_file_list;
  std::string kcore_bin = chip_out + "kcore_fw.bin";
  for (int i = 0; i < 16; i++) {
    kcore_file_list.push_back(kcore_bin);
  }

  if (TsmNpuPowerOn(dev, kcore_file_list) != RET_SUCCESS ||
      TsmSetMonitorInfo(dev) != RET_SUCCESS) {
    return false;
  }

  // delete new_model;
  g_runtime_initialized = true;
  printf("====init_txda_device====success=======\n");
  is_succ_init = true;
  PROFILE_CALL(addOrderPorfile, TIME_RUNTIME, TIME_INIT_END, &eventId);
  return true;
}

void set_device(int id) { device_id = id; }

bool cleanup_device() {
  if (!g_runtime_initialized) {
    return true;
  }

  uint32_t eventId = EVENT_INIT;
  PROFILE_CALL(addOrderPorfile, TIME_RUNTIME, TIME_RELEASE_START, &eventId);

  for (auto *dev : g_txda_devices) {
    // Reset and release each device
    TsmNpuPowerOff(dev);
    TsmResetDevice(dev);
    delete dev;
  }
  g_txda_devices.clear();
  TsmDeInitRuntime();
  g_runtime_initialized = false;
  printf("====cleanup_txda_runtime==== release success=======\n");

  PROFILE_CALL(addOrderPorfile, TIME_RUNTIME, TIME_RELEASE_END, &eventId);
  PROFILE_CALL(printProfileAll);
  return true;
}

int current_device() { return device_id; }

int current_stream(int id) { return stream_id; }

uint64_t get_device() { return (uint64_t)g_txda_devices[device_id]; }

int device_count() { return (int)1; }
// TODO:
bool is_available() { return true; }

void synchronize() {}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_device", &get_device, "get txda device");
  m.def("is_available", &is_available, "is tx device available");
  m.def("init_device", &init_device, "initialize tx device");
  m.def("current_device", &current_device, "get current tx device");
  m.def("current_stream", &current_stream, "get current tx stream");
  m.def("set_device", &set_device, "set tx device");
  m.def("cleanup_device", &cleanup_device, "cleanup tx device");
  m.def("synchronize", &synchronize, "synchronize all threads in block");
  m.def("device_count", &device_count, "get device count");
}

struct TXDADeviceAllocator final : at::Allocator {
  TXDADeviceAllocator() = default;

  at::DataPtr allocate(size_t nbytes) override {
    TsmDevicePtr data;
    int dev_id = current_device();
    TsmDeviceMalloc(g_txda_devices[dev_id], data, (uint64_t)nbytes);
    return {(void *)data, nullptr, &ReportAndDelete,
            at::Device(at::DeviceType::PrivateUse1, 0)};
  }

  static void ReportAndDelete(void *ptr) {
    if (!ptr) {
      return;
    }
    TsmDeviceFree((uint64_t)ptr);
  }

  at::DeleterFnPtr raw_deleter() const override { return &ReportAndDelete; }
  void copy_data(void *dest, const void *src, std::size_t count) const final {
    default_copy_data(dest, src, count);
  }
};

// register device allocator
static TXDADeviceAllocator global_txda_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_txda_alloc);

at::Tensor txda_to_device(const at::Tensor &self, at::Device device,
                          at::ScalarType dtype, bool non_blocking, bool copy,
                          c10::optional<at::MemoryFormat> memory_format) {
  TORCH_CHECK(self.is_cpu() ||
                  self.device().type() == c10::DeviceType::PrivateUse1,
              "only support data transfer between cpu and txda");
  TORCH_CHECK(device.is_cpu() || device.type() == c10::DeviceType::PrivateUse1,
              "only support data transfer between cpu and txda");
  // Some dummy asserts for the basic use case: inputs are the same size /
  // dtype, all contiguous.
  TORCH_CHECK(self.scalar_type() == dtype);
  TORCH_CHECK(self.is_contiguous());

  if (device != at::DeviceType::CPU) {
    return at::empty(self.sizes(), self.options());
  }
  auto out = at::empty(self.sizes(), dtype, self.options().layout(), device,
                       false, memory_format);
  memcpy(out.mutable_data_ptr(), self.mutable_data_ptr(), self.nbytes());
  return out;
}

// TsmMemcpyH2D() or TsmMemcpyD2H()
at::Tensor txda__copy_from(const at::Tensor &self, const at::Tensor &dst,
                           bool non_blocking) {
  TORCH_CHECK(self.is_cpu() ||
                  self.device().type() == c10::DeviceType::PrivateUse1,
              "only support data transfer between cpu and txda");
  TORCH_CHECK(dst.is_cpu() ||
                  dst.device().type() == c10::DeviceType::PrivateUse1,
              "only support data transfer between cpu and txda");

  TORCH_CHECK(self.sizes() == dst.sizes());
  TORCH_CHECK(self.scalar_type() == dst.scalar_type());
  TORCH_CHECK(self.is_contiguous() && dst.is_contiguous());

  if (self.is_cpu()) {
    // printf("H2D self: 0x%lx, dst: 0x%lx, size: 0x%lx\n",
    //        (uint64_t)self.storage().data_ptr().get(),
    //        (uint64_t)dst.storage().data_ptr().get(),
    //        (uint64_t)self.storage().nbytes());
    uint32_t eventId = EVENT_INIT;
    PROFILE_CALL(addOrderPorfile, TIME_RUNTIME, TIME_H2D_START, &eventId);
    auto ret = TsmMemcpyH2D((uint64_t)dst.storage().data_ptr().get(),
                            (const void *)self.storage().data_ptr().get(),
                            self.storage().nbytes());
    if (ret != RET_SUCCESS) {
      PyErr_SetString(PyExc_RuntimeError, "Failed to TsmMemcpyH2D");
    }
    PROFILE_CALL(addOrderPorfile, TIME_RUNTIME, TIME_H2D_END, &eventId);
    TORCH_CHECK(ret == RET_SUCCESS, "==H2DMemArray Error===");
  } else {
    // printf("D2H self: 0x%lx, dst: 0x%lx, size: 0x%lx\n",
    //        (uint64_t)self.storage().data_ptr().get(),
    //        (uint64_t)dst.storage().data_ptr().get(),
    //        (uint64_t)self.storage().nbytes());
    uint32_t eventId = EVENT_INIT;
    PROFILE_CALL(addOrderPorfile, TIME_RUNTIME, TIME_D2H_START, &eventId);
    auto ret = TsmMemcpyD2H((const void *)dst.storage().data_ptr().get(),
                            (uint64_t)self.storage().data_ptr().get(),
                            self.storage().nbytes());
    if (ret != RET_SUCCESS) {
      PyErr_SetString(PyExc_RuntimeError, "Failed to TsmMemcpyD2H");
    }
    PROFILE_CALL(addOrderPorfile, TIME_RUNTIME, TIME_D2H_END, &eventId);
    TORCH_CHECK(ret == RET_SUCCESS, "==D2HMemArray Error===");
  }
  return dst;
}

at::Tensor txda_empty_memory_format(
    at::IntArrayRef size, std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout, std::optional<at::Device> device,
    std::optional<bool> pin_memory,
    std::optional<at::MemoryFormat> memory_format) {
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_generic(size, &global_txda_alloc, private_use_ks,
                                   c10::dtype_or_default(dtype), memory_format);
}

at::Tensor txda_empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride,
                              std::optional<at::ScalarType> dtype_opt,
                              std::optional<at::Layout> layout_opt,
                              std::optional<at::Device> device_opt,
                              std::optional<bool> pin_memory_opt) {

  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  auto dtype = c10::dtype_or_default(dtype_opt);
  return at::detail::empty_strided_generic(size, stride, &global_txda_alloc,
                                           private_use_ks, dtype);
}

at::Tensor txda_as_strided(const at::Tensor &input, at::IntArrayRef size,
                           at::IntArrayRef stride,
                           c10::optional<int64_t> storage_offset) {
  return at::cpu::as_strided(input, size, stride, storage_offset);
}

at::Tensor &txda_fill__scalar(at::Tensor &self, const at::Scalar &value) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "only support txda");
  TORCH_CHECK(self.is_contiguous());
  TORCH_CHECK(self.scalar_type() == c10::ScalarType::Float);

  auto _data = static_cast<float *>(self.mutable_data_ptr());
  for (size_t idx = 0; idx < self.numel(); idx++) {
    _data[idx] = value.toFloat();
  }

  return self;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("to.Device", &txda_to_device);
  m.impl("fill_.Scalar", &txda_fill__scalar);
  m.impl("_copy_from", &txda__copy_from);
  m.impl("empty.memory_format", &txda_empty_memory_format);
  m.impl("empty_strided", &txda_empty_strided);
  m.impl("as_strided", &txda_as_strided);
}

void custom_cpu_fallback(const c10::OperatorHandle &op,
                         torch::jit::Stack *stack) {
  printf("custom_cpu_fallback \n");
  at::native::cpu_fallback(op, stack);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
}
