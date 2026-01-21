#include <c10/core/Allocator.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/impl/alloc_cpu.h>
#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>

#include <torch/csrc/Device.h>
#include <torch/csrc/autograd/autograd_not_implemented_fallback.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/extension.h>

#include <ATen/EmptyTensor.h>
#include <ATen/InferSize.h>
#include <ATen/core/GeneratorForPrivateuseone.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/native/cpu/DistributionTemplates.h>

#include <internal/internal_api.h>
#include <standard_api.h>

#include <mutex>
#include <unordered_map>

static c10::DeviceIndex aipu_device_index = 0;

namespace c10 {
namespace impl {

struct C10_API AIPUGuardImpl final : public DeviceGuardImplInterface {
  static constexpr DeviceType static_type = DeviceType::PrivateUse1;
  inline static int8_t current_device = 0;
  inline static int64_t current_stream = 0;

  DeviceType type() const override { return static_type; }

  void setDevice(Device d) const override {
    TORCH_CHECK(d.is_privateuseone(), "Device must be PrivateUse1 type");
    current_device = d.index();
  }

  void uncheckedSetDevice(Device d) const noexcept override {
    current_device = d.index();
  }

  Device getDevice() const override {
    return Device(DeviceType::PrivateUse1, current_device);
  }

  Device exchangeDevice(Device d) const override {
    Device old_device = getDevice();
    setDevice(d);
    return old_device;
  }

  Stream getStream(Device d) const noexcept override {
    int64_t stream_id = d.index();
    return Stream(Stream::UNSAFE, d, stream_id);
  }

  Stream exchangeStream(Stream s) const noexcept override {
    auto old_stream = getStream(s.device());
    current_stream = s.id();
    return old_stream;
  }

  DeviceIndex deviceCount() const noexcept override { return 1; }
};

} // namespace impl
} // namespace c10

namespace at {
namespace detail {

C10_REGISTER_GUARD_IMPL(PrivateUse1, c10::impl::AIPUGuardImpl);
}
} // namespace at

#define AIPU_DRIVER_HANDLE_ERROR(status)                                       \
  do {                                                                         \
    if (status != AIPU_STATUS_SUCCESS) {                                       \
      const char *error_message = nullptr;                                     \
      aipu_get_error_message(aipu_ctx_, status, &error_message);               \
      std::cout << error_message;                                              \
    }                                                                          \
  } while (false)

/*! \brief Return whether a string starts with the given prefix. */
inline bool StrStartsWith(const std::string &str, const std::string &prefix) {
  if (prefix.size() > str.size())
    return false;
  return std::equal(str.c_str(), str.c_str() + prefix.size(), prefix.c_str());
}

class Context final {
public:
  aipu_ctx_handle_t *process_ctx = nullptr;
  std::mutex inst_lock;
  Context() {
    if (process_ctx == nullptr) {
      std::lock_guard<std::mutex> lock(inst_lock);
      if (process_ctx == nullptr) {
        aipu_status_t status = aipu_init_context(&process_ctx);
        if (status != AIPU_STATUS_SUCCESS) {
          //
        }
      }
    }
  };
  ~Context() {
    if (process_ctx != nullptr) {
      std::lock_guard<std::mutex> lock(inst_lock);
      if (process_ctx != nullptr) {
        aipu_status_t status = aipu_deinit_context(process_ctx);
        if (status != AIPU_STATUS_SUCCESS) {
          //
        }
        process_ctx = nullptr;
      }
    }
  };
};

Context *context() {
  static const std::unique_ptr<Context> context([]() -> Context * {
    try {
      return new Context();
    } catch (...) {
    }
    return nullptr;
  }());

  return context.get();
}

using namespace at;

struct AIPUAllocator final : Allocator {
  AIPUAllocator() = default;

  DataPtr allocate(size_t nbytes) override {
    void *data = nullptr;
    status_ = aipu_malloc(aipu_ctx_, nbytes, 32, 0, &data);
    AIPU_DRIVER_HANDLE_ERROR(status_);

    return {data, data, &ReportAndDelete,
            Device(DeviceType::PrivateUse1, aipu_device_index)};
  }

  static void ReportAndDelete(void *ptr) {
    if (!ptr) {
      return;
    }
    status_ = aipu_free(aipu_ctx_, &ptr);
    AIPU_DRIVER_HANDLE_ERROR(status_);
  }

  DeleterFnPtr raw_deleter() const override { return &ReportAndDelete; }

  void copy_data(void *dest, const void *src, std::size_t count) const final {
    default_copy_data(dest, src, count);
  }

  static aipu_ctx_handle_t *aipu_ctx_;
  static aipu_status_t status_;
};

// Register our dummy allocator
aipu_ctx_handle_t *AIPUAllocator::aipu_ctx_ = context()->process_ctx;
aipu_status_t AIPUAllocator::status_ = AIPU_STATUS_SUCCESS;
static AIPUAllocator global_custom_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_custom_alloc);

Tensor custom_empty_symint(c10::IntArrayRef size,
                           std::optional<ScalarType> dtype,
                           std::optional<Layout> layout,
                           std::optional<Device> device,
                           std::optional<bool> pin_memory,
                           std::optional<MemoryFormat> memory_format) {
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_generic(size, &global_custom_alloc, private_use_ks,
                                   c10::dtype_or_default(dtype), memory_format);
}

Tensor custom_empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride,
                            std::optional<ScalarType> dtype_opt,
                            std::optional<Layout> layout_opt,
                            std::optional<Device> device_opt,
                            std::optional<bool> pin_memory_opt) {
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  auto dtype = c10::dtype_or_default(dtype_opt);
  return at::detail::empty_strided_generic(size, stride, &global_custom_alloc,
                                           private_use_ks, dtype);
}

Tensor aipu_view(const Tensor &self, c10::IntArrayRef size) {
  IntArrayRef self_sizes = self.sizes();
  IntArrayRef self_strides = self.strides();
  DimVector inferred_size = infer_size_dv(size, self.numel());
  std::optional<DimVector> stride =
      at::detail::computeStride(self_sizes, self_strides, inferred_size);
  TORCH_CHECK(
      stride.has_value(),
      "view size is "
      "not compatible with input tensor's size and stride (at least one "
      "dimension"
      " spans across two contiguous subspaces). Use .reshape(...) instead.");

  Tensor self_ = at::detail::make_tensor<c10::TensorImpl>(
      c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(),
      self.dtype());
  self_.unsafeGetTensorImpl()->set_sizes_and_strides(inferred_size, *stride);
  self_.unsafeGetTensorImpl()->set_storage_offset(self.storage_offset());
  return self_;
}

void aipu_contiguous_copy(const Tensor &self, const Tensor &dst,
                          aipu_memcpy_kind_t kind) {
  auto aipu_ctx_ = AIPUAllocator::aipu_ctx_;

  if (self.storage_offset() == 0 && dst.storage_offset() == 0) {
    auto status = aipu_memcpy(aipu_ctx_, dst.data_ptr(), self.data_ptr(),
                              self.nbytes(), kind);
    AIPU_DRIVER_HANDLE_ERROR(status);
  } else {
    auto aipu_ctx_ = AIPUAllocator::aipu_ctx_;
    char *data_ptr = nullptr;
    auto size = self.nbytes();
    if (kind == AIPU_MEMCPY_DEVICE_TO_HOST) {
      auto offset = self.storage_offset() * self.itemsize();
      auto status = aipu_get_va(aipu_ctx_, self.data_ptr() - offset, &data_ptr);
      AIPU_DRIVER_HANDLE_ERROR(status);
      data_ptr += offset;

      memcpy(dst.data_ptr(), data_ptr, size);
    } else {
      auto offset = dst.storage_offset() * dst.itemsize();
      auto status = aipu_get_va(aipu_ctx_, dst.data_ptr() - offset, &data_ptr);
      AIPU_DRIVER_HANDLE_ERROR(status);
      data_ptr += offset;

      memcpy(data_ptr, self.data_ptr(), size);
    }
  }
}

char *get_data_ptr(const Tensor &self) {
  if (StrStartsWith(self.device().str(), "aipu")) {
    auto aipu_ctx_ = AIPUAllocator::aipu_ctx_;
    char *data_ptr = nullptr;
    auto status = aipu_get_va(aipu_ctx_, self.data_ptr(), &data_ptr);
    AIPU_DRIVER_HANDLE_ERROR(status);
    return data_ptr;
  } else {
    return static_cast<char *>(self.data_ptr());
  }
}

Tensor make_contiguous(const Tensor &self) {
  if (self.is_contiguous()) {
    return self;
  }

  Tensor result = at::empty_like(self);
  const int64_t numel = self.numel();
  const int64_t ndim = self.dim();

  if (ndim == 0 || numel == 0) {
    if (numel == 1) {
      result.copy_(self);
    }
    return result;
  }

  char *self_data = get_data_ptr(self);
  char *result_data = get_data_ptr(result);
  const int64_t item_size = self.itemsize();
  const auto shape = self.sizes();
  const auto strides = self.strides();

  std::vector<int64_t> contig_strides(ndim);
  contig_strides[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; --i) {
    contig_strides[i] = contig_strides[i + 1] * shape[i + 1];
  }

  for (int64_t linear_idx = 0; linear_idx < numel; ++linear_idx) {
    int64_t offset = 0;
    int64_t remaining = linear_idx;

    for (int i = 0; i < ndim; ++i) {
      const int64_t dim_index = remaining / contig_strides[i];
      remaining %= contig_strides[i];
      offset += dim_index * strides[i];
    }

    const int64_t src_offset = offset * item_size;
    const int64_t dst_offset = linear_idx * item_size;
    for (int64_t byte = 0; byte < item_size; ++byte) {
      result_data[dst_offset + byte] = self_data[src_offset + byte];
    }
  }

  return result;
}

Tensor aipu_copy_from(const Tensor &self, const Tensor &dst,
                      bool non_blocking = false) {
  auto kind = AIPU_MEMCPY_HOST_TO_DEVICE;
  if (StrStartsWith(self.device().str(), "aipu")) {
    kind = AIPU_MEMCPY_DEVICE_TO_HOST;
    if (StrStartsWith(dst.device().str(), "aipu")) {
      kind = AIPU_MEMCPY_DEVICE_TO_DEVICE;
    }
  }

  if (self.is_contiguous() && dst.is_contiguous()) {
    aipu_contiguous_copy(self, dst, kind);
  } else {
    Tensor new_dst = dst.is_contiguous() ? dst : at::empty_like(dst);
    Tensor new_src = make_contiguous(self);
    aipu_contiguous_copy(new_src, new_dst, kind);
    if (!new_dst.is_same(dst)) {
      dst.copy_(new_dst);
    }
  }
  return dst;
}

Tensor aipu_copy_from_and_resize(const Tensor &self, const Tensor &dst) {
  if (self.sizes() != dst.sizes()) {
    auto new_dst =
        custom_empty_symint(self.sizes(), self.scalar_type(), c10::nullopt,
                            c10::nullopt, c10::nullopt, c10::nullopt);
    auto kind = AIPU_MEMCPY_HOST_TO_DEVICE;
    if (StrStartsWith(self.device().str(), "aipu")) {
      kind = AIPU_MEMCPY_DEVICE_TO_HOST;
      if (StrStartsWith(dst.device().str(), "aipu")) {
        kind = AIPU_MEMCPY_DEVICE_TO_DEVICE;
      }
    }
    auto aipu_ctx_ = AIPUAllocator::aipu_ctx_;
    auto status = aipu_memcpy(aipu_ctx_, new_dst.data_ptr(), self.data_ptr(),
                              self.nbytes(), kind);
    AIPU_DRIVER_HANDLE_ERROR(status);

    return new_dst;
  }
  return aipu_copy_from(self, dst, false);
}

template <template <typename> class RND>
Tensor &random_kernel(Tensor &self, double cond1, double cond2,
                      c10::optional<Generator> gen) {
  CPUGeneratorImpl *generator = get_generator_or_default<CPUGeneratorImpl>(
      gen, at::detail::getDefaultCPUGenerator());
  int64_t numel = self.numel();
  char *data_ptr = get_data_ptr(self);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      self.scalar_type(), "random_kernel_aipu", [&]() {
        RND<double> distribution(cond1, cond2);

        auto data = reinterpret_cast<scalar_t *>(data_ptr);
        for (int i = 0; i < numel; ++i) {
          data[i] = static_cast<scalar_t>(distribution(generator));
        }
      });
  return self;
}

template <template <typename> class RND>
Tensor &random_from_to_kernel(Tensor &self, int64_t from,
                              c10::optional<int64_t> to_opt,
                              c10::optional<Generator> gen) {
  CPUGeneratorImpl *generator = get_generator_or_default<CPUGeneratorImpl>(
      gen, at::detail::getDefaultCPUGenerator());
  int64_t numel = self.numel();
  uint64_t range = static_cast<int64_t>(*to_opt) - static_cast<int64_t>(from);
  char *data_ptr = get_data_ptr(self);

  AT_DISPATCH_ALL_TYPES_AND2(
      ScalarType::Bool, ScalarType::Half, self.scalar_type(),
      "random_from_to_kernel_aipu", [&]() {
        RND<scalar_t> distribution(range, from);

        auto data = reinterpret_cast<scalar_t *>(data_ptr);
        for (int i = 0; i < numel; ++i) {
          data[i] = static_cast<scalar_t>(distribution(generator));
        }
      });
  return self;
}

Scalar _local_scalar_dense_aipu(const Tensor &self) {
  Scalar r;
  auto aipu_ctx_ = AIPUAllocator::aipu_ctx_;
  char *data_ptr = nullptr;
  auto offset = self.storage_offset() * self.itemsize();
  auto status = aipu_get_va(aipu_ctx_, self.data_ptr() - offset, &data_ptr);
  AIPU_DRIVER_HANDLE_ERROR(status);
  data_ptr += offset;

  AT_DISPATCH_ALL_TYPES_AND2(
      ScalarType::Bool, ScalarType::Half, self.scalar_type(),
      "_local_scalar_dense_aipu", [&]() {
        auto data = reinterpret_cast<scalar_t *>(data_ptr);
        scalar_t value = static_cast<scalar_t>(*data);
        r = Scalar(value);
      });
  return r;
}

Tensor &fill_scalar_aipu(Tensor &self, const Scalar &value) {
  int64_t numel = self.numel();
  char *data_ptr = get_data_ptr(self);

  AT_DISPATCH_ALL_TYPES_AND2(
      ScalarType::Bool, ScalarType::Half, self.scalar_type(),
      "fill_scalar_aipu", [&]() {
        auto data = reinterpret_cast<scalar_t *>(data_ptr);
        std::fill(data, data + numel, value.to<scalar_t>());
      });
  return self;
}

Tensor reshape_aipu(const Tensor &self, IntArrayRef shape) {
  auto new_dst = custom_empty_symint(shape, self.scalar_type(), c10::nullopt,
                                     c10::nullopt, c10::nullopt, c10::nullopt);
  return aipu_copy_from(self, new_dst, false);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", &custom_empty_symint);
  m.impl("empty_strided", &custom_empty_strided);
  m.impl("as_strided", native::as_strided_tensorimpl);
  m.impl("aten::view", &aipu_view);
  m.impl("aten::uniform_", &random_kernel<uniform_real_distribution>);
  m.impl("aten::normal_", &random_kernel<normal_distribution>);
  m.impl("aten::_copy_from", &aipu_copy_from);
  m.impl("aten::_copy_from_and_resize", &aipu_copy_from_and_resize);
  m.impl("aten::random_.from",
         &random_from_to_kernel<uniform_int_from_to_distribution>);
  m.impl("aten::_local_scalar_dense", &_local_scalar_dense_aipu);
  m.impl("aten::fill_.Scalar", &fill_scalar_aipu);
  m.impl("aten::reshape", &reshape_aipu);
}

void custom_cpu_fallback(const c10::OperatorHandle &op,
                         torch::jit::Stack *stack) {
  at::native::cpu_fallback(op, stack);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
}

at::Generator make_custom_generator(c10::DeviceIndex device_index) {
  return at::detail::getDefaultCPUGenerator();
}

REGISTER_GENERATOR_PRIVATEUSE1(make_custom_generator)

C10_REGISTER_GUARD_IMPL(PrivateUse1, c10::impl::AIPUGuardImpl)

// Register the autograd dispatch key for operators that have no dispatches
TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
  m.impl("isfinite", torch::autograd::autogradNotImplementedFallback());
  m.impl("reshape", torch::autograd::autogradNotImplementedFallback());
  m.impl("isclose", torch::autograd::autogradNotImplementedFallback());
  m.impl("tile", torch::autograd::autogradNotImplementedFallback());
  m.impl("hstack", torch::autograd::autogradNotImplementedFallback());
  m.impl("vstack", torch::autograd::autogradNotImplementedFallback());
  m.impl("resolve_conj", torch::autograd::autogradNotImplementedFallback());
  m.impl("resolve_neg", torch::autograd::autogradNotImplementedFallback());
  m.impl("to.dtype", torch::autograd::autogradNotImplementedFallback());
  m.impl("where.ScalarSelf", torch::autograd::autogradNotImplementedFallback());
  m.impl("where.ScalarOther",
         torch::autograd::autogradNotImplementedFallback());
  m.impl("embedding_backward",
         torch::autograd::autogradNotImplementedFallback());
  m.impl("pad", torch::autograd::autogradNotImplementedFallback());
  m.impl("repeat_interleave.self_Tensor",
         torch::autograd::autogradNotImplementedFallback());
}

namespace aipu {

bool is_available() { return context() != nullptr; }

int device_count() { return is_available() ? 1 : 0; }

int current_device() { return 0; }

std::string get_device_name() { return "aipu"; }

// This change is a temporary adaptation for CUDA 8.0
// Need to do
py::tuple get_device_capability(int device_index = 0) {
  return py::make_tuple(8, 0);
}
} // namespace aipu

struct _DeviceGuard {
  _DeviceGuard(int index, int prev_index) : idx(index), prev_idx(prev_index) {}

  int idx = 0;
  int prev_idx = -1;
};

struct _Device {
  _Device(c10::Device device) { idx = device.index(); }
  _Device(int index) { idx = index; }

  int idx = 0;
  int prev_idx = -1;
};

static std::unordered_map<int, at::Generator> default_generators = {
    {0, at::detail::getDefaultCPUGenerator()}};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("device_count", &aipu::device_count, "aipu device count");
  m.def("is_available", &aipu::is_available, "aipu is available");
  m.def("current_device", &aipu::current_device, "aipu current device");
  m.def("get_device_name", &aipu::get_device_name,
        "Return the name of the AIPU backend");
  // This change is a temporary adaptation for CUDA 8.0
  // Need to do
  m.def("get_device_capability", &aipu::get_device_capability,
        py::arg("device") = 0, "Return the capability of the AIPU backend");

  m.def("_is_in_bad_fork", []() { return py::bool_(false); });
  m.def("manual_seed_all", [](int seed) { std::srand(seed); });
  m.attr("default_generators") = &default_generators;

  py::class_<_DeviceGuard>(m, "_DeviceGuard", py::module_local())
      .def(py::init(
          [](int index) { return std::make_unique<_DeviceGuard>(index, -1); }))
      .def("__enter__", [](_DeviceGuard &self) { ; })
      .def("__exit__",
           [](_DeviceGuard &self, pybind11::object type, pybind11::object value,
              pybind11::object traceback) { return py::bool_(false); });

  py::class_<_Device>(m, "device", py::module_local())
      .def(py::init(
          [](c10::Device device) { return std::make_unique<_Device>(device); }))
      .def(py::init([](int index) { return std::make_unique<_Device>(index); }))
      .def("__enter__", [](_Device &self) { ; })
      .def("__exit__",
           [](_Device &self, pybind11::object type, pybind11::object value,
              pybind11::object traceback) { return py::bool_(false); });
}
