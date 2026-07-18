// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef PROTON_COMMON_DEVICE_H_
#define PROTON_COMMON_DEVICE_H_

#include <cstdint>
#include <string>

namespace proton {

enum class DeviceType { HIP, CUDA, COUNT };

template <DeviceType T> struct DeviceTraits;

template <> struct DeviceTraits<DeviceType::CUDA> {
  constexpr static DeviceType type = DeviceType::CUDA;
  constexpr static const char *name = "CUDA";
};

template <> struct DeviceTraits<DeviceType::HIP> {
  constexpr static DeviceType type = DeviceType::HIP;
  constexpr static const char *name = "HIP";
};

struct Device {
  DeviceType type;
  uint64_t id;
  uint64_t clockRate;       // khz
  uint64_t memoryClockRate; // khz
  uint64_t busWidth;
  uint64_t numSms;
  std::string arch;

  Device() = default;

  Device(DeviceType type, uint64_t id, uint64_t clockRate,
         uint64_t memoryClockRate, uint64_t busWidth, uint64_t numSms,
         std::string arch)
      : type(type), id(id), clockRate(clockRate),
        memoryClockRate(memoryClockRate), busWidth(busWidth), numSms(numSms),
        arch(arch) {}
};

Device getDevice(DeviceType type, uint64_t index);

const std::string getDeviceTypeString(DeviceType type);

}; // namespace proton

#endif // PROTON_COMMON_DEVICE_H_
