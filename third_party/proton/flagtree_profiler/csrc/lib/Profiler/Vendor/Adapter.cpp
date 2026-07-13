#include "Profiler/Vendor/Adapter.h"

#include "Profiler/Vendor/CannAdapter.h"
#include "Utility/String.h"

namespace proton {

const VendorAdapter *VendorAdapterRegistry::find(const std::string &name) {
  auto lower = toLower(name);
  if (lower == "cann") {
    return &CannAdapter::instance();
  }
  return nullptr;
}

std::vector<std::string> VendorAdapterRegistry::names() { return {"cann"}; }

} // namespace proton
