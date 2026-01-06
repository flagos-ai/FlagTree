#pragma once

#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>

namespace mlir {
namespace flagtree {
// this is the unified hardware abstraction for hardware
// to determined if these abstraction is specified, using std::optional is
// needed using in passes: if(uh_flagtree->xxx()){...}

class UnifiedHardware {

public:
  UnifiedHardware() = default;
  virtual ~UnifiedHardware() = default;
  virtual bool isRegistered() const;
  virtual int getDMATag() const;
  virtual int getSharedMemoryTag() const;
  virtual bool getIncubatedTag() const;
  virtual std::string getReduceStrategy() const;
  virtual std::string getFlagTreeBackend() const;
#ifdef FLAGTREE_BACKEND
  static bool registered;
  int getDMATag();
  int getSharedMemoryTag();
  std::string getFlagTreeBackend() { return FLAGTREE_BACKEND; }
#else
  static constexpr bool registered = false;
  void *getDMATag() { return nullptr; }
  void *getSharedMemoryTag() { return nullptr; }
  std::string getFlagTreeBackend() { return "default"; }
#endif
};

std::unique_ptr<UnifiedHardware> createUnifiedHardwareManager();

} // namespace flagtree
} // namespace mlir

#define SET_REGISTER_FLAG(_Ty, FLAG) bool _Ty::registered = FLAG;

#define FLAGTREE_REGISTRAR_GET(_Ty, _Fn, _VAL)                                 \
  decltype(_VAL) _Ty::get##_Fn() { return static_cast<decltype(_VAL)>(_VAL); }

#ifdef FLAGTREE_BACKEND
#define FLAGTREE_REGISTRAR(fn_name, _VAL)                                      \
  using UnifiedHardwareType = mlir::flagtree::UnifiedHardware;                 \
  FLAGTREE_REGISTRAR_GET(UnifiedHardwareType, fn_name, _VAL)                   \
  SET_REGISTER_FLAG(UnifiedHardwareType, true)
#else
#define FLAGTREE_REGISTRAR(...)
#endif
