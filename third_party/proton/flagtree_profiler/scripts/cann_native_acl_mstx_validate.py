#!/usr/bin/env python3
"""Validate real CANN capture/import with ACLNN Add and MSTX.

This path does not require torch/torch_npu. It compiles a tiny C++ program that
initializes AscendCL, creates an ACL stream, emits an MSTX range, optionally
launches native ACLNN Add compute, and exits. CBLAS GEMM paths are retained only
as fallback diagnostics. The program is launched under external msprof, and
exported CSV files are post-imported through Proton's CANN importer.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Sequence

CPP_SOURCE = r"""
#include <acl/acl.h>
#include <acl/ops/acl_cblas.h>
#include <aclnnop/aclnn_add.h>
#include <mstx/ms_tools_ext.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <unistd.h>

void print_acl_error(const char *expr, aclError ret) {
  std::fprintf(stderr, "%s failed: %d\n", expr, static_cast<int>(ret));
  const char *recent = aclGetRecentErrMsg();
  if (recent != nullptr && recent[0] != '\0') {
    std::fprintf(stderr, "recent_acl_error: %s\n", recent);
  }
}

#define CHECK_ACL(expr)                                                        \
  do {                                                                         \
    auto _ret = (expr);                                                        \
    if (_ret != ACL_SUCCESS) {                                                 \
      print_acl_error(#expr, _ret);                                            \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define TRY_ACL(expr)                                                          \
  do {                                                                         \
    auto _ret = (expr);                                                        \
    if (_ret != ACL_SUCCESS) {                                                 \
      print_acl_error(#expr, _ret);                                            \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CHECK_ACLNN(expr)                                                      \
  do {                                                                         \
    auto _ret = (expr);                                                        \
    if (_ret != 0) {                                                           \
      std::fprintf(stderr, "%s failed: %d\n", #expr, static_cast<int>(_ret));  \
      const char *recent = aclGetRecentErrMsg();                               \
      if (recent != nullptr && recent[0] != '\0') {                            \
        std::fprintf(stderr, "recent_acl_error: %s\n", recent);                \
      }                                                                        \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int64_t square_dim_from_elements(int64_t elements) {
  if (elements <= 0) {
    return 0;
  }
  int64_t dim = 1;
  while ((dim + 1) <= elements / (dim + 1)) {
    ++dim;
  }
  return dim;
}

aclTensor *create_acl_tensor(const std::vector<int64_t> &shape,
                             aclDataType data_type, void *device_addr) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] =
        shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
  }
  return aclCreateTensor(shape.data(), shape.size(), data_type, strides.data(),
                         0, ACL_FORMAT_ND, shape.data(), shape.size(),
                         device_addr);
}

int run_aclnn_add_compute(aclrtStream stream, int iters, int64_t elements) {
  if (elements <= 0) {
    std::fprintf(stderr, "invalid ACLNN Add element count: %ld\n", elements);
    return 1;
  }
  const std::vector<int64_t> shape = {elements};
  const size_t bytes = static_cast<size_t>(elements) * sizeof(float);
  std::vector<float> host_a(static_cast<size_t>(elements), 1.0f);
  std::vector<float> host_b(static_cast<size_t>(elements), 2.0f);
  std::vector<float> host_out(static_cast<size_t>(elements), 0.0f);
  float alpha_value = 1.0f;

  void *dev_a = nullptr;
  void *dev_b = nullptr;
  void *dev_out = nullptr;
  CHECK_ACL(aclrtMalloc(&dev_a, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc(&dev_b, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc(&dev_out, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMemcpy(dev_a, bytes, host_a.data(), bytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(dev_b, bytes, host_b.data(), bytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  aclTensor *self = create_acl_tensor(shape, ACL_FLOAT, dev_a);
  aclTensor *other = create_acl_tensor(shape, ACL_FLOAT, dev_b);
  aclTensor *out = create_acl_tensor(shape, ACL_FLOAT, dev_out);
  aclScalar *alpha = aclCreateScalar(&alpha_value, ACL_FLOAT);
  if (!self || !other || !out || !alpha) {
    std::fprintf(stderr, "aclCreateTensor/aclCreateScalar failed\n");
    return 1;
  }

  for (int i = 0; i < iters; ++i) {
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    CHECK_ACLNN(aclnnAddGetWorkspaceSize(self, other, alpha, out,
                                         &workspace_size, &executor));
    void *workspace = nullptr;
    if (workspace_size > 0) {
      CHECK_ACL(aclrtMalloc(&workspace, workspace_size,
                            ACL_MEM_MALLOC_HUGE_FIRST));
    }
    CHECK_ACLNN(aclnnAdd(workspace, workspace_size, executor, stream));
    if (workspace != nullptr) {
      CHECK_ACL(aclrtFree(workspace));
    }
  }

  CHECK_ACL(aclrtSynchronizeStream(stream));
  CHECK_ACL(aclrtMemcpy(host_out.data(), bytes, dev_out, bytes,
                        ACL_MEMCPY_DEVICE_TO_HOST));

  std::printf("acl_compute_op=aclnnAdd\n");
  std::printf("acl_compute_elements=%ld\n", elements);
  std::printf("acl_compute_iters=%d\n", iters);
  std::printf("acl_compute_first_f32=%.1f\n", host_out[0]);

  aclDestroyScalar(alpha);
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  CHECK_ACL(aclrtFree(dev_a));
  CHECK_ACL(aclrtFree(dev_b));
  CHECK_ACL(aclrtFree(dev_out));
  return 0;
}

int run_acl_hgemm_compute(aclrtStream stream, int iters, int64_t elements) {
  int64_t dim64 = square_dim_from_elements(elements);
  if (dim64 <= 0 || dim64 > static_cast<int64_t>(INT32_MAX)) {
    std::fprintf(stderr, "invalid GEMM element count: %ld\n", elements);
    return 1;
  }
  const int m = static_cast<int>(dim64);
  const int n = static_cast<int>(dim64);
  const int k = static_cast<int>(dim64);

  const size_t matrix_elems = static_cast<size_t>(m) * static_cast<size_t>(n);
  const size_t bytes = matrix_elems * sizeof(aclFloat16);
  std::vector<aclFloat16> host_a(matrix_elems, aclFloatToFloat16(1.0f));
  std::vector<aclFloat16> host_b(matrix_elems, aclFloatToFloat16(1.0f));
  std::vector<aclFloat16> host_c(matrix_elems, aclFloatToFloat16(0.0f));
  aclFloat16 host_alpha = aclFloatToFloat16(1.0f);
  aclFloat16 host_beta = aclFloatToFloat16(0.0f);

  void *dev_a = nullptr;
  void *dev_b = nullptr;
  void *dev_c = nullptr;
  CHECK_ACL(aclrtMalloc(&dev_a, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc(&dev_b, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc(&dev_c, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMemcpy(dev_a, bytes, host_a.data(), bytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(dev_b, bytes, host_b.data(), bytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(dev_c, bytes, host_c.data(), bytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  const auto *alpha = &host_alpha;
  const auto *matrix_a = reinterpret_cast<const aclFloat16 *>(dev_a);
  const auto *matrix_b = reinterpret_cast<const aclFloat16 *>(dev_b);
  const auto *beta = &host_beta;
  auto *matrix_c = reinterpret_cast<aclFloat16 *>(dev_c);

  for (int i = 0; i < iters; ++i) {
    TRY_ACL(aclblasHgemm(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N, m, n, k,
                         alpha, matrix_a, -1, matrix_b, -1, beta, matrix_c,
                         -1, ACL_COMPUTE_HIGH_PRECISION, stream));
  }
  CHECK_ACL(aclrtSynchronizeStream(stream));
  CHECK_ACL(aclrtMemcpy(host_c.data(), bytes, dev_c, bytes,
                        ACL_MEMCPY_DEVICE_TO_HOST));

  auto first_bits = reinterpret_cast<const uint16_t *>(host_c.data())[0];
  std::printf("acl_gemm_op=aclblasHgemm\n");
  std::printf("acl_gemm_m=%d\n", m);
  std::printf("acl_gemm_n=%d\n", n);
  std::printf("acl_gemm_k=%d\n", k);
  std::printf("acl_gemm_iters=%d\n", iters);
  std::printf("acl_gemm_first_fp16_bits=0x%04x\n",
              static_cast<unsigned>(first_bits));

  CHECK_ACL(aclrtFree(dev_a));
  CHECK_ACL(aclrtFree(dev_b));
  CHECK_ACL(aclrtFree(dev_c));
  return 0;
}

int run_acl_s8gemm_compute(aclrtStream stream, int iters, int64_t elements) {
  int64_t dim64 = square_dim_from_elements(elements);
  if (dim64 <= 0 || dim64 > static_cast<int64_t>(INT32_MAX)) {
    std::fprintf(stderr, "invalid S8GEMM element count: %ld\n", elements);
    return 1;
  }
  const int m = static_cast<int>(dim64);
  const int n = static_cast<int>(dim64);
  const int k = static_cast<int>(dim64);

  const size_t matrix_elems = static_cast<size_t>(m) * static_cast<size_t>(n);
  const size_t bytes_ab = matrix_elems * sizeof(int8_t);
  const size_t bytes_c = matrix_elems * sizeof(int32_t);
  std::vector<int8_t> host_a(matrix_elems, 1);
  std::vector<int8_t> host_b(matrix_elems, 1);
  std::vector<int32_t> host_c(matrix_elems, 0);
  int32_t alpha = 1;
  int32_t beta = 0;

  void *dev_a = nullptr;
  void *dev_b = nullptr;
  void *dev_c = nullptr;
  CHECK_ACL(aclrtMalloc(&dev_a, bytes_ab, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc(&dev_b, bytes_ab, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc(&dev_c, bytes_c, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMemcpy(dev_a, bytes_ab, host_a.data(), bytes_ab,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(dev_b, bytes_ab, host_b.data(), bytes_ab,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(dev_c, bytes_c, host_c.data(), bytes_c,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  const auto *matrix_a = reinterpret_cast<const int8_t *>(dev_a);
  const auto *matrix_b = reinterpret_cast<const int8_t *>(dev_b);
  auto *matrix_c = reinterpret_cast<int32_t *>(dev_c);

  for (int i = 0; i < iters; ++i) {
    TRY_ACL(aclblasS8gemm(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N, m, n, k,
                          &alpha, matrix_a, -1, matrix_b, -1, &beta, matrix_c,
                          -1, ACL_COMPUTE_HIGH_PRECISION, stream));
  }
  CHECK_ACL(aclrtSynchronizeStream(stream));
  CHECK_ACL(aclrtMemcpy(host_c.data(), bytes_c, dev_c, bytes_c,
                        ACL_MEMCPY_DEVICE_TO_HOST));

  std::printf("acl_gemm_op=aclblasS8gemm\n");
  std::printf("acl_gemm_m=%d\n", m);
  std::printf("acl_gemm_n=%d\n", n);
  std::printf("acl_gemm_k=%d\n", k);
  std::printf("acl_gemm_iters=%d\n", iters);
  std::printf("acl_gemm_first_i32=%d\n", host_c[0]);

  CHECK_ACL(aclrtFree(dev_a));
  CHECK_ACL(aclrtFree(dev_b));
  CHECK_ACL(aclrtFree(dev_c));
  return 0;
}

int main(int argc, char **argv) {
  int device = 0;
  int iters = 20;
  int sleep_us = 1000;
  int compute_enabled = 1;
  int64_t elements = 1048576;
  int compute_kind = 0;
  if (argc > 1) {
    device = std::atoi(argv[1]);
  }
  if (argc > 2) {
    iters = std::atoi(argv[2]);
  }
  if (argc > 3) {
    sleep_us = std::atoi(argv[3]);
  }
  if (argc > 4) {
    compute_enabled = std::atoi(argv[4]);
  }
  if (argc > 5) {
    elements = std::atoll(argv[5]);
  }
  if (argc > 6) {
    compute_kind = std::atoi(argv[6]);
  }

  CHECK_ACL(aclInit(nullptr));
  CHECK_ACL(aclrtSetDevice(device));

  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream));

  auto domain = mstxDomainCreateA("proton");
  uint64_t range_id = 0;
  if (domain) {
    range_id = mstxDomainRangeStartA(domain, "proton_cann_native_acl_mstx_probe", stream);
  } else {
    range_id = mstxRangeStartA("proton_cann_native_acl_mstx_probe", stream);
  }

  std::printf("mstx_domain=%p\n", domain);
  std::printf("mstx_range_id=%lu\n", range_id);

  if (compute_enabled) {
    int compute_ret = 1;
    if (compute_kind == 1) {
      compute_ret = run_acl_hgemm_compute(stream, iters, elements);
    } else if (compute_kind == 2) {
      compute_ret = run_acl_s8gemm_compute(stream, iters, elements);
    } else if (compute_kind == 3) {
      compute_ret = run_aclnn_add_compute(stream, iters, elements);
    } else {
      compute_ret = run_aclnn_add_compute(stream, iters, elements);
      if (compute_ret != 0) {
        std::fprintf(stderr, "aclnnAdd failed; trying aclblasHgemm fallback.\n");
        compute_ret = run_acl_hgemm_compute(stream, iters, elements);
      }
      if (compute_ret != 0) {
        std::fprintf(stderr, "aclblasHgemm failed; trying aclblasS8gemm fallback.\n");
        compute_ret = run_acl_s8gemm_compute(stream, iters, elements);
      }
    }
    if (compute_ret != 0) {
      std::fprintf(stderr, "native_acl_compute failed for all attempted compute paths.\n");
      return 1;
    }
  } else {
    for (int i = 0; i < iters; ++i) {
      CHECK_ACL(aclrtSynchronizeStream(stream));
      usleep(sleep_us);
    }
  }

  if (range_id != 0) {
    if (domain) {
      mstxDomainRangeEnd(domain, range_id);
    } else {
      mstxRangeEnd(range_id);
    }
  }
  if (domain) {
    mstxDomainDestroy(domain);
  }

  CHECK_ACL(aclrtSynchronizeStream(stream));
  CHECK_ACL(aclrtDestroyStream(stream));
  CHECK_ACL(aclrtResetDevice(device));
  CHECK_ACL(aclFinalize());

  return range_id == 0 ? 2 : 0;
}
"""


def _make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/proton_cann_native_acl_mstx")
    parser.add_argument("--cann", default=os.environ.get("ASCEND_TOOLKIT_PATH", "/usr/local/Ascend/cann-8.5.0"))
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--msprof", default="msprof")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--sleep-us", type=int, default=1000)
    parser.add_argument(
        "--elements",
        type=int,
        default=1048576,
        help="Element count for ACLNN Add, or approximate matrix elements for GEMM fallbacks.",
    )
    parser.add_argument(
        "--skip-compute",
        action="store_true",
        help="Only emit ACL runtime calls and MSTX ranges; do not launch compute.",
    )
    parser.add_argument(
        "--compute-kind",
        choices=("auto", "aclnn-add", "hgemm", "s8gemm"),
        default="auto",
        help="Native compute path to run inside the MSTX range; auto prefers ACLNN Add.",
    )
    parser.add_argument(
        "--allow-compute-fail",
        action="store_true",
        help="Import msprof diagnostics and exit 0 even if native compute fails.",
    )
    parser.add_argument("--clean", action="store_true")
    return parser


def _run(cmd: Sequence[str]) -> None:
    print("+", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run([str(part) for part in cmd], check=True)


def _run_capture(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(str(part) for part in cmd), flush=True)
    completed = subprocess.run(
        [str(part) for part in cmd],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, end="")
    return completed


def _available_lib_flags(lib_dirs: Sequence[pathlib.Path], names: Sequence[str]) -> list[str]:
    flags: list[str] = []
    for name in names:
        for lib_dir in lib_dirs:
            if (lib_dir / f"lib{name}.so").exists() or (lib_dir / f"lib{name}.a").exists():
                flags.append(f"-l{name}")
                break
    return flags


def _compile_probe(cxx: str, cann: pathlib.Path, source: pathlib.Path, binary: pathlib.Path) -> None:
    include_dirs = [
        cann / "include",
        cann / "aarch64-linux" / "include",
        cann / "tools" / "mstx" / "include",
    ]
    lib_dirs = [
        cann / "aarch64-linux" / "lib64",
        cann / "tools" / "mstx" / "lib64",
    ]
    cmd = [cxx, "-std=c++17", source, "-o", binary]
    for path in include_dirs:
        cmd.extend(["-I", path])
    for path in lib_dirs:
        cmd.extend(["-L", path])
    for path in lib_dirs:
        cmd.append(f"-Wl,-rpath,{path}")
    cmd.extend(_available_lib_flags(lib_dirs, ["opapi", "nnopbase", "aclnn_ops"]))
    cmd.extend(["-lacl_cblas", "-lacl_op_compiler", "-lascendcl", "-lms_tools_ext", "-ldl", "-lpthread"])
    _run(cmd)


def main() -> int:
    args = _make_arg_parser().parse_args()
    out = pathlib.Path(args.out)
    cann = pathlib.Path(args.cann)
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o700)

    build_dir = out / "build"
    msprof_out = out / "msprof"
    build_dir.mkdir(parents=True, exist_ok=True)
    msprof_out.mkdir(parents=True, exist_ok=True)
    os.chmod(build_dir, 0o700)
    os.chmod(msprof_out, 0o700)

    source = build_dir / "cann_native_acl_mstx_probe.cpp"
    binary = build_dir / "cann_native_acl_mstx_probe"
    source.write_text(textwrap.dedent(CPP_SOURCE).strip() + "\n")
    _compile_probe(args.cxx, cann, source, binary)

    compute_kind_id = {
        "auto": "0",
        "hgemm": "1",
        "s8gemm": "2",
        "aclnn-add": "3",
    }[args.compute_kind]
    msprof_result = _run_capture([
        args.msprof,
        "--msproftx=on",
        f"--output={msprof_out}",
        binary,
        str(args.device),
        str(args.iters),
        str(args.sleep_us),
        "0" if args.skip_compute else "1",
        str(args.elements),
        compute_kind_id,
    ])
    compute_failed = (not args.skip_compute and "acl_compute_op=acl" not in msprof_result.stdout
                      and "acl_gemm_op=acl" not in msprof_result.stdout)
    if compute_failed:
        print(
            "native_compute_status failed: no native ACLNN/CBLAS compute path completed; "
            "continuing to import exported msprof diagnostics.",
            file=sys.stderr,
        )

    csv_files = sorted(msprof_out.rglob("*.csv"))
    print("exported_csv_count", len(csv_files))
    for path in csv_files[:40]:
        print("exported_csv", path)

    script_dir = pathlib.Path(__file__).resolve().parent
    post_import = script_dir / "cann_post_import_msprof.py"
    post_import_base = out / "post_import"
    _run([
        sys.executable,
        post_import,
        "--base",
        post_import_base,
        "--msprof-output",
        msprof_out,
        "--metrics",
        "aicore,bandwidth",
    ])

    print("DONE")
    print("native_probe", binary)
    print("post_import_vendor_json", post_import_base.with_suffix(".vendor.json"))
    if compute_failed and not args.allow_compute_fail:
        print(
            "native_compute_validation FAILED: CANN capture/import diagnostics "
            "were collected, but no native compute op completed.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
