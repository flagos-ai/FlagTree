#!/usr/bin/env python3
# flagtree tle
# Copyright (c) 2025  XCoreSigma Inc. All rights reserved.
"""
TLE Test Runner

Runs all TLE-related tests, including unit tests and end-to-end tests
"""

import sys
import os
import subprocess
from pathlib import Path


def run_command(cmd, description, env=None):
    """Run command and display results"""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)

    if result.stdout:
        print("📤 Output:")
        print(result.stdout)

    if result.stderr:
        print("❌ Errors:")
        print(result.stderr)

    return result.returncode == 0


def main():
    """Main function"""
    print("🚀 Starting TLE Test Suite")

    # Get current directory (should be /root/code/triton/python/test/tle/)
    script_dir = Path(__file__).parent.resolve()  # Use absolute path
    triton_root = script_dir.parent.parent  # This should be /root/code/triton

    # Add python to path
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{triton_root}/python:{env.get('PYTHONPATH', '')}"

    all_passed = True

    # 1. Run basic unit tests (relative to current directory)
    success = run_command("python -m pytest unit/test_tle.py  -v --tb=short", "TLE Basic Unit Tests", env)
    all_passed = all_passed and success

    # 2. Run end-to-end tests (if GPU available)
    try:
        import torch
        if torch.cuda.is_available():
            integration_tests = [
                ("integration/test_tle_pipeline_e2e.py", "TLE Pipeline End-to-End Tests"),
                ("integration/test_tle_local_store.py", "TLE Local Store Integration Tests"),
                ("integration/test_tle_tma_copy.py", "TLE TMA Copy Integration Tests"),
                ("integration/test_tle_gemm.py", "TLE TMA gemm Integration Tests"),
            ]

            for test_file, description in integration_tests:
                success = run_command(f"python -m pytest {test_file} -v --tb=short", description, env)
                all_passed = all_passed and success
        else:
            print("\n⚠️  Skipping end-to-end tests: CUDA GPU not detected")
    except ImportError:
        print("\n⚠️  Skipping end-to-end tests: PyTorch not installed")

    # 3. MLIR conversion tests
    print(f"\n{'='*60}")
    print("⚙️  MLIR Conversion Tests")
    print(f"{'='*60}")

    success = run_command("python mlir/verify_tle_conversion.py", "TLE MLIR Conversion Tests", env)
    all_passed = all_passed and success

    # 4. Import testing (skip syntax check for now)
    print(f"\n{'='*60}")
    print("📦 Module Import Test")
    print(f"{'='*60}")

    try:
        import triton.experimental.tle as tle
        print("✅ TLE module import successful")

        # Test basic functionality
        if hasattr(tle, 'scope') and hasattr(tle, 'pipeline'):
            print("✅ Core functionality available")
        else:
            print("❌ Core functionality not available")
            all_passed = False

    except Exception as e:
        print(f"❌ TLE module import failed: {e}")
        all_passed = False

    # Final results
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 All tests passed!")
        print("✅ TLE functionality is ready")
    else:
        print("❌ Some tests failed")
        print("🔧 Please check the above error messages and fix issues")
    print(f"{'='*60}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
