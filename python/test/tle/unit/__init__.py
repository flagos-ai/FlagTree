# Copyright (c) 2025  XCoreSigma Inc. All rights reserved.
# flagtree tle
"""
TLE Test Runner

Provides simplified test execution interface for developing and debugging TLE functionality
"""

import sys
import os

# Add TLE module path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def run_tle_tests():
    """Run all TLE tests"""
    import pytest

    # Run tests
    test_file = os.path.join(os.path.dirname(__file__), 'test_tle.py')
    result = pytest.main([test_file, "-v", "--tb=short"])

    return result == 0


if __name__ == "__main__":
    success = run_tle_tests()
    if success:
        print("✅ All TLE tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)
