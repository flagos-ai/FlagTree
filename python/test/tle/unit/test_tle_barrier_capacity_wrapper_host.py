"""Exercise the isolated real wrapper with the C ABI; no SDK initialization."""
import os
import ctypes
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(os.environ.get("FLAGTREE_TLE_CAPACITY_SOURCE_ROOT",Path(__file__).resolve().parents[4]))
source=ROOT/'python/triton/experimental/tle/language/flagcx_wrapper.py'
spec=importlib.util.spec_from_file_location("_tle_capacity_wrapper_probe",source)
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)


class WrapperCapacityTest(unittest.TestCase):
    def instance(self,functions):
        value=module.FLAGCXLibrary.__new__(module.FLAGCXLibrary)
        value._funcs=functions
        return value

    def test_missing_query_has_explicit_capability_and_error(self):
        value=self.instance({})
        self.assertFalse(value.supports_intra_barrier_count_query)
        with self.assertRaisesRegex(RuntimeError,"capacity query"):
            value.flagcxDevCommGetIntraBarrierCount(ctypes.c_void_p(4096))

    def test_capacity_and_handle_cross_actual_ctypes_signature(self):
        seen=[]

        @ctypes.CFUNCTYPE(ctypes.c_int,ctypes.c_void_p,ctypes.POINTER(ctypes.c_int))
        def query(handle,out):
            seen.append(handle);out[0]=129;return 0

        value=self.instance({"flagcxDevCommGetIntraBarrierCount":query})
        self.assertTrue(value.supports_intra_barrier_count_query)
        self.assertEqual(value.flagcxDevCommGetIntraBarrierCount(ctypes.c_void_p(4096)),129)
        self.assertEqual(seen,[4096])

    def test_query_status_error_is_preserved(self):
        @ctypes.CFUNCTYPE(ctypes.c_int,ctypes.c_void_p,ctypes.POINTER(ctypes.c_int))
        def query(handle,out):
            return 1

        value=self.instance({"flagcxDevCommGetIntraBarrierCount":query,
                             "flagcxGetErrorString":lambda status:b"capacity-query-fixture-error"})
        with self.assertRaisesRegex(RuntimeError,"capacity-query-fixture-error"):
            value.flagcxDevCommGetIntraBarrierCount(ctypes.c_void_p(4096))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
