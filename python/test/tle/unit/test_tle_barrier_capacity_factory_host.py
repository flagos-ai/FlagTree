"""Exercise the actual factory AST with a recording SDK, without CUDA imports."""
import os
import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(os.environ.get("FLAGTREE_TLE_CAPACITY_SOURCE_ROOT",Path(__file__).resolve().parents[4]))
SOURCE=ROOT/'python/triton/experimental/tle/language/communication.py'


def environment(capacity, supports=True, error=None):
    calls, contexts = [], []

    class SDK:
        supports_intra_barrier_count_query = supports

        def flagcxDevCommGetIntraBarrierCount(self,handle):
            calls.append("query")
            if error: raise RuntimeError(error)
            return capacity

        def __getattr__(self,name):
            def call(*args,**kwargs):
                calls.append(name)
                return SimpleNamespace(value=4096)
            return call

    namespace = dict(comm=object(),rank=0,dev_comm=None,dev_mem=None,win=None,
        _FLAGCX_WINDOW_ALIGNMENT=4096,FLAGCX_WIN_COLL_SYMMETRIC=1,
        flagcxDevCommRequirements=SimpleNamespace,flagcx=SDK(),
        torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda *a:calls.append("cuda_sync"))),
        dist=SimpleNamespace(is_initialized=lambda:False),
        _distributed_barrier=lambda:calls.append("host_barrier"))
    function = next(n for n in ast.parse(SOURCE.read_text()).body
                    if isinstance(n,ast.FunctionDef) and n.name == "create_dist_tensor")
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(SOURCE),"exec"),namespace)
    runtime = ModuleType("triton.runtime")

    def make_context(*args):
        result=SimpleNamespace(handles=args);contexts.append(result);return result

    runtime.DistributedRtContext = make_context
    package = ModuleType("triton");package.__path__=[];package.runtime=runtime
    buffer = SimpleNamespace(is_cuda=True,is_contiguous=lambda:True,data_ptr=lambda:4096,
                             numel=lambda:4096,element_size=lambda:1,device="cuda:0")

    def invoke(count):
        with patch.dict(sys.modules,{"triton":package,"triton.runtime":runtime}),redirect_stdout(io.StringIO()):
            return namespace["create_dist_tensor"](buffer,intra_barrier_count=count)

    return invoke,calls,contexts,namespace


class FactoryCapacityTest(unittest.TestCase):
    def test_undersized_capacity_rejected_and_resources_released(self):
        for actual in [0,36,128]:
            with self.subTest(actual=actual):
                invoke,calls,contexts,namespace = environment(actual)
                with self.assertRaisesRegex(RuntimeError,"provisioned"):
                    invoke(129)
                self.assertFalse(contexts)
                self.assertIsNone(namespace["dev_comm"])
                self.assertIsNone(namespace["win"])
                self.assertIn("flagcxDevCommDestroy",calls)
                self.assertIn("flagcxCommWindowDeregister",calls)

    def test_missing_query_rejected_before_cuda_or_sdk_side_effects(self):
        invoke,calls,contexts,_ = environment(36,supports=False)
        with self.assertRaisesRegex(RuntimeError,"capacity query"):
            invoke(17)
        self.assertEqual(calls,[])
        self.assertEqual(contexts,[])

    def test_actual_provisioned_capacity_is_published(self):
        invoke,calls,contexts,_ = environment(36)
        context=invoke(17)
        self.assertEqual(context.intra_barrier_count,36)
        self.assertEqual(len(contexts),1)
        self.assertLess(calls.index("query"),calls.index("flagcxDevMemCreate"))

    def test_requested_129_is_supported(self):
        invoke,_,_,_=environment(129)
        self.assertEqual(invoke(129).intra_barrier_count,129)

    def test_legacy_single_slot_sdk_is_supported(self):
        invoke,_,_,_=environment(None,supports=False)
        self.assertEqual(invoke(1).intra_barrier_count,1)

    def test_sdk_query_exception_releases_resources(self):
        invoke,calls,contexts,namespace=environment(None,error="SDK query failed")
        with self.assertRaisesRegex(RuntimeError,"SDK query failed"):
            invoke(17)
        self.assertEqual(contexts,[])
        self.assertIsNone(namespace["dev_comm"])
        self.assertIsNone(namespace["win"])
        self.assertIn("flagcxDevCommDestroy",calls)
        self.assertIn("flagcxCommWindowDeregister",calls)

    def test_active_context_is_still_rejected(self):
        invoke,calls,_,namespace=environment(129)
        namespace["dev_comm"]=object()
        with self.assertRaisesRegex(RuntimeError,"active FlagCX"):
            invoke(17)
        self.assertEqual(calls,[])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
