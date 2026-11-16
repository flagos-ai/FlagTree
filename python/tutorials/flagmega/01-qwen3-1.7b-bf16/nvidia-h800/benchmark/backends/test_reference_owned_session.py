"""Actual CPU process regressions with an explicit fake NVML interface."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from owned_session import OwnedSession, Subreaper, identity, scan

WORK = Path(__file__).resolve().parents[2]
OUT = WORK / '.local/agents/dispatch/repair_reference_monitor_whole_owned_session/process-tests'
MONITOR = WORK / 'benchmark/run_monitored_backend_v4.py'
spec = importlib.util.spec_from_file_location('owned_v4_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def fake_nvml(fail_read=False):
    module = ModuleType('pynvml')
    module.__file__ = __file__
    module.calls = []
    def result(name, value=None):
        def call(*args, **kwargs):
            module.calls.append(name)
            if fail_read and name == 'apps' and module.calls.count('apps') == 3:
                raise RuntimeError('CPU fixture NVML read error')
            return value
        return call
    fields = {'nvmlInit':('init',None), 'nvmlShutdown':('shutdown',None),
        'nvmlDeviceGetHandleByUUID':('handle',object()),
        'nvmlDeviceGetComputeRunningProcesses':('apps',[]),
        'nvmlDeviceGetMemoryInfo':('memory',SimpleNamespace(used=0,reserved=0)),
        'nvmlDeviceGetUtilizationRates':('util',SimpleNamespace(gpu=0,memory=0)),
        'nvmlDeviceGetUUID':('uuid','GPU-5b5e235a-09a2-fe8b-7754-916b7eb85976'),
        'nvmlDeviceGetIndex':('index',3), 'nvmlDeviceGetClockInfo':('clock',0),
        'nvmlDeviceGetPowerUsage':('power',0),
        'nvmlDeviceGetPciInfo':('pci',SimpleNamespace(busId='00000000:00:00.0')),
        'nvmlSystemGetNVMLVersion':('version','explicit-CPU-fake'),
        'nvmlSystemGetDriverVersion':('driver','no-GPU-driver')}
    for name,(label,value) in fields.items():
        setattr(module,name,result(label,value))
    module.nvmlMemory_v2=2
    module.NVML_CLOCK_SM=1
    module.NVML_CLOCK_MEM=2
    return module


class WholeSession(unittest.TestCase):
    def setUp(self):
        self.directory = OUT / self._testMethodName
        self.directory.mkdir(parents=True,exist_ok=False)
        self.guard = Subreaper()
        self.guard.__enter__()
        self.owned = []

    def tearDown(self):
        for owner in self.owned:
            result = owner.cleanup()
            self.assertTrue(result['terminal_qualified'], result)
            self.assertEqual(scan(owner.anchor['sid']), [])
        self.guard.__exit__(None,None,None)

    def run_monitor(self, body, fail_read=False, argv=None, **bounds):
        payload = self.directory/'payload.py'
        payload.write_text(body)
        job_id = 'CPU-' + self._testMethodName
        job = self.directory/'.local/jobs'/job_id
        config={'schema':'flagmega.monitored-reference-config/v4','CPU_fixture_only':True,
            'backend':'vllm','job_id':job_id,'assignment_id':'T097_actual_CPU_fixture',
            'agent_id':'/root/graph_profiler','GPU_UUID':'GPU-5b5e235a-09a2-fe8b-7754-916b7eb85976',
            'physical_index':3,'gpu_selector_kind':'nvml-index','cuda_visible_device':'3',
            'GPU_access':True,'root_lease_evidence':{'CPU_fixture_only':True,'GPU_authorized':False},
            'environment':{'CUDA_VISIBLE_DEVICES':'3','CUDA_DEVICE_ORDER':'PCI_BUS_ID'},
            'timeout_seconds':4,'drain_timeout_seconds':1,'cleanup_timeout_seconds':5,
            'source_pins':[{'path':str(MONITOR),'sha256':hashlib.sha256(MONITOR.read_bytes()).hexdigest()}],
            'argv':(argv or [sys.executable,'-B','-S',str(payload)]) +
                ['--gpu-selector-kind','nvml-index','--physical-gpu-uuid','GPU-5b5e235a-09a2-fe8b-7754-916b7eb85976',
                 '--cuda-visible-device','3','--device-identity-proof',str(job/'device-identity.json')],**bounds}
        path=self.directory/'config.json'
        path.write_text(json.dumps(config,indent=2)+'\n')
        nv=fake_nvml(fail_read)
        with mock.patch.dict(sys.modules,{'pynvml':nv}), contextlib.redirect_stdout(io.StringIO()):
            code=monitor.main(['--config',str(path),'--work-dir',str(self.directory)])
        receipt=json.loads((job/'receipt.json').read_text())
        (self.directory/'test-result.json').write_text(json.dumps({'CPU_fixture_only':True,'monitor_return':code,
            'receipt':str(job/'receipt.json'),'fake_NVML_calls':nv.calls,'actual_GPU_calls':0},indent=2)+'\n')
        return code,receipt

    def test_normal_payload_exit(self):
        code,r=self.run_monitor('pass\n')
        self.assertEqual(code,0)
        self.assertEqual(r['exit_code'],0)
        self.assertTrue(r['terminal'])
        self.assertEqual(r['owned_session_cleanup']['remaining_owned_members'],[])

    def test_nonzero_payload_preserved(self):
        code,r=self.run_monitor('raise SystemExit(7)\n')
        self.assertEqual(code,1)
        self.assertEqual(r['exit_code'],7)
        self.assertIn('code 7',r['error'])
        self.assertTrue(r['terminal'])

    def test_actual_exec_start_failure(self):
        code,r=self.run_monitor('',argv=[str(self.directory/'missing-CPU-executable')])
        self.assertEqual(code,1)
        self.assertFalse(r['payload_created'])
        self.assertFalse(r['terminal'])
        self.assertFalse(r['post_terminal_idle'])

    def test_first_read_error_cleans_session(self):
        code,r=self.run_monitor('import time;time.sleep(30)\n',fail_read=True)
        self.assertEqual(code,1)
        self.assertEqual(r['error'],'CPU fixture NVML read error')
        self.assertTrue(r['terminal'])
        self.assertEqual(r['owned_session_cleanup']['remaining_owned_members'],[])

    def test_deadline_escalates_pidfd_only(self):
        code,r=self.run_monitor('import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)\n',timeout_seconds=.3)
        self.assertEqual(code,1)
        self.assertIn('timeout',r['error'])
        self.assertTrue(r['terminal'])
        self.assertTrue(any(s['signal']==signal.SIGKILL and s['pidfd_used'] for s in r['owned_session_cleanup']['signals']))

    def test_natural_same_SID_new_PGID_drain(self):
        body="import subprocess,sys,time\nsubprocess.Popen([sys.executable,'-B','-S','-c','import os,time;os.setpgid(0,0);time.sleep(.35)'])\ntime.sleep(.15)\n"
        code,r=self.run_monitor(body)
        self.assertEqual(code,0)
        self.assertTrue(r['terminal'])
        self.assertTrue(r['owned_session_cleanup']['reaped_descendants'])
        self.assertEqual(r['owned_session_cleanup']['signals'],[])

    def test_foreign_session_and_reused_birth_rejected(self):
        for _ in range(2):
            child=subprocess.Popen([sys.executable,'-B','-S','-c','import time;time.sleep(30)'],start_new_session=True)
            self.owned.append(OwnedSession(child))
        target,foreign=self.owned
        foreign_row=foreign.members()[0]
        with self.assertRaisesRegex(ValueError,'foreign'):
            target.send_owned(foreign_row,signal.SIGTERM)
        changed=dict(target.members()[0],start_ticks=target.anchor['start_ticks']+1)
        with self.assertRaisesRegex(ValueError,'changed'):
            target.send_owned(changed,signal.SIGTERM)
        self.assertIsNotNone(identity(foreign_row['pid']))
        self.assertEqual(target.signals,[])
        (self.directory/'negative-proof.json').write_text(json.dumps({'target':target.anchor,'foreign':foreign.anchor,
            'changed_birth_candidate':changed,'no_signals':True,'foreign_live_after_rejections':identity(foreign_row['pid'])},indent=2)+'\n')

    def test_partial_cleanup_never_claims_terminal_or_postidle(self):
        instances=[]
        class Unproven(OwnedSession):
            def __init__(self,child):
                super().__init__(child)
                instances.append(self)
            def observe(self):
                raise ValueError('CPU fixture ownership proof unavailable')
            def cleanup(self,**kwargs):
                return self.report()
        try:
            with mock.patch.object(monitor,'OwnedSession',Unproven):
                code,r=self.run_monitor('import time;time.sleep(30)\n')
            self.assertEqual(code,1)
            self.assertFalse(r['terminal'])
            self.assertTrue(r['observed_last_NVML_samples_idle'])
            self.assertFalse(r['post_terminal_idle'])
            self.assertTrue(r['owned_session_cleanup']['remaining_owned_members'])
        finally:
            for instance in instances:
                self.owned.append(OwnedSession(instance.child))


if __name__ == '__main__':
    def audit(event,args):
        if event=='ctypes.dlopen' and args[0] is not None:
            raise RuntimeError('CPU fixtures permit libc process control only')
        if event=='subprocess.Popen':
            argv=args[1]
            allowed=Path(argv[0]).resolve()==Path(sys.executable).resolve() and '-B' in argv and '-S' in argv
            missing=Path(argv[0]).name=='missing-CPU-executable' and not Path(argv[0]).exists()
            if not (allowed or missing):raise RuntimeError('Only declared CPU fixtures may start')
    sys.addaudithook(audit)
    unittest.main()
