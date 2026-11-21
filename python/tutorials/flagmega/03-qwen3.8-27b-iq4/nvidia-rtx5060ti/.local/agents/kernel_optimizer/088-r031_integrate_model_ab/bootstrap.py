import sys, runpy, os, types
from pathlib import Path
A=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(A/'source-tree/python'))
sys.path.insert(1,str(A))
sys.path.insert(2,str(A.parents[8]/'python'))
os.environ.update(TRANSPORT_STAGE='3',WARP_GRID='17408',WARP_CAPACITY='2',BARRIER_BATCH_G='1',ADDRESS_SINGLE='1',PACKET_MODE='cached_tables',TRANSPORT_MAXNREG='40',PRODUCER_REGISTERS='32',TRANSPORT_KIND='global_idle',CP_PHASE_ALIGN='0',CP_ALIGNED_CARRIER='0',SINGLE_EPOCH='0',PRODUCER_PIPELINE='serial',OCCUPANCY_PTX_OPTIONS='--minnctapersm=9 --opt-level=2',PADDING_PREFETCH='0',BOUNDED_PREFETCH='1',PREFETCH_WINDOW='2',DOWN_PAIR_PACK='1',Q8_ALIGNED='1',Q8_WORK_QUEUE='1',CONTINUOUS_STREAM='0',NO_SHARED_STAGING='1')
package=types.ModuleType('triton._C')
package.__path__=[str(A.parent/'085-r027_register_40_no_spill/compiler-terminal/production')]
sys.modules['triton._C']=package
import triton
triton._C=package
runpy.run_path(sys.argv.pop(1),run_name='__main__')
