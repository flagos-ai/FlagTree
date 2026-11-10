"""Validate a live, frozen standard monitor ancestor; never flock/unlock.

Production uses real /proc and an opened read-only FD. Injectable proc_root in
validate_open_lock is for explicit CPU fixtures; the capture entry has no such
option. No process environment or command tail is recorded.
This host-only revision ignores only the lock-list ordinal and whitespace in
repeated holder comparison. The stable fields and all original checks remain.
"""
import hashlib,json,os,re,stat
from pathlib import Path
P=Path(__file__).resolve().parent

class LeaseVerificationError(ValueError):
 def __init__(self,message,observations):
  super().__init__(message);self.lease_observations=observations

class VerifiedLease:
 def __init__(self,handle,receipt):self.handle,self.receipt=handle,receipt
 def close(self):self.handle.close()

def _hash(data):return hashlib.sha256(data).hexdigest()
def _stat_process(proc,pid):
 raw=(proc/str(pid)/'stat').read_bytes();text=raw.decode();end=text.rfind(')')
 if end<0 or int(text.split('(',1)[0])!=pid:raise ValueError('Malformed process identity')
 fields=text[end+2:].split()
 if len(fields)<20:raise ValueError('Truncated process stat')
 return {'pid':pid,'ppid':int(fields[1]),'starttime':int(fields[19])}
def _chain(proc,pid):
 result=[];seen=set()
 while pid>0:
  if pid in seen or len(result)>=256:raise ValueError('Invalid process ancestry')
  seen.add(pid);row=_stat_process(proc,pid);result.append(row)
  if pid==1:break
  pid=row['ppid']
 if not result or result[-1]['pid']!=1:raise ValueError('Incomplete process ancestry')
 return result

def _holder(proc,device,inode):
 raw=(proc/'locks').read_bytes();matches=[]
 for line in raw.decode().splitlines():
  parts=line.split()
  # Waiting requests contain -> and are not current owners.
  if len(parts)!=8 or parts[1:4]!=['FLOCK','ADVISORY','WRITE']:continue
  try:
   major,minor,number=parts[5].split(':')
   identity=(int(major,16),int(minor,16),int(number))
   pid=int(parts[4])
  except (ValueError,IndexError):raise LeaseVerificationError('Unparseable FLOCK holder',[{'sha256':_hash(raw),'records':raw.decode().splitlines()}])
  if identity==(*device,inode):
   if parts[6:]!=['0','EOF'] or pid<=0:raise LeaseVerificationError('Invalid exclusive whole-file holder',[{'sha256':_hash(raw),'records':raw.decode().splitlines()}])
   matches.append((pid,line))
 if len(matches)!=1:raise LeaseVerificationError('Missing or ambiguous matching FLOCK/WRITE holder',[{'sha256':_hash(raw),'records':raw.decode().splitlines()}])
 return matches[0],_hash(raw)

def _file_pin(path):
 path=Path(path).resolve(strict=True);raw=path.read_bytes()
 return {'path':str(path),'sha256':_hash(raw),'nbytes':len(raw)}
def _monitor(proc,pid,expected):
 cmd=(proc/str(pid)/'cmdline').read_bytes();argv=[b.decode('utf-8') for b in cmd.split(b'\0') if b]
 if len(argv)<6:raise ValueError('Incomplete monitor command line')
 exe=Path(os.readlink(proc/str(pid)/'exe')).resolve(strict=True)
 if _hash(exe.read_bytes()) not in expected['python_sha256']:raise ValueError('Unapproved monitor Python executable')
 index=1
 while index<len(argv) and argv[index] in ('-u','-B'):index+=1
 if index>=len(argv) or argv[index].startswith('-'):raise ValueError('Not a direct standard monitor script invocation')
 cwd=Path(os.readlink(proc/str(pid)/'cwd')).resolve(strict=True)
 script=Path(argv[index]);script=script if script.is_absolute() else cwd/script
 actual=_file_pin(script)
 if actual!=expected['monitor']:raise ValueError('Wrong monitor source/path')
 if _file_pin(expected['helper']['path'])!=expected['helper']:raise ValueError('Changed standard monitor helper source')
 try:separator=argv.index('--',index+1)
 except ValueError:raise ValueError('Monitor child separator absent')
 options=argv[index+1:separator]
 if len(options)!=4 or set(options[::2])!={'--output','--policy'}:raise ValueError('Unadmitted monitor option prefix')
 values=dict(zip(options[::2],options[1::2]))
 def rooted(s):
  path=Path(s);return (path if path.is_absolute() else cwd/path).resolve(strict=True)
 output=rooted(values['--output']);policy=rooted(values['--policy'])
 before=policy.read_bytes();copied=(output/'isolation-policy.json').read_bytes()
 if before!=copied:raise ValueError('Monitor frozen policy copy mismatch')
 with (output/'telemetry.jsonl').open('rb') as stream:first=stream.readline()
 observation=json.loads(first)
 if (observation.get('observation_id')!=0 or observation.get('phase')!='prelaunch'
     or observation.get('root_pid')!=pid or observation.get('parse_error') is not None
     or observation.get('foreign_gpu_pids')!=[]):raise ValueError('No valid standard monitor prelaunch record')
 if (output/'monitor.json').exists():raise ValueError('Ancestor monitor already has terminal/rejected record')
 return {'monitor_source':actual,'helper_source':expected['helper'],'python_exe':str(exe),
         'argv_prefix':argv[:separator+1],'cmdline_sha256':_hash(cmd),
         'output':str(output),'policy':_file_pin(policy),'policy_copy_sha256':_hash(copied),
         'prelaunch_observation_sha256':_hash(first),'prelaunch_summary':{k:observation[k] for k in ('observation_id','phase','root_pid','clock','policy')}}

def validate_open_lock(handle,path,expected,*,proc_root=Path('/proc'),current_pid=None):
 proc=Path(proc_root);pid=os.getpid() if current_pid is None else current_pid
 info=os.fstat(handle.fileno());named=os.stat(path,follow_symlinks=False)
 if not stat.S_ISREG(info.st_mode) or (info.st_dev,info.st_ino)!=(named.st_dev,named.st_ino):raise ValueError('Lock path/open FD mismatch')
 device=(os.major(info.st_dev),os.minor(info.st_dev));boot=(proc/'sys/kernel/random/boot_id').read_text().strip()
 if not boot:raise ValueError('Missing boot identity')
 (owner,line),locks_sha=_holder(proc,device,info.st_ino)
 chain=_chain(proc,pid)
 if owner not in [p['pid'] for p in chain[1:]]:raise ValueError('Lock holder is not a strict actual capture ancestor')
 monitor=_monitor(proc,owner,expected)
 # Recheck all identity-bearing state after lock/ancestor/source/record reads.
 if _chain(proc,pid)!=chain:raise ValueError('Ancestry/PID starttime changed while verifying lease')
 before={'sha256':locks_sha,'holder_record':line}
 try:(again,line2),locks_sha2=_holder(proc,device,info.st_ino)
 except LeaseVerificationError as error:
  error.lease_observations.insert(0,before);raise
 observations=[before,{'sha256':locks_sha2,'holder_record':line2}]
 # /proc/locks field0 is a list ordinal, not an ownership identity. split()
 # ignores presentation whitespace; every other token is compared unchanged.
 if again!=owner or line2.split()[1:]!=line.split()[1:]:
  raise LeaseVerificationError('FLOCK owner changed during verification',observations)
 if _monitor(proc,owner,expected)!=monitor:raise ValueError('Monitor identity/source/record changed while verifying lease')
 final=os.fstat(handle.fileno());named=os.stat(path,follow_symlinks=False)
 if (final.st_dev,final.st_ino)!=(info.st_dev,info.st_ino) or (named.st_dev,named.st_ino)!=(info.st_dev,info.st_ino):raise ValueError('Lock inode changed')
 if (proc/'sys/kernel/random/boot_id').read_text().strip()!=boot:raise ValueError('Boot identity changed')
 return {'schema':'flagmega.verified-parent-monitor-lease/v1','status':'verified','boot_id':boot,
    'capture_pid':pid,'holder_pid':owner,'ancestry':chain,'lock_path':str(Path(path).resolve()),
    'lock_identity':{'device_major':device[0],'device_minor':device[1],'inode':info.st_ino},
    'holder_record':line,'holder_records':[line,line2],
    'holder_identity_tokens':line.split()[1:],
    'locks_snapshot_sha256':[locks_sha,locks_sha2],
    'lease_source':_file_pin(__file__),**monitor,
    'lifetime_basis':'Frozen standard monitor keeps its exclusive lock through child exit; child only holds a read FD'}

def require_parent_monitor_lock(path='/tmp/flagmega-gpu0.lock'):
 if os.environ.get('FLAGMEGA_GPU_LEASE_GRANTED')!='1':raise RuntimeError('Explicit root GPU lease flag required')
 expected=json.loads((P/'monitor-pin.json').read_text())
 fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW);handle=os.fdopen(fd,'r')
 try:return VerifiedLease(handle,validate_open_lock(handle,path,expected))
 except BaseException:handle.close();raise
