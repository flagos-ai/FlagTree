"""Birth-bound Linux process-session ownership; no CUDA or NVML imports.

The scan and pidfd revalidation follow the frozen whole-SID controller45665844.
The spawning monitor additionally adopts orphan descendants so it can reap them
after a payload leader exits. No foreign session or recycled PID is signalled.
"""
from pathlib import Path
import os
import signal
import time


def pidfd_library():
    # This Python build omits os.pidfd_open. Use the proven glibc interface,
    # retaining the same kernel pidfd safety; there is no PID-only fallback.
    import ctypes
    library = ctypes.CDLL(None, use_errno=True)
    if not hasattr(library, 'pidfd_open') or not hasattr(library, 'pidfd_send_signal'):
        raise RuntimeError('Linux/glibc pidfd API is required')
    library.pidfd_open.argtypes = [ctypes.c_int, ctypes.c_uint]
    library.pidfd_open.restype = ctypes.c_int
    library.pidfd_send_signal.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    library.pidfd_send_signal.restype = ctypes.c_int
    return library


def pidfd_open(pid):
    import ctypes
    result = pidfd_library().pidfd_open(pid, 0)
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result


def pidfd_signal(descriptor, number):
    import ctypes
    if pidfd_library().pidfd_send_signal(descriptor, number, None, 0) < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def identity(pid):
    try:
        raw = Path('/proc', str(pid), 'stat').read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = raw[raw.rfind(')') + 2:].split()
    return {'pid': pid, 'state': fields[0], 'ppid': int(fields[1]),
            'pgid': int(fields[2]), 'sid': int(fields[3]), 'start_ticks': int(fields[19])}


def scan(sid):
    members = []
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            row = identity(int(entry.name))
            if row is not None and row['sid'] == sid:
                members.append(row)
    return sorted(members, key=lambda row: row['pid'])


class Subreaper:
    """Use libc process control only; restore the caller's previous setting."""
    def __enter__(self):
        import ctypes
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.prctl.restype = ctypes.c_int
        self.previous = ctypes.c_int()
        if self.libc.prctl(37, ctypes.byref(self.previous), 0, 0, 0) < 0:
            raise OSError(ctypes.get_errno(), 'PR_GET_CHILD_SUBREAPER failed')
        if self.libc.prctl(36, 1, 0, 0, 0) < 0:
            raise OSError(ctypes.get_errno(), 'PR_SET_CHILD_SUBREAPER failed')
        self.entered = True
        return self

    def __exit__(self, *args):
        if getattr(self, 'entered', False):
            self.entered = False
            if self.libc.prctl(36, self.previous.value, 0, 0, 0) < 0:
                import ctypes
                raise OSError(ctypes.get_errno(), 'restore child subreaper failed')


class OwnedSession:
    def __init__(self, child):
        self.child = child
        self.anchor = identity(child.pid)
        if (self.anchor is None or self.anchor['sid'] != child.pid or
                self.anchor['pgid'] != child.pid or self.anchor['sid'] == os.getsid(0)):
            raise ValueError('new child session birth could not be anchored')
        self.observations = []
        self.signals = []
        self.reaped = []
        self.empty_scans = []
        self.cleanup_errors = []
        self.payload_exit_code = None

    def members(self):
        leader = identity(self.anchor['pid'])
        if leader is not None and any(leader[key] != self.anchor[key] for key in ('pid', 'sid', 'start_ticks')):
            raise ValueError('owned leader birth/SID changed; refuse reused PID')
        rows = scan(self.anchor['sid'])
        if any(row['start_ticks'] < self.anchor['start_ticks'] for row in rows):
            raise ValueError('session member predates anchored leader birth')
        return rows

    def observe(self):
        # Popen owns its direct child's wait status; never reap it independently.
        code = self.child.poll()
        if code is not None:
            self.payload_exit_code = code
        rows = self.members()
        for row in rows:
            if row['pid'] == self.child.pid or row['state'] != 'Z':
                continue
            try:
                pid, status = os.waitpid(row['pid'], os.WNOHANG)
            except ChildProcessError:
                continue  # A live owned parent still owns this zombie's wait.
            if pid:
                self.reaped.append({'birth': row, 'wait_status': status})
        rows = self.members()
        event = {'monotonic_ns': time.monotonic_ns(), 'members': rows,
                 'payload_exit_code': self.payload_exit_code}
        self.observations.append(event)
        if rows:
            self.empty_scans.clear()
        elif self.payload_exit_code is not None:
            self.empty_scans.append(event)
        return rows

    @property
    def terminal_qualified(self):
        return self.payload_exit_code is not None and len(self.empty_scans) >= 2

    def send_owned(self, row, number):
        if row['sid'] != self.anchor['sid'] or row['start_ticks'] < self.anchor['start_ticks']:
            raise ValueError('foreign SID/birth rejected before signal')
        try:
            descriptor = pidfd_open(row['pid'])
        except ProcessLookupError:
            return None
        try:
            current = identity(row['pid'])
            if current is None:
                return None
            if any(current[key] != row[key] for key in ('pid', 'sid', 'pgid', 'start_ticks')):
                raise ValueError('candidate birth/SID/PGID changed; refuse signal')
            if current['state'] == 'Z':
                return None
            pidfd_signal(descriptor, number)
            event = {'target_birth': current, 'signal': number,
                     'monotonic_ns': time.monotonic_ns(), 'pidfd_used': True}
            self.signals.append(event)
            return event
        except (FileNotFoundError, ProcessLookupError):
            return None
        finally:
            os.close(descriptor)

    def cleanup(self, timeout_seconds=8, poll_seconds=.05):
        """Repeated birth-safe waves, preserving parents briefly to reap workers."""
        start = time.monotonic()
        while time.monotonic() - start <= timeout_seconds:
            try:
                rows = self.observe()
                if self.terminal_qualified:
                    return self.report()
                elapsed = time.monotonic() - start
                separate = [row for row in rows if row['pgid'] != self.anchor['sid']]
                candidates = separate if elapsed < .5 and separate else rows
                number = signal.SIGTERM if elapsed < 2 else signal.SIGKILL
                for row in reversed(candidates):
                    self.send_owned(row, number)
            except (OSError, ValueError) as error:
                self.cleanup_errors.append(f'{type(error).__name__}: {error}')
                break  # A changed ownership boundary never permits a signal fallback.
            time.sleep(poll_seconds)
        return self.report()

    def report(self):
        try:
            remaining = self.members()
        except (OSError, ValueError) as error:
            remaining = None
            self.cleanup_errors.append(f'{type(error).__name__}: {error}')
        return {'anchor': self.anchor, 'payload_exit_code': self.payload_exit_code,
                'terminal_qualified': self.terminal_qualified and remaining == [] and not self.cleanup_errors,
                'all_owned_SID_members_absent': remaining == [], 'remaining_owned_members': remaining,
                'observations': self.observations, 'final_empty_scans': self.empty_scans[-2:],
                'signals': self.signals, 'reaped_descendants': self.reaped,
                'cleanup_errors': self.cleanup_errors}


class GatedLaunch:
    """Pre-exec transaction: prove ownership before releasing the backend.

    A frozen, single-process CPU bootstrap waits on a pipe. EOF/timeout exits it
    without executing the payload. The object exists before Popen, immediately
    retains its child, and anchors a cleanup guardian before the monitor's
    additional OwnedSession initialization. No guessed PID/SID is ever signalled.
    """
    def __init__(self):
        self.child = None
        self.guardian = None
        self.gate = None
        self.ready = None
        self.leader_pidfd = None
        self.ready_birth = None
        self.payload_released = False
        self.bootstrap_reaped = False
        self.bootstrap_exit_code = None

    def spawn(self, argv, *, cwd, env, stdout, stderr, timeout_seconds=3):
        import json
        import select
        import subprocess
        import sys
        # Check capability before creating any process. The bootstrap is the
        # same pinned helper source, invoked in a fresh strict-stdlib interpreter.
        pidfd_library()
        gate_read, self.gate = os.pipe()
        self.ready, ready_write = os.pipe()
        bootstrap = [sys.executable, '-B', '-S', str(Path(__file__).resolve()),
                     '--owned-bootstrap', str(gate_read), str(ready_write),
                     json.dumps(argv), str(timeout_seconds + 5)]
        try:
            self.child = subprocess.Popen(bootstrap, cwd=cwd, env=env,
                stdout=stdout, stderr=stderr, start_new_session=True,
                pass_fds=(gate_read, ready_write))
        finally:
            os.close(gate_read)
            os.close(ready_write)
        deadline = time.monotonic() + timeout_seconds
        data = b''
        while b'\n' not in data:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.ready], [], [], remaining)[0]:
                raise TimeoutError('CPU ownership bootstrap readiness deadline')
            part = os.read(self.ready, 4096)
            if not part:
                raise ValueError('CPU ownership bootstrap exited before birth proof')
            data += part
            if len(data) > 4096:
                raise ValueError('CPU bootstrap birth record exceeds bound')
        self.ready_birth = json.loads(data.decode())
        os.close(self.ready)
        self.ready = None
        current = identity(self.child.pid)
        keys = ('pid', 'ppid', 'sid', 'pgid', 'start_ticks')
        if (current is None or any(current[k] != self.ready_birth[k] for k in keys)
                or current['ppid'] != os.getpid()
                or current['sid'] != self.child.pid or current['pgid'] != self.child.pid
                or current['sid'] == os.getsid(0)):
            raise ValueError('CPU bootstrap parent/child birth proof unavailable or changed')
        self.leader_pidfd = pidfd_open(self.child.pid)
        confirmed = identity(self.child.pid)
        if confirmed is None or any(confirmed[k] != current[k] for k in keys):
            raise ValueError('CPU bootstrap birth changed after pidfd acquisition')
        self.guardian = OwnedSession(self.child)
        if any(self.guardian.anchor[k] != current[k] for k in keys):
            raise ValueError('CPU bootstrap guardian birth differs from proven child')
        return self.child

    def release(self, owner):
        if (self.guardian is None or owner.child is not self.child or
                any(owner.anchor[k] != self.guardian.anchor[k]
                    for k in ('pid', 'ppid', 'sid', 'pgid', 'start_ticks'))):
            raise ValueError('Additional ownership initialization differs from launch guardian')
        # Only this point permits original argv/environment to exec, preserving
        # PID, SID and birth; all preparation occurs before backend timing.
        os.write(self.gate, b'G')
        os.close(self.gate)
        self.gate = None
        self.payload_released = True

    def close_gate(self):
        if self.gate is not None:
            os.close(self.gate)
            self.gate = None

    def abort(self, timeout_seconds=8):
        self.close_gate()
        if self.guardian is not None:
            return self.guardian.cleanup(timeout_seconds=timeout_seconds)
        # Without honest ownership evidence, no PID-only signals or invented
        # anchor: the fixed bootstrap's closed gate prevents payload execution.
        if self.child is not None:
            self.bootstrap_exit_code = self.child.wait(timeout=timeout_seconds)
            self.bootstrap_reaped = True
        return {'terminal_qualified': False, 'all_owned_SID_members_absent': None,
                'remaining_owned_members': None,
                'cleanup_errors': ['OS ownership proof unavailable; backend gate never released'],
                'bootstrap_reaped': self.bootstrap_reaped,
                'bootstrap_exit_code': self.bootstrap_exit_code}

    def close(self):
        self.close_gate()
        for name in ('ready', 'leader_pidfd'):
            fd = getattr(self, name)
            if fd is not None:
                os.close(fd)
                setattr(self, name, None)

    def report(self):
        return {'bootstrap_created': self.child is not None,
                'bootstrap_child_PID': self.child.pid if self.child is not None else None,
                'bootstrap_birth': self.ready_birth,
                'guardian_anchor': self.guardian.anchor if self.guardian is not None else None,
                'birth_bound_pidfd_acquired': self.leader_pidfd is not None,
                'backend_payload_released': self.payload_released,
                'bootstrap_reaped_without_payload': self.bootstrap_reaped,
                'bootstrap_exit_code': self.bootstrap_exit_code}


def ownership_bootstrap(gate, ready, argv, timeout_seconds):
    """No fork, model import, or payload exec before authenticated pipe release."""
    import json
    import select
    try:
        birth = identity(os.getpid())
        if birth is None or birth['sid'] != os.getpid() or birth['pgid'] != os.getpid():
            raise ValueError('Bootstrap self birth/session unavailable')
        os.write(ready, (json.dumps(birth) + '\n').encode())
    finally:
        os.close(ready)
    try:
        if not select.select([gate], [], [], timeout_seconds)[0]:
            return 125
        if os.read(gate, 1) != b'G':
            return 125
    finally:
        os.close(gate)
    # exec preserves birth and session and uses the exact supplied argv/env.
    os.execvpe(argv[0], argv, os.environ)


if __name__ == '__main__':
    import json
    import sys
    if len(sys.argv) != 6 or sys.argv[1] != '--owned-bootstrap':
        raise SystemExit('This helper only accepts the owned CPU bootstrap protocol')
    raise SystemExit(ownership_bootstrap(int(sys.argv[2]), int(sys.argv[3]),
                                       json.loads(sys.argv[4]), float(sys.argv[5])))
