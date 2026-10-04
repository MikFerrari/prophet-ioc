"""Keeps a long job (synthetic_recovery.py, ...) from taking the workstation down, and reports its resource use.

Why: on 4 October a synthetic recovery run grew to 10.7 GB of host RAM. The kernel killed it once, then a second run
pushed the desktop into swap until the machine froze and had to be rebooted. The kernel OOM killer acts only when RAM
*and* swap are exhausted, long after the system has stopped responding, so the job has to watch itself.

guard(...) (call it before the first JAX computation, i.e. before ck.setup_jax):
- CPU: lower priority (nice) and an affinity to all cores but a few, so the desktop keeps free cores; XLA sizes its
  CPU thread pools from the affinity when the backend starts. OPENBLAS_NUM_THREADS=1 must be set before numpy is
  imported (the scripts do it at the top).
- RAM: a watchdog thread checks every `interval` s the process RSS and the system's available memory (RAM + free
  swap is not counted: swapping is what freezes the desktop). Over the limit it prints why and ends the process with
  os._exit (the main thread may be stuck inside an XLA compilation, where an exception would not be seen). The
  process also volunteers as the first victim of the kernel OOM killer (oom_score_adj 1000).
release() frees the compiled programs and heap of a finished stage; memory() / status() report the RSS, its peak and the available memory for the progress logs.

Linux only (/proc); elsewhere guard() only sets the priority.
"""

import os
import sys
import threading
import time
from typing import Callable, Optional

GB = 1024 ** 3


def _meminfo() -> dict:
    with open("/proc/meminfo") as f:
        return {line.split(":")[0]: int(line.split()[1]) * 1024 for line in f}


def _self_status() -> dict:
    """VmRSS and VmHWM (peak RSS) of this process, in bytes."""
    out = {}
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(("VmRSS:", "VmHWM:")):
                out[line.split(":")[0]] = int(line.split()[1]) * 1024
    return out


def memory() -> dict:
    """{rss, peak, available, total} in GB (empty off Linux)."""
    try:
        s, m = _self_status(), _meminfo()
    except OSError:
        return {}
    return {"rss": s.get("VmRSS", 0) / GB, "peak": s.get("VmHWM", 0) / GB, "available": m["MemAvailable"] / GB,
            "total": m["MemTotal"] / GB}


def status() -> str:
    """'RAM 3.1 GB (peak 4.0), 21.5 GB available' for the progress logs."""
    m = memory()
    return f"RAM {m['rss']:.1f} GB (peak {m['peak']:.1f}), {m['available']:.1f} GB available" if m else ""


class Guard:
    """The watchdog of guard(); stop() ends it."""

    def __init__(self, max_rss_gb: float, min_available_gb: float, interval: float,
                 on_exit: Optional[Callable[[str], None]]):
        self.max_rss_gb, self.min_available_gb, self.interval, self.on_exit = \
            max_rss_gb, min_available_gb, interval, on_exit
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="memory-guard", daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.wait(self.interval):
            m = memory()
            if not m:
                return
            reason = None
            if m["rss"] > self.max_rss_gb:
                reason = f"process RAM {m['rss']:.1f} GB > limit {self.max_rss_gb:.1f} GB"
            elif m["available"] < self.min_available_gb:
                reason = (f"system available RAM {m['available']:.1f} GB < {self.min_available_gb:.1f} GB "
                          f"(process {m['rss']:.1f} GB)")
            if reason:
                msg = (f"\n*** memory guard: {reason}. Stopping this run to protect the system "
                       f"(limits: resources.max_ram_gb / resources.min_available_gb). ***\n")
                try:
                    if self.on_exit is not None:
                        self.on_exit(msg)
                finally:
                    sys.stderr.write(msg)
                    sys.stderr.flush()
                    sys.stdout.flush()
                    os._exit(137)

    def stop(self):
        self._stop.set()


def guard(max_ram_gb: Optional[float] = None, max_ram_fraction: float = 0.4, min_available_gb: float = 3.0,
          free_cpus: int = 4, nice: int = 10, interval: float = 1.0,
          on_exit: Optional[Callable[[str], None]] = None) -> Optional[Guard]:
    """Lowers the priority, leaves `free_cpus` cores to the rest of the system and starts the memory watchdog: the
    run is stopped when its RSS exceeds max_ram_gb (None: max_ram_fraction of the physical RAM) or when the system's
    available RAM falls below min_available_gb. on_exit(message) is called first (e.g. to write the message to the
    run's log file). Returns the watchdog (None off Linux)."""
    try:
        os.nice(nice)
    except (OSError, AttributeError):
        pass
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass
    if hasattr(os, "sched_getaffinity"):
        cpus = sorted(os.sched_getaffinity(0))
        keep = max(1, len(cpus) - max(0, int(free_cpus)))
        os.sched_setaffinity(0, cpus[:keep])
    m = memory()
    if not m:
        return None
    limit = float(max_ram_gb) if max_ram_gb else max_ram_fraction * m["total"]
    print(f"resource guard: nice +{nice}, {len(os.sched_getaffinity(0))} of {os.cpu_count()} CPUs, stop at "
          f"{limit:.1f} GB process RAM or < {min_available_gb:.1f} GB available ({status()})", flush=True)
    return Guard(limit, float(min_available_gb), float(interval), on_exit)


def release():
    """Frees what a finished stage no longer needs: JAX's compiled programs (each likelihood compilation keeps several
    GB of host RAM), Python garbage, and the free heap that glibc would otherwise keep (malloc_trim)."""
    import gc
    try:
        import jax
        jax.clear_caches()
    except Exception:
        pass
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


class StageLog:
    """Timestamped progress lines with elapsed time and memory, printed and appended to a log file."""

    def __init__(self, path=None):
        self.path, self.t0 = path, time.perf_counter()

    def __call__(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')} +{(time.perf_counter() - self.t0) / 60:5.1f} min | {status()}] {msg}"
        print(line, flush=True)
        self.write(line + "\n")

    def write(self, text: str):
        if self.path is not None:
            with open(self.path, "a") as f:
                f.write(text)
