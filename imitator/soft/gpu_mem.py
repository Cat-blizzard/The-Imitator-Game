"""Per-process GPU memory peak via nvidia-smi --query-compute-apps (the GPU is shared with other jobs, so device-wide
memory.used would count them)."""
from __future__ import annotations

import subprocess
import threading
import time


def process_mib(pid: int) -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                      text=True, timeout=10)
    except Exception:
        return -1
    tot = 0
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
            tot += int(parts[1])
    return tot


class GpuMemPoller:
    def __init__(self, pid: int, interval: float = 0.25):
        self.pid, self.interval = pid, interval
        self.peak_mib = 0
        self.last_mib = 0
        self._stop = False
        self._th = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop:
            m = process_mib(self.pid)
            if m > 0:
                self.last_mib = m
                self.peak_mib = max(self.peak_mib, m)
            time.sleep(self.interval)

    def start(self):
        self._th.start()
        return self

    def stop(self):
        self._stop = True
        self._th.join(timeout=5)
        return self.peak_mib
