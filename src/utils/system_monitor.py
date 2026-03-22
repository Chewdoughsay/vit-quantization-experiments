"""
Background CPU, memory, and thermal monitoring for macOS training runs.

**Legacy** — used only by ``src.training.trainer.ViTTrainer`` in the
preliminary CIFAR-10 study.
"""
import time
import threading
import psutil
import subprocess
import numpy as np


class SystemMonitor:
    """Background thread monitor for CPU, memory, and macOS thermal throttling."""
    def __init__(self, interval=1.0):
        self.interval = interval
        self.running = False
        self.stats = {
            'cpu_percent': [],
            'memory_percent': [],
            'thermal_pressure': [],
            'timestamps': []
        }
        self.thread = None
        self.thermal_supported = True

    def _get_thermal_pressure(self):
        """Read macOS thermal pressure level via sysctl; returns 0 if unavailable."""
        if not self.thermal_supported:
            return 0

        try:
            output = subprocess.check_output(
                ['sysctl', '-n', 'machdep.cpu.thermal_level'],
                stderr=subprocess.DEVNULL
            )
            return int(output.strip())
        except Exception:
            self.thermal_supported = False
            return 0

    def _monitor_loop(self):
        start_time = time.time()
        while self.running:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
            thermal = self._get_thermal_pressure()

            self.stats['cpu_percent'].append(cpu)
            self.stats['memory_percent'].append(mem)
            self.stats['thermal_pressure'].append(thermal)
            self.stats['timestamps'].append(time.time() - start_time)

            time.sleep(self.interval)

    def start(self):
        """Start background monitoring thread; resets all stats."""
        self.running = True
        self.stats = {k: [] for k in self.stats}
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print("System Monitor started...")

    def stop(self):
        """Stop monitoring; returns (summary_dict, full_stats_dict)."""
        self.running = False
        if self.thread:
            self.thread.join()

        def safe_mean(data):
            return np.mean(data) if data else 0

        def safe_max(data):
            return np.max(data) if data else 0

        summary = {
            'avg_cpu': safe_mean(self.stats['cpu_percent']),
            'max_cpu': safe_max(self.stats['cpu_percent']),
            'avg_mem': safe_mean(self.stats['memory_percent']),
            'max_thermal': safe_max(self.stats['thermal_pressure']),
            'throttled': any(t > 0 for t in self.stats['thermal_pressure'])
        }
        print("System Monitor stopped.")
        return summary, self.stats