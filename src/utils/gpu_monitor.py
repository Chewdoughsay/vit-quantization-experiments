"""
Apple Silicon GPU and power monitoring via powermetrics (requires sudo).

Saves GPU utilization, GPU power, and CPU power to CSV for post-training
analysis.

**Legacy** — used only by ``src.training.trainer.ViTTrainer`` in the
preliminary CIFAR-10 study.
"""
import subprocess
import csv
import argparse
import re
import os
import threading
import time
from pathlib import Path
from datetime import datetime
import sys


class GPUMonitor:
    """Background powermetrics monitor for Apple Silicon GPU utilization and power (requires sudo)."""

    def __init__(self, output_file, interval=1000):
        self.output_file = Path(output_file)
        self.interval = interval
        self.running = False
        self.process = None
        self.thread = None
        self.has_sudo = False

        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def _check_sudo(self):
        """Return True if sudo access is available (prompts for password if needed)."""
        try:
            result = subprocess.run(
                ['sudo', '-v'],
                capture_output=True,
                timeout=60  # Give user 60 seconds to enter password
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            return False
        except Exception:
            return False

    def _monitor_loop(self):
        """
        Internal monitoring loop that runs in background thread.
        Parses powermetrics output and saves to CSV.
        """
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'gpu_utilization_percent', 'gpu_power_mW', 'cpu_power_mW'])

        cmd = [
            "sudo",
            "powermetrics",
            "-i", str(self.interval),
            "--samplers", "cpu_power,gpu_power",
            "--show-initial-usage"
        ]

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            current_metrics = {'gpu_res': 0.0, 'gpu_pwr': 0.0, 'cpu_pwr': 0.0}

            while self.running and self.process.poll() is None:
                line = self.process.stdout.readline()
                if not line:
                    break

                line_lower = line.lower()

                if "gpu active residency" in line_lower:
                    match = re.search(r'([\d\.]+)\s*%', line)
                    if match:
                        current_metrics['gpu_res'] = float(match.group(1))

                elif "gpu power" in line_lower:
                    match = re.search(r'([\d]+)\s*mw', line_lower)
                    if match:
                        current_metrics['gpu_pwr'] = float(match.group(1))

                elif "cpu power" in line_lower:
                    match = re.search(r'([\d]+)\s*mw', line_lower)
                    if match:
                        current_metrics['cpu_pwr'] = float(match.group(1))

                        timestamp = datetime.now().strftime('%H:%M:%S')
                        with open(self.output_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                timestamp,
                                current_metrics['gpu_res'],
                                current_metrics['gpu_pwr'],
                                current_metrics['cpu_pwr']
                            ])

        except Exception as e:
            if self.running:
                print(f"GPU Monitor error: {e}")

    def start(self):
        """
        Start GPU monitoring (prompts for sudo password).

        If sudo access is denied, prints a warning and returns without monitoring.
        Training will continue without GPU stats.
        """
        print("GPU Monitor requires sudo access for powermetrics...")

        if not self._check_sudo():
            print("Sudo access denied. GPU monitoring disabled.")
            print("(Training will continue without GPU stats)")
            self.has_sudo = False
            return

        self.has_sudo = True
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

        print("GPU Monitor started...")
        print(f"Saving to: {self.output_file}")

    def stop(self):
        """
        Stop GPU monitoring and clean up.

        Returns:
            dict: Summary statistics (empty dict if monitoring wasn't active)
        """
        if not self.has_sudo or not self.running:
            return {}

        self.running = False

        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

        if self.thread:
            self.thread.join(timeout=5)

        print("GPU Monitor stopped.")
        return {}


def monitor_stream(output_file, interval=1000):
    """Stream powermetrics output and save GPU/CPU power to CSV (requires sudo)."""
    print(f"Starting GPU Monitor (Robust Stream)...")
    print(f"Saving logs to: {output_file}")

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'gpu_utilization_percent', 'gpu_power_mW', 'cpu_power_mW'])

    cmd = [
        "powermetrics",
        "-i", str(interval),
        "--samplers", "cpu_power,gpu_power",
        "--show-initial-usage"
    ]

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        current_metrics = {'gpu_res': 0.0, 'gpu_pwr': 0.0, 'cpu_pwr': 0.0}

        while True:
            line = process.stdout.readline()
            if not line:
                break

            line_lower = line.lower()

            if "gpu active residency" in line_lower:
                match = re.search(r'([\d\.]+)\s*%', line)
                if match:
                    current_metrics['gpu_res'] = float(match.group(1))

            elif "gpu power" in line_lower:
                match = re.search(r'([\d]+)\s*mw', line_lower)
                if match:
                    current_metrics['gpu_pwr'] = float(match.group(1))

            # CPU Power triggers row save (appears last in powermetrics output)
            elif "cpu power" in line_lower:
                match = re.search(r'([\d]+)\s*mw', line_lower)
                if match:
                    current_metrics['cpu_pwr'] = float(match.group(1))

                    timestamp = datetime.now().strftime('%H:%M:%S')

                    print(
                        f"[{timestamp}] GPU Util: {current_metrics['gpu_res']:5.1f}% | GPU Pwr: {current_metrics['gpu_pwr']:5.0f} mW | CPU Pwr: {current_metrics['cpu_pwr']:5.0f} mW")

                    with open(output_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            timestamp,
                            current_metrics['gpu_res'],
                            current_metrics['gpu_pwr'],
                            current_metrics['cpu_pwr']
                        ])

    except KeyboardInterrupt:
        print("\nMonitor stopped.")
        process.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='monitor_log', help='Log file name')
    args = parser.parse_args()

    Path("../../results/logs").mkdir(parents=True, exist_ok=True)
    file_path = f"results/logs/{args.name}.csv"

    monitor_stream(file_path)

    # Example: sudo python src/utils/gpu_monitor.py --name run_fp16_extended