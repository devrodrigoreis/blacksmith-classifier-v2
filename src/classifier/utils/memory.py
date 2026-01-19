
import psutil
import time
import threading
import logging
import sys
import os

class MemoryMonitor:
    def __init__(self, max_memory_percent=90.0, check_interval=10):
        self.max_memory_percent = max_memory_percent
        self.check_interval = check_interval
        self.running = False
        self.thread = None

    def _monitor(self):
        process = psutil.Process()
        while self.running:
            mem_percent = process.memory_percent()
            if mem_percent > self.max_memory_percent:
                logging.error(f"Memory usage exceeded {self.max_memory_percent}% ({mem_percent:.2f}%). Terminating process.")
                # We might want to raise an exception or handle this more gracefully depending on the context,
                # but for now keeping original behavior of exit.
                os._exit(1) # Force exit
            time.sleep(self.check_interval)

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

def setup_cuda_memory():
    """Sets environment variables for optimized CUDA memory."""
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
