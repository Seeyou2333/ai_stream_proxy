import threading
from src.inference.detector import ONNXDetector

class GpuManager:
    def __init__(self, config, logger, max_tasks_per_gpu=2):
        self.cfg = config
        self.logger = logger
        self.device_ids = config.get('device_ids', [0])
        self.device_type = config.get('device_type', 'cpu').lower()
        self.max_tasks_per_gpu = max_tasks_per_gpu

        self.detectors = {}
        self.task_counts = {gid: 0 for gid in self.device_ids}
        self.lock = threading.Lock()

    def acquire(self):
        if self.device_type == 'cpu':
            return ONNXDetector(self.cfg, gpu_id=-1), -1

        with self.lock:
            candidates = [
                gid for gid, cnt in self.task_counts.items()
                if cnt < self.max_tasks_per_gpu
            ]

            if not candidates:
                return None, None

            gpu_id = min(candidates, key=lambda g: self.task_counts[g])

            if gpu_id not in self.detectors:
                self.detectors[gpu_id] = ONNXDetector(self.cfg, gpu_id)

            self.task_counts[gpu_id] += 1
            self.logger.info(f"GPU[{gpu_id}] 当前任务数: {self.task_counts[gpu_id]}")

            return self.detectors[gpu_id], gpu_id

    def release(self, gpu_id):
        if gpu_id == -1:
            return
        with self.lock:
            self.task_counts[gpu_id] = max(0, self.task_counts[gpu_id] - 1)
            self.logger.info(f"GPU[{gpu_id}] 释放任务，剩余: {self.task_counts[gpu_id]}")
