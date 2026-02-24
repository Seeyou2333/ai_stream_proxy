import threading
import uuid
import time
from src.manager.task import InferenceTask, TaskStatus

class InferenceTaskManager:
    def __init__(self, config, gpu_manager, logger, max_tasks_per_gpu=2):
        self.cfg = config
        self.gpu_manager = gpu_manager
        self.logger = logger
        self.max_tasks_per_gpu = max_tasks_per_gpu

        self.tasks = {}
        self.lock = threading.Lock()

    def start_task(self, input_url):
        with self.lock:
            self.cleanup()
            if not input_url.startswith("rtsp://"):
                raise RuntimeError("流地址非法")
            if self.url_exists(input_url):
                raise RuntimeError("流地址已存在")

            detector, gpu_id = self.gpu_manager.acquire()

            if detector is None:
                raise RuntimeError("没有可用 GPU 资源")

            task_id = str(uuid.uuid4().hex)
            task = InferenceTask(
                task_id,
                input_url,
                self.cfg,
                detector,
                gpu_id,
                self.logger
            )

            self.tasks[task_id] = task
            task.start()

            return task_id

    def stop_task(self, task_id):
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return False

            task.stop()
            self.gpu_manager.release(task.gpu_id)
            return True

    def get_status(self, task_id,enbale=True):
        if enbale:
            self.cleanup()
        task = self.tasks.get(task_id)
        if not task:
            return None

        return {
            "task_id": task.task_id,
            "input_url": task.input_url,
            "stream":task.stream,
            "status": task.status,
            "gpu_id": task.gpu_id,
            "fps":task.get_fps(),
            "uptime": time.time() - task.start_time if task.start_time else 0,
            "last_heartbeat":task.get_heartbeat()
        }

    def list_tasks(self):
        self.cleanup()
        return [self.get_status(tid,False) for tid in self.tasks]

    def cleanup_task(self, task_id):
        task = self.tasks.pop(task_id, None)
        if task:
            self.gpu_manager.release(task.gpu_id)

    def url_exists(self, input_url):
        return any(self.tasks[task_id].input_url == input_url for task_id in self.tasks)

    def cleanup(self):
        for task_id in self.tasks.copy():  
            if self.tasks[task_id].status in [TaskStatus.STOPPED, TaskStatus.ERROR]:
                self.cleanup_task(task_id)
                
    def stop_all(self):
        with self.lock:
            task_ids = list(self.tasks.keys())
            for task_id in task_ids:
                task = self.tasks.get(task_id)
                if not task:
                    continue
                try:
                    task.stop()
                except Exception as e:
                    self.logger.error(f"停止任务失败 {task_id}: {e}")
                self.gpu_manager.release(task.gpu_id)
            self.tasks.clear()

    
