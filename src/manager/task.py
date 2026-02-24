import threading
import time
from enum import Enum
from src.core.streamer import AiStreamer

class TaskStatus(str, Enum):
    INIT = "init"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"

class InferenceTask:
    def __init__(self, task_id, input_url, config, detector, gpu_id, logger):
        self.task_id = task_id
        self.input_url = input_url
        self.output_url = ""
        self.cfg = config
        self.detector = detector
        self.gpu_id = gpu_id
        self.logger = logger

        self.status = TaskStatus.INIT
        self.error_msg = None
        self.start_time = None
        

        self.streamer = None
        self.thread = None
        self.stream=""

    def start(self):
        def _run():
            try:
                self.logger.info(f"[Task {self.task_id}] 启动")
                self.status = TaskStatus.RUNNING
                self.start_time = time.time()

                self.streamer = AiStreamer(
                    self.cfg,
                    self.detector,
                    self.logger,
                    self.input_url,
                    self.gpu_id
                )
                self.output_url=self.streamer.output_url
                self.stream=self.streamer.stream
                self.streamer.run()

                if self.status != TaskStatus.STOPPED:
                    self.status = TaskStatus.STOPPED

            except Exception as e:
                self.status = TaskStatus.ERROR
                self.error_msg = str(e)
                self.logger.error(f"[Task {self.task_id}] 异常: {e}")

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()

    def stop(self):
        if self.streamer:
            self.streamer.running = False
        self.status = TaskStatus.STOPPED
        self.logger.info(f"[Task {self.task_id}] 已停止")


    def get_fps(self):
        return self.streamer.fps

    def get_heartbeat(self):
        return self.streamer.last_heartbeat