from fastapi import APIRouter
from src.api.response import ApiResponse
from src.api.request import StartTaskRequest, StopTaskRequest
from src.manager.task_manager import InferenceTaskManager

router = APIRouter(prefix="/aistream")

task_manager: InferenceTaskManager = None

def init(manager: InferenceTaskManager):
    global task_manager
    task_manager = manager

@router.post("/start")
def start_task(req: StartTaskRequest):
    try:
        task_id = task_manager.start_task(req.input_url)
        return ApiResponse.ok({
            "task_id": task_id
        })
    except Exception as e:
        logger.exception("开启任务失败")
        return ApiResponse.fail(str(e), code=500)

@router.post("/stop")
def stop_task(req: StopTaskRequest):
    success = task_manager.stop_task(req.task_id)
    if success:
        return ApiResponse.ok()
    return ApiResponse.fail("停止失败或任务不存在", code=404)


@router.get("/status")
def get_status(task_id: str):
    status = task_manager.get_status(task_id)
    if not status:
        return ApiResponse.fail("任务不存在", code=404)
    return ApiResponse.ok(status)

@router.get("/tasks")
def list_tasks():
    return ApiResponse.ok(task_manager.list_tasks())
