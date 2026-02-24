from pydantic import BaseModel

class StartTaskRequest(BaseModel):
    input_url: str

class StopTaskRequest(BaseModel):
    task_id: str
