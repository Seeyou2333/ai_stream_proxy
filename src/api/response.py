from typing import Any, Optional
from pydantic import BaseModel

class ApiResponse(BaseModel):
    success: bool
    code: Optional[int] = None
    msg: Optional[str] = None
    data: Optional[Any] = None

    @staticmethod
    def ok(data: Any = None):
        return ApiResponse(
            success=True,
            data=data
        )

    @staticmethod
    def fail(msg: str, code: int = -1):
        return ApiResponse(
            success=False,
            code=code,
            msg=msg
        )
