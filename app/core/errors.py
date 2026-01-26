from pydantic import BaseModel
from typing import Optional

class ErrorResponse(BaseModel):
    code: str
    message: str
    trace_id: Optional[str] = None
