from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class PlaygroundStatus(BaseModel):
    available: bool
    message: Optional[str] = None

class ResearchRequest(BaseModel):
    query: str
    max_tokens: Optional[int] = None
    timeout_seconds: Optional[int] = None 