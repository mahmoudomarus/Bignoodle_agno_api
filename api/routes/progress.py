from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

from agents.deep_research_agent import get_research_progress

router = APIRouter(prefix="/progress")

class ProgressResponse(BaseModel):
    id: str
    topic: str
    status: str
    completion_percentage: int
    current_stage: str
    searches_performed: list
    tasks_created: list
    tasks_completed: list
    sections_completed: list
    last_message: str
    
    class Config:
        schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "topic": "Quantum computing applications",
                "status": "in_progress",
                "completion_percentage": 65,
                "current_stage": "Analyzing research findings",
                "searches_performed": [
                    {"timestamp": 1620000000, "query": "quantum computing business applications", "tool": "tavily_search", "result_count": 12}
                ],
                "tasks_created": [
                    {"timestamp": 1620000100, "task_id": "task-1", "description": "Research quantum computing in finance"}
                ],
                "tasks_completed": [
                    {"timestamp": 1620001000, "task_id": "task-1"}
                ],
                "sections_completed": [
                    {"timestamp": 1620002000, "section_name": "Introduction to Quantum Computing"}
                ],
                "last_message": "Compiling information on quantum finance applications"
            }
        }


@router.get("/{session_id}", response_model=ProgressResponse)
async def get_session_progress(session_id: str):
    """
    Get the current progress for a specific research session.
    
    Args:
        session_id: The unique session identifier
        
    Returns:
        The current progress information
    """
    progress = get_research_progress(session_id)
    
    if "error" in progress:
        raise HTTPException(status_code=404, detail=f"No progress data found for session {session_id}")
    
    return progress 