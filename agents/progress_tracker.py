"""
Progress Tracker for Deep Research Agent

This module provides functionality for tracking and displaying research progress
similar to the example shown in the Manus UI.
"""

import json
import time
from typing import Dict, List, Optional, Any
from enum import Enum
import threading
import uuid

class ResearchStage(str, Enum):
    PLANNING = "planning"
    DATA_COLLECTION = "data_collection"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    REPORT_GENERATION = "report_generation"
    COMPLETE = "complete"

class ProgressTracker:
    """
    Tracks and reports the progress of research tasks.
    Provides a real-time view of what the agent is working on.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ProgressTracker, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        self.active_sessions = {}
        self.task_history = {}
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new tracking session for a research request"""
        session_id = session_id or str(uuid.uuid4())
        
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                "created_at": time.time(),
                "updated_at": time.time(),
                "current_stage": ResearchStage.PLANNING,
                "progress": 0.0,  # 0.0 to 1.0
                "tasks": [],
                "current_task": None,
                "completed_tasks": [],
                "status": "active"
            }
            self.task_history[session_id] = []
        
        return session_id
    
    def update_stage(self, session_id: str, stage: ResearchStage, progress: float = None) -> Dict:
        """Update the current research stage"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        session["current_stage"] = stage
        session["updated_at"] = time.time()
        
        if progress is not None:
            session["progress"] = max(0.0, min(1.0, progress))  # Clamp between 0 and 1
        elif stage == ResearchStage.COMPLETE:
            session["progress"] = 1.0
        
        return self.get_session_status(session_id)
    
    def add_task(self, session_id: str, task_name: str, task_description: str = None) -> Dict:
        """Add a new task to the current research session"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        task_id = str(uuid.uuid4())
        task = {
            "id": task_id,
            "name": task_name,
            "description": task_description or "",
            "status": "pending",
            "created_at": time.time(),
            "updated_at": time.time(),
            "completed_at": None
        }
        
        self.active_sessions[session_id]["tasks"].append(task)
        self.task_history[session_id].append({
            "type": "task_added",
            "task_id": task_id,
            "task_name": task_name,
            "timestamp": time.time()
        })
        
        return {"task_id": task_id, "status": "added"}
    
    def start_task(self, session_id: str, task_id: str) -> Dict:
        """Mark a task as in progress"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        
        # Find the task
        task = None
        for t in session["tasks"]:
            if t["id"] == task_id:
                task = t
                break
        
        if not task:
            return {"error": "Task not found"}
        
        # Update task
        task["status"] = "in_progress"
        task["updated_at"] = time.time()
        session["current_task"] = task_id
        
        self.task_history[session_id].append({
            "type": "task_started",
            "task_id": task_id,
            "task_name": task["name"],
            "timestamp": time.time()
        })
        
        return {"task_id": task_id, "status": "in_progress"}
    
    def complete_task(self, session_id: str, task_id: str, result: Any = None) -> Dict:
        """Mark a task as completed"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        
        # Find the task
        task = None
        task_index = -1
        for i, t in enumerate(session["tasks"]):
            if t["id"] == task_id:
                task = t
                task_index = i
                break
        
        if not task:
            return {"error": "Task not found"}
        
        # Update task
        task["status"] = "completed"
        task["updated_at"] = time.time()
        task["completed_at"] = time.time()
        task["result"] = result
        
        # Move to completed tasks
        if task_index >= 0:
            session["tasks"].pop(task_index)
            session["completed_tasks"].append(task)
        
        # Update current task if this was the current one
        if session["current_task"] == task_id:
            session["current_task"] = None
        
        self.task_history[session_id].append({
            "type": "task_completed",
            "task_id": task_id,
            "task_name": task["name"],
            "timestamp": time.time()
        })
        
        # Update progress based on completed tasks ratio
        total_tasks = len(session["completed_tasks"]) + len(session["tasks"])
        if total_tasks > 0:
            session["progress"] = len(session["completed_tasks"]) / total_tasks
        
        return {"task_id": task_id, "status": "completed"}
    
    def get_session_status(self, session_id: str) -> Dict:
        """Get the current status of a research session"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        
        # Calculate time elapsed
        elapsed_time = time.time() - session["created_at"]
        
        # Format response
        return {
            "session_id": session_id,
            "progress": session["progress"],
            "current_stage": session["current_stage"],
            "elapsed_time_seconds": elapsed_time,
            "active_tasks": len(session["tasks"]),
            "completed_tasks": len(session["completed_tasks"]),
            "current_task": session["current_task"],
            "status": session["status"]
        }
    
    def get_full_session_details(self, session_id: str) -> Dict:
        """Get detailed information about a research session"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        
        # Calculate time elapsed
        elapsed_time = time.time() - session["created_at"]
        
        # Format response
        return {
            "session_id": session_id,
            "progress": session["progress"],
            "current_stage": session["current_stage"],
            "elapsed_time_seconds": elapsed_time,
            "active_tasks": session["tasks"],
            "completed_tasks": session["completed_tasks"],
            "current_task": session["current_task"],
            "status": session["status"],
            "history": self.task_history.get(session_id, [])
        }
    
    def complete_session(self, session_id: str) -> Dict:
        """Mark a session as completed"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        session["status"] = "completed"
        session["progress"] = 1.0
        session["current_stage"] = ResearchStage.COMPLETE
        session["updated_at"] = time.time()
        
        return self.get_session_status(session_id)

# Singleton instance
progress_tracker = ProgressTracker() 