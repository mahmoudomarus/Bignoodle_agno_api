"""
Progress Tracker for Deep Research Agent

This module provides functionality for tracking and displaying research progress
similar to the example shown in the Manus UI.
"""

import json
import time
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import threading
import uuid

class ResearchStage(Enum):
    """Enum for tracking the stage of research"""
    PLANNING = "planning"
    DATA_COLLECTION = "data_collection"
    ANALYSIS = "analysis"
    REPORT_GENERATION = "report_generation"
    COMPLETE = "complete"
    ERROR = "error"

class ProgressTracker:
    """
    Tracks research progress across stages and tasks
    """
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_data: Dict[str, Dict[str, Any]] = {}
    
    def create_session(self) -> str:
        """Create a new tracking session and return its ID"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "stage": ResearchStage.PLANNING.value,
            "start_time": time.time(),
            "last_updated": time.time(),
            "tasks": {},
            "active_tasks": set(),
            "completed_tasks": set(),
            "history": []
        }
        return session_id
    
    def update_stage(self, session_id: str, stage: ResearchStage) -> Dict[str, Any]:
        """Update the current stage of research"""
        if session_id not in self.sessions:
            return {"error": f"Session {session_id} not found"}
            
        self.sessions[session_id]["stage"] = stage.value
        self.sessions[session_id]["last_updated"] = time.time()
        
        # Add to history
        self.sessions[session_id]["history"].append({
            "timestamp": time.time(),
            "event": f"Stage changed to {stage.value}"
        })
        
        return self.get_session_status(session_id)
    
    def add_task(self, session_id: str, name: str, description: str) -> Dict[str, Any]:
        """Add a new task to track"""
        if session_id not in self.sessions:
            return {"error": f"Session {session_id} not found"}
        
        task_id = str(uuid.uuid4())
        self.sessions[session_id]["tasks"][task_id] = {
            "name": name,
            "description": description,
            "status": "pending",
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "result": None
        }
        
        # Add to history
        self.sessions[session_id]["history"].append({
            "timestamp": time.time(),
            "event": f"Task added: {name}"
        })
        
        return {"task_id": task_id}
    
    def start_task(self, session_id: str, task_id: str) -> Dict[str, Any]:
        """Mark a task as started"""
        if session_id not in self.sessions:
            return {"error": f"Session {session_id} not found"}
        
        if task_id not in self.sessions[session_id]["tasks"]:
            return {"error": f"Task {task_id} not found in session {session_id}"}
        
        self.sessions[session_id]["tasks"][task_id]["status"] = "in-progress"
        self.sessions[session_id]["tasks"][task_id]["started_at"] = time.time()
        self.sessions[session_id]["active_tasks"].add(task_id)
        self.sessions[session_id]["last_updated"] = time.time()
        
        # Add to history
        task_name = self.sessions[session_id]["tasks"][task_id]["name"]
        self.sessions[session_id]["history"].append({
            "timestamp": time.time(),
            "event": f"Task started: {task_name}"
        })
        
        return {"status": "success"}
    
    def complete_task(self, session_id: str, task_id: str, result: str = None) -> Dict[str, Any]:
        """Mark a task as completed"""
        if session_id not in self.sessions:
            return {"error": f"Session {session_id} not found"}
        
        if task_id not in self.sessions[session_id]["tasks"]:
            return {"error": f"Task {task_id} not found in session {session_id}"}
        
        self.sessions[session_id]["tasks"][task_id]["status"] = "completed"
        self.sessions[session_id]["tasks"][task_id]["completed_at"] = time.time()
        self.sessions[session_id]["tasks"][task_id]["result"] = result
        
        if task_id in self.sessions[session_id]["active_tasks"]:
            self.sessions[session_id]["active_tasks"].remove(task_id)
        
        self.sessions[session_id]["completed_tasks"].add(task_id)
        self.sessions[session_id]["last_updated"] = time.time()
        
        # Add to history
        task_name = self.sessions[session_id]["tasks"][task_id]["name"]
        self.sessions[session_id]["history"].append({
            "timestamp": time.time(),
            "event": f"Task completed: {task_name}"
        })
        
        return {"status": "success"}
    
    def complete_session(self, session_id: str) -> Dict[str, Any]:
        """Mark session as complete"""
        if session_id not in self.sessions:
            return {"error": f"Session {session_id} not found"}
        
        self.sessions[session_id]["completed_at"] = time.time()
        self.sessions[session_id]["last_updated"] = time.time()
        duration = self.sessions[session_id]["completed_at"] - self.sessions[session_id]["start_time"]
        
        # Add to history
        self.sessions[session_id]["history"].append({
            "timestamp": time.time(),
            "event": f"Session completed in {duration:.2f} seconds"
        })
        
        return {"status": "success", "duration_seconds": duration}
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get the current status of a session"""
        if session_id not in self.sessions:
            return {"error": f"Session {session_id} not found"}
        
        session = self.sessions[session_id]
        
        # Calculate progress percentage
        total_tasks = len(session["tasks"])
        completed_tasks = len(session["completed_tasks"])
        progress_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # Current stage
        current_stage = session["stage"]
        
        # Active tasks
        active_task_details = []
        for task_id in session["active_tasks"]:
            if task_id in session["tasks"]:
                task = session["tasks"][task_id]
                active_task_details.append({
                    "name": task["name"],
                    "description": task["description"],
                    "started_at": task["started_at"]
                })
        
        # Time elapsed
        elapsed_time = time.time() - session["start_time"]
        
        return {
            "session_id": session_id,
            "stage": current_stage,
            "progress_percentage": progress_percentage,
            "active_tasks": active_task_details,
            "elapsed_time_seconds": elapsed_time,
            "last_updated": session["last_updated"]
        }
    
    def get_full_session_details(self, session_id: str) -> Dict[str, Any]:
        """Get detailed information about a session including history"""
        if session_id not in self.sessions:
            return {"error": f"Session {session_id} not found"}
        
        session = self.sessions[session_id]
        status = self.get_session_status(session_id)
        
        # All tasks
        all_tasks = []
        for task_id, task in session["tasks"].items():
            all_tasks.append({
                "id": task_id,
                "name": task["name"],
                "description": task["description"],
                "status": task["status"],
                "created_at": task["created_at"],
                "started_at": task["started_at"],
                "completed_at": task["completed_at"]
            })
        
        # Return all details including history
        return {
            **status,
            "tasks": all_tasks,
            "history": session["history"]
        }
    
    def store_session_data(self, session_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Store custom data for a session (like research results)"""
        if session_id not in self.sessions:
            return {"error": f"Session {session_id} not found"}
        
        if session_id not in self.session_data:
            self.session_data[session_id] = {}
        
        self.session_data[session_id].update(data)
        
        return {"status": "success"}
    
    def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve custom data for a session"""
        if session_id not in self.sessions:
            return None
        
        return self.session_data.get(session_id, {})

# Create a singleton instance
progress_tracker = ProgressTracker() 