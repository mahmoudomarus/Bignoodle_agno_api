import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime


class ResearchProgressTracker:
    """
    Tracks and manages progress of research tasks to provide real-time updates.
    """
    
    def __init__(self, session_id: str, total_steps: int = 5):
        """
        Initialize a new progress tracker.
        
        Args:
            session_id: Unique identifier for the research session
            total_steps: Total number of high-level steps in the research process
        """
        self.session_id = session_id
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.total_steps = total_steps
        self.completed_steps = 0
        self.current_step = 1
        self.step_names = [
            "Analyzing Research Question",
            "Gathering Information",
            "Processing Data",
            "Synthesizing Findings",
            "Formatting Results"
        ]
        self.search_queries = []
        self.sources_found = []
        self.status_updates = []
        self.completed = False
        
        # Initialize with first status update
        self.add_status_update("Research started", 0)
    
    def update_step(self, step_number: int, step_name: Optional[str] = None) -> None:
        """
        Update the current research step.
        
        Args:
            step_number: The new step number (1-based)
            step_name: Optional custom name for this step
        """
        if 1 <= step_number <= self.total_steps:
            self.current_step = step_number
            if step_name and step_number <= len(self.step_names):
                self.step_names[step_number - 1] = step_name
            
            self.add_status_update(f"Moving to step {step_number}: {self.get_current_step_name()}", 
                                  (step_number - 1) * (100 // self.total_steps))
    
    def get_current_step_name(self) -> str:
        """Get the name of the current step."""
        if 1 <= self.current_step <= len(self.step_names):
            return self.step_names[self.current_step - 1]
        return f"Step {self.current_step}"
    
    def complete_step(self, step_number: Optional[int] = None) -> None:
        """
        Mark a step as completed.
        
        Args:
            step_number: The step to mark complete (defaults to current step)
        """
        step = step_number if step_number is not None else self.current_step
        if step not in range(1, self.completed_steps + 2):
            return
            
        self.completed_steps = max(self.completed_steps, step)
        progress = min(95, self.completed_steps * (100 // self.total_steps))
        
        self.add_status_update(
            f"Completed step {step}: {self.step_names[step - 1]}", 
            progress
        )
        
        # Automatically move to next step if completing current
        if step == self.current_step and step < self.total_steps:
            self.current_step += 1
    
    def add_search_query(self, query: str, tool: str = "tavily_search") -> None:
        """
        Record a search query that was executed.
        
        Args:
            query: The search query
            tool: The tool used for searching
        """
        self.search_queries.append({
            "query": query,
            "tool": tool,
            "timestamp": time.time()
        })
        
        # Add status update with search count
        query_count = len(self.search_queries)
        self.add_status_update(
            f"Executed search query #{query_count}: '{query[:40] + '...' if len(query) > 40 else query}'",
            None  # Don't update progress percentage
        )
    
    def add_source(self, title: str, url: Optional[str] = None) -> None:
        """
        Add a source that was found during research.
        
        Args:
            title: Title of the source
            url: URL of the source if available
        """
        self.sources_found.append({
            "title": title,
            "url": url,
            "timestamp": time.time()
        })
        
        # Update status with source count
        source_count = len(self.sources_found)
        self.add_status_update(
            f"Found source #{source_count}: {title[:50] + '...' if len(title) > 50 else title}",
            None  # Don't update progress percentage
        )
    
    def add_status_update(self, message: str, progress: Optional[int] = None) -> None:
        """
        Add a status update message.
        
        Args:
            message: Status update message
            progress: Current progress percentage (0-100)
        """
        current_time = time.time()
        
        update = {
            "message": message,
            "timestamp": current_time,
            "elapsed_time": round(current_time - self.start_time, 2),
            "delta_time": round(current_time - self.last_update_time, 2)
        }
        
        if progress is not None:
            update["progress"] = progress
        
        self.status_updates.append(update)
        self.last_update_time = current_time
    
    def mark_complete(self) -> None:
        """Mark the research as complete."""
        self.completed = True
        self.add_status_update("Research completed", 100)
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get the current progress status.
        
        Returns:
            Dictionary with progress information
        """
        elapsed_time = time.time() - self.start_time
        
        # Calculate overall progress percentage
        if self.completed:
            progress = 100
        else:
            # Base progress on completed steps and current step
            base_progress = self.completed_steps * (100 // self.total_steps)
            
            # Add partial progress for current step
            if self.current_step > self.completed_steps:
                # Estimate current step progress based on sources and queries
                current_step_progress = min(90, (len(self.search_queries) + len(self.sources_found)) * 2)
                current_step_progress = min(current_step_progress, 95)  # Cap at 95%
                
                # Scale to the step's portion of the total
                step_portion = 100 // self.total_steps
                partial_progress = (current_step_progress * step_portion) // 100
                
                progress = min(95, base_progress + partial_progress)
            else:
                progress = base_progress
        
        return {
            "session_id": self.session_id,
            "current_step": self.current_step,
            "current_step_name": self.get_current_step_name(),
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
            "progress": progress,
            "elapsed_time": round(elapsed_time, 2),
            "elapsed_time_formatted": self._format_time(elapsed_time),
            "search_query_count": len(self.search_queries),
            "source_count": len(self.sources_found),
            "recent_searches": self.search_queries[-5:] if self.search_queries else [],
            "recent_sources": self.sources_found[-5:] if self.sources_found else [],
            "recent_updates": self.status_updates[-10:] if self.status_updates else [],
            "completed": self.completed,
            "timestamp": datetime.now().isoformat()
        }
    
    def _format_time(self, seconds: float) -> str:
        """
        Format time in seconds to a human-readable string.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"


# Global registry to store progress trackers by session ID
_progress_trackers: Dict[str, ResearchProgressTracker] = {}

def get_tracker(session_id: str, create_if_missing: bool = True) -> Optional[ResearchProgressTracker]:
    """
    Get progress tracker for a session. Creates one if it doesn't exist.
    
    Args:
        session_id: The session ID
        create_if_missing: Whether to create a new tracker if none exists
        
    Returns:
        The progress tracker, or None if not found and create_if_missing is False
    """
    if session_id in _progress_trackers:
        return _progress_trackers[session_id]
    elif create_if_missing:
        tracker = ResearchProgressTracker(session_id)
        _progress_trackers[session_id] = tracker
        return tracker
    return None

def get_all_active_trackers() -> List[Dict[str, Any]]:
    """
    Get progress information for all active research sessions.
    
    Returns:
        List of progress dictionaries for all active trackers
    """
    return [tracker.get_progress() for tracker in _progress_trackers.values()]

def clear_tracker(session_id: str) -> bool:
    """
    Remove a progress tracker.
    
    Args:
        session_id: The session ID to remove
        
    Returns:
        True if removed, False if not found
    """
    if session_id in _progress_trackers:
        del _progress_trackers[session_id]
        return True
    return False 