from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Body, HTTPException, Request, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import json
import asyncio
from agents.deep_research_agent import get_deep_research_agent
from agents import progress_tracker

router = APIRouter()

# Define model for the research request
class ResearchRequest(BaseModel):
    question: str
    model_id: Optional[str] = "gpt-4-1106-preview"
    additional_instructions: Optional[str] = None

class ProgressResponse(BaseModel):
    session_id: str
    current_step: int
    current_step_name: str
    progress: int
    elapsed_time: float
    elapsed_time_formatted: str
    search_query_count: int
    source_count: int
    recent_searches: List[Dict[str, Any]]
    recent_sources: List[Dict[str, Any]]
    recent_updates: List[Dict[str, Any]]
    completed: bool

# Store active WebSocket connections
active_connections: Dict[str, List[WebSocket]] = {}

async def broadcast_progress(session_id: str):
    """
    Broadcast progress updates to all connected WebSockets for a session.
    """
    if session_id not in active_connections or not active_connections[session_id]:
        return
    
    # Get the latest progress
    tracker = progress_tracker.get_tracker(session_id, create_if_missing=False)
    if not tracker:
        return
    
    # Get progress data
    progress_data = tracker.get_progress()
    
    # Send to all connected clients
    for websocket in active_connections[session_id]:
        try:
            await websocket.send_json(progress_data)
        except Exception:
            # Handle any errors (client disconnected, etc.)
            pass

@router.post("/deep-research")
async def deep_research(request: Request, research_req: ResearchRequest = Body(...)):
    """
    Perform a deep research on a given question
    """
    # Generate a user ID and session ID from the request
    user_id = "user_" + request.client.host.replace(".", "_")
    session_id = f"session_{user_id}_{hash(research_req.question)}"
    
    # Get or create a progress tracker for this session
    tracker = progress_tracker.get_tracker(session_id)
    
    # Set up the research agent
    agent = get_deep_research_agent(
        model_id=research_req.model_id,
        user_id=user_id,
        session_id=session_id,
        progress_tracker=tracker  # Pass the tracker to the agent
    )
    
    # Create a background task to run the research and send progress updates
    async def run_research_with_updates():
        try:
            # Start progress monitoring in background
            monitor_task = asyncio.create_task(monitor_progress(session_id))
            
            # Run the actual research
            response = await agent.arun({
                "input": research_req.question,
                "additional_instructions": research_req.additional_instructions or ""
            })
            
            # Mark research as complete
            tracker.mark_complete()
            
            # Send one final update
            await broadcast_progress(session_id)
            
            # Cancel the monitor task
            monitor_task.cancel()
            
            return response
        except Exception as e:
            # Log and propagate the error
            tracker.add_status_update(f"Error in research: {str(e)}", None)
            raise e
    
    # Start the research task in the background
    background_research = asyncio.create_task(run_research_with_updates())
    
    # Return the session ID to the client
    return {
        "session_id": session_id,
        "message": "Research started. You can track progress using the research-progress endpoint or WebSocket."
    }

async def monitor_progress(session_id: str):
    """
    Periodically check research progress and broadcast updates.
    """
    while True:
        await broadcast_progress(session_id)
        await asyncio.sleep(2)  # Update every 2 seconds

@router.get("/research-progress/{session_id}")
async def get_research_progress(session_id: str):
    """
    Get the current progress of a research session
    """
    tracker = progress_tracker.get_tracker(session_id, create_if_missing=False)
    if not tracker:
        raise HTTPException(status_code=404, detail=f"No research session found with ID: {session_id}")
    
    return tracker.get_progress()

@router.websocket("/ws/research-progress/{session_id}")
async def websocket_research_progress(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time progress updates
    """
    await websocket.accept()
    
    # Register the connection
    if session_id not in active_connections:
        active_connections[session_id] = []
    active_connections[session_id].append(websocket)
    
    try:
        # Send initial progress data
        tracker = progress_tracker.get_tracker(session_id, create_if_missing=False)
        if tracker:
            await websocket.send_json(tracker.get_progress())
        
        # Keep the connection open and handle messages
        while True:
            # Simply read and discard any incoming messages
            data = await websocket.receive_text()
            # If client sends "ping", respond with "pong" to keep connection alive
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        # Remove the connection when client disconnects
        if session_id in active_connections:
            active_connections[session_id].remove(websocket)
    except Exception as e:
        # Handle any other errors
        print(f"WebSocket error: {str(e)}")
    finally:
        # Ensure connection is removed when an error occurs
        if session_id in active_connections and websocket in active_connections[session_id]:
            active_connections[session_id].remove(websocket)

@router.get("/active-research-sessions")
async def get_active_sessions():
    """
    Get all active research sessions
    """
    return {"sessions": progress_tracker.get_all_active_trackers()} 