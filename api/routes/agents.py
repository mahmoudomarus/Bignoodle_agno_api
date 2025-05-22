from enum import Enum
from logging import getLogger
from typing import AsyncGenerator, List, Optional, Dict, Any

from agno.agent import Agent, AgentKnowledge
from fastapi import APIRouter, HTTPException, status, Request, Body, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.agno_assist import get_agno_assist_knowledge
from agents.selector import AgentType, get_agent, get_available_agents, get_agent_by_id, get_deep_research_agent_instance
from agents.deep_research_agent import (
    DeepResearchAgent, 
    create_supervisor_agent,
    create_researcher_agent
)
from api.models import ResearchRequest, PlaygroundStatus
from agents.progress_tracker import progress_tracker, ResearchStage

logger = getLogger(__name__)

######################################################
## Routes for the Agent Interface
######################################################

agents_router = APIRouter(prefix="/agents", tags=["Agents"])


class Model(str, Enum):
    gpt_4_1 = "gpt-4.1"
    o4_mini = "o4-mini"


@agents_router.get("", response_model=List[str])
async def list_agents():
    """
    Returns a list of all available agent IDs.

    Returns:
        List[str]: List of agent identifiers
    """
    return get_available_agents()


async def chat_response_streamer(agent: Agent, message: str) -> AsyncGenerator:
    """
    Stream agent responses chunk by chunk.

    Args:
        agent: The agent instance to interact with
        message: User message to process

    Yields:
        Text chunks from the agent response
    """
    run_response = await agent.arun(message, stream=True)
    async for chunk in run_response:
        # chunk.content only contains the text response from the Agent.
        # For advanced use cases, we should yield the entire chunk
        # that contains the tool calls and intermediate steps.
        yield chunk.content


class RunRequest(BaseModel):
    """Request model for an running an agent"""

    message: str
    stream: bool = True
    model: Model = Model.gpt_4_1
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@agents_router.post("/{agent_id}/runs", status_code=status.HTTP_200_OK)
async def create_agent_run(agent_id: AgentType, body: RunRequest):
    """
    Sends a message to a specific agent and returns the response.

    Args:
        agent_id: The ID of the agent to interact with
        body: Request parameters including the message

    Returns:
        Either a streaming response or the complete agent response
    """
    logger.debug(f"RunRequest: {body}")

    try:
        agent: Agent = get_agent(
            model_id=body.model.value,
            agent_id=agent_id,
            user_id=body.user_id,
            session_id=body.session_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    if body.stream:
        return StreamingResponse(
            chat_response_streamer(agent, body.message),
            media_type="text/event-stream",
        )
    else:
        response = await agent.arun(body.message, stream=False)
        # In this case, the response.content only contains the text response from the Agent.
        # For advanced use cases, we should yield the entire response
        # that contains the tool calls and intermediate steps.
        return response.content


@agents_router.post("/{agent_id}/knowledge/load", status_code=status.HTTP_200_OK)
async def load_agent_knowledge(agent_id: AgentType):
    """
    Loads the knowledge base for a specific agent.

    Args:
        agent_id: The ID of the agent to load knowledge for.

    Returns:
        A success message if the knowledge base is loaded.
    """
    agent_knowledge: Optional[AgentKnowledge] = None

    if agent_id == AgentType.AGNO_ASSIST:
        agent_knowledge = get_agno_assist_knowledge()
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Agent {agent_id} does not have a knowledge base.",
        )

    try:
        await agent_knowledge.aload(upsert=True)
    except Exception as e:
        logger.error(f"Error loading knowledge base for {agent_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load knowledge base for {agent_id}.",
        )

    return {"message": f"Knowledge base for {agent_id} loaded successfully."}


class ResearchProgressRequest(BaseModel):
    session_id: str

@agents_router.get("/research/progress/{session_id}")
async def get_research_progress(session_id: str):
    """
    Get the progress status of a research task
    """
    progress_data = progress_tracker.get_session_status(session_id)
    if "error" in progress_data:
        raise HTTPException(status_code=404, detail=progress_data["error"])
    
    return progress_data

@agents_router.get("/research/progress/{session_id}/details")
async def get_detailed_research_progress(session_id: str):
    """
    Get detailed progress information of a research task including history
    """
    progress_data = progress_tracker.get_full_session_details(session_id)
    if "error" in progress_data:
        raise HTTPException(status_code=404, detail=progress_data["error"])
    
    return progress_data

@agents_router.post("/deep-research")
async def execute_deep_research(request: ResearchRequest):
    """
    Execute a deep research request using the Deep Research Agent system.
    Returns a detailed report on the topic with cited sources.
    """
    try:
        # Initialize the Deep Research Agent with specified timeout
        agent = get_deep_research_agent_instance()
        
        # Set timeout to 5 minutes (300 seconds) if not specified
        timeout = request.timeout_seconds or 300
        
        # Create a background task to handle the research
        # This prevents the request from timing out
        background_tasks = BackgroundTasks()
        
        # Start research right away to create session ID
        session_id = agent.session_id or progress_tracker.create_session()
        
        # Execute the research with the specified parameters
        async def run_research():
            try:
                # Split research into chunks to avoid token limit issues
                progress_tracker.update_stage(session_id, ResearchStage.PLANNING)
                
                # Pass max_tokens if specified
                max_tokens_per_call = request.max_tokens
                
                # Execute the research with parameters
                results = agent.execute_research(request.query)
                
                # Update progress to indicate completion
                progress_tracker.update_stage(session_id, ResearchStage.COMPLETE)
                progress_tracker.complete_session(session_id)
                
            except Exception as e:
                logger.exception(f"Error in background research task: {e}")
                progress_tracker.update_stage(session_id, ResearchStage.ERROR)
                progress_tracker.add_task(
                    session_id, 
                    "Error", 
                    f"Research failed: {str(e)}"
                )
        
        # Add the research task to run in the background
        background_tasks.add_task(run_research)
        
        # Return the session ID immediately so client can track progress
        return {
            "session_id": session_id,
            "message": "Research started. Use the /research/progress/{session_id} endpoint to track progress."
        }
    except Exception as e:
        logger.exception(f"Error executing deep research: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute deep research: {str(e)}"
        )

@agents_router.get("/research/report/{session_id}")
async def get_research_report(session_id: str):
    """
    Get the completed research report for a session
    """
    # First check if research is complete
    progress_data = progress_tracker.get_session_status(session_id)
    if "error" in progress_data:
        raise HTTPException(status_code=404, detail=progress_data["error"])
    
    # If research is not complete, return appropriate message
    if progress_data["stage"] != ResearchStage.COMPLETE.value:
        if progress_data["stage"] == ResearchStage.ERROR.value:
            return {
                "status": "error",
                "message": "Research failed to complete. Please check session details for more information."
            }
        else:
            return {
                "status": "in_progress",
                "message": f"Research is still in progress (stage: {progress_data['stage']}). Please try again later.",
                "current_stage": progress_data["stage"],
                "progress_percentage": progress_data["progress_percentage"]
            }
    
    # Get the session data where we stored the report
    session_data = progress_tracker.get_session_data(session_id)
    if not session_data or "report" not in session_data:
        # Check if this is an older session without stored report
        from agents.selector import get_deep_research_agent_instance
        
        # Try to retrieve the report from agent's memory
        try:
            agent = get_deep_research_agent_instance()
            if hasattr(agent, "memory") and agent.memory:
                # Attempt to retrieve the report from memory
                return {
                    "status": "complete",
                    "report": "Report could not be retrieved from memory. Please run the research again."
                }
        except Exception as e:
            logger.exception(f"Error retrieving report from memory: {e}")
            return {
                "status": "error",
                "message": "Could not retrieve the report. It may have exceeded token limits or timed out."
            }
    
    # Return the report
    return {
        "status": "complete",
        "report": session_data.get("report", "Report not found"),
        "topics_researched": session_data.get("topics_researched", []),
        "time_taken_seconds": session_data.get("time_taken_seconds"),
        "token_usage": session_data.get("token_usage")
    }
