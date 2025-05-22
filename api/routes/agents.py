from enum import Enum
from logging import getLogger
from typing import AsyncGenerator, List, Optional, Dict, Any

from agno.agent import Agent, AgentKnowledge
from fastapi import APIRouter, HTTPException, status, Request, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.agno_assist import get_agno_assist_knowledge
from agents.selector import AgentType, get_agent, get_available_agents
from agents.deep_research_agent import (
    DeepResearchAgent, 
    get_deep_research_agent,
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
        # Initialize the Deep Research Agent
        agent = get_deep_research_agent()
        
        # Execute the research and get results
        results = agent.execute_research(request.query)
        
        # Return results along with the session_id for progress tracking
        return {
            "session_id": results.get("session_id"),
            "report": results.get("report"),
            "topics_researched": results.get("topics_researched", []),
            "time_taken_seconds": results.get("time_taken_seconds"),
            "token_usage": results.get("token_usage"),
        }
    except Exception as e:
        logger.exception(f"Error executing deep research: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute deep research: {str(e)}"
        )
