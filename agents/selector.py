from enum import Enum
from typing import List, Optional, Dict, Any

from agno.agent import Agent
from agents.web_agent import get_web_agent
from agents.agno_assist import get_agno_assist
from agents.finance_agent import get_finance_agent
from agents.deep_research_agent import get_deep_research_agent, DeepResearchAgent, create_researcher_agent


class AgentType(str, Enum):
    WEB_AGENT = "web_agent"
    AGNO_ASSIST = "agno_assist"
    FINANCE_AGENT = "finance_agent"
    DEEP_RESEARCH = "deep_research"


def get_available_agents() -> List[str]:
    """
    Returns a list of all available agent types.

    Returns:
        list[str]: List of available agent types.
    """
    # Only return the Deep Research agent type
    return [AgentType.DEEP_RESEARCH.value]


def get_agent_by_id(agent_id: str, model_id: str = "gpt-4.1", user_id: Optional[str] = None, session_id: Optional[str] = None) -> Agent:
    """
    Returns an agent by ID.

    Args:
        agent_id: ID of the agent to return.
        model_id: Model to use for the agent. Defaults to "gpt-4.1".
        user_id: User ID for memory/storage. Defaults to None.
        session_id: Session ID for memory/storage. Defaults to None.

    Returns:
        agent: The agent instance.

    Raises:
        ValueError: If an invalid agent ID is provided.
    """
    try:
        return get_agent(AgentType(agent_id), model_id, user_id, session_id)
    except ValueError:
        raise ValueError(f"Invalid agent ID: {agent_id}. Available agents: {get_available_agents()}")


def get_agent(
    agent_id: AgentType,
    model_id: str = "gpt-4.1",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    """
    Returns an agent by ID.

    Args:
        agent_id: ID of the agent to return.
        model_id: Model to use for the agent. Defaults to "gpt-4.1".
        user_id: User ID for memory/storage. Defaults to None.
        session_id: Session ID for memory/storage. Defaults to None.
        debug_mode: Enable debug mode. Defaults to True.

    Returns:
        agent: The agent instance.

    Raises:
        ValueError: If an invalid agent ID is provided.
    """
    if agent_id == AgentType.WEB_AGENT:
        return get_web_agent(model_id=model_id, user_id=user_id, session_id=session_id, debug_mode=debug_mode)
    elif agent_id == AgentType.AGNO_ASSIST:
        return get_agno_assist(model_id=model_id, user_id=user_id, session_id=session_id, debug_mode=debug_mode)
    elif agent_id == AgentType.FINANCE_AGENT:
        return get_finance_agent(model_id=model_id, user_id=user_id, session_id=session_id, debug_mode=debug_mode)
    elif agent_id == AgentType.DEEP_RESEARCH:
        return get_deep_research_agent(model_id=model_id, user_id=user_id, session_id=session_id, debug_mode=debug_mode)
    else:
        raise ValueError(f"Invalid agent type: {agent_id}. Available agents: {get_available_agents()}")


# Add a function to get a deep research agent instance for the report endpoint
def get_deep_research_agent_instance() -> Any:
    """
    Returns a DeepResearchAgent instance (not an Agno Agent).
    This is used for accessing completed research reports.
    
    Returns:
        A DeepResearchAgent instance
    """
    from agents.deep_research_agent import create_supervisor_agent
    
    # Create a supervisor agent
    supervisor_agent = create_supervisor_agent()
    
    # Create the DeepResearchAgent with the supervisor
    return DeepResearchAgent(
        supervisor_agent=supervisor_agent,
        researcher_agent_factory=create_researcher_agent
    )
