from agno.playground import Playground

from agents.agno_assist import get_agno_assist
from agents.deep_research_agent import get_deep_research_agent
from agents.finance_agent import get_finance_agent
from agents.web_agent import get_web_agent

######################################################
## Routes for the Playground Interface
######################################################

# Get Agents to serve in the playground
web_agent = get_web_agent(debug_mode=True)
agno_assist = get_agno_assist(debug_mode=True)
finance_agent = get_finance_agent(debug_mode=True)
deep_research_agent = get_deep_research_agent(debug_mode=True)

# Create a playground instance
playground = Playground(agents=[web_agent, agno_assist, finance_agent, deep_research_agent])

# Get the router for the playground
playground_router = playground.get_async_router()
