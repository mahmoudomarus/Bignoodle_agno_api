from agno.playground import Playground

# Only import the Deep Research agent
from agents.deep_research_agent import get_deep_research_agent

######################################################
## Routes for the Playground Interface
######################################################

# Only keep the Deep Research agent for the playground
deep_research_agent = get_deep_research_agent(debug_mode=True)

# Create a playground instance with only Deep Research agent
playground = Playground(agents=[deep_research_agent])

# Get the router for the playground
playground_router = playground.get_async_router()
