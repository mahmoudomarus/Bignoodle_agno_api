# Bignoodle Deep Research Agent API

This API powers the Deep Research Agent, an advanced AI-powered research platform that conducts comprehensive, in-depth research on any topic through a multi-agent architecture.

## System Architecture

### Multi-Agent Architecture

The system employs a **Supervisor-Researcher workflow**:

1. **Supervisor Agent** - Coordinates the research process:
   - Breaks down complex research questions
   - Dispatches specialized research tasks
   - Synthesizes findings into coherent reports

2. **Researcher Agents** - Specialized agents that:
   - Focus on specific research questions
   - Conduct thorough investigations using Tavily search
   - Report findings back to the Supervisor

3. **Progress Tracking System** - Monitors and displays real-time research progress

## Based on Agno Agent API

This system is built upon the Agno Agent API framework, which includes:
* A **FastAPI server** for handling API requests
* A **PostgreSQL database** for storing Agent sessions, knowledge, and memories
* A set of **pre-built Agents** to use as a starting point

For more information, checkout [Agno](https://agno.link/gh)

## Deployment

This API is deployed to Heroku and available at:
* Backend: `https://agno-agent-api-tabi-39e2dc704f77.herokuapp.com`
* Frontend: `https://deepresearch.bignoodle.com`

## Key Features

1. **Advanced Multi-Agent Architecture**:
   - Specialized researchers for different aspects of the topic
   - Coordinated by a supervisor agent

2. **Real-Time Progress Tracking**:
   - Visualization of research stages
   - Task completion monitoring
   - Time tracking

3. **Comprehensive Research Methodology**:
   - Strategic information acquisition using Tavily Search
   - Critical evaluation of sources
   - Advanced analytical processing
   - Knowledge synthesis

4. **Academic-Quality Reports**:
   - Executive summaries
   - Well-structured content
   - Proper citations
   - Evidence-based conclusions

## Development Setup

To setup your local virtual environment:

```sh
# Clone the repo
git clone https://github.com/mahmoudomarus/Bignoodle_agno_api.git
cd Bignoodle_agno_api

# Configure API keys
export OPENAI_API_KEY="YOUR_API_KEY_HERE"
export TAVILY_API_KEY="YOUR_API_KEY_HERE"

# Start the application using docker compose
docker compose up -d
```
