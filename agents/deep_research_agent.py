from textwrap import dedent
from typing import Dict, List, Optional, Any

from agno.agent import Agent
from agno.memory.v2.db.postgres import PostgresMemoryDb
from agno.memory.v2.memory import Memory
from agno.models.openai import OpenAIChat
from agno.storage.agent.postgres import PostgresAgentStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.base import Tool, ToolType

from agents.tavily_tools import TavilyTools
from db.session import db_url


class SupervisorToolKit(Tool):
    """
    A toolkit that allows the supervisor agent to spawn and coordinate researcher agents.
    """

    def __init__(self, create_researcher_fn: callable, model_id: str = "gpt-4.1", user_id: Optional[str] = None):
        self.create_researcher_fn = create_researcher_fn
        self.model_id = model_id
        self.user_id = user_id
        self.active_agents: Dict[str, Agent] = {}

        tool_types = [
            ToolType(
                name="create_research_task",
                description="Create a research task for a specialized researcher agent to execute.",
                function=self.create_research_task,
                args={
                    "task_id": {
                        "type": "string",
                        "description": "A unique identifier for this research task (e.g., 'financial-analysis-task-1')",
                    },
                    "research_question": {
                        "type": "string",
                        "description": "The specific research question to be answered",
                    },
                    "additional_instructions": {
                        "type": "string",
                        "description": "Any additional instructions for the researcher (optional)",
                    },
                },
            ),
            ToolType(
                name="execute_research_task",
                description="Execute a previously created research task and get results.",
                function=self.execute_research_task,
                args={
                    "task_id": {
                        "type": "string",
                        "description": "The task ID of a previously created research task",
                    },
                },
            ),
            ToolType(
                name="generate_research_report",
                description="Generate a well-formatted final research report from all collected research.",
                function=self.generate_research_report,
                args={
                    "title": {
                        "type": "string",
                        "description": "The title for the research report",
                    },
                    "sections": {
                        "type": "string",
                        "description": "JSON array of section objects with 'heading' and 'content' fields",
                    },
                    "format_style": {
                        "type": "string",
                        "description": "Desired formatting style (academic, business, journalistic)",
                    },
                },
            ),
        ]

        super().__init__(tool_types=tool_types)

    def create_research_task(self, task_id: str, research_question: str, additional_instructions: str = "") -> Dict[str, Any]:
        """
        Create a new research task to be executed by a researcher agent.
        
        Args:
            task_id: A unique identifier for this research task
            research_question: The specific research question to be answered
            additional_instructions: Any additional instructions for the researcher
            
        Returns:
            A dictionary with the task details and status
        """
        if task_id in self.active_agents:
            return {"status": "error", "message": f"Task ID '{task_id}' already exists. Choose a different ID."}
        
        # Create a new researcher agent for this task
        researcher = self.create_researcher_fn(
            model_id=self.model_id,
            user_id=self.user_id,
            session_id=f"research-task-{task_id}",
            research_question=research_question,
            additional_instructions=additional_instructions,
        )
        
        self.active_agents[task_id] = {
            "agent": researcher,
            "question": research_question,
            "instructions": additional_instructions,
            "status": "created",
            "results": None
        }
        
        return {
            "status": "success",
            "message": f"Research task '{task_id}' created successfully.",
            "task": {
                "id": task_id,
                "question": research_question,
                "status": "ready"
            }
        }
    
    def execute_research_task(self, task_id: str) -> Dict[str, Any]:
        """
        Execute a previously created research task and get results.
        
        Args:
            task_id: The ID of the task to execute
            
        Returns:
            The research results
        """
        if task_id not in self.active_agents:
            return {"status": "error", "message": f"Task ID '{task_id}' not found. Create it first using create_research_task."}
        
        task = self.active_agents[task_id]
        
        if task["status"] == "completed":
            return {"status": "success", "message": "Task already completed.", "results": task["results"]}
        
        # Execute the research task
        researcher = task["agent"]
        research_results = researcher.run(task["question"])
        
        # Store results
        task["results"] = research_results
        task["status"] = "completed"
        
        return {
            "status": "success",
            "message": f"Research task '{task_id}' completed successfully.",
            "results": research_results
        }
    
    def generate_research_report(self, title: str, sections: str, format_style: str = "academic") -> Dict[str, Any]:
        """
        Generate a well-formatted final research report from all collected research.
        
        Args:
            title: The title for the research report
            sections: JSON array of section objects with 'heading' and 'content' fields
            format_style: Desired formatting style (academic, business, journalistic)
            
        Returns:
            A formatted research report
        """
        import json
        try:
            sections_data = json.loads(sections)
        except json.JSONDecodeError:
            return {"status": "error", "message": "Invalid JSON format for sections."}
        
        # Generate the report
        report = f"# {title}\n\n"
        
        # Executive Summary
        report += "## Executive Summary\n\n"
        summary = "This report presents findings on " + title.lower() + ". "
        summary += "It covers " + ", ".join([s["heading"] for s in sections_data[:3]]) + " and other topics. "
        report += summary + "\n\n"
        
        # Table of Contents
        report += "## Table of Contents\n\n"
        for i, section in enumerate(sections_data):
            report += f"{i+1}. [{section['heading']}](#{''.join(section['heading'].lower().split())})\n"
        report += "\n\n"
        
        # Main Content
        for section in sections_data:
            report += f"## {section['heading']}\n\n{section['content']}\n\n"
        
        # Formatting based on style
        if format_style == "academic":
            report += "## References\n\n"
        elif format_style == "business":
            report += "## Recommendations\n\n"
            report += "## Next Steps\n\n"
        elif format_style == "journalistic":
            report += "## Sources\n\n"
        
        return {
            "status": "success",
            "report": report,
            "format": format_style
        }


def create_researcher_agent(
    model_id: str = "gpt-4.1", 
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    research_question: str = "",
    additional_instructions: str = "",
) -> Agent:
    """
    Create a specialized researcher agent for a specific research task.
    
    Args:
        model_id: The model ID to use for the agent
        user_id: The user ID
        session_id: The session ID
        research_question: The research question to be answered
        additional_instructions: Any additional instructions for the researcher
        
    Returns:
        A new researcher agent
    """
    return Agent(
        name="Specialized Researcher",
        agent_id="researcher_agent",
        user_id=user_id,
        session_id=session_id,
        model=OpenAIChat(id=model_id),
        tools=[
            TavilyTools(),
            DuckDuckGoTools(),
            YFinanceTools(
                stock_price=True,
                analyst_recommendations=True,
                stock_fundamentals=True,
                historical_prices=True,
                company_info=True,
                company_news=True,
            ),
        ],
        description=dedent(f"""\
            You are a specialized researcher focused on answering the following research question:
            "{research_question}"
            
            Your goal is to thoroughly investigate this question and provide comprehensive, accurate information.
        """),
        instructions=dedent(f"""\
            As a specialized researcher, your goal is to thoroughly investigate the following research question:
            
            RESEARCH QUESTION: "{research_question}"
            
            ADDITIONAL INSTRUCTIONS: {additional_instructions}
            
            Follow this research process:
            
            1. **Initial Information Gathering**:
               - Break down the research question into key components
               - Use tavily_search to gather high-quality, relevant information
               - If financial analysis is needed, use the yfinance tools
               
            2. **Deep Research**:
               - For each key component, conduct thorough research
               - Cross-reference information from multiple sources
               - Identify gaps in information and conduct targeted searches to fill them
               - For complex topics, explore different perspectives
               
            3. **Synthesis and Analysis**:
               - Combine all gathered information into a coherent narrative
               - Analyze the relationships between different pieces of information
               - Identify key insights and patterns
               - Draw evidence-based conclusions
            
            4. **Format Your Response**:
               - Start with a concise summary of your findings
               - Organize your research into logical sections with clear headings
               - Use bullet points for key facts and insights
               - Include tables or lists where appropriate for clarity
               - Include citations for all sources used
               - End with recommendations or implications if appropriate
               
            Your research should be thorough, accurate, and directly answer the research question.
        """),
        storage=PostgresAgentStorage(table_name="researcher_agent_sessions", db_url=db_url),
        add_history_to_messages=True,
        num_history_runs=3,
        read_chat_history=True,
        memory=Memory(
            model=OpenAIChat(id=model_id),
            db=PostgresMemoryDb(table_name="user_memories", db_url=db_url),
            delete_memories=True,
            clear_memories=True,
        ),
        enable_agentic_memory=True,
        markdown=True,
        add_datetime_to_instructions=True,
        debug_mode=True,
    )


def get_deep_research_agent(
    model_id: str = "gpt-4.1",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    """
    Create a Deep Research agent with multi-agent workflow capabilities.
    
    Args:
        model_id: The model ID to use for the agent
        user_id: The user ID
        session_id: The session ID
        debug_mode: Whether to enable debug mode
        
    Returns:
        A new Deep Research agent
    """
    # Create the supervisor tools that can spawn researcher agents
    supervisor_tools = SupervisorToolKit(
        create_researcher_fn=create_researcher_agent,
        model_id=model_id,
        user_id=user_id,
    )
    
    return Agent(
        name="Deep Research Agent",
        agent_id="deep_research_agent",
        user_id=user_id,
        session_id=session_id,
        model=OpenAIChat(id=model_id),
        tools=[
            supervisor_tools,
            TavilyTools(),
            DuckDuckGoTools(),
            YFinanceTools(
                stock_price=True,
                analyst_recommendations=True,
                stock_fundamentals=True,
                historical_prices=True,
                company_info=True,
                company_news=True,
            ),
        ],
        description=dedent("""\
            You are DeepResearch, an advanced AI research agent that coordinates a team of specialized researchers to produce comprehensive, well-formatted research reports on any topic.
            
            You excel at breaking down complex research requests into manageable tasks, coordinating research efforts, and synthesizing findings into professional reports.
        """),
        instructions=dedent("""\
            As DeepResearch, your goal is to provide in-depth, comprehensive research on any topic requested by the user. You'll coordinate a multi-agent research workflow to produce high-quality, well-structured research reports. Follow this process for each research request:

            1. **Planning Phase**:
               - Carefully analyze the user's research request
               - Break down the request into 3-5 key research questions or components
               - Create a structured research plan with clear sections
               - Each section should address a specific aspect of the overall research question
            
            2. **Research Coordination**:
               - For each research section, use the `create_research_task` tool to create a specialized researcher agent
               - Provide each researcher with a specific, focused question and clear instructions
               - Use the `execute_research_task` tool to have each researcher thoroughly investigate their assigned topic
               - Monitor the progress and results of each research task
            
            3. **Synthesis and Report Generation**:
               - Once all research tasks are complete, review and synthesize the findings
               - Identify key insights, patterns, and relationships across different research components
               - Organize the information into a cohesive narrative with a logical flow
               - Use the `generate_research_report` tool to create a well-formatted final report
               - Choose an appropriate formatting style based on the nature of the research:
                 * Academic: For educational, scientific, or scholarly topics
                 * Business: For market analysis, company research, or strategy topics
                 * Journalistic: For current events, trends, or news-related topics
            
            4. **Report Format**:
               - Ensure the final report includes:
                 * Executive Summary: A concise overview of key findings
                 * Table of Contents: For easy navigation
                 * Introduction: Context and research objectives
                 * Main Sections: Organized by topic with clear headings
                 * Visualizations: Suggest tables, charts, or diagrams where appropriate
                 * Conclusion: Summary of findings and implications
                 * References/Sources: Properly cited sources
                 
            5. **Special Handling for Financial Research**:
               - For financial research requests, include:
                 * Market data and financial metrics in tables
                 * Clear analysis of trends and patterns
                 * Risk assessments
                 * Comparative analysis where relevant
               - Use YFinance tools directly when appropriate for real-time financial data
            
            6. **Quality Assurance**:
               - Ensure all research is fact-based and well-supported by sources
               - Cross-reference information from multiple sources when possible
               - Clearly distinguish between facts and analysis/interpretation
               - Maintain an objective, balanced perspective
            
            Remember: Your value comes from your ability to coordinate multiple research streams, synthesize complex information, and present it in a clear, structured format. Always focus on delivering comprehensive, accurate, and actionable research.
            
            Additional Information:
            - You are interacting with the user_id: {current_user_id}
            - The user's name might be different from the user_id, you may ask for it if needed and add it to your memory if they share it with you.
        """),
        add_state_in_messages=True,
        storage=PostgresAgentStorage(table_name="deep_research_agent_sessions", db_url=db_url),
        add_history_to_messages=True,
        num_history_runs=5,
        read_chat_history=True,
        memory=Memory(
            model=OpenAIChat(id=model_id),
            db=PostgresMemoryDb(table_name="user_memories", db_url=db_url),
            delete_memories=True,
            clear_memories=True,
        ),
        enable_agentic_memory=True,
        markdown=True,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
    ) 