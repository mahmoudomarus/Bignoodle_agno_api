from textwrap import dedent
from typing import Dict, List, Optional, Any
import json
import os
import time
import logging
import random
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import uuid

from agno.agent import Agent
from agno.memory.v2.db.postgres import PostgresMemoryDb
from agno.memory.v2.memory import Memory
from agno.models.openai import OpenAIChat
from agno.storage.agent.postgres import PostgresAgentStorage
from agno.tools.yfinance import YFinanceTools
from agents.base_tools import Tool, ToolType

from agents.tavily_tools import TavilyTools
from db.session import db_url
from agents.progress_tracker import progress_tracker, ResearchStage


# Token usage tracker class for monitoring token consumption
class TokenUsageTracker:
    """Tracks token usage across multiple operations"""
    
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        
    def add_usage(self, prompt_tokens: int, completion_tokens: int):
        """Add token usage from a single operation"""
        # Convert to int if strings are provided
        if isinstance(prompt_tokens, str):
            try:
                prompt_tokens = len(prompt_tokens)
            except:
                prompt_tokens = 0
                
        if isinstance(completion_tokens, str):
            try:
                completion_tokens = len(completion_tokens)
            except:
                completion_tokens = 0
                
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        
    def add_tracker(self, tracker):
        """Add usage from another tracker or compatible object"""
        if hasattr(tracker, 'prompt_tokens'):
            self.prompt_tokens += tracker.prompt_tokens
        if hasattr(tracker, 'completion_tokens'):
            self.completion_tokens += tracker.completion_tokens
        if hasattr(tracker, 'total_tokens'):
            # If the object tracks total directly, ensure our total is recalculated
            pass
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        
    def get_usage(self):
        """Get the current token usage summary"""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }


class AdvancedReasoningTool(Tool):
    """
    A tool that provides advanced reasoning capabilities for complex research scenarios.
    """
    
    def __init__(self):
        tool_types = [
            ToolType(
                name="chain_of_thought_reasoning",
                description="Break down complex reasoning tasks into step-by-step logical deductions.",
                function=self.chain_of_thought,
                args={
                    "question": {
                        "type": "string",
                        "description": "The complex question or problem to analyze",
                    },
                    "context": {
                        "type": "string",
                        "description": "Relevant information and context for the reasoning task",
                    },
                },
            ),
            ToolType(
                name="compare_and_contrast",
                description="Compare and contrast multiple concepts, findings, or sources.",
                function=self.compare_and_contrast,
                args={
                    "items": {
                        "type": "string",
                        "description": "JSON array of items to compare, with each item having 'name' and 'description'",
                    },
                    "criteria": {
                        "type": "string",
                        "description": "Comma-separated list of criteria to use for comparison",
                    },
                },
            ),
            ToolType(
                name="synthesize_findings",
                description="Synthesize multiple pieces of information into coherent insights.",
                function=self.synthesize_findings,
                args={
                    "findings": {
                        "type": "string",
                        "description": "JSON array of findings to synthesize, each with 'source' and 'content'",
                    },
                    "perspective": {
                        "type": "string",
                        "description": "Analytical perspective to apply (e.g., 'critical', 'integrative')",
                    },
                },
            ),
        ]
        super().__init__(tool_types=tool_types)
    
    def chain_of_thought(self, question: str, context: str) -> Dict:
        """
        Apply step-by-step reasoning to break down a complex problem.
        
        Args:
            question: The complex question or problem to analyze
            context: Relevant information and context
            
        Returns:
            A structured reasoning process with steps and conclusion
        """
        # This would normally call an external reasoning service
        # Here we're providing a structured format for the agent to use
        return {
            "question": question,
            "reasoning_framework": "step-by-step logical deduction",
            "instructions": "Break this problem down into clear steps, analyzing each component separately before combining insights.",
            "reasoning_template": {
                "step_1": "Define key terms and concepts",
                "step_2": "Identify relevant factors and variables",
                "step_3": "Analyze relationships between factors",
                "step_4": "Consider alternative perspectives",
                "step_5": "Draw evidence-based conclusions",
            }
        }
    
    def compare_and_contrast(self, items: str, criteria: str) -> Dict:
        """
        Compare and contrast multiple items based on specified criteria.
        
        Args:
            items: JSON array of items to compare
            criteria: Criteria to use for comparison
            
        Returns:
            Structured comparison analysis
        """
        try:
            items_data = json.loads(items)
            criteria_list = [c.strip() for c in criteria.split(",")]
            
            # Create a comparison matrix template
            comparison = {
                "items": [item["name"] for item in items_data],
                "criteria": criteria_list,
                "analysis_framework": "structured comparison matrix",
                "instructions": "Fill in this comparison matrix, analyzing each item against each criterion. Then synthesize overall patterns and insights."
            }
            
            return comparison
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format for items."}
    
    def synthesize_findings(self, findings: str, perspective: str) -> Dict:
        """
        Synthesize multiple pieces of information into coherent insights.
        
        Args:
            findings: JSON array of findings to synthesize
            perspective: Analytical perspective to apply
            
        Returns:
            Synthesis framework and guidance
        """
        try:
            findings_data = json.loads(findings)
            
            return {
                "synthesis_framework": f"{perspective} analysis",
                "source_count": len(findings_data),
                "synthesis_process": {
                    "step_1": "Identify key themes across all sources",
                    "step_2": "Note agreements and contradictions between sources",
                    "step_3": "Evaluate the quality and reliability of each source",
                    "step_4": "Integrate insights into a cohesive narrative",
                    "step_5": f"Apply {perspective} perspective to draw deeper meaning"
                }
            }
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format for findings."}


class SupervisorToolKit(Tool):
    """
    A toolkit that allows the supervisor agent to spawn and coordinate researcher agents.
    """

    def __init__(self, create_researcher_fn: callable, model_id: str = "chatgpt-4o-latest", user_id: Optional[str] = None):
        self.create_researcher_fn = create_researcher_fn
        self.model_id = model_id
        self.user_id = user_id
        self.active_agents: Dict[str, Dict] = {}

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
                    "search_depth": {
                        "type": "string",
                        "description": "How deep to search: 'basic' for quick results or 'advanced' for thorough research",
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
                    "include_visualizations": {
                        "type": "boolean",
                        "description": "Whether to include tables and visualization suggestions",
                    },
                },
            ),
            ToolType(
                name="research_planning",
                description="Create a comprehensive research plan for a complex topic.",
                function=self.research_planning,
                args={
                    "topic": {
                        "type": "string",
                        "description": "The main research topic",
                    },
                    "objective": {
                        "type": "string",
                        "description": "The primary objective of the research",
                    },
                    "depth": {
                        "type": "string",
                        "description": "Desired depth: 'overview', 'comprehensive', or 'expert'",
                    },
                },
            ),
        ]

        super().__init__(tool_types=tool_types)

    def create_research_task(self, task_id: str, research_question: str, additional_instructions: str = "", search_depth: str = "advanced") -> Dict[str, Any]:
        """
        Create a new research task to be executed by a researcher agent.
        
        Args:
            task_id: A unique identifier for this research task
            research_question: The specific research question to be answered
            additional_instructions: Any additional instructions for the researcher
            search_depth: How deep to search ('basic' or 'advanced')
            
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
            search_depth=search_depth,
        )
        
        self.active_agents[task_id] = {
            "agent": researcher,
            "question": research_question,
            "instructions": additional_instructions,
            "search_depth": search_depth,
            "status": "created",
            "results": None
        }
        
        return {
            "status": "success",
            "message": f"Research task '{task_id}' created successfully.",
            "task": {
                "id": task_id,
                "question": research_question,
                "search_depth": search_depth,
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
    
    def generate_research_report(self, title: str, sections: str, format_style: str = "academic", include_visualizations: bool = True) -> Dict[str, Any]:
        """
        Generate a well-formatted final research report from all collected research.
        
        Args:
            title: The title for the research report
            sections: JSON array of section objects with 'heading' and 'content' fields
            format_style: Desired formatting style (academic, business, journalistic)
            include_visualizations: Whether to include tables and visualization suggestions
            
        Returns:
            A formatted research report
        """
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
            
            # Add visualization placeholders if requested
            if include_visualizations and "data" in section["content"].lower():
                report += "### Visualization\n\n"
                report += "_A visualization would be appropriate here to represent the key data points discussed above._\n\n"
        
        # Formatting based on style
        if format_style == "academic":
            report += "## References\n\n"
            report += "_All sources used in this research are cited using APA format._\n\n"
        elif format_style == "business":
            report += "## Recommendations\n\n"
            report += "_Based on the findings in this report, the following strategic recommendations are proposed:_\n\n"
            report += "## Next Steps\n\n"
            report += "_To implement these recommendations effectively, consider the following action items:_\n\n"
        elif format_style == "journalistic":
            report += "## Sources\n\n"
            report += "_This report is based on information gathered from the following sources:_\n\n"
        
        return {
            "status": "success",
            "report": report,
            "format": format_style,
            "visualization_ready": include_visualizations
        }

    def research_planning(self, topic: str, objective: str, depth: str = "comprehensive") -> Dict[str, Any]:
        """
        Create a comprehensive research plan for a complex topic.
        
        Args:
            topic: The main research topic
            objective: The primary objective of the research
            depth: Desired depth of research
            
        Returns:
            A detailed research plan
        """
        # Adjust number of sections based on depth
        if depth == "overview":
            section_count = 3
            search_depth = "basic"
        elif depth == "comprehensive":
            section_count = 5
            search_depth = "advanced"
        elif depth == "expert":
            section_count = 8
            search_depth = "advanced"
        else:
            section_count = 5
            search_depth = "advanced"
        
        return {
            "topic": topic,
            "objective": objective,
            "depth": depth,
            "recommended_approach": {
                "section_count": section_count,
                "search_depth": search_depth,
                "suggested_structure": {
                    "background": "Historical and contextual information",
                    "current_state": "Present status and recent developments",
                    "analysis": "In-depth examination of key aspects",
                    "comparisons": "Relevant comparisons or contrasts",
                    "implications": "Broader significance and implications",
                    "future_directions": "Potential future developments"
                },
                "recommended_tools": [
                    "TavilySearch (primary source for web information)",
                    "AdvancedReasoning (for complex analytical tasks)",
                    "YFinance (if financial data is relevant)"
                ]
            }
        }


def create_researcher_agent(
    model_id: str = "chatgpt-4o-latest", 
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    research_question: str = "",
    additional_instructions: str = "",
    search_depth: str = "advanced",
) -> Agent:
    """
    Create a specialized researcher agent for a specific research task.
    
    Args:
        model_id: The model ID to use for the agent
        user_id: The user ID
        session_id: The session ID
        research_question: The research question to be answered
        additional_instructions: Any additional instructions for the researcher
        search_depth: How deep to search ('basic' or 'advanced')
        
    Returns:
        A new researcher agent
    """
    # Configure Tavily with appropriate search depth
    tavily_tools = TavilyTools(
        search_depth=search_depth,
        max_results=20 if search_depth == "advanced" else 10,
        include_answer=True,
        include_raw_content=True,
    )
    
    # Create advanced reasoning tools
    reasoning_tools = AdvancedReasoningTool()
    
    return Agent(
        name="Specialized Researcher",
        agent_id="researcher_agent",
        user_id=user_id,
        session_id=session_id,
        model=OpenAIChat(id=model_id),
        tools=[
            tavily_tools,  # Primary research tool (listed first for priority)
            reasoning_tools,  # Advanced reasoning capabilities
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
            You are an elite research specialist tasked with investigating the following research question with exhaustive thoroughness:
            "{research_question}"
            
            Your mission is to conduct comprehensive, in-depth research on this specific question, exploring every relevant aspect and providing authoritative information supported by evidence.
            
            You EXCLUSIVELY use Tavily search for all web-based information gathering. No other search tools are available.
        """),
        instructions=dedent(f"""\
            As an elite research specialist, your objective is to conduct exhaustive investigation into the following research question:
            
            RESEARCH QUESTION: "{research_question}"
            
            ADDITIONAL INSTRUCTIONS: {additional_instructions}
            
            SEARCH DEPTH: {search_depth} ({"Conduct exhaustive, multi-faceted research with multiple query approaches and source triangulation" if search_depth == "advanced" else "Conduct targeted, efficient research focusing on authoritative sources and key information"})
            
            Follow this rigorous research methodology:
            
            1. **Strategic Information Acquisition**:
               - Decompose the research question into its fundamental components and sub-questions
               - EXCLUSIVELY use tavily_search as your PRIMARY research tool for all web-based information
               - Employ systematic search strategies with multiple query formulations to ensure comprehensive coverage
               - Use precise, technical terminology in searches to access specialized information
               - For financial or quantitative analysis, utilize the yfinance tools
               - Document all search queries and information sources systematically
               
            2. **Comprehensive Analysis & Critical Evaluation**:
               - For each component of the research question:
                 * Conduct multiple searches using varied terminology and approaches
                 * Systematically cross-reference information across 3+ independent sources
                 * Evaluate source credibility using academic standards (authority, currency, objectivity)
                 * Identify consensus views AND points of disagreement in the literature
               - Apply advanced analytical frameworks:
                 * Use chain_of_thought_reasoning to break down complex conceptual problems
                 * Apply compare_and_contrast to evaluate competing theories or perspectives
                 * Utilize synthesize_findings to integrate information across disciplinary boundaries
               - Identify and address knowledge gaps through targeted follow-up research
               - Consider methodological limitations in existing research
               
            3. **Evidence Synthesis & Knowledge Integration**:
               - Systematically organize findings into a coherent knowledge structure
               - Apply disciplinary frameworks appropriate to the research domain
               - Evaluate the weight of evidence for key claims and conclusions
               - Identify meta-patterns across multiple information sources
               - Clearly distinguish between established facts, expert consensus, and emerging theories
               - Acknowledge limitations, uncertainties, and areas of scholarly disagreement
            
            4. **Scholarly Communication Format**:
               - Present findings with exceptional clarity and academic rigor:
                 * Begin with a concise executive summary of key findings
                 * Organize information into logical sections with descriptive headings
                 * Use precise terminology with definitions where needed
                 * Present quantitative information in appropriate tables or structured formats
                 * Include bullet points for key findings and insights
               - Maintain impeccable citation practices:
                 * Include specific citations for EVERY factual claim or assertion
                 * Format citations consistently with complete source information
                 * Include URLs for all web-based sources
                 * Specify the exact source for each major claim
               - Conclude with evidence-based implications, limitations, and recommendations
               
            Your research must be thorough, precise, and directly address all aspects of the research question.
            CRITICAL: You MUST use Tavily search as your EXCLUSIVE primary tool for all web-based research.
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


def create_supervisor_agent(
    model_id: str = "chatgpt-4o-latest",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Agent:
    """
    Create a supervisor agent that coordinates researcher agents.
    
    Args:
        model_id: The model ID to use for the agent
        user_id: The user ID
        session_id: The session ID
        
    Returns:
        A new supervisor agent
    """
    # Create the supervisor tools
    supervisor_tools = SupervisorToolKit(
        create_researcher_fn=create_researcher_agent,
        model_id=model_id,
        user_id=user_id,
    )
    
    # Create advanced reasoning tools
    reasoning_tools = AdvancedReasoningTool()
    
    return Agent(
        name="Research Supervisor",
        agent_id="supervisor_agent",
        user_id=user_id,
        session_id=session_id,
        model=OpenAIChat(id=model_id),
        tools=[
            supervisor_tools,
            reasoning_tools,
        ],
        description=dedent("""\
            You are a Research Supervisor responsible for coordinating complex research projects.
            Your role is to plan, delegate, and synthesize research across multiple domains and sources.
            You oversee a team of specialized researcher agents who execute detailed investigations under your guidance.
            
            You excel at:
            1. Breaking down complex research questions into structured components
            2. Creating comprehensive research plans with clear objectives
            3. Assigning specific research tasks to specialized agents
            4. Synthesizing diverse findings into cohesive reports
            5. Ensuring research quality, comprehensiveness, and accuracy
        """),
        instructions=dedent("""\
            As a Research Supervisor, your mission is to coordinate sophisticated research operations across multiple researcher agents.
            Follow this research methodology:
            
            1. **Research Planning**:
               - Analyze the research question thoroughly
               - Break complex topics into 3-8 manageable research components
               - Create a structured research plan with clear objectives for each component
               - Identify potential information sources and search strategies
            
            2. **Task Delegation**:
               - Use create_research_task to create targeted research tasks
               - Provide each researcher with specific questions and methodological guidance
               - Set the search_depth to "advanced" for comprehensive results
               - Include special instructions for domain-specific considerations
            
            3. **Results Analysis**:
               - Critically evaluate research findings from each task
               - Identify knowledge gaps requiring further investigation
               - Assess source quality and reliability
               - Cross-reference information across multiple sources
            
            4. **Synthesis & Integration**:
               - Integrate findings from all research components
               - Identify patterns, connections, and insights across sources
               - Resolve contradictions or inconsistencies
               - Structure information logically and hierarchically
            
            5. **Report Generation**:
               - Use generate_research_report to create comprehensive reports
               - Ensure proper citation of all sources
               - Include executive summaries and key findings
               - Present information in clear, well-structured formats
            
            Your goal is to produce thorough, authoritative research that comprehensively addresses the original question.
        """),
        storage=PostgresAgentStorage(table_name="supervisor_agent_sessions", db_url=db_url),
        add_history_to_messages=True,
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
    model_id: str = "chatgpt-4o-latest",
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
    
    # Configure Tavily with advanced search capabilities
    tavily_tools = TavilyTools(
        search_depth="advanced",
        max_results=20,  # Increased from 15 to 20 for more comprehensive results
        include_answer=True,
        include_raw_content=True,
    )
    
    # Create advanced reasoning tools
    reasoning_tools = AdvancedReasoningTool()
    
    return Agent(
        name="Deep Research Agent",
        agent_id="deep_research_agent",
        user_id=user_id,
        session_id=session_id,
        model=OpenAIChat(id=model_id),
        tools=[
            supervisor_tools,  # Coordinator for multi-agent research
            tavily_tools,      # Primary search tool (placed second for importance)
            reasoning_tools,   # Advanced reasoning capabilities
            YFinanceTools(     # Financial analysis tools
                stock_price=True,
                analyst_recommendations=True,
                stock_fundamentals=True,
                historical_prices=True,
                company_info=True,
                company_news=True,
            ),
        ],
        description=dedent("""\
            You are DeepResearch, an exceptionally powerful AI research agent designed to conduct comprehensive, in-depth research on any topic. You coordinate a team of specialized researcher agents to produce thorough, well-formatted research reports with academic-level rigor and precision.
            
            Your capabilities include:
            1. Breaking down complex research questions into structured components
            2. Coordinating multiple specialized researcher agents simultaneously
            3. Applying advanced reasoning frameworks to analyze findings
            4. Synthesizing information from diverse sources into cohesive reports
            5. Providing evidence-based conclusions supported by citations
            
            You maintain the highest standards of scholarship, including thorough source verification, critical analysis of information, and proper citation practices. Your research is comprehensive, nuanced, and exhaustive, leaving no stone unturned.
            
            Your audience typically comes from the world of crypto investing and crypto collectables. They like to trade Bitcoin Ordinals and NFTs, often using the Magic Eden marketplace (magiceden.us) to research bitcoin ordinals projects and bitcoin runes projects. You may get questions about Bitmaps, $NAT tokens, DMT (digital matter theory), the TAP protocol which is associated with TRAC and Bitcoin, and HIROS. Be prepared to provide detailed research on these crypto-related topics.
            
            CRITICAL: You MUST use Tavily search as your EXCLUSIVE research tool for all web-based information gathering. No other web search tools are available. Only use YFinance for specialized financial data queries.
        """),
        instructions=dedent("""\
            As DeepResearch, your mission is to deliver exhaustive, authoritative research on any topic requested by the user. You'll orchestrate a sophisticated multi-agent research workflow to produce scholarly-level research reports. For each research request, follow this rigorous methodology:

            1. **Research Planning & Question Decomposition**:
               - Begin by thoroughly analyzing the user's research request, identifying core questions and implicit information needs
               - Use the `research_planning` tool to create a comprehensive research strategy with clear objectives
               - Decompose complex topics into 5-10 interrelated research components, ensuring comprehensive coverage
               - Create a logical hierarchy of research questions that builds from foundational understanding to specialized insights
               - Prioritize DEPTH, THOROUGHNESS, and ACADEMIC RIGOR in your approach
               - For each component, identify specific information needs and potential sources of evidence
            
            2. **Multi-Agent Research Orchestration**:
               - For each research component, use `create_research_task` to spawn a specialized researcher agent
               - Provide each researcher with precise, targeted questions and methodological instructions
               - ALWAYS set search_depth to "advanced" for maximum thoroughness
               - Provide specific guidance on search strategies and domain-specific considerations
               - Instruct researchers to EXCLUSIVELY use tavily_search as their PRIMARY research tool
               - Sequence research tasks to build on prior findings when logical
               - Monitor progress and results, providing additional guidance as needed
            
            3. **Advanced Analytical Processing**:
               - Apply sophisticated reasoning frameworks to complex information:
                 * Use chain_of_thought_reasoning for step-by-step analytical breakdowns of difficult concepts
                 * Apply compare_and_contrast to systematically evaluate competing perspectives, theories, or options
                 * Employ synthesize_findings to integrate information across disciplinary boundaries
               - Critically evaluate source credibility using academic standards:
                 * Publication reputation and peer-review status
                 * Author credentials and expertise
                 * Methodological rigor and transparency
                 * Recency and relevance to the question at hand
               - Identify patterns, contradictions, and gaps across sources
               - Apply domain-appropriate analytical frameworks and methodologies
            
            4. **Comprehensive Synthesis & Report Construction**:
               - After thorough research, systematically integrate all findings into a cohesive knowledge structure
               - Identify meta-patterns, themes, and insights that emerge across research components
               - Evaluate the overall weight of evidence for key conclusions
               - Acknowledge areas of uncertainty, conflicting evidence, or knowledge gaps
               - Use the `generate_research_report` tool to create a publication-quality final report
               - ALWAYS enable include_visualizations for enhanced data presentation
               - Select the optimal formatting style based on research purpose:
                 * Academic: For scholarly, scientific, or educational investigations (default style)
                 * Business: For market analysis, strategic planning, or organizational research
                 * Journalistic: For current events, trend analysis, or public interest topics
            
            5. **Publication-Quality Report Standards**:
               - Structure your reports with exceptional clarity and scholarly organization:
                 * Executive Summary: Concise overview of research question, methodology, and key findings
                 * Table of Contents: Hierarchical organization of report sections
                 * Introduction: Context, significance, and scope of the research
                 * Literature Review/Background: Synthesis of existing knowledge on the topic
                 * Methodology: Transparent explanation of research approach
                 * Findings/Results: Systematically presented evidence organized by themes
                 * Analysis/Discussion: Critical interpretation of findings with supporting evidence
                 * Conclusions: Evidence-based answers to the research questions
                 * Limitations: Honest assessment of research constraints and uncertainties
                 * References: Comprehensive bibliography with properly formatted citations
               - Include precise citations for EVERY factual claim, insight, or quotation
               - Always include complete URLs for web sources to enable verification
               - Format information using academic conventions (tables, headings, etc.)
               - Use visualizations to clarify complex relationships or quantitative information
               
            6. **Domain-Specific Research Excellence**:
               - For financial/economic research:
                 * Utilize YFinance tools for precise market and company data
                 * Present quantitative data in standardized financial formats
                 * Apply appropriate financial analytical frameworks (e.g., SWOT, Porter's Five Forces)
                 * Include risk analysis and confidence intervals for projections
                 * Compare multiple data sources to establish reliability
               - For scientific/technical research:
                 * Prioritize peer-reviewed and authoritative technical sources
                 * Explain complex technical concepts with precision and clarity
                 * Present competing theories or models with fair representation
                 * Include disciplinary consensus views alongside emerging research
                 * Incorporate appropriate technical terminology with definitions
               - For historical/social research:
                 * Consider multiple perspectives and interpretive frameworks
                 * Acknowledge cultural and historical context of sources
                 * Distinguish between primary and secondary sources
                 * Address potential biases in historical accounts
                 * Consider socio-political factors influencing the topic
            
            7. **Research Integrity & Source Validation**:
               - Apply rigorous source evaluation standards:
                 * Currency: Prioritize recent sources for rapidly evolving topics
                 * Authority: Evaluate author credentials and institutional affiliations
                 * Accuracy: Cross-verify facts across multiple independent sources
                 * Objectivity: Assess potential biases or conflicts of interest
                 * Coverage: Ensure comprehensive treatment of the topic
               - Maintain intellectual honesty throughout:
                 * Explicitly distinguish between facts, expert opinions, and your analysis
                 * Acknowledge contradictory evidence and alternative interpretations
                 * Clearly mark areas of uncertainty or limited evidence
                 * Identify methodological limitations in source materials
                 * Present competing viewpoints fairly and accurately
               - Document your research process transparently
            
            RESEARCH APPROACH: Your methodology should mirror the standards of doctoral-level academic research, emphasizing comprehensiveness, methodological rigor, critical analysis, and evidence-based conclusions. You leave no aspect of the topic unexplored and no stone unturned in your pursuit of authoritative understanding.
            
            Additional Information:
            - You are interacting with the user_id: {current_user_id}
            - The user's name might be different from the user_id, you may ask for it if needed and add it to your memory if they share it with you.
            - The current date and time is: {current_datetime}
            
            AUDIENCE CONTEXT: Your users typically come from the world of crypto investing and crypto collectibles with specific interests in:
            
            - Bitcoin Ordinals: Digital collectibles on Bitcoin blockchain
              Resource: https://help.magiceden.io/en/articles/7154941-bitcoin-ordinals-a-beginner-s-guide
              
            - NFTs: Non-fungible tokens across various blockchains
            
            - Magic Eden marketplace: A popular platform for trading Bitcoin Ordinals
              Resource: https://magiceden.us
              
            - Bitcoin Runes: A new token standard on Bitcoin
              Resource: https://community.magiceden.io/learn/runes-guide
              
            - Bitmaps: An ordinals metaverse project
              Resource: https://help.magiceden.io/en/articles/8175699-understanding-bitmap-and-the-ordinals-metaverse
              
            - $NAT tokens: Native Bitcoin tokens
              Resource: https://natgmi.com/#faq
              
            - DMT (Digital Matter Theory): A Bitcoin-native project
              Resource: https://digital-matter-theory.gitbook.io/digital-matter-theory
              
            - TAP protocol: Associated with TRAC and Bitcoin
              Resource: https://sovryn.com/all-things-sovryn/tap-protocol-bitcoin
              
            - HIROS: A Bitcoin project
              Resource: https://superfan.gitbook.io/hiros
              
            When researching these topics, prioritize finding the most current information as the crypto space evolves rapidly. Include market trends, recent developments, and technical analysis when relevant. Always cite reliable sources and verify information across multiple references when possible.
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


class DeepResearchAgent:
    """
    Deep Research Agent that coordinates a supervisor and multiple researcher agents
    to conduct in-depth research on a topic with well-cited, rigorous methodology.
    """

    def __init__(
        self,
        supervisor_agent: Agent,
        tools: List[Tool] = None,
        researcher_agent_factory=None,
        max_iterations: int = 5,
    ):
        self.supervisor_agent = supervisor_agent
        self.tools = tools or []
        self.researcher_agent_factory = researcher_agent_factory or create_researcher_agent
        self.max_iterations = max_iterations
        self.token_usage = TokenUsageTracker()
        self.session_id = None
        
    async def execute_research(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute research on the given question using a simplified approach.
        
        Args:
            question: The research question to answer
            session_id: Optional session ID to use
            
        Returns:
            Dictionary with research results
        """
        import traceback
        
        if not session_id:
            session_id = str(uuid.uuid4())
            
        start_time = time.time()
        
        # Initialize progress tracking
        self.progress_tracker.initialize_session(session_id, question)
        self.progress_tracker.update_stage(session_id, ResearchStage.PLANNING)
        
        try:
            self.logger.info(f"Starting research for question: {question}")
            
            # Update to research stage before making the call
            self.progress_tracker.update_stage(session_id, ResearchStage.RESEARCH)
            
            # Create a comprehensive research prompt
            research_prompt = f"""
            You are a world-class research agent. Your task is to thoroughly investigate the following question:
            
            QUESTION: {question}
            
            Conduct comprehensive research on this topic and provide a detailed, well-structured report.
            
            Your report should include:
            
            1. EXECUTIVE SUMMARY: A brief overview of your findings
            2. BACKGROUND: Relevant context and information about the topic
            3. MAIN FINDINGS: Detailed information addressing the question
            4. ANALYSIS: Your expert analysis of the information
            5. CONCLUSIONS: Clear conclusions based on your research
            6. REFERENCES: Sources of information (if applicable)
            
            Be thorough, accurate, and objective in your research. Provide specific details and examples where relevant.
            """
            
            # Make a single call to OpenAI for the research
            self.logger.info("Making research call to OpenAI")
            research_result = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant."},
                    {"role": "user", "content": research_prompt}
                ],
                temperature=0.2,
                max_tokens=4000
            )
            
            # Extract the research report
            research_report = research_result.choices[0].message.content
            
            # Update progress to complete
            self.progress_tracker.update_stage(session_id, ResearchStage.COMPLETE)
            
            elapsed_time = time.time() - start_time
            
            return {
                "success": True,
                "question": question,
                "research_report": research_report,
                "elapsed_time": elapsed_time,
                "session_id": session_id
            }
            
        except Exception as e:
            error_msg = f"Error in research execution: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            # Update progress to error
            self.progress_tracker.update_stage(session_id, ResearchStage.ERROR)
            
            elapsed_time = time.time() - start_time
            
            return {
                "success": False,
                "question": question,
                "error": error_msg,
                "elapsed_time": elapsed_time,
                "session_id": session_id
            }

# Add this helper function at the top of the file
def run_with_retry(agent, prompt, max_attempts=5):
    """Run an agent with retry logic for rate limit errors"""
    @retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((RateLimitError, APIError, APIConnectionError)),
        before_sleep=before_sleep_log(logging.getLogger(), logging.WARNING)
    )
    def _run_with_retry():
        try:
            return agent.run(prompt)
        except Exception as e:
            if "rate limit" in str(e).lower():
                logging.warning(f"Rate limit hit, retrying: {str(e)}")
                raise RateLimitError(f"OpenAI rate limit: {str(e)}")
            elif "api" in str(e).lower() and ("error" in str(e).lower() or "connection" in str(e).lower()):
                logging.warning(f"API error, retrying: {str(e)}")
                raise APIError(f"OpenAI API error: {str(e)}")
            else:
                raise e
    
    try:
        return _run_with_retry()
    except tenacity.RetryError as e:
        # If all retries failed, return a simple object with content attribute
        logging.error(f"All retries failed: {str(e)}")
        return type('obj', (object,), {'content': f"Error after {max_attempts} retries: {str(e.last_attempt.exception())}"})

# Define custom exceptions for retry logic
class RateLimitError(Exception):
    """Rate limit error from OpenAI"""
    pass

class APIError(Exception):
    """Generic API error from OpenAI"""
    pass

class APIConnectionError(Exception):
    """API connection error from OpenAI"""
    pass