from textwrap import dedent
from typing import Dict, List, Optional, Any
import json
import time
import uuid

from agno.agent import Agent
from agno.memory.v2.db.postgres import PostgresMemoryDb
from agno.memory.v2.memory import Memory
from agno.models.openai import OpenAIChat
from agno.storage.agent.postgres import PostgresAgentStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agents.base_tools import Tool, ToolType

from agents.tavily_tools import TavilyTools
from db.session import db_url

# Global progress tracking dict
# Maps session_id to progress info
research_progress = {}

class ResearchProgressTracker:
    """
    Tracks progress of research tasks and exposes methods to update and retrieve status.
    """
    
    @staticmethod
    def initialize_progress(session_id: str, research_topic: str) -> str:
        """Initialize a new progress tracker for a research session"""
        progress_id = str(uuid.uuid4())
        research_progress[session_id] = {
            "id": progress_id,
            "topic": research_topic,
            "start_time": time.time(),
            "status": "initializing",
            "completion_percentage": 0,
            "current_stage": "Planning research approach",
            "searches_performed": [],
            "tasks_created": [],
            "tasks_completed": [],
            "sections_completed": [],
            "latest_update": time.time(),
            "last_message": "Starting comprehensive research"
        }
        return progress_id
    
    @staticmethod
    def update_progress(session_id: str, update_data: Dict[str, Any]) -> None:
        """Update progress for a research session"""
        if session_id not in research_progress:
            return
        
        progress = research_progress[session_id]
        progress.update(update_data)
        progress["latest_update"] = time.time()
    
    @staticmethod
    def add_search(session_id: str, query: str, tool: str, result_count: int) -> None:
        """Record a search that was performed"""
        if session_id not in research_progress:
            return
        
        progress = research_progress[session_id]
        progress["searches_performed"].append({
            "timestamp": time.time(),
            "query": query,
            "tool": tool,
            "result_count": result_count
        })
        progress["latest_update"] = time.time()
    
    @staticmethod
    def add_task(session_id: str, task_id: str, description: str) -> None:
        """Record a research task that was created"""
        if session_id not in research_progress:
            return
        
        progress = research_progress[session_id]
        progress["tasks_created"].append({
            "timestamp": time.time(),
            "task_id": task_id,
            "description": description
        })
        progress["latest_update"] = time.time()
    
    @staticmethod
    def complete_task(session_id: str, task_id: str) -> None:
        """Mark a research task as completed"""
        if session_id not in research_progress:
            return
        
        progress = research_progress[session_id]
        progress["tasks_completed"].append({
            "timestamp": time.time(),
            "task_id": task_id
        })
        
        # Update completion percentage based on tasks
        if progress["tasks_created"]:
            completion = len(progress["tasks_completed"]) / len(progress["tasks_created"])
            progress["completion_percentage"] = int(70 * completion)  # Tasks account for 70% of work
        
        progress["latest_update"] = time.time()
    
    @staticmethod
    def complete_section(session_id: str, section_name: str) -> None:
        """Mark a report section as completed"""
        if session_id not in research_progress:
            return
        
        progress = research_progress[session_id]
        progress["sections_completed"].append({
            "timestamp": time.time(),
            "section_name": section_name
        })
        progress["latest_update"] = time.time()
    
    @staticmethod
    def set_stage(session_id: str, stage: str, percentage: Optional[int] = None) -> None:
        """Update the current research stage"""
        if session_id not in research_progress:
            return
        
        progress = research_progress[session_id]
        progress["current_stage"] = stage
        if percentage is not None:
            progress["completion_percentage"] = percentage
        progress["latest_update"] = time.time()
        progress["last_message"] = stage
    
    @staticmethod
    def get_progress(session_id: str) -> Dict[str, Any]:
        """Get the current progress for a research session"""
        if session_id not in research_progress:
            return {"error": "No progress data for this session"}
        
        return research_progress[session_id]
    
    @staticmethod
    def finalize(session_id: str) -> None:
        """Mark research as complete"""
        if session_id not in research_progress:
            return
        
        progress = research_progress[session_id]
        progress["status"] = "completed"
        progress["completion_percentage"] = 100
        progress["current_stage"] = "Research completed"
        progress["latest_update"] = time.time()
        progress["last_message"] = "Research report completed"


class ProgressTrackingTavilyTools(TavilyTools):
    """
    Extended version of TavilyTools that tracks search progress.
    """
    
    def __init__(self, session_id: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.session_id = session_id
    
    def search(self, query: str, search_depth: Optional[str] = None) -> Dict[str, Any]:
        """
        Search with Tavily and track the search in progress.
        """
        result = super().search(query, search_depth)
        
        if self.session_id:
            ResearchProgressTracker.add_search(
                self.session_id, 
                query, 
                "tavily_search", 
                result.get("result_count", 0)
            )
            ResearchProgressTracker.set_stage(
                self.session_id,
                f"Searching for information on: {query}",
            )
        
        return result
    
    def search_news(self, query: str, max_results: Optional[int] = None) -> Dict[str, Any]:
        """
        Search news with Tavily and track the search in progress.
        """
        result = super().search_news(query, max_results)
        
        if self.session_id:
            ResearchProgressTracker.add_search(
                self.session_id, 
                query, 
                "tavily_news_search", 
                result.get("result_count", 0)
            )
            ResearchProgressTracker.set_stage(
                self.session_id,
                f"Searching for news articles about: {query}",
            )
        
        return result
    
    def search_topic(self, topic: str, include_domains: str = "", exclude_domains: str = "", time_period: str = "recent") -> Dict[str, Any]:
        """
        Perform a topic search with Tavily and track the search in progress.
        """
        result = super().search_topic(topic, include_domains, exclude_domains, time_period)
        
        if self.session_id:
            ResearchProgressTracker.add_search(
                self.session_id, 
                topic, 
                "tavily_topic_search", 
                result.get("result_count", 0)
            )
            ResearchProgressTracker.set_stage(
                self.session_id,
                f"Researching topic in depth: {topic}",
            )
        
        return result


class ProgressTrackingSupervisorToolKit(SupervisorToolKit):
    """
    Extended version of SupervisorToolKit that tracks research progress.
    """
    
    def __init__(self, create_researcher_fn: callable, session_id: Optional[str] = None, **kwargs):
        super().__init__(create_researcher_fn, **kwargs)
        self.session_id = session_id
    
    def create_research_task(self, task_id: str, research_question: str, additional_instructions: str = "", search_depth: str = "advanced") -> Dict[str, Any]:
        """
        Create a research task and track it in progress.
        """
        result = super().create_research_task(task_id, research_question, additional_instructions, search_depth)
        
        if self.session_id and result.get("status") == "success":
            ResearchProgressTracker.add_task(
                self.session_id,
                task_id,
                research_question
            )
            ResearchProgressTracker.set_stage(
                self.session_id,
                f"Created research task: {research_question}",
            )
        
        return result
    
    def execute_research_task(self, task_id: str) -> Dict[str, Any]:
        """
        Execute a research task and track its completion.
        """
        if self.session_id:
            ResearchProgressTracker.set_stage(
                self.session_id,
                f"Executing research task {task_id}",
            )
            
        result = super().execute_research_task(task_id)
        
        if self.session_id and result.get("status") == "success":
            ResearchProgressTracker.complete_task(
                self.session_id,
                task_id
            )
            ResearchProgressTracker.set_stage(
                self.session_id,
                f"Completed research task {task_id}",
            )
        
        return result
    
    def generate_research_report(self, title: str, sections: str, format_style: str = "academic", include_visualizations: bool = True) -> Dict[str, Any]:
        """
        Generate a research report and track its completion.
        """
        if self.session_id:
            ResearchProgressTracker.set_stage(
                self.session_id,
                "Generating final research report",
                90  # Report generation is near the end
            )
            
        result = super().generate_research_report(title, sections, format_style, include_visualizations)
        
        if self.session_id and result.get("status") == "success":
            ResearchProgressTracker.set_stage(
                self.session_id,
                "Finalizing research report formatting",
                95
            )
            
            # Extract section names from the JSON
            try:
                sections_data = json.loads(sections)
                for section in sections_data:
                    if "heading" in section:
                        ResearchProgressTracker.complete_section(
                            self.session_id,
                            section["heading"]
                        )
            except (json.JSONDecodeError, KeyError):
                pass
        
        return result
    
    def research_planning(self, topic: str, objective: str, depth: str = "comprehensive") -> Dict[str, Any]:
        """
        Create a research plan and track it in progress.
        """
        if self.session_id:
            ResearchProgressTracker.set_stage(
                self.session_id,
                f"Planning research approach for: {topic}",
                10  # Planning is early in the process
            )
            
        result = super().research_planning(topic, objective, depth)
        
        if self.session_id:
            ResearchProgressTracker.set_stage(
                self.session_id,
                "Research plan created, preparing to gather information",
                15
            )
        
        return result


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

    def __init__(self, create_researcher_fn: callable, model_id: str = "gpt-4.1", user_id: Optional[str] = None):
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


def create_progress_tracking_researcher(
    model_id: str = "gpt-4.1", 
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    research_question: str = "",
    additional_instructions: str = "",
    search_depth: str = "advanced",
) -> Agent:
    """
    Create a specialized researcher agent with progress tracking.
    """
    # Configure Tavily with appropriate search depth and progress tracking
    tavily_tools = ProgressTrackingTavilyTools(
        session_id=session_id,
        search_depth=search_depth,
        max_results=20 if search_depth == "advanced" else 10,
        include_answer=True,
        include_raw_content=True,
    )
    
    # Create advanced reasoning tools
    reasoning_tools = AdvancedReasoningTool()
    
    # Track progress
    if session_id:
        ResearchProgressTracker.set_stage(
            session_id,
            f"Creating specialized researcher for: {research_question}",
        )
    
    # Create the researcher agent with Tavily first in the tools list for priority
    researcher = Agent(
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
            # DuckDuckGo is now completely removed to ensure it's not used
        ],
        description=dedent(f"""\
            You are an elite research specialist tasked with investigating the following research question with exhaustive thoroughness:
            "{research_question}"
            
            Your mission is to conduct comprehensive, in-depth research on this specific question, exploring every relevant aspect and providing authoritative information supported by evidence.
            
            You MUST use Tavily search as your EXCLUSIVE primary research tool for all web-based information gathering. DuckDuckGo search is NOT available to you.
        """),
        instructions=dedent(f"""\
            As an elite research specialist, your objective is to conduct exhaustive investigation into the following research question:
            
            RESEARCH QUESTION: "{research_question}"
            
            ADDITIONAL INSTRUCTIONS: {additional_instructions}
            
            SEARCH DEPTH: {search_depth} ({"Conduct exhaustive, multi-faceted research with multiple query approaches and source triangulation" if search_depth == "advanced" else "Conduct targeted, efficient research focusing on authoritative sources and key information"})
            
            IMPORTANT: You MUST use tavily_search, tavily_news_search, or tavily_topic_search as your research tools. DuckDuckGo search is NOT available to you.
            
            Follow this rigorous research methodology:
            
            1. **Strategic Information Acquisition**:
               - Decompose the research question into its fundamental components and sub-questions
               - EXCLUSIVELY use tavily_search (or other Tavily tools) for all web-based information
               - Employ systematic search strategies with multiple query formulations to ensure comprehensive coverage
               - Use precise, technical terminology in searches to access specialized information
               - For financial or quantitative analysis, utilize the yfinance tools
               - Document all search queries and information sources systematically
               
            [Rest of instructions remain the same]
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
    
    return researcher


def get_deep_research_agent(
    model_id: str = "gpt-4.1",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    """
    Create a Deep Research agent with multi-agent workflow capabilities and progress tracking.
    
    Args:
        model_id: The model ID to use for the agent
        user_id: The user ID
        session_id: The session ID
        debug_mode: Whether to enable debug mode
        
    Returns:
        A new Deep Research agent
    """
    # Initialize progress tracking for this session
    if session_id:
        ResearchProgressTracker.initialize_progress(session_id, "Deep Research")
        ResearchProgressTracker.set_stage(session_id, "Initializing Deep Research agent", 5)
    
    # Create the supervisor tools with progress tracking
    supervisor_tools = ProgressTrackingSupervisorToolKit(
        create_researcher_fn=create_progress_tracking_researcher,
        session_id=session_id,
        model_id=model_id,
        user_id=user_id,
    )
    
    # Configure Tavily with progress tracking
    tavily_tools = ProgressTrackingTavilyTools(
        session_id=session_id,
        search_depth="advanced",
        max_results=20,
        include_answer=True,
        include_raw_content=True,
    )
    
    # Create advanced reasoning tools
    reasoning_tools = AdvancedReasoningTool()
    
    # Create the Deep Research agent with Tavily prioritized
    deep_research_agent = Agent(
        name="Deep Research Agent",
        agent_id="deep_research_agent",
        user_id=user_id,
        session_id=session_id,
        model=OpenAIChat(id=model_id),
        tools=[
            tavily_tools,      # PRIMARY search tool (placed FIRST for priority)
            supervisor_tools,  # Coordinator for multi-agent research
            reasoning_tools,   # Advanced reasoning capabilities
            YFinanceTools(     # Financial analysis tools
                stock_price=True,
                analyst_recommendations=True,
                stock_fundamentals=True,
                historical_prices=True,
                company_info=True,
                company_news=True,
            ),
            # DuckDuckGo is now completely removed to ensure it's not used
        ],
        description=dedent("""\
            You are DeepResearch, an exceptionally powerful AI research agent designed to conduct comprehensive, in-depth research on any topic. You coordinate specialized researcher agents to produce thorough, well-formatted research reports with academic-level rigor and precision.
            
            Your capabilities include:
            1. Breaking down complex research questions into structured components
            2. Coordinating multiple specialized researcher agents simultaneously
            3. Applying advanced reasoning frameworks to analyze findings
            4. Synthesizing information from diverse sources into cohesive reports
            5. Providing evidence-based conclusions supported by citations
            
            You MUST use Tavily search (tavily_search, tavily_news_search, or tavily_topic_search) as your PRIMARY and EXCLUSIVE research tool for all web-based information gathering. DuckDuckGo search is NOT available to you.
        """),
        instructions=dedent("""\
            As DeepResearch, your mission is to deliver exhaustive, authoritative research on any topic requested by the user. You'll orchestrate a sophisticated research workflow using Tavily search tools to produce scholarly-level reports. For each research request, follow this methodology:

            CRITICAL: You MUST use tavily_search, tavily_news_search, or tavily_topic_search for all web-based research. These are your ONLY available search tools - DuckDuckGo is NOT available to you.

            1. **Research Planning & Question Decomposition**:
               - Begin by thoroughly analyzing the user's research request
               - Decompose complex topics into 5-7 interrelated research components
               - Create clear research questions that build from fundamental to specialized insights
               - Prioritize DEPTH and ACADEMIC RIGOR in your approach
               - For each component, identify specific information needs
            
            2. **Information Gathering with Tavily Search**:
               - Use tavily_search as your PRIMARY search tool for general information
               - Use tavily_news_search for current events and recent developments
               - Use tavily_topic_search for specialized domain knowledge
               - Use varied search queries to ensure comprehensive coverage
               - Keep searches focused on specific aspects to improve relevance
               - For financial data, use the YFinance tools
            
            3. **Analysis & Critical Evaluation**:
               - Apply advanced reasoning frameworks to complex information
               - Evaluate source credibility using academic standards
               - Cross-verify important information across multiple sources
               - Identify patterns, contradictions, and gaps in the research
               
            4. **Report Construction**:
               - Integrate all findings into a cohesive knowledge structure
               - Create a well-structured report with clear sections
               - Include visualizations for data and complex relationships
               - Use academic formatting with proper citations
               - Complete your research within 10-15 minutes maximum
               
            EFFICIENCY GUIDELINES:
            - Focus on quality over quantity in your searches
            - Prioritize authoritative sources to save verification time
            - Process information in batches rather than sequentially
            - Complete research and analysis within 10-15 minutes total
            - Provide incremental updates to show work in progress
            
            REMEMBER: Tavily search tools (tavily_search, tavily_news_search, tavily_topic_search) are your ONLY options for web search. DuckDuckGo search is NOT available to you. Always prioritize Tavily search tools for information gathering.
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
    
    if session_id:
        ResearchProgressTracker.set_stage(session_id, "Deep Research agent initialized and ready", 10)
    
    return deep_research_agent

# Function to get progress for a specific session
def get_research_progress(session_id: str) -> Dict[str, Any]:
    """
    Get the current progress for a research session.
    
    Args:
        session_id: The session ID
        
    Returns:
        Progress information
    """
    return ResearchProgressTracker.get_progress(session_id)