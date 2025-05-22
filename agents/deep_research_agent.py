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
import re
from datetime import datetime

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

# Constants
DEFAULT_MODEL_ID = "o4-mini"  # Use this constant for consistent model naming
DEFAULT_TOOL_MODEL = "o4-mini"  # Model used for tools functionality

# Token usage tracker class for monitoring token consumption
class TokenUsageTracker:
    """Tracks token usage across multiple operations"""
    
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self._encoding = None  # Cache the encoding object
        
    def _get_encoding(self):
        """Get or create a cached tiktoken encoding object"""
        if self._encoding is None:
            try:
                import tiktoken
                self._encoding = tiktoken.encoding_for_model("gpt-4o")
            except (ImportError, Exception):
                # If tiktoken is not available or any other error
                return None
        return self._encoding
        
    def add_usage(self, prompt_tokens: int, completion_tokens: int):
        """Add token usage from a single operation"""
        # Convert to int if strings are provided
        if isinstance(prompt_tokens, str):
            encoding = self._get_encoding()
            if encoding:
                prompt_tokens = len(encoding.encode(prompt_tokens))
            else:
                # Fallback if tiktoken not available
                prompt_tokens = len(prompt_tokens) // 4  # Rough estimate
                
        if isinstance(completion_tokens, str):
            encoding = self._get_encoding()
            if encoding:
                completion_tokens = len(encoding.encode(completion_tokens))
            else:
                # Fallback if tiktoken not available
                completion_tokens = len(completion_tokens) // 4  # Rough estimate
                
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
        
        # Support for OpenAI response objects
        if hasattr(tracker, 'usage'):
            usage = tracker.usage
            if hasattr(usage, 'prompt_tokens'):
                self.prompt_tokens += usage.prompt_tokens
            if hasattr(usage, 'completion_tokens'):
                self.completion_tokens += usage.completion_tokens
            if hasattr(usage, 'total_tokens'):
                # Don't add total directly, recalculate
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
        # Create an OpenAI client to actually perform the reasoning
        try:
            from api.settings import settings
            import openai
            
            client = openai.OpenAI(api_key=settings.openai_api_key)
            
            # Construct a reasoning prompt that forces step by step thinking
            reasoning_prompt = f"""
            You need to break down this complex question using step-by-step logical reasoning.
            
            QUESTION: {question}
            
            CONTEXT: {context}
            
            Think through this problem systematically:
            1. First, clearly define any key terms or concepts in the question
            2. Identify the core issues or variables that need to be considered
            3. Analyze how these elements relate to each other
            4. Consider alternative perspectives or interpretations
            5. Draw evidence-based conclusions
            
            Provide your detailed reasoning process and final conclusion.
            """
            
            # Make the API call to get detailed reasoning
            response = client.chat.completions.create(
                model=DEFAULT_TOOL_MODEL,  # Using constant instead of hardcoded model name
                messages=[
                    {"role": "system", "content": "You are a logical reasoning assistant that breaks down complex problems step-by-step."},
                    {"role": "user", "content": reasoning_prompt}
                ],
                temperature=0.2,
                max_tokens=1000,
            )
            
            # Extract the reasoning from the response
            reasoning = response.choices[0].message.content if response.choices else "Error performing reasoning"
            
            # Return a structured result with the actual reasoning
            return {
                "question": question,
                "reasoning_process": reasoning,
                "conclusion": "See the final part of the reasoning process above."
            }
            
        except Exception as e:
            # Fall back to template if there's an error
            return {
                "question": question,
                "error": f"Error performing reasoning: {str(e)}",
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
            
            # Create a comparison prompt for the LLM
            comparison_prompt = f"""
            You need to compare and contrast the following items across specific criteria.
            
            ITEMS TO COMPARE:
            {json.dumps(items_data, indent=2)}
            
            COMPARISON CRITERIA:
            {criteria}
            
            For each item and criterion:
            1. Analyze how the item performs or relates to that criterion
            2. Note similarities and differences between items
            3. Highlight strengths and weaknesses
            
            Then provide an overall synthesis of patterns, insights, and conclusions from this comparison.
            """
            
            # Create an OpenAI client to perform the comparison
            from api.settings import settings
            import openai
            
            client = openai.OpenAI(api_key=settings.openai_api_key)
            
            # Make the API call
            response = client.chat.completions.create(
                model=DEFAULT_TOOL_MODEL,  # Using constant
                messages=[
                    {"role": "system", "content": "You are a comparative analysis expert who excels at structured comparison."},
                    {"role": "user", "content": comparison_prompt}
                ],
                temperature=0.2,
                max_tokens=1500,
            )
            
            # Extract the comparison from the response
            comparison_analysis = response.choices[0].message.content if response.choices else "Error performing comparison"
            
            # Return a structured result with the actual comparison
            return {
                "items": [item["name"] for item in items_data],
                "criteria": criteria_list,
                "comparison_analysis": comparison_analysis
            }
            
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format for items. Please provide properly formatted data."}
        except Exception as e:
            # Fall back to template if there's an error
            comparison = {
                "items": items,
                "criteria": criteria,
                "error": f"Error performing comparison: {str(e)}",
                "analysis_framework": "structured comparison matrix",
                "instructions": "Fill in this comparison matrix, analyzing each item against each criterion. Then synthesize overall patterns and insights."
            }
            
            return comparison
    
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
            
            # Create a synthesis prompt for the LLM
            synthesis_prompt = f"""
            You need to synthesize the following research findings from a {perspective} perspective.
            
            RESEARCH FINDINGS:
            {json.dumps(findings_data, indent=2)}
            
            ANALYTICAL PERSPECTIVE: {perspective}
            
            Your synthesis should:
            1. Identify key themes and patterns across all sources
            2. Note agreements and contradictions between sources
            3. Evaluate the quality and reliability of each source
            4. Integrate insights into a cohesive narrative
            5. Apply a {perspective} perspective to draw deeper meaning
            
            Provide a comprehensive synthesis that goes beyond summarizing to generate new insights.
            """
            
            # Create an OpenAI client to perform the synthesis
            from api.settings import settings
            import openai
            
            client = openai.OpenAI(api_key=settings.openai_api_key)
            
            # Make the API call
            response = client.chat.completions.create(
                model=DEFAULT_TOOL_MODEL,  # Using constant
                messages=[
                    {"role": "system", "content": f"You are a research synthesis expert with expertise in {perspective} analysis."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                temperature=0.2,
                max_tokens=1500,
            )
            
            # Extract the synthesis from the response
            synthesis_analysis = response.choices[0].message.content if response.choices else "Error performing synthesis"
            
            # Return a structured result with the actual synthesis
            return {
                "source_count": len(findings_data),
                "analytical_perspective": perspective,
                "synthesis": synthesis_analysis
            }
            
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format for findings. Please provide properly formatted data."}
        except Exception as e:
            # Fall back to template if there's an error
            return {
                "synthesis_framework": f"{perspective} analysis",
                "source_count": "unknown (invalid JSON)",
                "error": f"Error performing synthesis: {str(e)}",
                "synthesis_process": {
                    "step_1": "Identify key themes across all sources",
                    "step_2": "Note agreements and contradictions between sources",
                    "step_3": "Evaluate the quality and reliability of each source",
                    "step_4": "Integrate insights into a cohesive narrative",
                    "step_5": f"Apply {perspective} perspective to draw deeper meaning"
                }
            }


class SupervisorToolKit(Tool):
    """
    A toolkit that allows the supervisor agent to spawn and coordinate researcher agents.
    """

    def __init__(self, create_researcher_fn: callable, model_id: str = DEFAULT_MODEL_ID, user_id: Optional[str] = None):
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
    model_id: str = DEFAULT_MODEL_ID, 
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
            delete_memories=False,
            clear_memories=True,
        ),
        enable_agentic_memory=True,
        markdown=True,
        add_datetime_to_instructions=True,
        debug_mode=False,  # Changed to False for production
    )


def create_supervisor_agent(
    model_id: str = DEFAULT_MODEL_ID,  # Changed from gpt-4o to DEFAULT_MODEL_ID
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
            delete_memories=False,
            clear_memories=True,
        ),
        enable_agentic_memory=True,
        markdown=True,
        add_datetime_to_instructions=True,
        debug_mode=False,  # Changed to False for production
    )


class CryptoAwareAgent(Agent):
    """
    Enhanced Agent that forces using Tavily search tools and disables financial tools for crypto-related queries.
    This prevents wrong tool selection issues and ensures proper research is conducted.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Define crypto keywords for detection
        self.crypto_terms = [
            "protocol", "blockchain", "bitcoin", "ethereum", "crypto", "token", "nft", 
            "defi", "ordinal", "web3", "dao", "dapp", "smart contract", "wallet", 
            "mining", "staking", "validator", "consensus", "mainnet", "testnet",
            "btc", "eth", "sol", "bnb", "xrp", "ada", "avax", "tron", "tap protocol",
            "tap", "trac", "dmt", "digital matter theory", "bitmaps", "nat token", "hiros",
            "snat", "sovyrn", "satoshi", "bitcoin ordinals", "$nat", "runes"
        ]
    
    def _is_crypto_related(self, text):
        """Detect if text contains crypto-related terms"""
        if not text:
            return False
            
        text_lower = text.lower()
        for term in self.crypto_terms:
            if term.lower() in text_lower:
                return True
        return False
    
    def _is_information_seeking(self, text):
        """Detect if this is an information-seeking query that should trigger search"""
        if not text:
            return False
            
        info_patterns = [
            "what is", "how does", "explain", "tell me about", "describe", 
            "who is", "when was", "where is", "why is", "research", "information on",
            "details about", "history of", "can you provide", "find information",
            "?", "learn about", "understand", "overview of"
        ]
        
        text_lower = text.lower()
        for pattern in info_patterns:
            if pattern in text_lower:
                return True
        return False
    
    def _find_tavily_search_tool(self):
        """Find the tavily search tool in the available tools"""
        for tool in self.tools:
            if hasattr(tool, 'tool_types'):
                for tt in tool.tool_types:
                    if 'tavily_search' in tt.name or 'search' in tt.name:
                        return tt.name
        return None
    
    def run(self, prompt: str, **kwargs):
        """
        Override the run method to:
        1. Force using Tavily search for any information-seeking query
        2. Completely disable financial tools for crypto queries
        3. Force tool usage for all queries requiring research
        """
        # Always check if this is a crypto-related and/or information-seeking query
        is_crypto_query = self._is_crypto_related(prompt)
        is_info_seeking = self._is_information_seeking(prompt)
        
        # Find search tool
        tavily_search_name = self._find_tavily_search_tool()
        
        # Create a completely modified prompt if this is an information query
        if is_info_seeking or "research" in prompt.lower():
            # This is the critical change - for ANY information query, we inject
            # instructions that FORCE using tavily search first
            modified_prompt = f"""
[CRITICAL TOOL USAGE INSTRUCTION - HIGHEST PRIORITY]

You are responding to an information request: "{prompt}"

Before answering, you MUST:
1. Use {tavily_search_name} as your FIRST tool to gather current, accurate information
2. NEVER answer solely from memory without verifying with search
3. Use multiple search queries to explore different aspects of the topic
4. Cite your sources with proper links

For ANY information question, search is MANDATORY - not optional.

"""
            
            # Add crypto-specific instructions if needed
            if is_crypto_query:
                term = next((term for term in self.crypto_terms if term.lower() in prompt.lower()), "crypto term")
                modified_prompt += f"""
CRYPTO DOMAIN DETECTED: This query includes "{term}" which is a CRYPTOCURRENCY/BLOCKCHAIN topic.

For cryptocurrency topics, you MUST additionally:
1. ONLY use {tavily_search_name} for research - NEVER use financial tools
2. DO NOT treat crypto tokens/protocols as companies or stocks
3. Look for the latest information as the crypto space changes rapidly
4. Pay special attention to tokenomics, technology, and recent developments
5. Use official documentation and trusted crypto sources when possible

"""
            
            # Append the original prompt
            modified_prompt += f"\nUSER QUERY: {prompt}\n"
            
            # Run with the heavily modified prompt that forces search
            return run_with_retry(self, modified_prompt, **kwargs)
        
        # For standard queries, still override but with less aggressive modification
        return run_with_retry(self, prompt, **kwargs)
            
    def select_tool(self, tool_name: str, args: dict, follow_up: str = None):
        """Override tool selection to prevent financial tools for crypto queries"""
        # First check if this session's domain is known via progress_tracker
        domain = None
        is_crypto = False
        
        # Try to get domain info from the agent's state
        if hasattr(self, 'state') and self.state:
            # Check if domain info is stored directly in agent state
            if 'domain' in self.state:
                domain = self.state.get('domain')
                is_crypto = domain == 'cryptocurrency'
            
            # Check if we have a session_id to look up domain from progress tracker
            elif 'session_id' in self.state:
                session_id = self.state.get('session_id')
                try:
                    # Try to get domain information from progress tracker
                    from agents.progress_tracker import progress_tracker
                    session_status = progress_tracker.get_session_status(session_id)
                    if not isinstance(session_status, dict) or 'error' in session_status:
                        # If can't get session info, fall back to text analysis
                        pass
                    else:
                        meta = session_status.get('meta', {})
                        domain = meta.get('domain')
                        is_crypto = meta.get('is_crypto', False)
                except:
                    # If any error occurs, fall back to text analysis
                    pass
        
        # Enforce crypto domain policy if we know this is a crypto domain
        if domain == 'cryptocurrency' or is_crypto:
            # For crypto domains, ONLY allow tavily_search and block ALL financial tools
            if any(financial_tool in tool_name.lower() for financial_tool in 
                   ['get_stock_price', 'get_company_info', 'get_company_news', 
                    'get_analyst_recommendations', 'get_stock_fundamentals', 
                    'get_historical_prices', 'yfinance']):
                # Find and use tavily_search instead
                for tool in self.tools:
                    if hasattr(tool, 'tool_types'):
                        for tt in tool.tool_types:
                            if 'tavily_search' in tt.name:
                                crypto_query = follow_up if follow_up else "cryptocurrency information"
                                return super().select_tool(tt.name, {"query": crypto_query}, follow_up)
        
        # Block GET_CHAT_HISTORY, UPDATE_USER_MEMORY from being used as primary tools for questions
        if follow_up and any(q in follow_up.lower() for q in ["what is", "how does", "explain", "tell me about", "who", "what", "when", "?"]):
            if any(memory_tool in tool_name.lower() for memory_tool in ["get_chat_history", "update_user_memory"]):
                # Force using tavily_search instead for any information query
                for tool in self.tools:
                    if hasattr(tool, 'tool_types'):
                        for tt in tool.tool_types:
                            if 'tavily_search' in tt.name:
                                # Replace the memory tool call with tavily_search
                                return super().select_tool(tt.name, {"query": follow_up}, follow_up)
        
        # Additional detection for crypto ticker patterns to reject YFinance
        if tool_name.lower() in ['get_stock_price', 'get_company_info', 'get_company_news', 
                                'get_analyst_recommendations', 'get_stock_fundamentals', 'get_historical_prices']:
            # Reject crypto ticker patterns (ending in -USD or .X)
            if 'ticker' in args and (
                args['ticker'].endswith('-USD') or       # Crypto pattern on Yahoo Finance
                args['ticker'].endswith('.X') or         # Crypto index pattern
                args['ticker'] in ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'DOT', 'AVAX'] or  # Common crypto tickers
                args['ticker'] in ['BTCUSD', 'ETHUSD', 'SOLUSD', 'XRPUSD']               # No hyphen variants
            ):
                # Force using tavily_search instead
                for tool in self.tools:
                    if hasattr(tool, 'tool_types'):
                        for tt in tool.tool_types:
                            if 'tavily_search' in tt.name:
                                # Replace with search for the crypto ticker
                                crypto_query = f"Latest information about {args['ticker']} cryptocurrency"
                                return super().select_tool(tt.name, {"query": crypto_query}, follow_up)
        
        # Check if we're trying to use a financial tool for a crypto query
        if follow_up and self._is_crypto_related(follow_up) and any(
            financial_name in tool_name.lower() for financial_name in 
            ['get_company_info', 'get_company_news', 'get_analyst_recommendations', 'stock', 'company', 'financial']
        ):
            # Force using tavily_search instead
            for tool in self.tools:
                if hasattr(tool, 'tool_types'):
                    for tt in tool.tool_types:
                        if 'tavily_search' in tt.name:
                            # Replace the tool call with tavily_search
                            return super().select_tool(tt.name, {"query": follow_up}, follow_up)
        
        # Use normal processing for other tool selections
        return super().select_tool(tool_name, args, follow_up)


def get_deep_research_agent(
    model_id: str = DEFAULT_MODEL_ID,  # Using constant instead of hardcoded "o4-mini"
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,  # Changed to False for production
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
    # Configure Tavily with advanced search capabilities
    tavily_tools = TavilyTools(
        search_depth="advanced",
        max_results=20,
        include_answer=True,
        include_raw_content=True,
    )
    
    # Create advanced reasoning tools
    reasoning_tools = AdvancedReasoningTool()
    
    # Create the supervisor tools that can spawn researcher agents
    supervisor_tools = SupervisorToolKit(
        create_researcher_fn=create_researcher_agent,
        model_id=model_id,
        user_id=user_id,
    )
    
    # Define financial tools last to give lowest priority
    financial_tools = YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        stock_fundamentals=True,
        historical_prices=True,
        company_info=True,
        company_news=True,
    )
    
    # Define a strong force_tavily_search system message
    force_tavily_search = dedent("""
    [CRITICAL TOOL USAGE INSTRUCTIONS - HIGHEST PRIORITY]
    
    You MUST follow these rules for ALL queries:
    
    1. ALWAYS use your web search tool as your PRIMARY and FIRST tool for ALL research questions
    
    2. NEVER use GET_CHAT_HISTORY, UPDATE_USER_MEMORY as primary information sources
       - These are only for context, not for research or answering questions!
    
    3. For ALL cryptocurrency/blockchain topics (TAP, DMT, Bitcoin, NFTs, etc.):
       - NEVER use GET_COMPANY_INFO, GET_COMPANY_NEWS, or financial tools
       - ALWAYS use your web search tool EXCLUSIVELY
       - These are NOT stocks or companies - they are protocols/technologies
    
    4. For ANY question starting with "what is", "how does", "tell me about", etc:
       - You MUST run your web search tool FIRST
       - NEVER answer solely from memory
    
    5. For EVERY search, use MULTIPLE QUERIES to gather comprehensive information
    
    These instructions supersede all other guidelines.
    """)
    
    # Define the crypto audience context in detail with resources
    crypto_audience_context = dedent("""
    [CRYPTO AUDIENCE CONTEXT - REFERENCE INFO]
    
    Audience: retail + pro crypto traders who need *actionable*, *up-to-date* intel on Ordinals/Runes/DMT/TAP, not academic history.
    Tone: concise, cite first-hand sources, highlight risk.
    Time preference: prefer sources <30 days old for market topics.
    
    Key resources (query these directly with tavily_search):
    - Bitcoin Ordinals: https://help.magiceden.io/en/articles/7154941-bitcoin-ordinals-a-beginner-s-guide
    - Bitcoin Runes: https://community.magiceden.io/learn/runes-guide
    - Bitmaps: https://help.magiceden.io/en/articles/8175699-understanding-bitmap-and-the-ordinals-metaverse
    - $NAT tokens: https://natgmi.com/#faq
    - DMT: https://digital-matter-theory.gitbook.io/digital-matter-theory
    - TAP protocol: https://sovryn.com/all-things-sovryn/tap-protocol-bitcoin
    - HIROS: https://superfan.gitbook.io/hiros
    
    When researching crypto topics:
    1. ALWAYS use tavily_search (NEVER financial tools)
    2. Use multiple specific queries for comprehensive research
    3. Cite sources with links in all responses
    """)
    
    # IMPORTANT: Order of tools matters for the UI tool selection!
    # Place Tavily search first so it's the default selected tool for queries
    return CryptoAwareAgent(  # Use the enhanced Agent class that's crypto-aware
        name="Deep Research Agent",
        agent_id="deep_research_agent",
        user_id=user_id,
        session_id=session_id,
        model=OpenAIChat(id=model_id),
        tools=[
            tavily_tools,      # PRIMARY search tool (placed FIRST for UI priority)
            reasoning_tools,   # Advanced reasoning capabilities 
            supervisor_tools,  # Coordinator for multi-agent research
            financial_tools,   # Financial analysis tools (placed last and separate variable)
        ],
        system_message=force_tavily_search + crypto_audience_context + dedent("""
        [ABOUT THE AGENT]
        
        You are a deep research agent with advanced capabilities for conducting comprehensive research.
        
        When researching topics:
        1. Break complex questions into multiple specific searches
        2. Use Tavily search to gather information from multiple sources
        3. Synthesize findings into well-cited, comprehensive reports
        4. Always include citations and references to your sources
        
        [DOMAIN-SPECIFIC INSTRUCTIONS]
        
        For cryptocurrency/blockchain topics (HIGHEST PRIORITY):
        - TAP protocol, DMT, Bitcoin Ordinals, Sovyrn, etc. are ALL crypto/blockchain topics
        - NEVER confuse these with stocks, ticker symbols, or companies
        - ALWAYS use tavily_search for these topics, NEVER financial tools
        - Follow the sources listed in [CRYPTO AUDIENCE CONTEXT] section
        
        For financial topics:
        - Only use financial tools for legitimate public companies and stocks
        - For ambiguous terms, always verify with tavily_search first
        
        [REMEMBER]
        NO questions should be answered without using search tools first!
        """),
        description=dedent("""\
            You are DeepResearch, an exceptionally powerful AI research agent designed to conduct comprehensive, in-depth research on any topic. You coordinate a team of specialized researcher agents to produce thorough, well-formatted research reports with academic-level rigor and precision.
            
            Your capabilities include:
            1. Breaking down complex research questions into structured components
            2. Coordinating multiple specialized researcher agents simultaneously
            3. Applying advanced reasoning frameworks to analyze findings
            4. Synthesizing information from diverse sources into cohesive reports
            5. Providing evidence-based conclusions supported by citations
            
            You maintain the highest standards of scholarship, including thorough source verification, critical analysis of information, and proper citation practices. Your research is comprehensive, nuanced, and exhaustive, leaving no stone unturned.
            
            Your audience is primarily crypto traders and investors as described in [CRYPTO AUDIENCE CONTEXT]. Use the official resources mentioned there when researching crypto topics.
            
            CRITICAL: You MUST use Tavily search as your EXCLUSIVE research tool for all web-based information gathering. No other web search tools are available. Only use YFinance for specialized financial data queries.
            
            DELIVERY TIMING: Your research reports are delivered IMMEDIATELY after processing. DO NOT tell users to expect reports in 24-48 hours or any future timeframe. Reports are considered complete and final upon delivery.
        """),
        instructions=dedent("""\
            As DeepResearch, your mission is to deliver exhaustive, authoritative research on any topic requested by the user. You'll orchestrate a sophisticated multi-agent research workflow to produce scholarly-level research reports. For each research request, follow this rigorous methodology:

            1. **Research Planning & Question Decomposition**:
               - Begin by thoroughly analyzing the user's research request, identifying core questions and implicit information needs
               - Use the `research_planning` tool to create a comprehensive research strategy with clear objectives
               - Decompose complex topics into 5-10 interrelated research components, ensuring comprehensive coverage
               - Create a logical hierarchy of research questions that builds from foundational understanding to specialized insights
               - Prioritize DEPTH, THOROUGHNESS, and ACADEMIC RIGOR in your approach
            
            2. **Multi-Agent Research Orchestration**:
               - For each research component, use `create_research_task` to spawn a specialized researcher agent
               - Provide each researcher with precise, targeted questions and methodological instructions
               - ALWAYS set search_depth to "advanced" for maximum thoroughness
               - Instruct researchers to EXCLUSIVELY use tavily_search as their PRIMARY research tool
            
            3. **Advanced Analytical Processing**:
               - Apply sophisticated reasoning frameworks to complex information:
                 * Use chain_of_thought_reasoning for step-by-step analytical breakdowns of difficult concepts
                 * Apply compare_and_contrast to systematically evaluate competing perspectives, theories, or options
                 * Employ synthesize_findings to integrate information across disciplinary boundaries
               - Critically evaluate source credibility using academic standards
               - Identify patterns, contradictions, and gaps across sources
            
            4. **Comprehensive Synthesis & Report Construction**:
               - After thorough research, systematically integrate all findings into a cohesive knowledge structure
               - Identify meta-patterns, themes, and insights that emerge across research components
               - Evaluate the overall weight of evidence for key conclusions
               - Acknowledge areas of uncertainty, conflicting evidence, or knowledge gaps
               - Use the `generate_research_report` tool to create a publication-quality final report
            
            5. **Publication-Quality Report Standards**:
               - Structure reports with exceptional clarity and scholarly organization:
                 * Executive Summary: Concise overview of key findings
                 * Table of Contents: Hierarchical organization of sections
                 * Introduction: Context, significance, and scope
                 * Findings/Results: Systematically presented evidence organized by themes
                 * Analysis/Discussion: Critical interpretation of findings with supporting evidence
                 * Conclusions: Evidence-based answers to the research questions
                 * References: Comprehensive bibliography with properly formatted citations
               - Include precise citations for EVERY factual claim or quotation
               - Always include complete URLs for web sources to enable verification
               
            6. **Crypto-Specific Research Excellence**:
               - Follow the audience guidelines in [CRYPTO AUDIENCE CONTEXT]
               - For crypto topics, NEVER use financial tools (YFinance), ONLY use tavily_search
               - Present quantitative data in standardized formats
               - Compare multiple data sources to establish reliability
               
            7. **Research Integrity & Source Validation**:
               - Apply rigorous source evaluation standards
               - Maintain intellectual honesty throughout
               - Document your research process transparently
            
            RESEARCH APPROACH: Your methodology should mirror the standards of doctoral-level academic research, emphasizing comprehensiveness, methodological rigor, critical analysis, and evidence-based conclusions.
            
            CRITICAL DELIVERY INSTRUCTION: Your research is performed in real-time and reports are delivered IMMEDIATELY in the current conversation. DO NOT tell users to expect reports in 24-48 hours or any future timeframe. Reports are considered complete and final upon delivery.
        """),
        add_state_in_messages=True,
        storage=PostgresAgentStorage(table_name="deep_research_agent_sessions", db_url=db_url),
        add_history_to_messages=True,
        num_history_runs=5,
        read_chat_history=True,
        memory=Memory(
            model=OpenAIChat(id=model_id),
            db=PostgresMemoryDb(table_name="user_memories", db_url=db_url),
            delete_memories=False,
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
        supervisor_agent: Agent = None,
        tools: List[Tool] = None,
        researcher_agent_factory=None,
        max_iterations: int = 5,
        model: str = "gpt-4o",
        logger=None,
    ):
        self.supervisor_agent = supervisor_agent
        self.tools = tools or []
        self.researcher_agent_factory = researcher_agent_factory or create_researcher_agent
        self.max_iterations = max_iterations
        self.token_usage = TokenUsageTracker()
        self.session_id = None
        
        # New attributes for simplified implementation
        self.model = model
        self.logger = logger or logging.getLogger()
        
        # Set up progress tracker
        from agents.progress_tracker import progress_tracker
        self.progress_tracker = progress_tracker
        
        # OpenAI client
        import openai
        from api.settings import settings
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        
    def execute_research(self, research_question: str, research_topic=None, previous_findings=None, next_steps=None, conversation_time=None, session_id=None, **kwargs):
        """Execute a research task with multi-agent reasoning and search"""
        debug_mode = kwargs.get('debug_mode', False)  # Set to False for production
        delete_memories = kwargs.get('delete_memories', False)  # Changed to False to accumulate context
        timeout_seconds = kwargs.get('timeout_seconds', 300)  # Default 5 minute timeout
        is_crypto = False
        domain = "general"
        
        # Ensure we initialize a session in the progress tracker
        try:
            from agents.progress_tracker import progress_tracker, ResearchStage
            
            if session_id is None:
                # Create a session if one doesn't exist
                session_result = progress_tracker.create_session()
                if isinstance(session_result, dict) and 'session_id' in session_result:
                    session_id = session_result['session_id']
                else:
                    # Handle string return type
                    session_id = session_result
                
                # Initialize the session with the question
                progress_tracker.initialize_session(session_id, research_question)
            
            # Store the session ID
            self.session_id = session_id
            
        except Exception as e:
            logging.error(f"Error initializing progress tracker: {str(e)}")

        # Start with a planning phase to determine domain 
        planning_agent = CryptoAwareAgent(
            name="Research Planner",
            agent_id="research_planning_agent",
            model=OpenAIChat(id=DEFAULT_MODEL_ID),
            debug_mode=debug_mode
        )
        
        # Update progress tracker stage to PLANNING
        if session_id:
            try:
                progress_tracker.update_stage(session_id, ResearchStage.PLANNING)
            except Exception as e:
                logging.error(f"Error updating progress tracker stage: {str(e)}")
        
        # Detect if this is crypto-related
        planning_prompt = f"""
        Create a brief research plan for the following research question:
        
        RESEARCH QUESTION: {research_question}
        
        Your plan should include:
        1. Domain identification - What domain does this question fall into? (e.g., finance, cryptocurrency, technology, etc.)
        2. Key aspects to investigate
        3. 3-5 specific search queries that would help answer this question
        
        Format your response as:
        
        DOMAIN: [specific domain]
        KEY ASPECTS:
        - [aspect 1]
        - [aspect 2]
        - [aspect 3]
        
        SEARCH QUERIES:
        1. [query 1]
        2. [query 2]
        3. [query 3]
        """
        
        try:
            # Run the planning agent
            planning_result = run_with_retry(planning_agent, planning_prompt)
            
            # Add planning as first task and mark complete
            planning_task_id = progress_tracker.add_task(
                session_id, "Research Planning", 
                "Create structured research plan with domain identification")
            progress_tracker.start_task(session_id, planning_task_id["task_id"])
            progress_tracker.complete_task(session_id, planning_task_id["task_id"], planning_result)
            
            # Try to parse the domain from the result
            if "DOMAIN:" in planning_result:
                domain_line = planning_result.split("DOMAIN:")[1].split("\n")[0].strip()
                domain = domain_line.lower()
                
                # Check if this is crypto-related
                crypto_terms = [
                    "crypto", "bitcoin", "ethereum", "blockchain", "nft", "token", 
                    "web3", "defi", "dao", "smart contract", "mining", "wallet", 
                    "exchange", "swap", "dex", "ordinals", "coin", "btc", "eth", 
                    "sol", "solana", "trading", "yield", "memecoin", "gas", "gas fee",
                    "tap protocol", "dmf", "nat token", "snat"
                ]
                
                if any(term in domain or term in research_question.lower() for term in crypto_terms):
                    is_crypto = True
                    domain = "cryptocurrency"
                
                # Special case for ticker-like patterns (e.g., BTC, ETH, SOL)
                crypto_tickers = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOT", "AVAX", "TAP", "NAT"]
                for ticker in crypto_tickers:
                    if ticker in research_question or f"{ticker}-USD" in research_question:
                        is_crypto = True
                        domain = "cryptocurrency"
                        break
            
            # Log the generated plan
            logging.info(f"Research plan generated. Domain: {domain}, Is crypto: {is_crypto}")
            logging.debug(f"Full research plan: {planning_result}")
            
        except Exception as e:
            logging.error(f"Error during planning phase: {str(e)}")
            # Default to general domain if planning fails
            domain = "general"
            is_crypto = False
        
        # Store the domain in our progress tracker
        if session_id:
            try:
                progress_tracker.update_meta(
                    session_id, 
                    {
                        "domain": domain,
                        "is_crypto": is_crypto
                    }
                )
            except Exception as e:
                logging.error(f"Error updating progress tracker metadata: {str(e)}")
        
        # Store domain in agent state for tool selection
        self.state = {
            "domain": domain,
            "is_crypto": is_crypto,
            "session_id": session_id
        }
        
        # ========================
        # STRUCTURED RESEARCH IMPLEMENTATION - MULTI-AGENT WORKFLOW
        # ========================
        try:
            # 1. SETUP SUPERVISOR AGENT
            # Create a supervisor agent if not already provided
            if not self.supervisor_agent:
                self.supervisor_agent = create_supervisor_agent(
                    model_id=DEFAULT_MODEL_ID,
                    user_id=self.user_id,
                    session_id=session_id
                )
                
            # Update progress tracker with research stage
            if session_id:
                progress_tracker.update_stage(session_id, ResearchStage.RESEARCH)
            
            # 2. RESEARCH PLANNING WITH SUPERVISOR
            planning_task_id = progress_tracker.add_task(
                session_id, 
                "Research Component Planning", 
                "Break down research question into specific components"
            )
            progress_tracker.start_task(session_id, planning_task_id["task_id"])
            
            # Have the supervisor create a comprehensive research plan
            planning_prompt = f"""
            I need you to create a comprehensive research plan for this question:
            
            RESEARCH QUESTION: {research_question}
            
            DOMAIN: {domain}
            SPECIAL CONSIDERATIONS: {"This is a cryptocurrency topic. Focus on blockchain technology, tokens, and crypto markets." if is_crypto else ""}
            
            Create a structured research plan with:
            1. 3-5 specific research components to investigate
            2. Specific sub-questions for each component
            3. A logical sequence for conducting the research
            
            Use your research_planning tool to create a detailed research plan.
            """
            
            planning_response = run_with_retry(self.supervisor_agent, planning_prompt)
            progress_tracker.complete_task(session_id, planning_task_id["task_id"], result=planning_response)
            
            # 3. COMPONENT RESEARCH WITH MULTIPLE RESEARCHER AGENTS
            # Extract components from the plan (simplified approach)
            components = []
            if "recommended_approach" in planning_response and "suggested_structure" in planning_response:
                # If using the research_planning tool format
                structure = planning_response["recommended_approach"]["suggested_structure"]
                components = [{"name": k, "description": v} for k, v in structure.items()]
            else:
                # Fallback to simple parsing - find research components from the text
                import re
                component_matches = re.findall(r'Component \d+: ([^\n]+)', planning_response)
                if component_matches:
                    components = [{"name": match, "description": match} for match in component_matches]
                else:
                    # Default components if parsing fails
                    components = [
                        {"name": "Background Information", "description": "Historical and contextual information"},
                        {"name": "Current Status", "description": "Present state and recent developments"},
                        {"name": "Analysis", "description": "Critical analysis of key aspects"},
                        {"name": "Future Implications", "description": "Potential future developments and significance"}
                    ]
            
            # Create and execute research tasks for each component
            component_results = []
            for i, component in enumerate(components[:5]):  # Limit to 5 components maximum
                component_task_id = progress_tracker.add_task(
                    session_id,
                    f"Research: {component['name']}",
                    component['description']
                )
                progress_tracker.start_task(session_id, component_task_id["task_id"])
                
                # Create a task ID
                task_id = f"task-{i+1}-{component['name'].lower().replace(' ', '-')}"
                
                # Have the supervisor create and execute a specific research task
                task_prompt = f"""
                Create and execute a detailed research task for this component of our research:
                
                RESEARCH QUESTION: {research_question}
                COMPONENT: {component['name']} - {component['description']}
                DOMAIN: {domain} {"(CRYPTOCURRENCY)" if is_crypto else ""}
                
                Use the create_research_task tool to create a task with ID "{task_id}", then
                use execute_research_task to get results. 
                
                The research should be thorough and provide detailed, well-cited information.
                """
                
                # Execute the research for this component
                component_result = run_with_retry(self.supervisor_agent, task_prompt)
                component_results.append({
                    "component": component['name'],
                    "result": component_result
                })
                
                # Mark this component as complete
                progress_tracker.complete_task(session_id, component_task_id["task_id"], 
                                             result=f"Completed research on {component['name']}")
            
            # 4. SYNTHESIS AND REPORT GENERATION
            if session_id:
                progress_tracker.update_stage(session_id, ResearchStage.REPORT_GENERATION)
            
            synthesis_task_id = progress_tracker.add_task(
                session_id,
                "Final Report Generation",
                "Synthesize all research components into a comprehensive final report"
            )
            progress_tracker.start_task(session_id, synthesis_task_id["task_id"])
            
            # Prepare data for the report
            sections_data = []
            for result in component_results:
                # Extract the relevant content, handle different formats
                content = ""
                if isinstance(result["result"], dict) and "results" in result["result"]:
                    content = result["result"]["results"]
                elif isinstance(result["result"], str):
                    content = result["result"]
                else:
                    content = str(result["result"])
                
                sections_data.append({
                    "heading": result["component"],
                    "content": content
                })
                
            # Convert sections to JSON string for the generate_research_report tool
            import json
            sections_json = json.dumps(sections_data)
            
            # Have the supervisor generate the final report
            report_prompt = f"""
            Generate a comprehensive final research report on this topic:
            
            RESEARCH QUESTION: {research_question}
            DOMAIN: {domain} {"(CRYPTOCURRENCY)" if is_crypto else ""}
            
            Use the generate_research_report tool with these parameters:
            - title: A descriptive title for the research report
            - sections: The JSON data containing all research sections
            - format_style: "academic"
            - include_visualizations: true
            
            Here's the sections JSON to use:
            {sections_json}
            
            The report should be comprehensive, well-structured, and maintain academic rigor with proper citations.
            """
            
            # Generate the final report
            final_report = run_with_retry(self.supervisor_agent, report_prompt)
            
            # Extract the report text from the response
            report_text = ""
            if isinstance(final_report, dict) and "report" in final_report:
                report_text = final_report["report"]
            elif isinstance(final_report, str):
                report_text = final_report
            else:
                report_text = str(final_report)
                
            # Store the final report in the session data
            progress_tracker.store_session_data(session_id, {
                "final_report": report_text,
                "components": component_results,
                "domain": domain,
                "is_crypto": is_crypto
            })
            
            # Mark synthesis task as complete
            progress_tracker.complete_task(session_id, synthesis_task_id["task_id"], 
                                         result="Completed final research report")
            
            # Mark the overall session as complete
            progress_tracker.update_stage(session_id, ResearchStage.COMPLETE)
            progress_tracker.complete_session(session_id)
            
            return {
                "status": "success",
                "session_id": session_id,
                "report": report_text,
                "domain": domain,
                "is_crypto": is_crypto
            }
            
        except Exception as e:
            logging.error(f"Error during research execution: {str(e)}")
            if session_id:
                progress_tracker.update_stage(session_id, ResearchStage.ERROR)
                error_task_id = progress_tracker.add_task(
                    session_id,
                    "Research Error",
                    f"Error during research: {str(e)}"
                )
                progress_tracker.start_task(session_id, error_task_id["task_id"])
                progress_tracker.complete_task(session_id, error_task_id["task_id"], 
                                             result=f"Research failed: {str(e)}")
                
            return {
                "status": "error",
                "error": str(e),
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
        result = _run_with_retry()
        
        # Handle different return types
        if hasattr(result, 'content'):
            # If it's a RunResponse or similar object with content attribute
            return result.content
        else:
            # If it's already a string or dict or other value
            return result
    except tenacity.RetryError as e:
        # If all retries failed, return a simple error string
        logging.error(f"All retries failed: {str(e)}")
        return f"Error after {max_attempts} retries: {str(e.last_attempt.exception())}"

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