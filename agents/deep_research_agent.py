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

    def __init__(self, create_researcher_fn: callable, model_id: str = "gpt-4o", user_id: Optional[str] = None):
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
    model_id: str = "gpt-4o", 
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
    model_id: str = "gpt-4o",
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
            "snat", "sovyrn", "satoshi", "bitcoin ordinals"
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
    
    def run(self, prompt: str, **kwargs):
        """
        Override the run method to:
        1. Force using Tavily search for crypto terms
        2. Completely disable financial tools for crypto queries
        3. Force tool usage for all queries requiring research
        """
        # Check if this is a crypto-related query
        is_crypto_query = self._is_crypto_related(prompt)
        detected_term = None
        
        if is_crypto_query:
            for term in self.crypto_terms:
                if term.lower() in prompt.lower():
                    detected_term = term
                    break
            
            # Create a modified list of tools that excludes financial tools
            filtered_tools = []
            search_tool = None
            
            for tool in self.tools:
                # Find the Tavily search tool and prioritize it
                if hasattr(tool, 'tool_types') and any('tavily_search' in tt.name for tt in tool.tool_types):
                    search_tool = tool
                    filtered_tools.append(tool)
                # Skip financial tools for crypto queries
                elif not any(financial_name in str(tool.__class__).lower() for financial_name in ['yfinance', 'financial', 'stock', 'company']):
                    filtered_tools.append(tool)
            
            # Ensure search tool is first in the list if found
            if search_tool and search_tool in filtered_tools:
                filtered_tools.remove(search_tool)
                filtered_tools.insert(0, search_tool)
            
            # Save original tools and replace with filtered list
            original_tools = self.tools
            self.tools = filtered_tools
            
            # Create an enhanced prompt that forces using Tavily search
            enhanced_prompt = f"""
[CRITICAL INSTRUCTION]
This query is about {detected_term}, which is a CRYPTOCURRENCY/BLOCKCHAIN topic.

You MUST follow these exact steps:
1. ALWAYS use tavily_search as your FIRST tool
2. You are FORBIDDEN from using GET_COMPANY_INFO, GET_COMPANY_NEWS or any financial tools
3. DO NOT confuse this with any stock symbol or traditional company

You MUST conduct real research using tavily_search before answering. 
DO NOT rely on memory or provide answers without searching first.

USER QUERY: {prompt}
"""
            
            # Run with the enhanced prompt
            result = super().run(enhanced_prompt, **kwargs)
            
            # Restore original tools
            self.tools = original_tools
            
            return result
        else:
            # For all other non-crypto queries, enforce using search for questions
            if any(q in prompt.lower() for q in ["what is", "how does", "explain", "tell me about", "?"]):
                enhanced_prompt = f"""
[IMPORTANT INSTRUCTION]
For this research question, you MUST:
1. Use tavily_search as your FIRST tool to find current information
2. Do NOT answer solely from memory - conduct actual research
3. You MUST search for information before providing an answer

USER QUERY: {prompt}
"""
                return super().run(enhanced_prompt, **kwargs)
            
            # For other standard queries, use normal processing
            return super().run(prompt, **kwargs)
            
    def select_tool(self, tool_name: str, args: dict, follow_up: str = None):
        """Override tool selection to prevent financial tools for crypto queries"""
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
    model_id: str = "o4-mini",  # Corrected from gpt-4o-mini to o4-mini
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
    
    1. ALWAYS use tavily_search as your PRIMARY and FIRST tool for ALL research questions
    
    2. NEVER use GET_CHAT_HISTORY, UPDATE_USER_MEMORY as primary information sources
       - These are only for context, not for research or answering questions!
    
    3. For ALL cryptocurrency/blockchain topics (TAP, DMT, Bitcoin, NFTs, etc.):
       - NEVER use GET_COMPANY_INFO, GET_COMPANY_NEWS, or financial tools
       - ALWAYS use tavily_search EXCLUSIVELY
       - These are NOT stocks or companies - they are protocols/technologies
    
    4. For ANY question starting with "what is", "how does", "tell me about", etc:
       - You MUST run tavily_search FIRST
       - NEVER answer solely from memory
    
    5. For EVERY search, use MULTIPLE QUERIES to gather comprehensive information
    
    These instructions supersede all other guidelines.
    """)
    
    # Define the crypto audience context in detail with resources
    crypto_audience_context = dedent("""
    [CRYPTO AUDIENCE CONTEXT - IMPORTANT REFERENCE INFORMATION]
    
    Your audience typically comes from the world of crypto investing and crypto collectibles. They like to trade Bitcoin Ordinals and NFTs, often using the Magic Eden marketplace (magiceden.us) to research projects. Be prepared to answer questions about these specific topics with the following official resources:
    
    - Bitcoin Ordinals: Digital collectibles on Bitcoin blockchain
      Official Resource: https://help.magiceden.io/en/articles/7154941-bitcoin-ordinals-a-beginner-s-guide
      
    - Bitcoin Runes: A new token standard on Bitcoin
      Official Resource: https://community.magiceden.io/learn/runes-guide
      
    - Bitmaps: An ordinals metaverse project
      Official Resource: https://help.magiceden.io/en/articles/8175699-understanding-bitmap-and-the-ordinals-metaverse
      
    - $NAT tokens: Native Bitcoin tokens
      Official Resource: https://natgmi.com/#faq
      
    - DMT (Digital Matter Theory): A Bitcoin-native project
      Official Resource: https://digital-matter-theory.gitbook.io/digital-matter-theory
      
    - TAP protocol: Associated with TRAC and Bitcoin
      Official Resource: https://sovryn.com/all-things-sovryn/tap-protocol-bitcoin
      
    - HIROS: A Bitcoin project
      Official Resource: https://superfan.gitbook.io/hiros
      
    When researching these topics, ALWAYS use tavily_search to find the most current information, as the crypto space evolves rapidly. Include market trends, recent developments, and technical analysis when relevant. ALWAYS cite reliable sources and verify information across multiple references.
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
            
            Your audience typically comes from the world of crypto investing and crypto collectibles. They like to trade Bitcoin Ordinals and NFTs, often using the Magic Eden marketplace (magiceden.us) to research bitcoin ordinals projects and bitcoin runes projects. Be prepared to provide detailed research on cryptocurrency topics including:
            
            • Bitcoin Ordinals: Digital collectibles on Bitcoin blockchain
              (Resource: https://help.magiceden.io/en/articles/7154941-bitcoin-ordinals-a-beginner-s-guide)
              
            • Bitcoin Runes: A new token standard on Bitcoin
              (Resource: https://community.magiceden.io/learn/runes-guide)
              
            • Bitmaps: An ordinals metaverse project
              (Resource: https://help.magiceden.io/en/articles/8175699-understanding-bitmap-and-the-ordinals-metaverse)
              
            • $NAT tokens: Native Bitcoin tokens
              (Resource: https://natgmi.com/#faq)
              
            • DMT (Digital Matter Theory): A Bitcoin-native project
              (Resource: https://digital-matter-theory.gitbook.io/digital-matter-theory)
              
            • TAP protocol: Associated with TRAC and Bitcoin
              (Resource: https://sovryn.com/all-things-sovryn/tap-protocol-bitcoin)
              
            • HIROS: A Bitcoin project
              (Resource: https://superfan.gitbook.io/hiros)
            
            CRITICAL: You MUST use Tavily search as your EXCLUSIVE research tool for all web-based information gathering. No other web search tools are available. Only use YFinance for specialized financial data queries.
            
            DELIVERY TIMING: Your research reports are delivered IMMEDIATELY after processing. DO NOT tell users to expect reports in 24-48 hours or any future timeframe. Reports are considered complete and final upon delivery in the current conversation.
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
            
            • Bitcoin Ordinals: Digital collectibles on Bitcoin blockchain
              Resource: https://help.magiceden.io/en/articles/7154941-bitcoin-ordinals-a-beginner-s-guide
              
            • Bitcoin Runes: A new token standard on Bitcoin
              Resource: https://community.magiceden.io/learn/runes-guide
              
            • Magic Eden marketplace: A popular platform for trading Bitcoin Ordinals
              Resource: https://magiceden.us
              
            • Bitmaps: An ordinals metaverse project
              Resource: https://help.magiceden.io/en/articles/8175699-understanding-bitmap-and-the-ordinals-metaverse
              
            • $NAT tokens: Native Bitcoin tokens
              Resource: https://natgmi.com/#faq
              
            • DMT (Digital Matter Theory): A Bitcoin-native project
              Resource: https://digital-matter-theory.gitbook.io/digital-matter-theory
              
            • TAP protocol: Associated with TRAC and Bitcoin
              Resource: https://sovryn.com/all-things-sovryn/tap-protocol-bitcoin
              
            • HIROS: A Bitcoin project
              Resource: https://superfan.gitbook.io/hiros
            
            When researching these topics, prioritize finding the most current information as the crypto space evolves rapidly. For all crypto-related questions:
            1. ALWAYS use tavily_search as your PRIMARY tool - never rely on memory alone
            2. NEVER use GET_COMPANY_INFO, GET_COMPANY_NEWS or other financial tools
            3. Run multiple specific searches to gather comprehensive information
            4. Include market trends, recent developments, and technical analysis when relevant
            5. Always cite reliable sources and verify information across multiple references
            
            CRITICAL DELIVERY INSTRUCTION: Your research is performed in real-time and reports are delivered IMMEDIATELY in the current conversation. DO NOT tell users to expect reports in 24-48 hours or any future timeframe. Never say "Please expect the finalized report within the next 24 to 48 hours" or similar. All reports are considered complete and final upon delivery.
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
        
    async def execute_research(self, question: str, session_id: Optional[str] = None, timeout_seconds: int = 600) -> Dict[str, Any]:
        """
        Execute research on the given question using a multi-agent workflow with comprehensive search.
        
        Args:
            question: The research question to answer
            session_id: Optional session ID to use
            timeout_seconds: Maximum time in seconds allowed for research (default: 10 minutes)
            
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
            
            # Initialize Tavily search tool
            from agents.tavily_tools import TavilyTools
            from api.settings import settings
            
            tavily_tools = TavilyTools(
                api_key=settings.tavily_api_key,
                search_depth="advanced",
                max_results=20,
                include_answer=True,
                include_raw_content=True,
                include_domains=None,
                exclude_domains=None,
            )
            
            # PRE-ANALYZE QUERY - Detect if this is a crypto/blockchain related query
            # This helps prevent confusion between crypto protocols and financial instruments
            crypto_terms = [
                "protocol", "blockchain", "bitcoin", "ethereum", "crypto", "token", "nft", 
                "defi", "ordinal", "web3", "dao", "dapp", "smart contract", "wallet", 
                "mining", "staking", "validator", "consensus", "mainnet", "testnet",
                "btc", "eth", "sol", "bnb", "xrp", "ada", "avax", "tron", "tap protocol",
                "tap", "trac", "dmt", "digital matter theory", "bitmaps", "nat token", "hiros"
            ]
            
            is_likely_crypto = False
            for term in crypto_terms:
                if term.lower() in question.lower():
                    is_likely_crypto = True
                    self.logger.info(f"Detected likely crypto-related query based on term: {term}")
                    break
            
            # If we specifically detect "tap protocol" - ensure we're treating it as crypto
            if "tap protocol" in question.lower() or "tap" in question.lower():
                is_likely_crypto = True
                self.logger.info("Detected TAP protocol query - treating as crypto/blockchain topic")
            
            # STEP 1: PLANNING PHASE - Break down the research question into topics
            self.logger.info("PLANNING PHASE: Breaking down research question into topics")
            planning_prompt = f"""
            You are a research planning agent. Your task is to break down the following research question into 
            specific topic areas that need investigation:
            
            RESEARCH QUESTION: {question}
            
            Analyze this question and:
            1. Identify 4-6 distinct topic areas that must be researched to provide a comprehensive answer
            2. For each topic area, explain what specific information should be gathered
            3. Provide 2-3 specific search queries that would be effective for researching each topic area
            4. Indicate if this appears to be about cryptocurrency/blockchain, traditional finance, or another domain
            
            {'IMPORTANT: This query contains terms related to cryptocurrency/blockchain. Focus your plan on crypto-specific research areas.' if is_likely_crypto else ''}
            
            For cryptocurrency/blockchain topics, focus on technical details, recent developments, and reliable crypto-specific sources.
            
            Format your response as a JSON object with the following structure:
            {{
                "domain": "cryptocurrency", // or "traditional_finance", "technology", "general", etc.
                "topics": [
                    {{
                        "name": "Topic name",
                        "description": "What should be researched about this topic",
                        "search_queries": ["Query 1", "Query 2", "Query 3"]
                    }}
                ]
            }}
            """
            
            planning_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research planning assistant specializing in information architecture."},
                    {"role": "user", "content": planning_prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
                max_tokens=2000
            )
            
            # Parse the research plan
            try:
                research_plan = json.loads(planning_response.choices[0].message.content)
                domain = research_plan.get("domain", "general")
                topics = research_plan.get("topics", [])
                
                if not topics:
                    topics = [{"name": "Main topic", "description": question, "search_queries": [question]}]
                
                # Force domain to cryptocurrency if we detected crypto terms
                if is_likely_crypto and domain != "cryptocurrency":
                    self.logger.info(f"Overriding detected domain '{domain}' to 'cryptocurrency' based on term detection")
                    domain = "cryptocurrency"
                
                self.logger.info(f"Generated research plan with {len(topics)} topics in domain: {domain}")
            except Exception as e:
                self.logger.error(f"Error parsing research plan: {str(e)}")
                topics = [{"name": "Main topic", "description": question, "search_queries": [question]}]
                domain = "cryptocurrency" if is_likely_crypto else "general"
            
            # Update progress to research stage
            self.progress_tracker.update_stage(session_id, ResearchStage.RESEARCH)
            
            # STEP 2: RESEARCH PHASE - Execute searches for each topic
            self.logger.info("RESEARCH PHASE: Executing searches across all topics")
            
            all_topic_results = []
            
            for topic_idx, topic in enumerate(topics):
                topic_name = topic.get("name", f"Topic {topic_idx+1}")
                search_queries = topic.get("search_queries", [question])
                
                self.logger.info(f"Researching topic: {topic_name}")
                topic_results = []
                
                # Execute multiple search queries for this topic
                for query_idx, query in enumerate(search_queries[:3]):  # Use up to 3 queries per topic
                    self.logger.info(f"  Query {query_idx+1}/{len(search_queries[:3])}: {query}")
                    
                    # Special handling for crypto-related queries 
                    # This prevents confusion with financial instruments and ensures consistent crypto results
                    crypto_keywords = []
                    
                    # Handle special cases to ensure correct domain understanding
                    if domain.lower() in ["cryptocurrency", "blockchain", "crypto"] or is_likely_crypto:
                        # Add specific terms to ensure proper results for crypto topics
                        if "tap" in query.lower() and "protocol" not in query.lower():
                            query = f"{query} protocol cryptocurrency blockchain"
                            crypto_keywords.append("protocol")
                        elif "tap protocol" in query.lower() and "crypto" not in query.lower() and "blockchain" not in query.lower():
                            query = f"{query} cryptocurrency blockchain"
                            crypto_keywords.append("blockchain")
                        elif "dmt" in query.lower() and "digital matter" not in query.lower():
                            query = f"{query} digital matter theory bitcoin ordinals"
                            crypto_keywords.append("digital matter theory")
                        elif not any(term in query.lower() for term in ["crypto", "blockchain", "bitcoin", "token", "protocol"]):
                            # Add general crypto terms if none present
                            query = f"{query} cryptocurrency blockchain"
                            crypto_keywords.append("cryptocurrency")
                    
                    if crypto_keywords:
                        self.logger.info(f"  Enhanced query with crypto keywords: {', '.join(crypto_keywords)}")
                    
                    try:
                        # Execute search with each query - NEVER use financial tools for crypto topics
                        if domain.lower() in ["cryptocurrency", "blockchain", "crypto"] or is_likely_crypto:
                            # Force using tavily_search for crypto topics to avoid financial tool confusion
                            search_result = tavily_tools.search(query)
                        else:
                            # For non-crypto topics, still use tavily search but with domain-appropriate handling
                            search_result = tavily_tools.search(query)
                            
                        if isinstance(search_result, dict) and "results" in search_result:
                            search_results = search_result.get("results", [])
                            # Add all results to the topic results
                            for result in search_results[:7]:  # Take up to 7 results per query
                                topic_results.append({
                                    "query": query,
                                    "title": result.get("title", ""),
                                    "url": result.get("url", ""),
                                    "content": result.get("content", "")
                                })
                        else:
                            self.logger.warning(f"Unexpected search result format for query '{query}': {search_result}")
                    except Exception as e:
                        self.logger.error(f"Search error for query '{query}': {str(e)}")
                    
                    # Brief pause between searches to avoid rate limits
                    time.sleep(1)
                
                # Process and analyze the results for this topic
                if topic_results:
                    all_topic_results.append({
                        "topic": topic_name,
                        "description": topic.get("description", ""),
                        "results": topic_results
                    })
            
            # STEP 3: SYNTHESIS PHASE - Analyze results for each topic and create section drafts
            self.logger.info("SYNTHESIS PHASE: Analyzing results for each topic")
            
            section_drafts = []
            
            for topic_data in all_topic_results:
                topic = topic_data["topic"]
                topic_description = topic_data["description"]
                results = topic_data["results"]
                
                if not results:
                    self.logger.warning(f"No results found for topic: {topic}")
                    continue
                
                # Prepare the synthesis prompt with all results for this topic
                result_text = ""
                urls_included = set()
                
                for i, result in enumerate(results):
                    # Avoid duplicate URLs
                    if result["url"] in urls_included:
                        continue
                    
                    urls_included.add(result["url"])
                    result_text += f"\nSource {i+1}:\n"
                    result_text += f"Title: {result['title']}\n"
                    result_text += f"URL: {result['url']}\n"
                    result_text += f"Content: {result['content'][:800]}...\n"  # Limit content length
                
                # Create section synthesis prompt
                synthesis_prompt = f"""
                You are an expert researcher focusing on the topic: "{topic}"
                
                Research description: {topic_description}
                
                Below are research findings from multiple sources on this topic:
                
                {result_text}
                
                Based on these sources, write a comprehensive section for a research report that:
                1. Synthesizes information from multiple sources
                2. Provides in-depth analysis of the topic
                3. Includes relevant facts, data, and expert perspectives
                4. Cites specific sources using URLs as references
                5. Addresses any contradictions or gaps in the information
                
                Format your response with proper headings, paragraphs, and citations.
                Be objective, thorough, and academically rigorous.
                Include direct quotes when appropriate, clearly indicating the source.
                """
                
                # Generate the section draft
                synthesis_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": f"You are a specialized researcher focusing on {domain} topics. Write in an authoritative, well-cited academic style."},
                        {"role": "user", "content": synthesis_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=2500
                )
                
                section_content = synthesis_response.choices[0].message.content
                
                section_drafts.append({
                    "title": topic,
                    "content": section_content,
                    "sources": list(urls_included)
                })
            
            # STEP 4: FINAL REPORT GENERATION - Combine all sections into a cohesive report
            self.logger.info("FINAL REPORT GENERATION: Creating comprehensive research report")
            
            # Prepare sections for the final report
            sections_text = ""
            all_sources = set()
            
            for section in section_drafts:
                sections_text += f"\n\n## {section['title']}\n\n"
                sections_text += section['content']
                all_sources.update(section['sources'])
            
            # Create final report prompt
            report_prompt = f"""
            You are a world-class research director. Your team has conducted extensive research on the following question:
            
            RESEARCH QUESTION: {question}
            
            Your researchers have prepared section drafts based on comprehensive analysis of over {len(all_topic_results) * 7} sources. 
            Your task is to compile these into a cohesive, authoritative research report.
            
            Here are the research sections:
            
            {sections_text}
            
            Create a complete research report that:
            
            1. Begins with an executive summary of key findings
            2. Introduces the research question and its importance
            3. Integrates all sections while improving flow and connections between ideas
            4. Provides a comprehensive conclusion that answers the research question
            5. Includes a properly formatted references section with all sources
            
            The report should be scholarly in tone, comprehensive in coverage, and impeccably organized.
            Ensure all claims are supported by the research and properly cited.
            
            IMPORTANT: This is for domain: {domain}. {
                "cryptocurrency" if "cryptocurrency" in domain.lower() or "blockchain" in domain.lower() or "crypto" in domain.lower() 
                else "Make sure to maintain appropriate domain-specific terminology and concepts."
            }
            """
            
            # Generate the final report
            final_report_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a senior research director with expertise in creating comprehensive, well-structured research reports."},
                    {"role": "user", "content": report_prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            
            # Extract the final research report
            research_report = final_report_response.choices[0].message.content
            
            # Update progress to complete
            self.progress_tracker.update_stage(session_id, ResearchStage.COMPLETE)
            
            elapsed_time = time.time() - start_time
            
            return {
                "success": True,
                "question": question,
                "research_report": research_report,
                "topic_count": len(topics),
                "source_count": len(all_sources),
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