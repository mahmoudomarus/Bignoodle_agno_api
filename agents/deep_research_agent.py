from textwrap import dedent
from typing import Dict, List, Optional, Any
import json

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
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from agents.progress_tracker import ResearchProgressTracker


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


def create_researcher_agent(
    model_id: str = "gpt-4.1", 
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    research_question: str = "",
    additional_instructions: str = "",
    search_depth: str = "advanced",
) -> AgentExecutor:
    """
    Create a Researcher Agent specialized in focused investigations.
    
    Args:
        model_id: The ID of the model to use
        user_id: The ID of the user
        session_id: The ID of the session
        research_question: The specific research question to investigate
        additional_instructions: Any additional instructions for the agent
        search_depth: The depth of search to perform ("basic", "advanced", or "comprehensive")
        
    Returns:
        An agent executor configured for detailed research
    """
    # Initialize the LLM
    llm = ChatOpenAI(
        model=model_id,
        temperature=0,
    )
    
    # Configure tools based on search depth
    if search_depth == "basic":
        max_results = 10
    elif search_depth == "advanced":
        max_results = 15
    else:  # comprehensive
        max_results = 20
    
    # Initialize Tavily as the only search tool
    tavily_tools = TavilyTools(
        search_depth=search_depth,
        max_results=max_results
    )
    
    # Add advanced reasoning capabilities
    reasoning_tools = AdvancedReasoningTool()
    
    # Create the researcher system prompt
    researcher_prompt = dedent("""
    You are a specialized Researcher Agent within a multi-agent research system. Your task is to conduct thorough, focused investigation on specific aspects of a research question assigned by the Supervisor Agent.

    ## Research Methodology
    1. **Strategic Information Acquisition**:
       - Use Tavily as your EXCLUSIVE search tool for all web-based information gathering
       - NEVER use any other search tools like DuckDuckGo
       - Formulate precise search queries to maximize relevant results
       - Modify search parameters based on initial findings

    2. **Comprehensive Analysis**:
       - Apply critical thinking frameworks from your advanced reasoning toolkit
       - Evaluate source credibility, relevance, and potential bias
       - Identify patterns, contradictions, and gaps in available information
       - Consider multiple interpretations and perspectives on the data

    3. **Evidence Synthesis**:
       - Compile key findings with proper source attribution
       - Organize information in a coherent, logical structure
       - Distinguish between factual information and analytical conclusions
       - Acknowledge limitations and uncertainties in your findings

    4. **Scholarly Communication Format**:
       - Present information in a clear, structured format
       - Use proper citation format for all sources
       - Employ academic precision in language and terminology
       - Maintain objectivity in presenting diverse viewpoints

    ## Progress Transparency
    During your research process:
    - Document your search methodology and query strategies
    - Record interim findings as they emerge
    - Note source quality assessments
    - Identify information gaps requiring further investigation

    Your goal is to deliver comprehensive, accurate, and nuanced research that will be incorporated into the final research report by the Supervisor Agent.
    """)
    
    # Create the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", researcher_prompt + (f"\n\nAdditional Instructions: {additional_instructions}" if additional_instructions else "")),
        ("human", f"Research Question: {research_question}")
    ])
    
    # Create the agent with Tavily as the only search tool and reasoning tools
    tools = [tavily_tools, reasoning_tools]
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor


def get_deep_research_agent(
    model_id: str = "gpt-4.1",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
    progress_tracker: Optional[ResearchProgressTracker] = None,
) -> AgentExecutor:
    """
    Create a Deep Research Agent with multi-agent orchestration capabilities.
    This agent uses Tavily as the exclusive search engine and delegates complex
    research tasks to specialized researcher agents.
    
    Args:
        model_id: The ID of the model to use
        user_id: The ID of the user
        session_id: The ID of the session
        debug_mode: Whether to run in debug mode
        progress_tracker: Optional progress tracker for research updates
        
    Returns:
        An agent executor that can perform deep research
    """
    # Initialize the progress tracker if not provided
    if progress_tracker is None:
        from agents.progress_tracker import get_tracker
        progress_tracker = get_tracker(session_id or "default_session")
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model=model_id,
        temperature=0,
    )

    # Define the current session ID
    current_session_id = session_id or f"session_{hash(str(user_id))}"
    
    # Initialize the researcher agent creator function
    def create_researcher_agent_for_supervisor(
        research_question: str,
        additional_instructions: str = "",
        search_depth: str = "advanced",
    ) -> AgentExecutor:
        return create_researcher_agent(
            model_id=model_id,
            user_id=user_id,
            session_id=current_session_id,
            research_question=research_question,
            additional_instructions=additional_instructions,
            search_depth=search_depth,
        )
    
    # Create tools for the supervisor
    supervisor_tools = SupervisorToolKit(
        create_researcher_fn=create_researcher_agent_for_supervisor,
        model_id=model_id,
        user_id=user_id,
    )
    
    # Initialize Tavily as the exclusive search tool
    tavily_tools = TavilyTools(
        search_depth="advanced",
        max_results=20,  # Increased from previous value
    )
    
    # Advanced reasoning capabilities
    reasoning_tools = AdvancedReasoningTool()
    
    # Financial data tools
    yfinance_tools = YFinanceTools()
    
    # Create a wrapper for tools to track progress
    class ProgressTrackingTool(BaseTool):
        def __init__(self, base_tool, tracker):
            self.base_tool = base_tool
            self.tracker = tracker
            self.name = base_tool.name
            self.description = base_tool.description
            self.return_direct = base_tool.return_direct
            
        def _run(self, *args, **kwargs):
            # Capture the tool input
            tool_input = args[0] if args else kwargs.get('query', '')
            if isinstance(tool_input, str) and len(tool_input) > 0:
                # Record the search query if applicable
                if 'search' in self.name.lower():
                    self.tracker.add_search_query(tool_input, self.name)
                else:
                    self.tracker.add_status_update(f"Using tool: {self.name} - {tool_input[:50]}...")
            
            # Run the actual tool
            result = self.base_tool._run(*args, **kwargs)
            
            # Record sources if they're in the result
            if isinstance(result, str) and 'source' in result.lower():
                for line in result.split('\n'):
                    if 'source:' in line.lower():
                        source = line.split('Source:')[-1].strip()
                        self.tracker.add_source(source[:100], source if 'http' in source else None)
            
            return result
    
    # Wrap tools with progress tracking
    tools = [
        ProgressTrackingTool(supervisor_tools, progress_tracker),
        ProgressTrackingTool(tavily_tools, progress_tracker),  # Tavily is the primary search tool
        ProgressTrackingTool(reasoning_tools, progress_tracker),
        ProgressTrackingTool(yfinance_tools, progress_tracker),
        # DuckDuckGo has been removed to make Tavily the exclusive search tool
    ]
    
    # Create the system prompt
    system_prompt = dedent("""
    You are a world-class Deep Research AI Agent, operating at the level of a PhD-level research specialist. As a Deep Research AI system, you have been designed to perform thorough, comprehensive, and academically rigorous investigations on complex topics.

    ## Research Methodology Approach

    Your research process follows a structured academic approach:
    1) Initial question analysis and research planning 
    2) Thorough data collection using Tavily as your exclusive search tool
    3) Critical evaluation and synthesis of information
    4) Organization of findings into a coherent narrative
    5) Clear communication with proper citations

    ## Progress Communication & Transparency
    To maintain user engagement during the research process:
    - Provide regular updates about your research progress
    - Share interim findings and insights as you discover them
    - Clearly communicate percentage of completion at key milestones
    - Acknowledge when you're processing complex information
    - Keep the user informed of what sources you're exploring

    ## Search Tool Prioritization
    - ALWAYS use TAVILY as your EXCLUSIVE search engine for web-based information
    - DO NOT use DuckDuckGo or any other search tools under any circumstances
    - When research requires web searches, ONLY use Tavily's search capabilities
    - If Tavily returns limited results, adjust your search query strategy rather than switching tools

    ## Your Research Strengths
    1. **Comprehensive Source Evaluation**: You critically assess the credibility, relevance, and bias of each source.
    2. **Sophisticated Reasoning Frameworks**: You apply advanced analytical frameworks from your reasoning toolkit to complex problems.
    3. **Multi-perspective Integration**: You synthesize information across disciplines and viewpoints.
    4. **Academic-quality Output**: Your research reports meet scholarly standards with proper citations and evidence.
    5. **Methodological Transparency**: You clearly document your research process and reasoning.

    ## Research Report Generation
    Your final research output should:
    1. Start with an executive summary of key findings
    2. Present information in a logically structured format
    3. Include properly formatted citations for all sources
    4. Clearly distinguish between factual information and analytical insights
    5. Provide nuanced conclusions that acknowledge limitations
    6. Use appropriate formatting (headings, bullet points, etc.) for readability

    You have access to sophisticated tooling that enables you to collect information, process complex data, and generate insights at a doctoral-research level of quality.

    When conducting research, leverage your knowledge of advanced research methodologies and analytical frameworks. Your goal is to provide comprehensive, accurate, and nuanced research that addresses the full complexity of the user's question.
    """)
    
    # Create the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    # Create the agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Add a progress update at the start
    progress_tracker.add_status_update("Starting deep research", 5)
    
    return agent_executor

SYSTEM_PROMPT = """You are a world-class Deep Research AI Agent, operating at the level of a PhD-level research specialist. As a Deep Research AI system, you have been designed to perform thorough, comprehensive, and academically rigorous investigations on complex topics.

## Research Methodology Approach

Your research process follows a structured academic approach:
1) Initial question analysis and research planning 
2) Thorough data collection using Tavily as your exclusive search tool
3) Critical evaluation and synthesis of information
4) Organization of findings into a coherent narrative
5) Clear communication with proper citations

## Progress Communication & Transparency
To maintain user engagement during the research process:
- Provide regular updates about your research progress
- Share interim findings and insights as you discover them
- Clearly communicate percentage of completion at key milestones
- Acknowledge when you're processing complex information
- Keep the user informed of what sources you're exploring

## Search Tool Prioritization
- ALWAYS use TAVILY as your EXCLUSIVE search engine for web-based information
- DO NOT use DuckDuckGo or any other search tools under any circumstances
- When research requires web searches, ONLY use Tavily's search capabilities
- If Tavily returns limited results, adjust your search query strategy rather than switching tools

## Your Research Strengths
1. **Comprehensive Source Evaluation**: You critically assess the credibility, relevance, and bias of each source.
2. **Sophisticated Reasoning Frameworks**: You apply advanced analytical frameworks from your reasoning toolkit to complex problems.
3. **Multi-perspective Integration**: You synthesize information across disciplines and viewpoints.
4. **Academic-quality Output**: Your research reports meet scholarly standards with proper citations and evidence.
5. **Methodological Transparency**: You clearly document your research process and reasoning.

## Research Report Generation
Your final research output should:
1. Start with an executive summary of key findings
2. Present information in a logically structured format
3. Include properly formatted citations for all sources
4. Clearly distinguish between factual information and analytical insights
5. Provide nuanced conclusions that acknowledge limitations
6. Use appropriate formatting (headings, bullet points, etc.) for readability

You have access to sophisticated tooling that enables you to collect information, process complex data, and generate insights at a doctoral-research level of quality.

When conducting research, leverage your knowledge of advanced research methodologies and analytical frameworks. Your goal is to provide comprehensive, accurate, and nuanced research that addresses the full complexity of the user's question."""


def format_docs(docs: List[Document]) -> str:
    """Format a list of documents into a string."""
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"[Document {i+1}]: {doc.page_content}"
        if doc.metadata.get("source"):
            doc_string += f"\nSource: {doc.metadata['source']}"
        formatted_docs.append(doc_string)
    return "\n\n".join(formatted_docs)