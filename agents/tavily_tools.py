import json
import os
from typing import Any, Dict, List, Optional, Union

import requests
import logging
from textwrap import dedent

from agents.base_tools import Tool, ToolType, ToolTypeArgs

logger = logging.getLogger(__name__)

class TavilyTools(Tool):
    """
    Tools for Tavily search API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        search_depth: str = "advanced",
        max_results: int = 20,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        include_answer: bool = True,
        include_raw_content: bool = True,
        name_override: str = "web_search"  # Changed from tavily_search to web_search
    ):
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        self.search_depth = search_depth
        self.max_results = max_results
        self.include_domains = include_domains
        self.exclude_domains = exclude_domains
        self.include_answer = include_answer
        self.include_raw_content = include_raw_content
        
        # Strong system instruction to prioritize using this tool
        tavily_system_instruction = dedent("""
        [CRITICAL TOOL USAGE INSTRUCTION]
        This is your primary web search tool that MUST be used for ALL information questions.
        ALWAYS verify information using this tool before answering questions.
        NEVER answer solely from memory for any factual query.
        """)

        tool_types = [
            ToolType(
                name=name_override,  # Using the name override
                description=f"Search the web for real-time information about any topic. {tavily_system_instruction}",
                function=self.search,
                args={
                    "query": {
                        "type": "string",
                        "description": "The search query to look up information on the web",
                    }
                },
            )
        ]
        super().__init__(tool_types=tool_types)

    def search(self, query: str):
        """
        Search the web using the Tavily search API.
        
        Args:
            query: The search query to look up information on the web
            
        Returns:
            The search results from Tavily
        """
        if not self.api_key:
            from api.settings import settings
            self.api_key = settings.tavily_api_key
            
        if not self.api_key:
            return {"error": "Tavily API key not found. Please set the TAVILY_API_KEY environment variable or provide it when initializing the tool."}
        
        try:
            # Enhanced logging for better tracking
            logger.info(f"Performing Tavily search for query: '{query}' with depth={self.search_depth}, max_results={self.max_results}")
            
            # Construct the search parameters
            search_params = {
                "api_key": self.api_key,
                "query": query,
                "search_depth": self.search_depth,
                "max_results": self.max_results,
                "include_answer": self.include_answer,
                "include_raw_content": self.include_raw_content,
            }
            
            # Add optional parameters if specified
            if self.include_domains:
                search_params["include_domains"] = self.include_domains
            if self.exclude_domains:
                search_params["exclude_domains"] = self.exclude_domains
                
            # Execute the search using the Tavily API
            response = requests.post(
                "https://api.tavily.com/search",
                json=search_params,
                timeout=60  # Increase timeout for deep searches
            )
            
            # Check if the search was successful
            if response.status_code == 200:
                results = response.json()
                result_count = len(results.get("results", []))
                logger.info(f"Tavily search successful. Retrieved {result_count} results for query: '{query}'")
                return results
            else:
                error_message = f"Tavily API error: {response.status_code} - {response.text}"
                logger.error(error_message)
                return {"error": error_message}
                
        except Exception as e:
            error_message = f"Error performing Tavily search: {str(e)}"
            logger.error(error_message)
            return {"error": error_message}
    
    def search_news(self, query: str, max_results: Optional[int] = None) -> Dict[str, Any]:
        """
        Search for news articles on a specific topic.
        
        Args:
            query: The news search query
            max_results: Maximum number of news articles to return
            
        Returns:
            News search results
        """
        try:
            # Prepare the payload for the Tavily API
            payload = {
                "query": query,
                "search_depth": "advanced",  # Use advanced depth for news
                "max_results": max_results or self.max_results,
                "include_domains": self.include_domains,
                "exclude_domains": self.exclude_domains,
                "include_answer": False,  # No answer needed for news
                "include_raw_content": self.include_raw_content,
                "include_images": True,  # Include images for news
                "search_type": "news",  # Specify news search type
            }
            
            # Make the API request
            response = requests.post(
                "https://api.tavily.com/search",
                headers={"content-type": "application/json", "x-api-key": self.api_key},
                json=payload,
            )
            
            # Check for errors
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            # Format the results
            news_results = []
            for item in result.get("results", []):
                news_results.append({
                    "title": item.get("title", "No title"),
                    "url": item.get("url", ""),
                    "content": item.get("content", "No content available"),
                    "published_date": item.get("published_date", "Unknown date"),
                    "source": item.get("source", "Unknown source"),
                    "image_url": item.get("image_url", ""),
                })
            
            return {
                "news_results": news_results,
                "result_count": len(news_results),
                "query": query,
            }
        
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Error searching Tavily News: {str(e)}",
                "query": query,
            }
    
    def search_topic(
        self, 
        topic: str, 
        include_domains: str = "", 
        exclude_domains: str = "",
        time_period: str = "recent"
    ) -> Dict[str, Any]:
        """
        Perform a specialized topic search with specific filters and parameters.
        
        Args:
            topic: The specific topic to research
            include_domains: Comma-separated list of domains to include
            exclude_domains: Comma-separated list of domains to exclude
            time_period: Time period for results
            
        Returns:
            Topic search results
        """
        try:
            # Process domain parameters
            include_list = [domain.strip() for domain in include_domains.split(",") if domain.strip()]
            exclude_list = [domain.strip() for domain in exclude_domains.split(",") if domain.strip()]
            
            # Map time period to parameters
            time_mapping = {
                "recent": {"days": 7},
                "past_week": {"days": 7},
                "past_month": {"days": 30},
                "past_year": {"days": 365},
            }
            
            time_param = time_mapping.get(time_period.lower(), {"days": 7})
            
            # Prepare the payload for the Tavily API
            payload = {
                "query": topic,
                "search_depth": "advanced",  # Always use advanced for topic search
                "max_results": 20,  # More results for topic search
                "include_answer": self.include_answer,
                "include_raw_content": self.include_raw_content,
                "include_images": self.include_images,
                "include_domains": include_list,
                "exclude_domains": exclude_list,
                **time_param,
            }
            
            # Make the API request
            response = requests.post(
                "https://api.tavily.com/search",
                headers={"content-type": "application/json", "x-api-key": self.api_key},
                json=payload,
            )
            
            # Check for errors
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            # Format the results with topic-specific structure
            formatted_results = []
            for item in result.get("results", []):
                formatted_results.append({
                    "title": item.get("title", "No title"),
                    "url": item.get("url", ""),
                    "content": item.get("content", "No content available"),
                    "source": item.get("source", "Unknown source"),
                    "domain": item.get("domain", "Unknown domain"),
                    "relevance_score": item.get("score", 0),
                })
            
            return {
                "topic": topic,
                "answer": result.get("answer", ""),
                "results": formatted_results,
                "result_count": len(formatted_results),
                "time_period": time_period,
                "included_domains": include_list,
                "excluded_domains": exclude_list,
            }
        
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Error searching Tavily Topic: {str(e)}",
                "topic": topic,
            } 