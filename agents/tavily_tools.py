import json
import os
from typing import Any, Dict, List, Optional, Union

import requests

from agents.base_tools import Tool, ToolType, ToolTypeArgs

# For progress tracking
from agents.progress_tracker import ResearchProgressTracker


class TavilyTools(Tool):
    """
    A class that provides tools for interacting with the Tavily search API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        search_depth: str = "basic",
        max_results: int = 15,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        include_answer: bool = True,
        include_raw_content: bool = True,
        include_images: bool = False,
        progress_tracker: Optional[ResearchProgressTracker] = None,
    ):
        """
        Initialize the TavilyTools class.

        Args:
            api_key: The Tavily API key. If not provided, it will be read from environment variables.
            search_depth: The depth of the search. Either "basic" or "advanced" or "comprehensive".
            max_results: The maximum number of results to return.
            include_domains: A list of domains to include in the search.
            exclude_domains: A list of domains to exclude from the search.
            include_answer: Whether to include an answer in the response.
            include_raw_content: Whether to include raw content in the response.
            include_images: Whether to include images in the response.
            progress_tracker: Optional progress tracker for research updates
        """
        # Use provided API key, or try environment variable, or use fallback
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY") or "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"
        
        self.search_depth = search_depth
        self.max_results = max_results
        self.include_domains = include_domains or []
        self.exclude_domains = exclude_domains or []
        self.include_answer = include_answer
        self.include_raw_content = include_raw_content
        self.include_images = include_images
        self.progress_tracker = progress_tracker
        
        tool_types = [
            ToolType(
                name="tavily_search",
                description="Search the web for information on a specific topic using Tavily's powerful search engine. This is the PRIMARY research tool for web-based information gathering.",
                function=self.search,
                args={
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "search_depth": {
                        "type": "string",
                        "description": "Search depth: 'basic' for quick results, 'advanced' for thorough research",
                    }
                },
            ),
            ToolType(
                name="tavily_news_search",
                description="Search for recent news articles on a specific topic using Tavily's news search capabilities.",
                function=self.search_news,
                args={
                    "query": {
                        "type": "string",
                        "description": "The news search query",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of news articles to return",
                    }
                },
            ),
            ToolType(
                name="tavily_topic_search",
                description="Perform a specialized topic search with specific filters and parameters.",
                function=self.search_topic,
                args={
                    "topic": {
                        "type": "string",
                        "description": "The specific topic to research",
                    },
                    "include_domains": {
                        "type": "string",
                        "description": "Comma-separated list of domains to include (e.g., 'nytimes.com,forbes.com')",
                    },
                    "exclude_domains": {
                        "type": "string",
                        "description": "Comma-separated list of domains to exclude",
                    },
                    "time_period": {
                        "type": "string",
                        "description": "Time period for results (recent, past_week, past_month, past_year)",
                    }
                },
            ),
        ]
        super().__init__(tool_types=tool_types)
    
    def search(self, query: str, search_depth: Optional[str] = None) -> Dict[str, Any]:
        """
        Search the web for information on a specific topic using Tavily's API.
        
        Args:
            query: The search query
            search_depth: Search depth ('basic' or 'advanced')
            
        Returns:
            Search results
        """
        # Update progress tracker if available
        if self.progress_tracker:
            self.progress_tracker.add_search_query(query, "tavily_search")
            
        try:
            search_depth = search_depth or self.search_depth
            
            response = requests.post(
                "https://api.tavily.com/search",
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": self.api_key
                },
                json={
                    "query": query,
                    "search_depth": search_depth,
                    "max_results": self.max_results,
                    "include_domains": self.include_domains,
                    "exclude_domains": self.exclude_domains,
                    "include_answer": self.include_answer,
                    "include_raw_content": self.include_raw_content,
                    "include_images": self.include_images
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Format results
            formatted_results = [
                {
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "url": r.get("url", ""),
                    "score": r.get("score", 0),
                    "raw_content": r.get("raw_content", "") if self.include_raw_content else ""
                }
                for r in result.get("results", [])
            ]
            
            # Update progress tracker if available
            if self.progress_tracker and result.get("results"):
                for r in result.get("results", []):
                    if "title" in r and "url" in r:
                        self.progress_tracker.add_source(r["title"], r["url"])
            
            return {
                "answer": result.get("answer", ""),
                "results": formatted_results
            }
        
        except requests.exceptions.RequestException as e:
            # Log the error and provide a fallback response if API fails
            if self.progress_tracker:
                self.progress_tracker.add_status_update(f"Search API error: {str(e)}", None)
            return {
                "error": f"Error searching Tavily: {str(e)}",
                "results": []
            }
    
    def search_news(self, query: str, max_results: Optional[int] = None) -> Dict[str, Any]:
        """
        Search for news articles on a specific topic.
        
        Args:
            query: The news search query
            max_results: Maximum number of news articles to return
            
        Returns:
            News search results
        """
        # Update progress tracker if available
        if self.progress_tracker:
            self.progress_tracker.add_search_query(query, "tavily_news_search")
            
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": self.api_key
                },
                json={
                    "query": query,
                    "search_depth": self.search_depth,
                    "max_results": max_results or self.max_results,
                    "include_domains": self.include_domains,
                    "exclude_domains": self.exclude_domains,
                    "search_type": "news"
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Format news results
            news_results = [
                {
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "url": r.get("url", ""),
                    "published_date": r.get("published_date", ""),
                    "source": r.get("source", "Unknown source"),
                    "image_url": r.get("image_url", "")
                }
                for r in result.get("results", [])
            ]
            
            # Update progress tracker if available
            if self.progress_tracker and result.get("results"):
                for r in result.get("results", []):
                    if "title" in r and "url" in r:
                        self.progress_tracker.add_source(r["title"], r["url"])
            
            return {
                "news_results": news_results,
                "result_count": len(news_results),
                "query": query,
            }
        
        except requests.exceptions.RequestException as e:
            # Log the error and provide a fallback response if API fails
            if self.progress_tracker:
                self.progress_tracker.add_status_update(f"Search API error: {str(e)}", None)
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
        # Update progress tracker if available
        if self.progress_tracker:
            self.progress_tracker.add_search_query(topic, "tavily_topic_search")
            
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
            
            # Update progress tracker if available
            if self.progress_tracker and formatted_results:
                for result in formatted_results:
                    if "title" in result and "url" in result:
                        self.progress_tracker.add_source(result["title"], result["url"])
            
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
            # Log the error and provide a fallback response if API fails
            if self.progress_tracker:
                self.progress_tracker.add_status_update(f"Search API error: {str(e)}", None)
            return {
                "error": f"Error searching Tavily Topic: {str(e)}",
                "topic": topic,
            } 