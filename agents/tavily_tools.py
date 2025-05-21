from typing import Dict, List, Optional

from agno.tools.base import Tool, ToolType, ToolTypeList

import requests


class TavilyTools(Tool):
    """
    A tool for searching the web using Tavily AI's search API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        search_depth: str = "advanced",
        max_results: int = 10,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        include_answer: bool = True,
        include_raw_content: bool = False,
        search_topic: bool = True,
        search_news: bool = True,
    ):
        """
        Initialize the Tavily search tool.

        Args:
            api_key: Tavily API key, if None it will try to load from environment variable TAVILY_API_KEY.
            search_depth: "basic" or "advanced" - determines the depth of the search.
            max_results: Maximum number of results to return.
            include_domains: List of domains to include in the search.
            exclude_domains: List of domains to exclude from the search.
            include_answer: Whether to include Tavily's generated answer.
            include_raw_content: Whether to include the raw content of the results.
            search_topic: Whether to enable searching on general topics.
            search_news: Whether to enable news search.
        """
        self.api_key = api_key or "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"
        self.search_depth = search_depth
        self.max_results = max_results
        self.include_domains = include_domains or []
        self.exclude_domains = exclude_domains or []
        self.include_answer = include_answer
        self.include_raw_content = include_raw_content
        self.search_topic = search_topic
        self.search_news = search_news

        # Define the tool types
        tool_types: ToolTypeList = []

        if search_topic:
            tool_types.append(
                ToolType(
                    name="tavily_search",
                    description="Search the web for information on a topic. Perfect for research questions, current events, and general knowledge.",
                    function=self.search,
                    args={
                        "query": {
                            "type": "string",
                            "description": "The search query to look up information about.",
                        }
                    },
                )
            )

        if search_news:
            tool_types.append(
                ToolType(
                    name="tavily_news_search",
                    description="Search for recent news articles. Best for current events, breaking news, and recent developments.",
                    function=self.news_search,
                    args={
                        "query": {
                            "type": "string",
                            "description": "The news search query.",
                        }
                    },
                )
            )

        super().__init__(tool_types=tool_types)

    def search(self, query: str) -> Dict:
        """
        Search the web for information on a topic using Tavily.

        Args:
            query: The search query.

        Returns:
            Dictionary containing search results and metadata.
        """
        url = "https://api.tavily.com/search"
        params = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": self.search_depth,
            "max_results": self.max_results,
            "include_answer": self.include_answer,
            "include_raw_content": self.include_raw_content,
        }

        if self.include_domains:
            params["include_domains"] = self.include_domains
        if self.exclude_domains:
            params["exclude_domains"] = self.exclude_domains

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Format the results in a more readable way
            formatted_results = {
                "answer": data.get("answer", "No summary available."),
                "results": []
            }

            for result in data.get("results", []):
                formatted_result = {
                    "title": result.get("title", "No title"),
                    "url": result.get("url", "No URL"),
                    "content": result.get("content", "No content available"),
                    "score": result.get("score", 0),
                }
                formatted_results["results"].append(formatted_result)

            return formatted_results
        except Exception as e:
            return {"error": f"Error searching Tavily: {str(e)}"}

    def news_search(self, query: str) -> Dict:
        """
        Search for recent news articles using Tavily.

        Args:
            query: The news search query.

        Returns:
            Dictionary containing news search results.
        """
        url = "https://api.tavily.com/search"
        params = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": self.search_depth,
            "max_results": self.max_results,
            "include_answer": self.include_answer,
            "include_raw_content": self.include_raw_content,
            "search_type": "news"
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Format the results in a more readable way
            formatted_results = {
                "answer": data.get("answer", "No news summary available."),
                "results": []
            }

            for result in data.get("results", []):
                formatted_result = {
                    "title": result.get("title", "No title"),
                    "url": result.get("url", "No URL"),
                    "content": result.get("content", "No content available"),
                    "published_date": result.get("published_date", "No date available"),
                    "score": result.get("score", 0),
                }
                formatted_results["results"].append(formatted_result)

            return formatted_results
        except Exception as e:
            return {"error": f"Error searching Tavily news: {str(e)}"} 