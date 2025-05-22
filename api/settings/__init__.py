"""
API settings module for storing configuration and environment variables.
"""
import os
from typing import Optional, Dict, Any

# Create a settings object to hold API keys and configuration
class Settings:
    """Settings object for API keys and configuration."""
    
    def __init__(self):
        # Load environment variables
        self.tavily_api_key = os.environ.get("TAVILY_API_KEY", "")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        self.serpapi_api_key = os.environ.get("SERPAPI_API_KEY", "")
        self.perplexity_api_key = os.environ.get("PERPLEXITY_API_KEY", "")
        
        # Other settings
        self.debug = os.environ.get("DEBUG", "false").lower() == "true"
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value by key name."""
        if hasattr(self, key):
            return getattr(self, key)
        return default

# Create an instance of the settings object
settings = Settings() 