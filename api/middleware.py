from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

def setup_cors_middleware(app: FastAPI) -> None:
    """
    Configure CORS middleware for the FastAPI application.
    This ensures all routes have proper CORS headers.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://bignoodle-agent-ui-1.onrender.com",
            "https://bignoodle-agent-ui.onrender.com",
            "https://deepresearch.bignoodle.com",
            "http://localhost:3000",
            "https://app.agno.com"
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    ) 