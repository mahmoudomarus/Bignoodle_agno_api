from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

def setup_cors(app: FastAPI) -> None:
    """
    Setup CORS middleware for the application.
    
    Args:
        app: FastAPI application
    """
    # Define allowed origins
    allow_origins = [
        "http://localhost:3000",
        "https://localhost:3000",
        "http://127.0.0.1:3000",
        "https://127.0.0.1:3000",
        "https://deepresearch.bignoodle.com",
        "http://deepresearch.bignoodle.com",
    ]
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Type", "X-Research-Progress"],
        max_age=86400,  # 24 hours
    ) 