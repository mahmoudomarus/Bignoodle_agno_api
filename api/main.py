from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from api.routes.v1_router import v1_router
from api.middleware import setup_cors_middleware


def create_app() -> FastAPI:
    """Create a FastAPI App"""

    # Create FastAPI App
    app: FastAPI = FastAPI(
        title="agent-api",
        version="1.0",
        docs_url="/docs", 
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Add CORS middleware first (before routers)
    setup_cors_middleware(app)

    # Add v1 router
    app.include_router(v1_router)

    return app


# Create a FastAPI app
app = create_app()
