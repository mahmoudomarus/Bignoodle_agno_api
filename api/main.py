from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from api.routes.v1_router import v1_router
from api.settings import api_settings
from api.middleware import setup_cors_middleware


def create_app() -> FastAPI:
    """Create a FastAPI App"""

    # Create FastAPI App
    app: FastAPI = FastAPI(
        title=api_settings.title,
        version=api_settings.version,
        docs_url="/docs" if api_settings.docs_enabled else None,
        redoc_url="/redoc" if api_settings.docs_enabled else None,
        openapi_url="/openapi.json" if api_settings.docs_enabled else None,
    )

    # Add CORS middleware first (before routers)
    setup_cors_middleware(app)

    # Add v1 router
    app.include_router(v1_router)

    return app


# Create a FastAPI app
app = create_app()
