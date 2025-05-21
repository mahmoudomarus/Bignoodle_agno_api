from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import traceback

from api.middleware import setup_cors
from api.playground import router as playground_router
from api.research import router as research_router

app = FastAPI()

# Setup CORS
setup_cors(app)

# Include routers
app.include_router(playground_router, prefix="/v1/playground")
app.include_router(research_router, prefix="/v1/research")

@app.get("/v1/status")
async def status():
    return {"status": "healthy"}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for the application."""
    error_detail = traceback.format_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "traceback": error_detail},
    )
