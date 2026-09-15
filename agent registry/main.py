"""
SatQuery AI - Agent Registry Main Application Entry Point.
Provides RESTful APIs for registering, discovering, updating, health checking,
and deregistering remote sensing specialist agents.
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from database.db import init_db, SessionLocal
from services.registry_service import (
    AgentRegistryService,
    AgentAlreadyExistsError,
    AgentNotFoundError,
    InvalidMetadataError
)
from api.routes import router as agent_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes database tables and seeds mock specialist agents on startup."""
    init_db()
    db = SessionLocal()
    try:
        AgentRegistryService.seed_mock_agents(db)
    finally:
        db.close()
    yield


app = FastAPI(
    title="SatQuery AI - Remote Sensing Agent Registry API",
    description=(
        "Centralized registry service for domain-adapted remote sensing specialist agents "
        "(VQA, Region Grounding, Bi-temporal Change Detection, Optical-SAR Fusion). "
        "Supports dynamic discovery, capability search, health telemetry, and router integration."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for web GUI integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Step 12: Exception Handling ---

@app.exception_handler(AgentAlreadyExistsError)
async def agent_already_exists_handler(request: Request, exc: AgentAlreadyExistsError):
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={
            "detail": str(exc),
            "error_code": "AGENT_ALREADY_EXISTS",
            "agent_id": exc.agent_id
        }
    )


@app.exception_handler(AgentNotFoundError)
async def agent_not_found_handler(request: Request, exc: AgentNotFoundError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "detail": str(exc),
            "error_code": "AGENT_NOT_FOUND",
            "agent_id": exc.agent_id
        }
    )


@app.exception_handler(InvalidMetadataError)
async def invalid_metadata_handler(request: Request, exc: InvalidMetadataError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "detail": str(exc),
            "error_code": "INVALID_METADATA"
        }
    )


# Include Agent Registry API routes
app.include_router(agent_router)


@app.get("/", tags=["Health"])
def root_status():
    """Service health check endpoint."""
    return {
        "service": "SatQuery AI Agent Registry",
        "status": "online",
        "docs_url": "/docs",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
