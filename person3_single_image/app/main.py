"""Main FastAPI entry point for SatQuery AI - Person 3 Single Image Module."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router


def create_app() -> FastAPI:
    """Application factory for Person 3 Single Image Remote-Sensing Module."""
    application = FastAPI(
        title="SatQuery AI — Person 3 (Single Image Remote Sensing)",
        description=(
            "Interactive Vision-Language Assistant for Remote Sensing. "
            "Handles single-image VQA, Captioning, Visual Grounding, and Evidence Generation. "
            "Currently equipped with lightweight mock interfaces, architected for direct plug-in "
            "of BigData-KSU/RS-llava-v1.5-7b-LoRA and GeoChat."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Enable CORS for seamless integration with frontend or teammate microservices
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    application.include_router(router)

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
