from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
from app.core.config import settings
from app.api.v1.api import api_router
from app.services.model_loader import model_loader
from app.core.logging import setup_logging
from app.core.errors import ErrorResponse
from app.core.middleware import TraceIDMiddleware
import logging

# Setup Logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - run model loading in a separate thread to avoid blocking the event loop
    logger.info("Starting up ID Validator Service...")
    await asyncio.to_thread(model_loader.load_models)
    yield
    # Shutdown
    logger.info("Shutting down ID Validator Service...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# Add TraceID Middleware
app.add_middleware(TraceIDMiddleware)

# Global Exception Handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    error_code = str(exc.status_code)
    error_message = str(exc.detail)
    
    if isinstance(exc.detail, dict):
        # If detail is already a dict (structured error from our endpoints)
        error_code = str(exc.detail.get("code", exc.status_code))
        error_message = str(exc.detail.get("message", exc.detail))
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            code=error_code,
            message=error_message,
            trace_id=None # Can be populated with Request ID middleware later
        ).model_dump()
    )

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)


@app.api_route("/", methods=["GET", "HEAD"])
async def root(request: Request):
    """Root endpoint - service status check."""
    if request.method == "HEAD":
        return Response(status_code=200)
    return {"message": "Id Validator Service is running"}


@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check(request: Request):
    """Health check endpoint for monitoring systems."""
    if request.method == "HEAD":
        return Response(status_code=200)
    return {"status": "healthy", "service": settings.PROJECT_NAME}


@app.api_route("/ready", methods=["GET", "HEAD"])
async def readiness_check(request: Request):
    """Readiness check - verifies models are loaded."""
    is_ready = model_loader._models_loaded
    status_code = 200 if is_ready else 503
    if request.method == "HEAD":
        return Response(status_code=status_code)
    return {
        "status": "ready" if is_ready else "not_ready",
        "models_loaded": is_ready
    }


@app.get("/metrics")
async def metrics():
    """Basic metrics endpoint (placeholder for Prometheus integration)."""
    return {"status": "ok", "message": "endpoint is healthy"}
