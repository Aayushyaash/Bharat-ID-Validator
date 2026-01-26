from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
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
    # Startup
    logger.info("Starting up ID Validator Service...")
    model_loader.load_models()
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

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Id Validator Service is running"}

@app.get("/metrics")
async def metrics():
    return {"status": "ok", "message": "endpoint is healthy"}
