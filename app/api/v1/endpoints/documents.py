from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, Form
from fastapi.concurrency import run_in_threadpool
from app.services.model_loader import get_model_loader, ModelLoader
from app.services import pipeline, extraction, combined_pipeline, image_utils
from app.schemas.document import DocumentResponse, ExtractionResponse, ClassifyAndExtractResponse
from app.core.config import settings
import logging
import cv2
import numpy as np

router = APIRouter()
logger = logging.getLogger(__name__)


def _validate_file_size(file: UploadFile) -> None:
    """
    Validates that the uploaded file does not exceed MAX_FILE_SIZE.
    
    Args:
        file: The uploaded file to validate
        
    Raises:
        HTTPException: 413 if file size exceeds limit
    """
    if file.size and file.size > settings.MAX_FILE_SIZE:
        max_mb = settings.MAX_FILE_SIZE / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail={
                "code": "FILE_TOO_LARGE",
                "message": f"File size exceeds maximum allowed size of {max_mb:.1f}MB"
            }
        )


@router.post("/classify", response_model=DocumentResponse)
async def classify_document_endpoint(
    file: UploadFile = File(...),
    loader: ModelLoader = Depends(get_model_loader)
):
    """
    Classifies the uploaded document image.
    """
    try:
        # Validate file size before reading into memory
        _validate_file_size(file)
        
        # Async I/O
        image = await image_utils.read_image_file(file)
        
        if image is None:
            raise HTTPException(status_code=400, detail={"code": "INVALID_FILE_TYPE", "message": "File must be an image"})
        
        # Offload CPU task
        result = await run_in_threadpool(pipeline.classify_document_sync, image, loader)
        
        # Add filename since sync function doesn't know about UploadFile
        result["filename"] = file.filename
        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail={"code": "INTERNAL_ERROR", "message": "Classification failed due to an internal error"}
        )


@router.post("/extract", response_model=ExtractionResponse)
async def extract_document_data(
    file: UploadFile = File(...),
    document_type: str = Form(...),
    loader: ModelLoader = Depends(get_model_loader)
):
    """
    Extracts text fields from the document.
    """
    try:
        # Validate file size before reading into memory
        _validate_file_size(file)
        
        # Async I/O
        image = await image_utils.read_image_file(file)
        
        if image is None:
            raise HTTPException(status_code=400, detail={"code": "INVALID_FILE_TYPE", "message": "File must be an image"})
        
        # Offload CPU task
        try:
            # Use extract_data (aliased to v2) for compatibility with tests mocking 'extract_data'
            result = await run_in_threadpool(extraction.extract_data, image, document_type, loader)
            return result
        except ValueError as ve:
            raise HTTPException(
                status_code=400, 
                detail={"code": "INVALID_DOCUMENT_TYPE", "message": str(ve)}
            )
        except RuntimeError as re:
            raise HTTPException(
                status_code=503, 
                detail={"code": "MODEL_NOT_LOADED", "message": str(re)}
            )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail={"code": "INTERNAL_ERROR", "message": "Extraction failed due to an internal error"}
        )

@router.post("/classify-and-extract", response_model=ClassifyAndExtractResponse)
async def classify_and_extract_endpoint(
    file: UploadFile = File(...),
    loader: ModelLoader = Depends(get_model_loader)
):
    """
    Combined endpoint: Classifies the document and extracts fields in one call.
    """
    try:
        # Validate file size before reading into memory
        _validate_file_size(file)
        
        # Async I/O
        image = await image_utils.read_image_file(file)
        
        if image is None:
            raise HTTPException(status_code=400, detail={"code": "INVALID_FILE_TYPE", "message": "File must be an image"})
        
        # Offload CPU task
        result = await run_in_threadpool(combined_pipeline.process_document_sync, image, loader, file.filename)
        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Combined pipeline failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail={"code": "INTERNAL_ERROR", "message": "Document processing failed due to an internal error"}
        )
