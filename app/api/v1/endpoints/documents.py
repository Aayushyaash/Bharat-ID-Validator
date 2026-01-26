from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, Form
from fastapi.concurrency import run_in_threadpool
from app.services.model_loader import get_model_loader, ModelLoader
from app.services import pipeline, extraction, combined_pipeline, image_utils
from app.schemas.document import DocumentResponse, ExtractionResponse, ClassifyAndExtractResponse
import logging
import cv2
import numpy as np

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/classify", response_model=DocumentResponse)
async def classify_document_endpoint(
    file: UploadFile = File(...),
    loader: ModelLoader = Depends(get_model_loader)
):
    """
    Classifies the uploaded document image.
    """
    try:
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
        raise HTTPException(status_code=500, detail=str(e))

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
            raise HTTPException(status_code=400, detail=str(ve))
        except RuntimeError as re:
            raise HTTPException(status_code=503, detail=str(re))
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/classify-and-extract", response_model=ClassifyAndExtractResponse)
async def classify_and_extract_endpoint(
    file: UploadFile = File(...),
    loader: ModelLoader = Depends(get_model_loader)
):
    """
    Combined endpoint: Classifies the document and extracts fields in one call.
    """
    try:
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
        raise HTTPException(status_code=500, detail=str(e))
