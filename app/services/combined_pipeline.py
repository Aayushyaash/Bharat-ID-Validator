from app.services.model_loader import ModelLoader
from app.services import pipeline as pipeline # Changed from pipeline_v2
from app.services import extraction as extraction # Changed from extraction_v2
from app.core.config import settings
import logging
import numpy as np

logger = logging.getLogger(__name__)


def normalize_doc_type_for_extraction(doc_type_raw: str) -> str:
    """
    Normalizes document type string for extraction model lookup.
    
    Handles spelling variations (e.g., 'aadhar' -> 'aadhaar') and converts
    to lowercase for consistent model key matching.
    
    Args:
        doc_type_raw: Raw document type string from classification (e.g., 'Aadhaar_Front')
        
    Returns:
        Normalized document type string (e.g., 'aadhaar_front')
    """
    normalized = doc_type_raw.lower()
    if "aadhar" in normalized:
        normalized = normalized.replace("aadhar", "aadhaar")
    return normalized


def process_document_sync(
    image: np.ndarray,
    loader: ModelLoader,
    filename: str = "unknown"
) -> dict:
    """
    Combined classification and extraction pipeline (synchronous).
    
    Performs document classification first, and if confidence meets threshold,
    proceeds with field extraction. This is more efficient than separate calls
    as it shares the corrected image and allows early exit for invalid documents.
    
    Args:
        image: Input document image as numpy array (BGR format from OpenCV)
        loader: ModelLoader instance containing all loaded ML models
        filename: Original filename for inclusion in response (default: "unknown")
        
    Returns:
        Dictionary containing:
            - filename: Original filename
            - document_type: Generalized document type (e.g., 'aadhaar', 'pan_card')
            - confidence: Classification confidence score (0.0 to 1.0)
            - is_valid: Whether confidence meets threshold
            - fields: Extracted field data dict or None if extraction skipped/failed
            - message: Status message or error description
            
    Note:
        If classification confidence is below threshold, extraction is skipped
        entirely for performance optimization (~80-90% faster for invalid docs).
    """
    # --- CLASSIFICATION PHASE ---
    
    # 1. Orientation
    corrected_image, _ = pipeline.detect_orientation(image, loader)
    
    # 2. Classification
    # run_classification doesn't do orientation, so this is safe.
    doc_type_raw, confidence = pipeline.run_classification(corrected_image, loader)
    
    is_valid = confidence >= settings.CONFIDENCE_THRESHOLD
    
    if not is_valid:
        doc_type_mapped = "unknown"
        return {
            "filename": filename,
            "document_type": doc_type_mapped,
            "confidence": confidence,
            "is_valid": False,
            "fields": None,
            "message": "Classification confidence below threshold - extraction skipped"
        }
        
    doc_type_mapped = pipeline.format_response(doc_type_raw)
    
    # --- EXTRACTION PHASE ---
    fields = None
    message = None
    
    try:
        doc_type_normalized = normalize_doc_type_for_extraction(doc_type_raw)
        
        # Call extraction with Corrected Image
        extraction_result = extraction.extract_data_v2(
            corrected_image, 
            doc_type_normalized, 
            loader
        )
        
        fields = extraction_result.get("fields", {})
        message = extraction_result.get("message")
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        message = str(e)
        
    return {
        "filename": filename,
        "document_type": doc_type_mapped,
        "confidence": confidence,
        "is_valid": is_valid,
        "fields": fields,
        "message": message
    }

# Legacy alias
classify_and_extract = None # Async wrapper removed, logic is now sync. Endpoints must use run_in_threadpool
