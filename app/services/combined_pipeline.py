from app.services.model_loader import ModelLoader
from app.services import pipeline as pipeline # Changed from pipeline_v2
from app.services import extraction as extraction # Changed from extraction_v2
from app.core.config import settings
import logging
import numpy as np

logger = logging.getLogger(__name__)

def normalize_doc_type_for_extraction(doc_type_raw: str) -> str:
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
    Combined classification and extraction pipeline (Synchronous).
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
            "extraction_message": "Classification confidence below threshold - extraction skipped"
        }
        
    doc_type_mapped = pipeline.format_response(doc_type_raw)
    
    # --- EXTRACTION PHASE ---
    fields = None
    extraction_message = None
    
    try:
        doc_type_normalized = normalize_doc_type_for_extraction(doc_type_raw)
        
        # Call extraction with Corrected Image
        extraction_result = extraction.extract_data_v2(
            corrected_image, 
            doc_type_normalized, 
            loader
        )
        
        fields = extraction_result.get("fields", {})
        extraction_message = extraction_result.get("message")
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        extraction_message = str(e)
        
    return {
        "filename": filename,
        "document_type": doc_type_mapped,
        "confidence": confidence,
        "is_valid": is_valid,
        "fields": fields,
        "extraction_message": extraction_message
    }

# Legacy alias
classify_and_extract = None # Async wrapper removed, logic is now sync. Endpoints must use run_in_threadpool
