from app.services.model_loader import ModelLoader
from app.services import image_utils
from app.core.config import settings
import logging
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)

def _extract_class_id_from_paddle_result(res_item: Any) -> int:
    """
    Helper to safely extract class ID from PaddleOCR's variable output format.
    """
    class_id = 0
    if isinstance(res_item, dict):
        class_id = res_item.get('class_ids', [0])[0]
    elif hasattr(res_item, 'keys'):
        if 'class_ids' in res_item:
            class_id = res_item['class_ids'][0]
    else:
        if hasattr(res_item, 'json'):
            res_json = res_item.json
            if isinstance(res_json, dict) and 'class_ids' in res_json:
                    class_id = res_json['class_ids'][0]
        else:
            class_id = getattr(res_item, 'class_ids', [0])[0]
    return class_id

def detect_orientation(image: np.ndarray, loader: ModelLoader) -> tuple[np.ndarray, int]:
    """
    Detects document orientation and corrects rotation if necessary.
    
    Uses PaddleOCR's document orientation model to detect if the document
    is rotated (0°, 90°, 180°, or 270°) and applies the inverse rotation
    to correct it.
    
    Args:
        image: Input document image as numpy array (BGR format)
        loader: ModelLoader instance with doc_orientation_model loaded
        
    Returns:
        Tuple of (corrected_image, detected_angle):
            - corrected_image: Orientation-corrected image (or original if no correction needed)
            - detected_angle: The detected rotation angle (0, 90, 180, or 270)
            
    Note:
        Returns original image unchanged if:
            - Orientation model is not loaded
            - Detection fails with an exception
            - Document is already upright (0° detected)
    """
    orientation_angle = 0
    corrected_image = image
    
    if not loader.doc_orientation_model:
        logger.warning("Document Orientation Model not loaded. Skipping orientation detection.")
        return corrected_image, orientation_angle

    try:
        # Predict using the dedicated document orientation model
        result = loader.doc_orientation_model.predict(image)
        
        if result:
             res_item = result[0]
             class_id = _extract_class_id_from_paddle_result(res_item)

             # Map class_id to angle
             # 0 -> 0, 1 -> 90, 2 -> 180, 3 -> 270
             angle_map = {0: 0, 1: 90, 2: 180, 3: 270}
             detected_angle = angle_map.get(class_id, 0)

             if detected_angle != 0:
                logger.info(f"Detected rotation: {detected_angle}. Correcting...")
                # Correction logic
                correction_angle = 0
                if detected_angle == 90:
                    correction_angle = 270 # or -90
                elif detected_angle == 180:
                    correction_angle = 180
                elif detected_angle == 270:
                    correction_angle = 90
                
                if correction_angle != 0:
                    corrected_image = image_utils.rotate_image(image, correction_angle)
    
    except Exception as e:
        logger.error(f"Document Orientation processing error: {e}", exc_info=True)
        # Return original image on error
        return image, 0

    return corrected_image, orientation_angle

def run_classification(image: np.ndarray, loader: ModelLoader) -> tuple[str, float]:
    """
    Runs the YOLO classification model on a document image.
    
    Performs inference using the loaded YOLO model and extracts the
    predicted document type and confidence score. Handles both
    classification-style (probs) and detection-style (boxes) YOLO outputs.
    
    Args:
        image: Input document image as numpy array (BGR format, should be orientation-corrected)
        loader: ModelLoader instance with yolo_model loaded
        
    Returns:
        Tuple of (document_type, confidence):
            - document_type: Detected document class name (e.g., 'Aadhaar_Front', 'Pan_Card')
                            Returns 'unknown' if classification fails
            - confidence: Confidence score between 0.0 and 1.0
                         Returns 0.0 if classification fails
                         
    Note:
        Does NOT perform orientation correction - caller should pre-process image
        using detect_orientation() if needed.
    """
    doc_type = "unknown"
    confidence = 0.0
    
    if not loader.yolo_model:
        logger.error("YOLO model not loaded. Cannot classify.")
        return doc_type, confidence

    try:
        results = loader.yolo_model(image)
        
        if results:
            r = results[0]
            # Check if it's classification (probs) or detection (boxes)
            if hasattr(r, 'probs') and r.probs is not None:
                top1 = r.probs.top1
                conf = r.probs.top1conf.item()
                class_id = int(top1)
                confidence = conf
                
                # Prefer model's internal names
                if hasattr(r, 'names') and r.names:
                    doc_type = r.names.get(class_id, str(class_id))
                else:
                    names = loader.model_config.get("classes", {})
                    doc_type = names.get(str(class_id), str(class_id))
                
                logger.info(f"Model detected specific type: {doc_type}")

            elif hasattr(r, 'boxes') and r.boxes is not None:
                if len(r.boxes) > 0:
                    best_idx = r.boxes.conf.argmax().item()
                    conf = r.boxes.conf[best_idx].item()
                    cls_id = int(r.boxes.cls[best_idx].item())
                    
                    confidence = conf
                    names = r.names 
                    doc_type = names.get(cls_id, str(cls_id))
                    logger.info(f"Model detected specific type (box): {doc_type}")

    except Exception as e:
        logger.error(f"YOLO inference error: {e}", exc_info=True)
        
    return doc_type, confidence

def format_response(doc_type: str) -> str:
    """
    Generalizes document type string for API response.
    
    Converts specific document type variants (e.g., 'Aadhaar_Front', 'Aadhaar_Back')
    into generalized types (e.g., 'aadhaar') for consistent API responses.
    Also handles spelling normalization (e.g., 'aadhar' -> 'aadhaar').
    
    Args:
        doc_type: Raw document type string from classification model
        
    Returns:
        Generalized document type string (lowercase, no front/back suffix)
        
    Examples:
        >>> format_response("Aadhaar_Front")
        'aadhaar'
        >>> format_response("Pan_Card")
        'pan_card'
        >>> format_response("unknown")
        'unknown'
    """
    raw_type = doc_type.lower()
    final_doc_type = raw_type.replace("_front", "").replace("_back", "")

    if "aadhar" in final_doc_type:
        final_doc_type = "aadhaar"
    elif "pan" in final_doc_type:
        final_doc_type = "pan_card"
    elif final_doc_type == "unknown":
        pass # Keep as unknown
        
    return final_doc_type

def classify_document_sync(image: np.ndarray, loader: ModelLoader) -> dict:
    """
    Complete synchronous document classification pipeline.
    
    Performs the full classification workflow: orientation detection,
    YOLO classification, confidence threshold validation, and response formatting.
    
    Args:
        image: Input document image as numpy array (BGR format)
        loader: ModelLoader instance with all required models loaded
        
    Returns:
        Dictionary containing:
            - document_type: Generalized type ('aadhaar', 'pan_card', etc.) or 'unknown'
            - confidence: Classification confidence score (0.0 to 1.0)
            - is_valid: True if confidence >= CONFIDENCE_THRESHOLD
            
    Raises:
        Exception: Re-raises any exception from orientation detection or classification
        
    Note:
        Returns 'unknown' as document_type if confidence is below threshold,
        even if the model detected a specific document type.
    """
    try:
        # 1. Orientation
        corrected_image, _ = detect_orientation(image, loader)

        # 2. Classification
        doc_type, confidence = run_classification(corrected_image, loader)

        # 3. Validation & Threshold Logic
        is_valid = confidence >= settings.CONFIDENCE_THRESHOLD
        logger.info(f"Classification result: detected_type={doc_type}, confidence={confidence:.4f}, valid={is_valid}")

        # 4. Response Formatting
        if not is_valid:
            logger.warning(f"Confidence {confidence:.4f} below threshold {settings.CONFIDENCE_THRESHOLD}. Returning 'unknown'.")
            final_doc_type = "unknown"
        else:
            final_doc_type = format_response(doc_type)

        return {
            "document_type": final_doc_type,
            "confidence": confidence,
            "is_valid": is_valid
        }
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise e
