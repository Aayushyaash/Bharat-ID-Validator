import numpy as np
import cv2
import logging
from app.services.model_loader import ModelLoader
from app.services import pipeline as pipeline  # Changed from pipeline_v2
from app.services import text_utils, image_utils
from app.core.config import settings

logger = logging.getLogger(__name__)

def format_output(
    extracted_fields: dict[str, str],
    doc_type: str,
    config: dict
) -> dict[str, str]:
    """
    Format extracted fields according to document type schema.
    
    Maps internal field names to display names and orders fields according
    to the output schema defined in config. Missing fields are filled with "N/A".
    
    Args:
        extracted_fields: Dictionary of {internal_field_name: extracted_text}
        doc_type: Document type key for schema lookup (e.g., 'aadhaar_front')
        config: Model configuration dictionary containing output_schema
        
    Returns:
        Ordered dictionary of {display_name: value} with all expected fields
        
    Example:
        Input: {'name': 'John Doe', 'dob': '01/01/1990'}
        Output: {'Name': 'John Doe', 'DOB': '01/01/1990', 'Gender': 'N/A', ...}
    """
    output_schema = config.get("output_schema", {}).get(doc_type, {})
    field_order = output_schema.get("fields", [])
    field_mapping = output_schema.get("field_mapping", {})
    
    # Create reverse mapping (display name -> internal name)
    reverse_mapping = {v: k for k, v in field_mapping.items()}
    
    # Build ordered output
    formatted_output = {}
    for display_name in field_order:
        # Check if this display name has an internal name mapping
        internal_name = reverse_mapping.get(display_name, display_name)
        
        # Get value or "N/A"
        value = extracted_fields.get(internal_name, "N/A")
        formatted_output[display_name] = value
    
    return formatted_output

def _validate_and_get_model(doc_type: str, loader: ModelLoader) -> tuple:
    """
    Validates document type and retrieves necessary model configuration.
    
    Performs validation to ensure the document type is supported and all
    required models are loaded before extraction begins.
    
    Args:
        doc_type: Document type key (e.g., 'aadhaar_front', 'pan_card')
        loader: ModelLoader instance with models and configuration
        
    Returns:
        Tuple of (detection_model, allowed_fields, output_mapping):
            - detection_model: YOLO detection model for this document type
            - allowed_fields: Set of field names to extract
            - output_mapping: Dict mapping internal names to display names
            
    Raises:
        ValueError: If document type is not supported
        RuntimeError: If required detection or OCR model is not loaded
    """
    config = loader.model_config
    doc_mapping = config.get("doc_type_to_model", {})
    field_filters = config.get("field_filters", {})

    if doc_type not in doc_mapping:
        raise ValueError(f"Unsupported document type: {doc_type}")

    model_key = doc_mapping[doc_type]
    allowed_fields = set(field_filters.get(doc_type, []))
    
    # Retrieve Output Mapping
    model_config_entry = config.get("models", {}).get(model_key, {})
    output_mapping = model_config_entry.get("output_mapping", {})
    
    # Retrieve Models
    detection_model = loader.detection_models.get(model_key)
    if not detection_model:
        raise RuntimeError(f"Detection model '{model_key}' not loaded.")
    
    if not loader.ocr_rec_model:
        raise RuntimeError("OCR Recognition model not loaded.")
        
    return detection_model, allowed_fields, output_mapping

def _detect_and_deduplicate(
    image: np.ndarray, 
    detection_model, 
    allowed_fields: set, 
    output_mapping: dict,
    doc_type: str
) -> dict:
    """
    Runs YOLO detection and performs two-stage deduplication.
    
    Stage 1: Keeps only the highest confidence bounding box per raw class name
    Stage 2: Filters by allowed fields and maps to display names, keeping highest
             confidence if multiple raw labels map to the same display name
    
    Args:
        image: Input document image as numpy array (BGR format)
        detection_model: YOLO detection model for field detection
        allowed_fields: Set of raw field names to keep
        output_mapping: Dict mapping raw field names to display names
        doc_type: Document type for logging purposes
        
    Returns:
        Dictionary of {display_name: (box, confidence, raw_label)} for each detected field
        Returns empty dict if no valid bounding boxes detected
    """
    results = detection_model(image)
    
    if not results or len(results[0].boxes) == 0:
        logger.warning(f"No bounding boxes detected for {doc_type}")
        return {}

    r = results[0]
    boxes = r.boxes
    names = r.names

    # Stage 1: Keep only highest confidence box per RAW class name
    best_boxes_raw = {}  # {raw_class_name: (box, confidence)}
    
    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        raw_label = names[cls_id]
        
        if raw_label not in best_boxes_raw or conf > best_boxes_raw[raw_label][1]:
            best_boxes_raw[raw_label] = (box, conf)

    # Stage 2: Filter and Map to Display Name
    filtered_boxes = {}  # {display_name: (box, confidence, raw_label)}
    
    for raw_label, (box, conf) in best_boxes_raw.items():
        if raw_label not in allowed_fields:
            continue
        
        display_name = output_mapping.get(raw_label, raw_label)
        
        if display_name not in filtered_boxes or conf > filtered_boxes[display_name][1]:
            filtered_boxes[display_name] = (box, conf, raw_label)
            
    return filtered_boxes

def _process_single_roi(image: np.ndarray, box, display_name: str, loader: ModelLoader) -> str:
    """
    Process a single Region of Interest (ROI) through the OCR pipeline.
    
    Implements a two-phase OCR strategy:
    - Phase 1: Fast TextRecognition (~100-300ms) for single-line fields
    - Phase 2: Full PaddleOCR with detection (fallback for multi-line text)
    
    Also handles vertical text detection using aspect ratio heuristics.
    
    Args:
        image: Full document image as numpy array (BGR format)
        box: YOLO detection box with xyxy coordinates
        display_name: Field name for logging purposes
        loader: ModelLoader instance with OCR models
        
    Returns:
        Extracted text string, or empty string if extraction fails
        
    Note:
        Phase 2 fallback adds ~1-3 seconds but significantly improves accuracy
        for multi-line addresses and complex text layouts.
    """
    # Get coordinates
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    
    # Ensure valid crop
    h_img, w_img = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    
    if x2 <= x1 or y2 <= y1:
        return ""

    crop = image[y1:y2, x1:x2]
    
    # Heuristic Rotation (Vertical text)
    h_crop, w_crop = crop.shape[:2]
    if w_crop > 0 and h_crop > 2.5 * w_crop:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        
    text_val = ""
    
    # Phase 1: Fast TextRecognition
    try:
        rec_result = loader.ocr_rec_model.predict(crop)
        if rec_result:
            if isinstance(rec_result, list) and len(rec_result) > 0:
                result_obj = rec_result[0]
                if hasattr(result_obj, 'rec_text'):
                    text_val = result_obj.rec_text
                elif isinstance(result_obj, dict) and 'rec_text' in result_obj:
                    text_val = result_obj['rec_text']
                elif isinstance(result_obj, tuple):
                    text_val = result_obj[0]
                elif isinstance(result_obj, str):
                    text_val = result_obj
    except Exception as e:
        logger.error(f"Phase 1 OCR failed for {display_name}: {e}")

    # Phase 2: PaddleOCR Fallback
    if not text_val and loader.paddleocr_model:
        logger.info(f"Triggering PaddleOCR fallback for '{display_name}'")
        try:
            extended_crop = image_utils.extend_image(crop, padding_h=1, padding_w=20)
            paddle_result = loader.paddleocr_model.predict(extended_crop)
            
            if paddle_result:
                rec_texts = []
                for res in paddle_result:
                    if hasattr(res, 'rec_texts'):
                        rec_texts.extend(res.rec_texts)
                    elif isinstance(res, dict) and 'rec_texts' in res:
                        rec_texts.extend(res['rec_texts'])
                
                if rec_texts:
                    raw_text = " ".join(rec_texts)
                    text_val = text_utils.format_ocr_text(raw_text)
        except Exception as e:
            logger.error(f"Phase 2 OCR failed for {display_name}: {e}")
            
    return text_val

def extract_data_v2(image: np.ndarray, doc_type: str, loader: ModelLoader) -> dict:
    """
    Main extraction pipeline for document field extraction.
    
    Performs the complete extraction workflow:
    1. Validates document type and loads appropriate models
    2. Corrects document orientation (if orientation model available)
    3. Detects field bounding boxes using YOLO
    4. Applies two-stage deduplication (by raw label, then by display name)
    5. Extracts text from each ROI using two-phase OCR
    6. Formats output according to document schema
    
    Args:
        image: Input document image as numpy array (BGR format)
        doc_type: Document type key (e.g., 'aadhaar_front', 'pan_card', 'voter_id')
        loader: ModelLoader instance with all required models
        
    Returns:
        Dictionary containing:
            - document_type: The input document type
            - fields: Dict of {display_name: extracted_value} with "N/A" for missing fields
            - message: Optional status message (present if no boxes detected)
            
    Raises:
        ValueError: If document type is not supported
        RuntimeError: If required models are not loaded
    """
    # 1. Validation
    detection_model, allowed_fields, output_mapping = _validate_and_get_model(doc_type, loader)
    
    # 2. Orientation Correction
    if loader.doc_orientation_model:
        try:
            image, _ = pipeline.detect_orientation(image, loader)
        except Exception as e:
            logger.warning(f"Orientation correction failed: {e}")

    # 3. Detection & Deduplication
    filtered_boxes = _detect_and_deduplicate(image, detection_model, allowed_fields, output_mapping, doc_type)
    
    if not filtered_boxes:
         return {
            "document_type": doc_type,
            "fields": {}, 
            "message": "No valid bounding boxes detected"
        }

    # 4. OCR Processing
    extracted_fields = {}
    for display_name, (box, conf, raw_label) in filtered_boxes.items():
        text_val = _process_single_roi(image, box, display_name, loader)
        if text_val:
            extracted_fields[display_name] = text_val
        else:
            logger.warning(f"Empty text extracted for field {display_name}")

    # 5. Formatting
    formatted_fields = format_output(extracted_fields, doc_type, loader.model_config)
    
    return {
        "document_type": doc_type,
        "fields": formatted_fields
    }

# Main entry point name match
extract_data = extract_data_v2
