import json
import logging
from pathlib import Path
from ultralytics import YOLO
from paddleocr import TextRecognition, DocImgOrientationClassification, PaddleOCR

from app.core.config import settings

logger = logging.getLogger(__name__)

class ModelLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.model_config = self._load_config()
        self.detection_models: dict[str, YOLO] = {}
        self.class_mappings: dict[str, dict[int, str]] = {}
        
        # Specific PaddleOCR Modules
        self.doc_orientation_model: DocImgOrientationClassification | None = None
        self.ocr_rec_model: TextRecognition | None = None
        
        # Full PaddleOCR Pipeline (fallback for multi-line text)
        self.paddleocr_model = None  # Will be PaddleOCR instance
        
        self._initialized = True

    def _load_config(self):
        if not settings.MODEL_CONFIG_PATH.exists():
            logger.warning(f"Config file not found at {settings.MODEL_CONFIG_PATH}")
            return {}
        with open(settings.MODEL_CONFIG_PATH, 'r') as f:
            return json.load(f)

    def load_models(self):
        logger.info("Loading models...")
        
        # 1. Load YOLO Models from Config
        models_cfg = self.model_config.get("models", {})
        
        for model_name, cfg in models_cfg.items():
            path_str = cfg.get("path")
            
            if path_str:
                # Resolve path relative to project root (settings.BASE_DIR)
                model_path = settings.BASE_DIR / path_str
                
                if model_path.exists():
                    logger.info(f"Loading YOLO model: {model_name} from {model_path}")
                    try:
                        model = YOLO(str(model_path))
                        self.detection_models[model_name] = model
                        
                        # Store explicit classes from config if provided
                        explicit_classes = cfg.get("classes")
                        if explicit_classes and isinstance(explicit_classes, list):
                            # Map list index to class name: {0: "Name", 1: "DOB"...}
                            self.class_mappings[model_name] = {i: name for i, name in enumerate(explicit_classes)}
                            
                    except Exception as e:
                        logger.error(f"Failed to load {model_name}: {e}")
                else:
                    logger.error(f"Model file not found: {model_path}")

        # 2. Load Document Orientation Model (Endpoint 1)
        logger.info("Loading Document Orientation Model (PP-LCNet_x1_0_doc_ori)...")
        try:
            self.doc_orientation_model = DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori")
            logger.info("Document Orientation Model loaded.")
        except Exception as e:
            logger.error(f"Failed to load Document Orientation Model: {e}")
            raise # Fail fast if core model fails

        # 3. Load Text Recognition Model (Endpoint 2)
        logger.info("Loading Text Recognition Model (PP-OCRv5_server_rec)...")
        try:
            self.ocr_rec_model = TextRecognition(model_name="PP-OCRv5_server_rec")
            logger.info("Text Recognition Model loaded.")
        except Exception as e:
            logger.error(f"Failed to load Text Recognition Model: {e}")
            raise # Fail fast

        # 4. Load Full PaddleOCR Pipeline (fallback for multi-line text)
        logger.info("Loading full PaddleOCR pipeline for fallback...")
        try:
            self.paddleocr_model = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=True,
                use_textline_orientation=False,
                text_detection_model_name="PP-OCRv5_server_det",
                text_recognition_model_name="PP-OCRv5_server_rec",
                lang='en'
            )
            logger.info("Full PaddleOCR pipeline loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load full PaddleOCR pipeline: {e}")
            raise

        logger.info("Models loaded.")

    @property
    def yolo_model(self):
        """Legacy accessor for Phase 1 pipeline compatibility"""
        return self.detection_models.get("Id_Classifier")

model_loader = ModelLoader()

def get_model_loader() -> ModelLoader:
    return model_loader