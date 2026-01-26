import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from app.services.extraction import extract_data
from app.services.model_loader import ModelLoader

@pytest.fixture
def mock_loader():
    loader = MagicMock(spec=ModelLoader)
    loader.model_config = {
        "models": {
            "Aadhaar": {
                "output_mapping": {
                    "Aadhaar_Name": "Name",
                    "0": "DOB"
                }
            }
        },
        "doc_type_to_model": {"aadhaar_front": "Aadhaar"},
        "field_filters": {"aadhaar_front": ["Name", "DOB"]},
        "output_schema": {
            "aadhaar_front": {
                "fields": ["Name", "DOB"],
                "field_mapping": {
                    "Name": "Name",
                    "DOB": "DOB"
                }
            }
        }
    }
    loader.detection_models = {}
    loader.ocr_rec_model = MagicMock()
    loader.doc_orientation_model = MagicMock()
    return loader

@pytest.fixture
def mock_detection_model():
    model = MagicMock()
    # Create a mock result object structure mimicking Ultralytics YOLO result
    mock_result = MagicMock()
    mock_result.boxes = []
    mock_result.names = {0: "Name", 1: "DOB", 2: "Address"} # Address not in allowed list
    model.return_value = [mock_result]
    return model

def test_field_filtering(mock_loader, mock_detection_model):
    """Verify that fields not in allowed list are skipped."""
    # Disable output mapping for this test to test filtering logic in isolation
    mock_loader.model_config["models"]["Aadhaar"]["output_mapping"] = {}
    
    mock_loader.detection_models["Aadhaar"] = mock_detection_model
    
    # Mock boxes: Class 0 (Name), Class 2 (Address)
    # Address should be filtered out
    box1 = MagicMock()
    box1.cls = np.array([0.0])
    box1.xyxy = np.array([[10, 10, 50, 50]])
    
    box2 = MagicMock()
    box2.cls = np.array([2.0]) # Address
    box2.xyxy = np.array([[60, 60, 100, 100]])
    
    mock_detection_model.return_value[0].boxes = [box1, box2]
    
    # Mock OCR response
    result_obj = MagicMock()
    result_obj.rec_text = "Test Value"
    mock_loader.ocr_rec_model.predict.return_value = [result_obj]
    
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    
    result = extract_data(image, "aadhaar_front", mock_loader)
    
    assert "Name" in result["fields"]
    assert "Address" not in result["fields"]
    assert result["fields"]["Name"] == "Test Value"

def test_output_mapping(mock_loader, mock_detection_model):
    """Verify that raw labels are mapped correctly."""
    # Update filters to accept raw labels
    mock_loader.model_config["field_filters"]["aadhaar_front"] = ["Aadhaar_Name", "0"]
    
    mock_loader.detection_models["Aadhaar"] = mock_detection_model
    
    # Update names to reflect raw model output (e.g. "Aadhaar_Name", "0")
    mock_detection_model.return_value[0].names = {0: "Aadhaar_Name", 1: "0", 2: "Unknown"}
    
    # Box 1: "Aadhaar_Name" -> Should map to "Name"
    box1 = MagicMock()
    box1.cls = np.array([0.0])
    box1.xyxy = np.array([[10, 10, 50, 50]])
    
    # Box 2: "0" -> Should map to "DOB"
    box2 = MagicMock()
    box2.cls = np.array([1.0])
    box2.xyxy = np.array([[60, 60, 100, 100]])

    mock_detection_model.return_value[0].boxes = [box1, box2]
    
    # Mock OCR
    result_obj = MagicMock()
    result_obj.rec_text = "Mapped Value"
    mock_loader.ocr_rec_model.predict.return_value = [result_obj]
    
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    
    result = extract_data(image, "aadhaar_front", mock_loader)
    
    assert "Name" in result["fields"]
    assert "DOB" in result["fields"]

def test_rotation_logic(mock_loader, mock_detection_model):
    """Verify rotation occurs when h > 2.5 * w."""
    mock_loader.detection_models["Aadhaar"] = mock_detection_model
    
    # Create a box that yields a tall crop: 10x100 (w=10, h=90) -> h > 2.5*w
    box1 = MagicMock()
    box1.cls = np.array([0.0])
    box1.xyxy = np.array([[10, 10, 20, 100]]) # x1, y1, x2, y2
    
    mock_detection_model.return_value[0].boxes = [box1]
    
    # Mock OCR
    result_obj = MagicMock()
    result_obj.rec_text = "Rotated"
    mock_loader.ocr_rec_model.predict.return_value = [result_obj]
    
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # We mock cv2.rotate to verify it's called
    with patch('cv2.rotate', wraps=lambda img, code: img) as mock_rotate:
        extract_data(image, "aadhaar_front", mock_loader)
        mock_rotate.assert_called_once()

def test_no_boxes_detected(mock_loader, mock_detection_model):
    """Verify early exit when no boxes are detected."""
    mock_loader.detection_models["Aadhaar"] = mock_detection_model
    mock_detection_model.return_value[0].boxes = []
    
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    result = extract_data(image, "aadhaar_front", mock_loader)
    
    assert result["message"] == "No valid bounding boxes detected"
    assert result["fields"] == {}

def test_invalid_doc_type(mock_loader):
    """Verify error raised for invalid document type."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    with pytest.raises(ValueError, match="Unsupported document type"):
        extract_data(image, "invalid_type", mock_loader)