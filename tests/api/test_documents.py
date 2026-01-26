import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import io
import cv2
import numpy as np
from app.main import app

client = TestClient(app)

@pytest.fixture
def mock_extract_data():
    with patch("app.services.extraction.extract_data") as mock:
        yield mock

def create_dummy_image():
    """Create a valid JPEG image in memory."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode('.jpg', img)
    return io.BytesIO(img_encoded.tobytes())

def test_extract_aadhaar_front(mock_extract_data):
    """Test successful extraction for Aadhaar Front."""
    mock_extract_data.return_value = {
        "document_type": "aadhaar_front",
        "fields": {
            "Aadhaar": "1234 5678 9012",
            "Name": "John Doe",
            "DOB": "01/01/1990",
            "Gender": "Male"
        }
    }
    
    file = create_dummy_image()
    
    response = client.post(
        "/api/v1/documents/extract",
        files={"file": ("test.jpg", file, "image/jpeg")},
        data={"document_type": "aadhaar_front"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["document_type"] == "aadhaar_front"
    assert data["fields"]["Name"] == "John Doe"
    assert "Address" not in data["fields"]

def test_extract_aadhaar_back(mock_extract_data):
    """Test successful extraction for Aadhaar Back."""
    mock_extract_data.return_value = {
        "document_type": "aadhaar_back",
        "fields": {
            "Aadhaar": "1234 5678 9012",
            "Address": "123 Main St"
        }
    }
    
    file = create_dummy_image()
    
    response = client.post(
        "/api/v1/documents/extract",
        files={"file": ("test.jpg", file, "image/jpeg")},
        data={"document_type": "aadhaar_back"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["fields"]["Address"] == "123 Main St"

def test_extract_no_boxes(mock_extract_data):
    """Test scenario where no boxes are detected."""
    mock_extract_data.return_value = {
        "document_type": "aadhaar_front",
        "fields": {},
        "message": "No valid bounding boxes detected"
    }
    
    file = create_dummy_image()
    
    response = client.post(
        "/api/v1/documents/extract",
        files={"file": ("test.jpg", file, "image/jpeg")},
        data={"document_type": "aadhaar_front"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "No valid bounding boxes detected"
    assert data["fields"] == {}

def test_extract_invalid_doc_type(mock_extract_data):
    """Test error handling for invalid doc type."""
    mock_extract_data.side_effect = ValueError("Unsupported document type: invalid")
    
    file = create_dummy_image()
    
    response = client.post(
        "/api/v1/documents/extract",
        files={"file": ("test.jpg", file, "image/jpeg")},
        data={"document_type": "invalid"}
    )
    
    assert response.status_code == 400
    assert "Unsupported document type" in response.json()["message"]

def test_extract_service_error(mock_extract_data):
    """Test error handling for runtime errors (e.g. model not loaded)."""
    mock_extract_data.side_effect = RuntimeError("Model not loaded")
    
    file = create_dummy_image()
    
    response = client.post(
        "/api/v1/documents/extract",
        files={"file": ("test.jpg", file, "image/jpeg")},
        data={"document_type": "aadhaar_front"}
    )
    
    assert response.status_code == 503
    assert "Model not loaded" in response.json()["message"]