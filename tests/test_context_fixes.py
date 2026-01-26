from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch, AsyncMock
import pytest

client = TestClient(app)

# Mock ModelLoader to prevent actual model loading during tests
@pytest.fixture(autouse=True)
def mock_model_loader():
    with patch("app.services.model_loader.ModelLoader.load_models") as mock_load:
        yield mock_load

def test_threshold_logic_low_confidence():
    # Mock pipeline to simulate low confidence
    # pipeline.classify_document_sync is what's called now
    with patch("app.services.pipeline.classify_document_sync") as mock_classify:
        mock_classify.return_value = {
            "document_type": "unknown",
            "confidence": 0.50,
            "is_valid": False
        }
        
        # We also need to mock read_image_file since we are hitting the endpoint
        with patch("app.services.image_utils.read_image_file", new_callable=AsyncMock) as mock_read:
            mock_read.return_value = b"some numpy array" # Dummy
            
            file_content = b"fake image"
            response = client.post(
                "/api/v1/documents/classify",
                files={"file": ("test.jpg", file_content, "image/jpeg")}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["document_type"] == "unknown"
            assert data["is_valid"] is False
            assert data["confidence"] == 0.50

def test_invalid_file_type():
    # Test file upload with invalid content type/content (not an image)
    # The endpoint calls read_image_file. If it returns None, endpoint raises 400.
    
    # We rely on integration behavior of read_image_file here (it uses cv2.imdecode)
    # So we don't mock read_image_file, we let it run on bad input.
    
    response = client.post(
        "/api/v1/documents/classify",
        files={"file": ("test.txt", b"text content", "text/plain")}
    )
    assert response.status_code == 400
    error = response.json()
    assert error["code"] == "INVALID_FILE_TYPE"
    assert "File must be an image" in str(error["message"])

def test_validator_enforcement():
    # This tests the schema validator.
    from app.schemas.document import DocumentResponse
    
    # Valid
    DocumentResponse(filename="x", document_type="aadhaar", confidence=1.0, is_valid=True)
    
    # Invalid type check (Pydantic validates types)
    with pytest.raises(ValueError):
        DocumentResponse(filename="x", document_type="aadhaar", confidence="not_a_float", is_valid=True)