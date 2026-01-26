from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Id Validator Service is running"}

    @patch("app.services.pipeline.classify_document")
    def test_classify_document_valid(mock_classify):
        # Mock return value
        mock_classify.return_value = {
            "filename": "test.jpg",
            "document_type": "aadhaar",
            "confidence": 0.99,
            "is_valid": True
        }
        
        # Create dummy image bytes
        file_content = b"fake image data"
        
        response = client.post(
            "/api/v1/documents/classify",
            files={"file": ("test.jpg", file_content, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["document_type"] == "aadhaar"
        assert data["is_valid"] is True
        # orientation_angle was removed from response schema
def test_classify_document_invalid_file_type():
    response = client.post(
        "/api/v1/documents/classify",
        files={"file": ("test.txt", b"text data", "text/plain")}
    )
    assert response.status_code == 400
