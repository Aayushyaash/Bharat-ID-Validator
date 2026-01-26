from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch, AsyncMock
import pytest

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_model_loader():
    with patch("app.services.model_loader.ModelLoader.load_models"):
        yield

def test_low_confidence_handling():
    """
    Scenario: Model returns low confidence.
    Expected: 'unknown' doc type, is_valid=False.
    """
    with patch("app.services.pipeline.run_classification", return_value=("Aadhaar_Front", 0.50)), \
         patch("app.services.pipeline.detect_orientation", return_value=(None, 0)), \
         patch("app.services.image_utils.read_image_file", new_callable=AsyncMock):
        
        response = client.post(
            "/api/v1/documents/classify",
            files={"file": ("test.jpg", b"fake", "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["document_type"] == "unknown"
        assert data["is_valid"] is False
        assert data["confidence"] == 0.50

def test_model_failure_graceful_degradation():
    """
    Scenario: ModelLoader has no models loaded (None).
    Expected: 'unknown' doc type, 0.0 confidence, 200 OK.
    """
    # We don't need to patch run_classification here; we need to mock the loader passed to pipeline
    # But our endpoint injects the global loader. We can patch get_model_loader or manipulate the mock.
    # The easiest way is to ensure the injected loader returns None for models.
    
    # Create a mock loader with None models
    mock_loader = AsyncMock()
    mock_loader.yolo_model = None
    mock_loader.doc_orientation_model = None
    
    # Patch the dependency
    app.dependency_overrides["app.services.model_loader.get_model_loader"] = lambda: mock_loader
    
    try:
        with patch("app.services.image_utils.read_image_file", new_callable=AsyncMock):
            response = client.post(
                "/api/v1/documents/classify",
                files={"file": ("test.jpg", b"fake", "image/jpeg")}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["document_type"] == "unknown"
            assert data["confidence"] == 0.0
            assert data["is_valid"] is False
    finally:
        app.dependency_overrides = {}

def test_trace_id_injection():
    """
    Scenario: Send a request.
    Expected: Response headers contain X-Trace-ID.
    """
    with patch("app.services.pipeline.run_classification", return_value=("Aadhaar_Front", 0.99)), \
         patch("app.services.pipeline.detect_orientation", return_value=(None, 0)), \
         patch("app.services.image_utils.read_image_file", new_callable=AsyncMock):
         
         response = client.post(
            "/api/v1/documents/classify",
            files={"file": ("test.jpg", b"fake", "image/jpeg")}
         )
         
         assert "X-Trace-ID" in response.headers
         assert len(response.headers["X-Trace-ID"]) > 0
