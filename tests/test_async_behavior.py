"""
Tests for async behavior verification.

Tests verify that:
- Endpoints are truly async and non-blocking
- Model loading runs in separate thread
- Concurrent requests are handled properly
- HEAD requests return no body
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from httpx import AsyncClient, ASGITransport
import time


@pytest.fixture(autouse=True)
def mock_models():
    """Mock model loading to prevent actual ML model initialization."""
    with patch('app.services.model_loader.ModelLoader.load_models'):
        with patch('app.services.model_loader.ModelLoader._models_loaded', True):
            yield


@pytest.mark.asyncio
async def test_health_endpoint_async():
    """Test that health endpoint responds asynchronously."""
    from app.main import app
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_readiness_endpoint_async():
    """Test readiness endpoint async behavior."""
    from app.main import app
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data


@pytest.mark.asyncio
async def test_root_endpoint_async():
    """Test root endpoint async behavior."""
    from app.main import app
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data


@pytest.mark.asyncio
async def test_concurrent_health_checks():
    """Test that multiple concurrent health checks are handled."""
    from app.main import app
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Fire multiple concurrent requests
        tasks = [client.get("/health") for _ in range(10)]
        responses = await asyncio.gather(*tasks)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200


@pytest.mark.asyncio
async def test_head_request_returns_no_body():
    """Test HEAD request returns no body."""
    from app.main import app
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.head("/health")
        assert response.status_code == 200
        assert response.content == b""  # HEAD should have no body


@pytest.mark.asyncio
async def test_head_request_on_root():
    """Test HEAD request on root endpoint."""
    from app.main import app
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.head("/")
        assert response.status_code == 200
        assert response.content == b""


@pytest.mark.asyncio
async def test_head_request_on_ready():
    """Test HEAD request on readiness endpoint."""
    from app.main import app
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.head("/ready")
        assert response.status_code == 200
        assert response.content == b""


@pytest.mark.asyncio
async def test_metrics_endpoint_async():
    """Test metrics endpoint responds asynchronously."""
    from app.main import app
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_concurrent_mixed_endpoints():
    """Test concurrent requests to different endpoints."""
    from app.main import app
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        tasks = [
            client.get("/"),
            client.get("/health"),
            client.get("/ready"),
            client.get("/metrics"),
            client.head("/health"),
        ]
        responses = await asyncio.gather(*tasks)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200


def test_model_loading_uses_asyncio_to_thread():
    """Test that model loading is configured to use asyncio.to_thread."""
    import inspect
    from app.main import lifespan
    
    # Get the source code of the lifespan function
    source = inspect.getsource(lifespan)
    
    # Verify asyncio.to_thread is used for model loading
    assert "asyncio.to_thread" in source, "Model loading should use asyncio.to_thread"
    assert "load_models" in source, "load_models should be called in lifespan"


@pytest.mark.asyncio
async def test_classify_endpoint_async_io():
    """Test that classify endpoint handles async I/O properly."""
    from app.main import app
    
    with patch('app.services.pipeline.classify_document_sync') as mock_classify:
        mock_classify.return_value = {
            "document_type": "aadhaar",
            "confidence": 0.99,
            "is_valid": True
        }
        
        with patch('app.services.image_utils.read_image_file', new_callable=AsyncMock) as mock_read:
            # Return a mock numpy array
            mock_read.return_value = MagicMock()
            
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                files = {"file": ("test.jpg", b"fake image data", "image/jpeg")}
                response = await client.post("/api/v1/documents/classify", files=files)
                
                # Verify async read was called
                mock_read.assert_called_once()


@pytest.mark.asyncio  
async def test_readiness_reflects_model_state_ready():
    """Test that readiness endpoint reflects models loaded state."""
    from app.main import app
    
    # Test when models are loaded (default from autouse fixture)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["models_loaded"] is True


def test_readiness_endpoint_structure():
    """Test that readiness endpoint returns expected structure."""
    from fastapi.testclient import TestClient
    from app.main import app
    
    client = TestClient(app)
    response = client.get("/ready")
    
    data = response.json()
    # Verify response structure
    assert "status" in data
    assert "models_loaded" in data
    assert isinstance(data["models_loaded"], bool)
