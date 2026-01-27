"""
Edge case tests for document processing.

Tests cover:
- Corrupt/invalid images
- Zero-size bounding boxes
- Empty OCR results
- Overlapping detections
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import numpy as np
import io
import struct
import zlib


@pytest.fixture(autouse=True)
def mock_models():
    """Mock model loading to prevent actual ML model initialization."""
    with patch('app.services.model_loader.ModelLoader.load_models'):
        with patch('app.services.model_loader.ModelLoader._models_loaded', True):
            yield


@pytest.fixture
def client():
    """Create test client with mocked models."""
    from app.main import app
    return TestClient(app)


def create_minimal_png() -> bytes:
    """Create a minimal valid 1x1 PNG image."""
    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk_len = struct.pack('>I', len(data))
        chunk_crc = struct.pack('>I', zlib.crc32(chunk_type + data) & 0xffffffff)
        return chunk_len + chunk_type + data + chunk_crc
    
    signature = b'\x89PNG\r\n\x1a\n'
    ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)  # 1x1 RGB
    ihdr = png_chunk(b'IHDR', ihdr_data)
    
    raw_data = b'\x00\xff\xff\xff'  # filter byte + white pixel
    compressed = zlib.compress(raw_data)
    idat = png_chunk(b'IDAT', compressed)
    iend = png_chunk(b'IEND', b'')
    
    return signature + ihdr + idat + iend


def create_corrupt_file() -> bytes:
    """Create corrupt/invalid file data."""
    return b'not an image at all - just random bytes 12345'


class TestCorruptImages:
    """Tests for handling corrupt or invalid image files."""
    
    def test_classify_with_corrupt_file(self, client):
        """Test classification endpoint rejects corrupt files."""
        corrupt_data = create_corrupt_file()
        
        response = client.post(
            "/api/v1/documents/classify",
            files={"file": ("corrupt.jpg", io.BytesIO(corrupt_data), "image/jpeg")}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert data["code"] == "INVALID_FILE_TYPE"
    
    def test_extract_with_corrupt_file(self, client):
        """Test extraction endpoint rejects corrupt files."""
        corrupt_data = create_corrupt_file()
        
        response = client.post(
            "/api/v1/documents/extract",
            files={"file": ("corrupt.png", io.BytesIO(corrupt_data), "image/png")},
            data={"document_type": "aadhaar_front"}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert data["code"] == "INVALID_FILE_TYPE"
    
    def test_combined_with_corrupt_file(self, client):
        """Test combined endpoint rejects corrupt files."""
        corrupt_data = create_corrupt_file()
        
        response = client.post(
            "/api/v1/documents/classify-and-extract",
            files={"file": ("bad.jpg", io.BytesIO(corrupt_data), "image/jpeg")}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert data["code"] == "INVALID_FILE_TYPE"


class TestZeroSizeBoundingBoxes:
    """Tests for handling zero-size or invalid bounding boxes."""
    
    def test_extraction_with_zero_size_box(self):
        """Test that zero-size bounding boxes are handled gracefully."""
        from app.services.extraction import _process_single_roi
        
        # Create a mock image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Create a mock box with zero size (x1 == x2)
        mock_box = MagicMock()
        mock_box.xyxy = MagicMock()
        mock_box.xyxy.__getitem__ = MagicMock(return_value=MagicMock(
            tolist=MagicMock(return_value=[50, 50, 50, 50])  # Zero size box
        ))
        
        mock_loader = MagicMock()
        
        # Should return empty string, not crash
        result = _process_single_roi(image, mock_box, "TestField", mock_loader)
        assert result == ""
    
    def test_extraction_with_out_of_bounds_coordinates(self):
        """Test that out-of-bounds coordinates are clamped to valid range."""
        from app.services.extraction import _process_single_roi
        
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Mock box with coordinates outside image bounds
        mock_box = MagicMock()
        mock_box.xyxy = MagicMock()
        mock_box.xyxy.__getitem__ = MagicMock(return_value=MagicMock(
            tolist=MagicMock(return_value=[80, 80, 150, 150])  # Extends beyond 100x100
        ))
        
        mock_loader = MagicMock()
        mock_loader.ocr_rec_model = MagicMock()
        mock_loader.ocr_rec_model.predict = MagicMock(return_value=[])
        mock_loader.paddleocr_model = None
        
        # Should handle gracefully (clamp to valid bounds)
        result = _process_single_roi(image, mock_box, "TestField", mock_loader)
        assert isinstance(result, str)


class TestEmptyDetections:
    """Tests for handling empty or no detections."""
    
    def test_extraction_no_boxes_detected(self):
        """Test extraction when YOLO finds no bounding boxes."""
        from app.services.extraction import _detect_and_deduplicate
        
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Mock detection model returning empty boxes
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes = []
        mock_model.return_value = [mock_result]
        
        result = _detect_and_deduplicate(
            image, 
            mock_model, 
            {"name", "dob"}, 
            {"name": "Name", "dob": "DOB"},
            "aadhaar_front"
        )
        
        assert result == {}
    
    def test_extraction_empty_ocr_result(self):
        """Test handling when OCR returns empty results."""
        from app.services.extraction import _process_single_roi
        
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        mock_box = MagicMock()
        mock_box.xyxy = MagicMock()
        mock_box.xyxy.__getitem__ = MagicMock(return_value=MagicMock(
            tolist=MagicMock(return_value=[10, 10, 50, 50])
        ))
        
        mock_loader = MagicMock()
        mock_loader.ocr_rec_model = MagicMock()
        mock_loader.ocr_rec_model.predict = MagicMock(return_value=[])
        mock_loader.paddleocr_model = None  # No fallback
        
        result = _process_single_roi(image, mock_box, "TestField", mock_loader)
        assert result == ""
    
    def test_extraction_with_none_detection_result(self):
        """Test handling when detection model returns None."""
        from app.services.extraction import _detect_and_deduplicate
        
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        mock_model = MagicMock()
        mock_model.return_value = None
        
        result = _detect_and_deduplicate(
            image,
            mock_model,
            {"name"},
            {"name": "Name"},
            "aadhaar_front"
        )
        
        assert result == {}


class TestOverlappingDetections:
    """Tests for handling overlapping bounding box detections."""
    
    def test_deduplication_keeps_highest_confidence(self):
        """Test that deduplication keeps only the highest confidence box per field."""
        from app.services.extraction import _detect_and_deduplicate
        
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Create mock boxes with same class but different confidences
        mock_box1 = MagicMock()
        mock_box1.cls = MagicMock()
        mock_box1.cls.__getitem__ = MagicMock(return_value=MagicMock(item=MagicMock(return_value=0)))
        mock_box1.conf = MagicMock()
        mock_box1.conf.__getitem__ = MagicMock(return_value=MagicMock(item=MagicMock(return_value=0.7)))
        
        mock_box2 = MagicMock()
        mock_box2.cls = MagicMock()
        mock_box2.cls.__getitem__ = MagicMock(return_value=MagicMock(item=MagicMock(return_value=0)))
        mock_box2.conf = MagicMock()
        mock_box2.conf.__getitem__ = MagicMock(return_value=MagicMock(item=MagicMock(return_value=0.9)))
        
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes = [mock_box1, mock_box2]
        mock_result.names = {0: "name"}
        mock_model.return_value = [mock_result]
        
        result = _detect_and_deduplicate(
            image,
            mock_model,
            {"name"},
            {"name": "Name"},
            "aadhaar_front"
        )
        
        # Should keep only one entry with highest confidence
        assert len(result) == 1
        assert "Name" in result
        # Verify it's the higher confidence one (0.9)
        assert result["Name"][1] == 0.9


class TestFileSizeValidation:
    """Tests for file size validation."""
    
    def test_large_file_rejected(self, client):
        """Test that files exceeding MAX_FILE_SIZE are rejected."""
        # Create data larger than default 10MB limit
        from app.core.config import settings
        large_data = b'x' * (settings.MAX_FILE_SIZE + 1024)
        
        response = client.post(
            "/api/v1/documents/classify",
            files={"file": ("large.jpg", io.BytesIO(large_data), "image/jpeg")}
        )
        
        assert response.status_code == 413
        data = response.json()
        assert data["code"] == "FILE_TOO_LARGE"
