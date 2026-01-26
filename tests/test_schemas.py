import pytest
from app.schemas.document import DocumentResponse

def test_document_response_instantiation():
    """
    Scenario: Instantiate response with valid data.
    """
    resp = DocumentResponse(
        filename="test.jpg",
        document_type="aadhaar",
        confidence=0.99,
        is_valid=True
    )
    assert resp.document_type == "aadhaar"
    assert resp.confidence == 0.99
    assert resp.is_valid is True