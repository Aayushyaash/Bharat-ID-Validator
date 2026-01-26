from pydantic import BaseModel
from typing import Any

class DocumentResponse(BaseModel):
    filename: str
    document_type: str
    confidence: float
    is_valid: bool

class ExtractionResponse(BaseModel):
    document_type: str
    fields: dict[str, str | None]
    message: str | None = None

class ClassifyAndExtractResponse(BaseModel):
    """Response model for combined classification and extraction endpoint."""
    # Classification Results
    filename: str
    document_type: str
    confidence: float
    is_valid: bool
    
    # Extraction Results
    fields: dict[str, str] | None
    extraction_message: str | None = None