from pydantic import BaseModel
from typing import Any


class DocumentResponse(BaseModel):
    """Response model for document classification endpoint."""
    filename: str
    document_type: str
    confidence: float
    is_valid: bool


class ExtractionResponse(BaseModel):
    """Response model for document extraction endpoint."""
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
    
    # Extraction Results (unified with ExtractionResponse)
    fields: dict[str, str | None] | None
    message: str | None = None