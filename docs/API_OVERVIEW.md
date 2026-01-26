# API Overview

## ID Validator Service API Reference

The ID Validator Service provides document classification and text extraction capabilities for various identity documents using computer vision and OCR technology.

## Base URL

```
http://localhost:8000
```

## Endpoints

### Root Endpoint

#### GET /

**Description:** Health check endpoint to verify the service is running.

**Response:**
```json
{
  "message": "Id Validator Service is running"
}
```

### Documents API

#### POST /api/v1/documents/classify

**Description:** Classifies the uploaded document image to determine its type (aadhaar, pan_card, driving_license, passport, voter_id).

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body:
  - `file`: Image file to classify

**Response:**
```json
{
  "filename": "document.jpg",
  "document_type": "aadhaar_front",
  "confidence": 0.995,
  "is_valid": true
}
```

**Response Fields:**
- `filename`: Name of the uploaded file
- `document_type`: Type of document detected (e.g., "aadhaar_front", "pan_card", "passport")
- `confidence`: Confidence score of the classification (0.0 to 1.0)
- `is_valid`: Boolean indicating if confidence meets the threshold

**Error Responses:**
- `400 Bad Request`: Invalid file type or upload error
- `500 Internal Server Error`: Processing failure
- `503 Service Unavailable`: Runtime errors during extraction

#### POST /api/v1/documents/extract

**Description:** Extracts text fields from a document of a known type.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body:
  - `file`: Image file to extract from
  - `document_type`: Type of document (e.g., "aadhaar_front", "pan_card")

**Response:**
```json
{
  "document_type": "aadhaar_front",
  "fields": {
    "Name": "John Doe",
    "Aadhaar": "1234 5678 9012",
    "Gender": "Male",
    "DOB": "01/01/1990"
  },
  "message": "No valid bounding boxes detected"
}
```

**Response Fields:**
- `document_type`: Type of document processed
- `fields`: Dictionary of extracted fields with their values
- `message`: Optional message about the extraction process (can contain meaningful text like "No valid bounding boxes detected")

**Error Responses:**
- `400 Bad Request`: Invalid document type or file
- `400 Bad Request`: Unsupported document type
- `500 Internal Server Error`: Processing failure

#### POST /api/v1/documents/classify-and-extract

**Description:** Combined endpoint that first classifies the document and then extracts fields in a single call.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body:
  - `file`: Image file to process

**Response:**
```json
{
  "filename": "document.jpg",
  "document_type": "aadhaar_front",
  "confidence": 0.995,
  "is_valid": true,
  "fields": {
    "Name": "John Doe",
    "Aadhaar": "1234 5678 9012",
    "Gender": "Male",
    "DOB": "01/01/1990"
  },
  "extraction_message": null
}
```

**Response Fields:**
- `filename`: Name of the uploaded file
- `document_type`: Type of document detected
- `confidence`: Confidence score of the classification
- `is_valid`: Boolean indicating if confidence meets the threshold
- `fields`: Dictionary of extracted fields with their values
- `extraction_message`: Optional message about the extraction process

**Error Responses:**
- `400 Bad Request`: Invalid file type
- `500 Internal Server Error`: Processing failure

### Metrics Endpoint

#### GET /api/v1/metrics

**Description:** Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "ok",
  "message": "endpoint is healthy"
}
```

## Error Handling

All API endpoints follow a consistent error response format:

```json
{
  "code": "ERROR_CODE",
  "message": "Human-readable error message",
  "trace_id": null
}
```

Common error codes:
- `INVALID_FILE_TYPE`: Uploaded file is not a valid image
- `UNSUPPORTED_DOCUMENT_TYPE`: Document type is not supported
- Generic error messages for internal server errors

## Configuration

The service uses the following configuration parameters:
- `CONFIDENCE_THRESHOLD`: Minimum confidence required for a valid classification (default: 0.98)
- `API_V1_STR`: API version prefix (default: "/api/v1")
- `PROJECT_NAME`: Name of the project (default: "Id Validator Service")
- `LOG_EXCLUDED_PATHS`: Paths whose access logs will be filtered out (default: {"/metrics", "/", "/health", "/ready"})

## Authentication

The API does not require authentication for basic operations.