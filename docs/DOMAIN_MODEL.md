# Domain Model

## ID Validator Service Domain Entities

This document describes the domain models, services, and relationships within the ID Validator Service.

## Core Entities

### Document
Represents an uploaded document image that needs to be processed.

**Attributes:**
- `filename`: Name of the uploaded file
- `document_type`: Type of document (e.g., "aadhaar_front", "aadhaar_back", "pan_card", "passport")
- `confidence`: Confidence score of the classification (0.0 to 1.0)
- `is_valid`: Boolean indicating if confidence meets the threshold
- `fields`: Dictionary of extracted fields with their values

### DocumentResponse
Response model for document classification operations.

**Attributes:**
- `filename`: Name of the uploaded file
- `document_type`: Type of document detected
- `confidence`: Confidence score of the classification
- `is_valid`: Boolean indicating if confidence meets the threshold

### ExtractionResponse
Response model for document extraction operations.

**Attributes:**
- `document_type`: Type of document processed
- `fields`: Dictionary of extracted fields with their values
- `message`: Optional message about the extraction process

### ClassifyAndExtractResponse
Response model for combined classification and extraction operations.

**Attributes:**
- `filename`: Name of the uploaded file
- `document_type`: Type of document detected
- `confidence`: Confidence score of the classification
- `is_valid`: Boolean indicating if confidence meets the threshold
- `fields`: Dictionary of extracted fields with their values
- `extraction_message`: Optional message about the extraction process

### ErrorResponse
Standard error response model.

**Attributes:**
- `code`: Error code
- `message`: Human-readable error message
- `trace_id`: Optional trace identifier

## Services

### ModelLoader Service
Singleton service responsible for loading and managing ML models.

**Responsibilities:**
- Loading YOLO models for document classification and field detection
- Loading PaddleOCR models for text recognition
- Managing document orientation detection model
- Providing access to loaded models

**Key Methods:**
- `load_models()`: Loads all required models at startup
- `get_model_loader()`: Singleton accessor

### Pipeline Service
Handles document classification pipeline.

**Responsibilities:**
- Detecting document orientation and correcting it
- Running YOLO classification model to identify document type
- Validating classification results against confidence threshold
- Formatting classification responses

**Key Methods:**
- `classify_document_sync()`: Synchronous classification of document
- `detect_orientation()`: Detects and corrects document orientation
- `run_classification()`: Runs YOLO classification model

### Extraction Service
Handles text extraction from documents.

**Responsibilities:**
- Validating document type and retrieving appropriate models
- Running YOLO detection to locate fields in document
- Performing OCR on detected regions
- Formatting extracted fields according to document schema
- Handling two-phase OCR strategy (fast recognition followed by fallback)

**Key Methods:**
- `extract_data()`: Main extraction method
- `_validate_and_get_model()`: Validates document type and retrieves models
- `_detect_and_deduplicate()`: Runs YOLO detection and deduplicates results
- `_process_single_roi()`: Processes a single region of interest for OCR

### CombinedPipeline Service
Handles combined classification and extraction in a single pipeline.

**Responsibilities:**
- Coordinating classification and extraction phases
- Managing document orientation correction for both phases
- Normalizing document types between classification and extraction
- Providing unified response for combined operations

**Key Methods:**
- `process_document_sync()`: Combined classification and extraction
- `normalize_doc_type_for_extraction()`: Normalizes document types between phases

### ImageUtils Service
Provides image processing utilities.

**Responsibilities:**
- Reading uploaded image files
- Rotating images to correct orientation
- Extending images with padding for better OCR results

**Key Methods:**
- `read_image_file()`: Reads uploaded file as OpenCV image
- `rotate_image()`: Rotates image by specified angle
- `extend_image()`: Adds padding to improve OCR accuracy

### TextUtils Service
Provides text processing utilities.

**Responsibilities:**
- Formatting OCR text output
- Removing leading labels from extracted text
- Cleaning up spacing and punctuation in extracted text
- Handling transitions between letters and numbers

**Key Methods:**
- `format_ocr_text()`: Formats and cleans OCR text output

## Configuration

### Settings
Application configuration settings.

**Key Properties:**
- `PROJECT_NAME`: Name of the project ("Id Validator Service")
- `API_V1_STR`: API version prefix ("/api/v1")
- `BASE_MODEL_PATH`: Path to models directory
- `MODEL_CONFIG_PATH`: Path to model configuration file
- `CONFIDENCE_THRESHOLD`: Minimum confidence required for valid classification (0.98)

## Models and Configuration

### Model Configuration (config.json)
Defines the mapping between document types and models, field filters, and output schemas.

**Structure:**
- `models`: Defines all available models with their types, paths, classes, and output mappings
- `doc_type_to_model`: Maps document types to appropriate detection models
- `field_filters`: Specifies which fields to extract for each document type
- `output_schema`: Defines the output format for each document type

**Supported Models:**
- `Id_Classifier`: Classification model for identifying document types
- `Aadhaar`: Detection model for Aadhaar card fields
- `Pan_Card`: Detection model for PAN card fields
- `Passport`: Detection model for Passport fields
- `Voter_Id`: Detection model for Voter ID fields
- `Driving_License`: Detection model for Driving License fields

## Relationships

```
Document (uploaded image)
    ↓
ModelLoader Service (loads appropriate models)
    ↓
Pipeline Service (classifies document type)
    ↓
Extraction Service (extracts fields based on document type)
    ↓
DocumentResponse/ExtractionResponse (returns results)

CombinedPipeline Service coordinates both Pipeline and Extraction services
```

## Supported Document Types

The service supports the following document types:

1. **Aadhaar Card** (Front and Back)
   - **aadhaar_front**: Name, Aadhaar Number, Date of Birth, Gender
   - **aadhaar_back**: Aadhaar Number, Address

2. **PAN Card** (Front)
   - Fields: PAN Number, Name, Father's Name, Date of Birth

3. **Driving License** (Front and Back)
   - Fields: Name, DL Number, Date of Birth, Blood Group, Address, RTO, State, Vehicle Type

4. **Passport**
   - Fields: Name, Code, Date of Birth, Gender, Nationality, Address, MRZ codes

5. **Voter ID**
   - Fields: Name, Voter ID, Date of Birth, Gender, Address, Father's Name

## Architecture Layers

1. **API Layer**: FastAPI endpoints in `app/api/v1/endpoints/documents.py`
2. **Schema Layer**: Pydantic models in `app/schemas/document.py`
3. **Service Layer**: Business logic in `app/services/`
4. **Core Layer**: Configuration and utilities in `app/core/`
5. **Model Layer**: ML models and configuration in `models/`