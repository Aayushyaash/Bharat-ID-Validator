# Project Index

This index organizes all files in the ID Validator Service project by architectural layer.

## Core Logic (`core-logic`)

Business logic, algorithms, domain models, and core services:

- `app/services/pipeline.py` - Document classification pipeline with orientation detection and YOLO classification
- `app/services/extraction.py` - Document field extraction logic with two-phase OCR strategy
- `app/services/combined_pipeline.py` - Combined classification and extraction pipeline
- `app/services/model_loader.py` - Singleton model loader for YOLO and PaddleOCR models
- `app/services/image_utils.py` - Image processing utilities (rotation, extension, reading)
- `app/services/text_utils.py` - Text formatting and cleaning utilities
- `models/Id_Classifier.pt` - YOLO model for document type classification
- `models/Aadhaar_Card.pt` - YOLO model for Aadhaar card field detection
- `models/Pan_Card.pt` - YOLO model for PAN card field detection
- `models/Passport.pt` - YOLO model for Passport field detection
- `models/Voter_Id.pt` - YOLO model for Voter ID field detection
- `models/Driving_License.pt` - YOLO model for Driving License field detection
- `models/config.json` - Configuration file mapping document types to models and defining field schemas

## API Handlers (`api-handlers`)

Controllers, routes, endpoints, and handlers:

- `app/main.py` - FastAPI application entry point with CORS, middleware, and exception handling
- `app/api/v1/api.py` - API router configuration
- `app/api/v1/endpoints/documents.py` - Document processing endpoints (classify, extract, classify-and-extract)

## Infrastructure (`infrastructure`)

Configuration, deployment, CI/CD, and environment files:

- `.env.example` - Example environment variables file
- `.gitignore` - Git ignore rules
- `pyproject.toml` - Project configuration (Poetry, build, static analysis)
- `mkdocs.yml` - MkDocs documentation configuration
- `venv/` - Python virtual environment directory (excluded from index)

## Data Schema (`data-schema`)

Schema definitions and data structures:

- `app/schemas/document.py` - Pydantic models for API request/response validation
- `models/config.json` - Model configuration schema (included in core-logic as well)

## Tests (`tests`)

Unit, integration, and end-to-end tests:

- `tests/test_classification.py` - Tests for document classification functionality
- `tests/test_context_fixes.py` - Tests for context-related fixes
- `tests/test_integration.py` - Integration tests
- `tests/test_schemas.py` - Tests for schema validation
- `tests/test_unwarp.py` - Tests for document unwarping functionality
- `tests/api/test_documents.py` - API tests for document endpoints
- `tests/services/test_extraction.py` - Service-level tests for extraction functionality
- `tests/output/` - Test output directory

## Docs (`docs`)

Documentation files:

- `README.md` - Main project documentation with setup and usage instructions
- `docs/API_OVERVIEW.md` - Complete API reference
- `docs/ARCHITECTURE.md` - Technical design documentation
- `docs/DIAGRAMS.md` - Architecture and flow diagrams
- `docs/DOMAIN_MODEL.md` - Domain entities and relationships
- `docs\New Text Document.txt` - Documentation review report

## Config (`config`)

Configuration files:

- `requirements.txt` - Pinned Python dependencies
- `requirements.lock` - Locked dependencies file
- `app/core/config.py` - Application settings and configuration
- `.qwen/` - Qwen-specific configuration directory
- `.rules/` - Rule configuration directory
- `.gemini/` - Gemini-specific configuration directory

## Other Directories

- `app/core/` - Core utilities (config, logging, errors, middleware)
- `samples/` - Sample image files for testing
- `logs/` - Application logs directory (JSON format, ignored by git)
- `app/__pycache__/` - Python cache directory (excluded from index)
- `.git/` - Git repository directory (excluded from index)
- `.pytest_cache/` - Pytest cache directory (excluded from index)