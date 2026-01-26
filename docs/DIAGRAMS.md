# ID Validator Service - Architecture Diagrams

## 1. System Architecture Component Diagram

This diagram shows the high-level architecture of the ID Validator Service, including the main layers and their relationships.

```mermaid
graph TB
    subgraph "Client Layer"
        Client["Client Application"]
    end
    
    subgraph "API Layer"
        API["FastAPI Application"]
        Router["API Router v1"]
        Endpoints["Document Endpoints"]
    end
    
    subgraph "Service Layer"
        ModelLoader["Model Loader Service"]
        Pipeline["Classification Pipeline"]
        Extraction["Extraction Service"]
        Combined["Combined Pipeline"]
        ImageUtils["Image Utilities"]
        TextUtils["Text Utilities"]
    end
    
    subgraph "Model Layer"
        YOLO["YOLO Models"]
        PaddleOCR["PaddleOCR Models"]
        Config["Model Configuration"]
    end
    
    subgraph "Data Layer"
        Schema["Pydantic Schemas"]
    end
    
    Client --> API
    API --> Router
    Router --> Endpoints
    Endpoints --> ModelLoader
    Endpoints --> Pipeline
    Endpoints --> Extraction
    Endpoints --> Combined
    ModelLoader --> YOLO
    ModelLoader --> PaddleOCR
    ModelLoader --> Config
    Pipeline --> ImageUtils
    Extraction --> ImageUtils
    Extraction --> TextUtils
    Combined --> Pipeline
    Combined --> Extraction
    Endpoints --> Schema
```

## 2. API Request Flow Sequence Diagram

This sequence diagram shows the flow of requests through the API endpoints, from client to response.

```mermaid
sequenceDiagram
    participant Client as Client
    participant API as FastAPI App
    participant Router as API Router
    participant Endpoint as Document Endpoint
    participant ModelLoader as Model Loader
    participant Pipeline as Classification Pipeline
    participant Extraction as Extraction Service
    participant ImageUtils as Image Utils
    participant Response as Response
    
    Client->>API: POST /api/v1/documents/classify
    API->>Router: Route Request
    Router->>Endpoint: Forward to classify endpoint
    Endpoint->>ModelLoader: Get model loader
    Endpoint->>ImageUtils: Read image file
    ImageUtils-->>Endpoint: Return image array
    Endpoint->>Pipeline: classify_document_sync(image, loader)
    Pipeline->>ImageUtils: detect_orientation(image, loader)
    ImageUtils-->>Pipeline: Return corrected image
    Pipeline->>Pipeline: run_classification(corrected_image, loader)
    Pipeline-->>Endpoint: Return classification result
    Endpoint-->>API: Return DocumentResponse
    API-->>Client: 200 OK with classification result
```

## 3. Document Classification Flow Sequence Diagram

This sequence diagram shows the detailed flow for document classification.

```mermaid
sequenceDiagram
    participant Endpoint as classify_document_endpoint
    participant ImageUtils as ImageUtils Service
    participant Pipeline as Pipeline Service
    participant ModelLoader as ModelLoader Service
    participant YOLO as YOLO Classifier
    participant Response as DocumentResponse

    Endpoint->>ImageUtils: read_image_file(file)
    ImageUtils-->>Endpoint: Return numpy array
    Endpoint->>Pipeline: classify_document_sync(image, loader)
    Pipeline->>Pipeline: detect_orientation(image, loader)
    Pipeline->>ModelLoader: Access doc_orientation_model
    ModelLoader-->>Pipeline: Return orientation model
    Pipeline->>Pipeline: run_classification(corrected_image, loader)
    Pipeline->>ModelLoader: Access yolo_model
    ModelLoader-->>Pipeline: Return YOLO classifier
    Pipeline->>YOLO: Run inference
    YOLO-->>Pipeline: Return predictions
    Pipeline->>Pipeline: format_response(doc_type)
    Pipeline-->>Endpoint: Return classification result
    Endpoint->>Response: Create DocumentResponse
    Response-->>Endpoint: Return formatted response
```

## 4. Document Extraction Flow Sequence Diagram

This sequence diagram shows the detailed flow for document field extraction.

```mermaid
sequenceDiagram
    participant Endpoint as extract_document_data
    participant ImageUtils as ImageUtils Service
    participant Extraction as Extraction Service
    participant ModelLoader as ModelLoader Service
    participant YOLO as YOLO Detector
    participant OCR as OCR Model
    participant Response as ExtractionResponse
    
    Endpoint->>ImageUtils: read_image_file(file)
    ImageUtils-->>Endpoint: Return numpy array
    Endpoint->>Extraction: extract_data(image, document_type, loader)
    Extraction->>Extraction: _validate_and_get_model(doc_type, loader)
    Extraction->>ModelLoader: Access detection_models
    ModelLoader-->>Extraction: Return detection model
    Extraction->>Extraction: detect_orientation(image, loader)
    Extraction->>Extraction: _detect_and_deduplicate(image, detection_model, allowed_fields, output_mapping, doc_type)
    Extraction->>YOLO: Run detection
    YOLO-->>Extraction: Return bounding boxes
    loop For each detected field
        Extraction->>Extraction: _process_single_roi(image, box, display_name, loader)
        Extraction->>OCR: Run OCR on cropped region
        OCR-->>Extraction: Return extracted text
    end
    Extraction->>Extraction: format_output(extracted_fields, doc_type, config)
    Extraction-->>Endpoint: Return extraction result
    Endpoint->>Response: Create ExtractionResponse
    Response-->>Endpoint: Return formatted response
```

## 5. Combined Classification and Extraction Flow

This sequence diagram shows the flow for the combined endpoint that performs both classification and extraction.

```mermaid
sequenceDiagram
    participant Endpoint as classify_and_extract_document_endpoint
    participant ImageUtils as ImageUtils Service
    participant Combined as CombinedPipeline Service
    participant Pipeline as Classification Pipeline
    participant Extraction as Extraction Service
    participant ModelLoader as ModelLoader Service
    participant Response as ClassifyAndExtractResponse

    Endpoint->>ImageUtils: read_image_file(file)
    ImageUtils-->>Endpoint: Return numpy array
    Endpoint->>Combined: process_document_sync(image, loader, filename)

    Note over Combined: Classification Phase
    Combined->>Pipeline: detect_orientation(image, loader)
    Combined->>Pipeline: run_classification(corrected_image, loader)
    Pipeline->>ModelLoader: Access YOLO classifier
    ModelLoader-->>Pipeline: Return classifier
    Pipeline-->>Combined: Return doc_type, confidence

    Note over Combined: Validation
    Combined->>Combined: Check confidence threshold

    alt confidence too low
        Combined-->>Endpoint: Return with is_valid=false
    else sufficient confidence
        Note over Combined: Extraction Phase
        Combined->>Extraction: extract_data_v2(corrected_image, doc_type_normalized, loader)
        Extraction->>ModelLoader: Access detection models
        ModelLoader-->>Extraction: Return models
        Extraction-->>Combined: Return extracted fields
        Combined->>Response: Create ClassifyAndExtractResponse
        Response-->>Endpoint: Return formatted response
    end
    Endpoint-->>Endpoint: Add filename to response
```

## 6. Module Dependency Diagram

This diagram shows the dependencies between different modules in the application.

```mermaid
graph LR
    Main["app.main"] --> APIRouter["app.api.v1.api"]
    APIRouter --> Endpoints["app.api.v1.endpoints.documents"]
    Endpoints --> ModelLoader["app.services.model_loader"]
    Endpoints --> Pipeline["app.services.pipeline"]
    Endpoints --> Extraction["app.services.extraction"]
    Endpoints --> Combined["app.services.combined_pipeline"]
    Endpoints --> ImageUtils["app.services.image_utils"]
    Endpoints --> DocumentSchema["app.schemas.document"]
    
    ModelLoader --> Config["app.core.config"]
    Pipeline --> ImageUtils
    Pipeline --> Config
    Extraction --> ImageUtils
    Extraction --> TextUtils["app.services.text_utils"]
    Extraction --> Pipeline
    Combined --> Pipeline
    Combined --> Extraction
    Combined --> Config
    
    ModelLoader -.-> YOLO["YOLO Models"]
    ModelLoader -.-> PaddleOCR["PaddleOCR Models"]
    ModelLoader -.-> ModelConfig["models/config.json"]
    
    classDef core fill:#e1f5fe
    classDef service fill:#f3e5f5
    classDef model fill:#e8f5e8
    classDef api fill:#fff3e0
    
    class Main,APIRouter,Endpoints api
    class ModelLoader,Pipeline,Extraction,Combined,ImageUtils,TextUtils service
    class Config core
    class YOLO,PaddleOCR,ModelConfig model
```

## 7. Service Interaction Flowchart

This flowchart shows the logical flow and decision points in the document processing services.

```mermaid
flowchart TD
    Start([Start Processing]) --> Validate{Validate Input}
    
    Validate -->|Invalid| ReturnError[Return Error Response]
    Validate -->|Valid| LoadModels{Load Models Available?}
    
    LoadModels -->|Not Loaded| LoadError[Return Model Loading Error]
    LoadModels -->|Loaded| CorrectOrientation[Correct Image Orientation]
    
    CorrectOrientation --> ClassifyDoc[Run Document Classification]
    ClassifyDoc --> CheckConfidence{Confidence > Threshold?}
    
    CheckConfidence -->|No| ReturnInvalid[Return Invalid Classification]
    CheckConfidence -->|Yes| NormalizeType[Normalize Document Type]
    
    NormalizeType --> ExtractFields[Extract Document Fields]
    ExtractFields --> FormatOutput[Format Output According to Schema]
    FormatOutput --> ReturnSuccess[Return Success Response]
    
    ReturnError --> End([End])
    LoadError --> End
    ReturnInvalid --> End
    ReturnSuccess --> End
    
    classDef startEnd fill:#e1f5fe
    classDef process fill:#f3e5f5
    classDef decision fill:#fff3e0
    classDef error fill:#ffebee
    
    class Start,End startEnd
    class Validate,CorrectOrientation,ClassifyDoc,NormalizeType,ExtractFields,FormatOutput process
    class CheckConfidence,LoadModels decision
    class ReturnError,LoadError,ReturnInvalid error
```

## 8. Data Flow Architecture

This diagram shows how data flows through the system from input to output.

```mermaid
graph LR
    subgraph "Input"
        ImageFile["Image File Upload"]
    end
    
    subgraph "Preprocessing"
        ImageRead["Read Image to Array"]
        OrientationDetect["Detect & Correct Orientation"]
    end
    
    subgraph "Processing"
        Classification["Document Classification"]
        FieldDetection["Field Detection"]
        OCRProcessing["OCR Processing"]
    end
    
    subgraph "Post-processing"
        FormatOutput["Format Output"]
        Validation["Validate Results"]
    end
    
    subgraph "Output"
        Response["Structured Response"]
    end
    
    ImageFile --> ImageRead
    ImageRead --> OrientationDetect
    OrientationDetect --> Classification
    OrientationDetect --> FieldDetection
    Classification --> FieldDetection
    FieldDetection --> OCRProcessing
    OCRProcessing --> FormatOutput
    FormatOutput --> Validation
    Validation --> Response
```