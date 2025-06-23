"""
Pydantic Schemas for Documents

Request/response schemas for document-related API endpoints.
Provides data validation, serialization, and OpenAPI documentation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator, root_validator

from src.models.document import DocumentStatus, AnalysisType


# Base schemas
class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    
    class Config:
        orm_mode = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


# Document schemas
class DocumentCreate(BaseModel):
    """Schema for creating a new document."""
    
    filename: str = Field(..., min_length=1, max_length=255)
    file_size: int = Field(..., gt=0)
    mime_type: str = Field(..., min_length=1, max_length=100)
    file_hash: str = Field(..., min_length=64, max_length=64)
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    
    @validator('file_hash')
    def validate_file_hash(cls, v):
        """Validate SHA-256 hash format."""
        if len(v) != 64:
            raise ValueError('File hash must be 64 characters (SHA-256)')
        try:
            int(v, 16)  # Verify it's valid hex
        except ValueError:
            raise ValueError('File hash must be valid hexadecimal')
        return v.lower()
    
    @validator('tags')
    def validate_tags(cls, v):
        """Validate tags format."""
        if v is not None:
            if not isinstance(v, list):
                raise ValueError('Tags must be a list')
            if len(v) > 20:
                raise ValueError('Maximum 20 tags allowed')
            for tag in v:
                if not isinstance(tag, str) or len(tag.strip()) == 0:
                    raise ValueError('Each tag must be a non-empty string')
        return v


class DocumentUpdate(BaseModel):
    """Schema for updating document metadata."""
    
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    status: Optional[DocumentStatus] = None
    
    @validator('tags')
    def validate_tags(cls, v):
        """Validate tags format."""
        if v is not None:
            if not isinstance(v, list):
                raise ValueError('Tags must be a list')
            if len(v) > 20:
                raise ValueError('Maximum 20 tags allowed')
            for tag in v:
                if not isinstance(tag, str) or len(tag.strip()) == 0:
                    raise ValueError('Each tag must be a non-empty string')
        return v


class DocumentResponse(BaseSchema):
    """Schema for document response."""
    
    id: UUID
    filename: str
    original_filename: str
    file_size: int
    mime_type: str
    file_hash: str
    status: DocumentStatus
    processing_started_at: Optional[datetime]
    processing_completed_at: Optional[datetime]
    word_count: Optional[int]
    metadata: Optional[Dict[str, Any]]
    tags: Optional[List[str]]
    created_at: datetime
    updated_at: datetime


class DocumentSummary(BaseSchema):
    """Schema for document summary (lighter version)."""
    
    id: UUID
    filename: str
    file_size: int
    mime_type: str
    status: DocumentStatus
    word_count: Optional[int]
    created_at: datetime


class DocumentContent(BaseSchema):
    """Schema for document content response."""
    
    id: UUID
    filename: str
    raw_text: Optional[str]
    clean_text: Optional[str]
    word_count: Optional[int]


# Document Analysis schemas
class AnalysisCreate(BaseModel):
    """Schema for creating a new analysis."""
    
    document_id: UUID
    analysis_type: AnalysisType
    model_name: str = Field(..., min_length=1, max_length=100)
    model_version: Optional[str] = Field(None, max_length=50)
    
    @validator('model_name')
    def validate_model_name(cls, v):
        """Validate model name format."""
        allowed_models = [
            'gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo',
            'claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku',
            'gemini-pro', 'gemini-pro-vision',
        ]
        if v not in allowed_models:
            raise ValueError(f'Model must be one of: {", ".join(allowed_models)}')
        return v


class AnalysisResponse(BaseSchema):
    """Schema for analysis response."""
    
    id: UUID
    document_id: UUID
    analysis_type: AnalysisType
    model_name: str
    model_version: Optional[str]
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    processing_time: Optional[float]
    result: Optional[Dict[str, Any]]
    confidence_score: Optional[float]
    error_message: Optional[str]
    retry_count: int
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cost_usd: Optional[float]
    created_at: datetime


class AnalysisSummary(BaseSchema):
    """Schema for analysis summary."""
    
    id: UUID
    analysis_type: AnalysisType
    model_name: str
    status: str
    confidence_score: Optional[float]
    processing_time: Optional[float]
    created_at: datetime


# Batch operations
class BatchAnalysisCreate(BaseModel):
    """Schema for creating batch analysis."""
    
    document_ids: List[UUID] = Field(..., min_items=1, max_items=100)
    analysis_type: AnalysisType
    model_name: str = Field(..., min_length=1, max_length=100)
    model_version: Optional[str] = Field(None, max_length=50)
    
    @validator('document_ids')
    def validate_document_ids(cls, v):
        """Validate document IDs list."""
        if len(set(v)) != len(v):
            raise ValueError('Document IDs must be unique')
        return v


class BatchAnalysisResponse(BaseSchema):
    """Schema for batch analysis response."""
    
    batch_id: str
    total_documents: int
    status: str
    created_analyses: List[UUID]
    failed_documents: List[Dict[str, Any]]


# File upload schemas
class FileUploadResponse(BaseSchema):
    """Schema for file upload response."""
    
    document: DocumentResponse
    upload_url: Optional[str] = None  # For direct upload to cloud storage


# Search and filtering
class DocumentFilter(BaseModel):
    """Schema for document filtering."""
    
    status: Optional[List[DocumentStatus]] = None
    mime_types: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    file_size_min: Optional[int] = Field(None, ge=0)
    file_size_max: Optional[int] = Field(None, ge=0)
    
    @validator('file_size_max')
    def validate_file_size_range(cls, v, values):
        """Validate file size range."""
        if v is not None and 'file_size_min' in values and values['file_size_min'] is not None:
            if v < values['file_size_min']:
                raise ValueError('file_size_max must be greater than file_size_min')
        return v


class DocumentSearchQuery(BaseModel):
    """Schema for document search."""
    
    query: Optional[str] = Field(None, min_length=1, max_length=500)
    filters: Optional[DocumentFilter] = None
    sort_by: str = Field('created_at', regex='^(created_at|updated_at|filename|file_size)$')
    sort_order: str = Field('desc', regex='^(asc|desc)$')
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)


class DocumentSearchResponse(BaseSchema):
    """Schema for document search response."""
    
    documents: List[DocumentSummary]
    total: int
    limit: int
    offset: int
    has_more: bool


# Statistics and analytics
class DocumentStats(BaseSchema):
    """Schema for document statistics."""
    
    total_documents: int
    total_size: int  # in bytes
    documents_by_status: Dict[str, int]
    documents_by_type: Dict[str, int]
    recent_uploads: int  # Last 24 hours
    processing_queue_size: int


class AnalysisStats(BaseSchema):
    """Schema for analysis statistics."""
    
    total_analyses: int
    analyses_by_type: Dict[str, int]
    analyses_by_status: Dict[str, int]
    avg_processing_time: Optional[float]
    total_cost_usd: Optional[float]
    success_rate: float  # Percentage


# Error responses
class ErrorDetail(BaseModel):
    """Schema for error details."""
    
    type: str
    message: str
    field: Optional[str] = None


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    
    error: str
    details: Optional[List[ErrorDetail]] = None
    request_id: Optional[str] = None
