"""
Document Analysis Models

SQLAlchemy models for storing documents, analysis results,
and related metadata in the persistent knowledge system.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    Boolean,
    JSON,
    Float,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from .base import BaseModel, SoftDeleteMixin, GUID


class DocumentStatus(str, Enum):
    """Document processing status."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class AnalysisType(str, Enum):
    """Types of analysis that can be performed."""
    CONTENT_EXTRACTION = "content_extraction"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    ENTITY_EXTRACTION = "entity_extraction"
    RELATIONSHIP_MAPPING = "relationship_mapping"
    LEGAL_ANALYSIS = "legal_analysis"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"


class Document(BaseModel, SoftDeleteMixin):
    """
    Core document model for storing uploaded files and metadata.
    """
    
    __tablename__ = "case_documents"
    
    # Basic document information
    filename = Column(String(255), nullable=False, index=True)
    original_filename = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String(100), nullable=False)
    
    # File storage information
    file_path = Column(String(500), nullable=True)  # Local file path
    storage_url = Column(String(500), nullable=True)  # Cloud storage URL
    file_hash = Column(String(64), nullable=False, index=True)  # SHA-256 hash
    
    # Processing status
    status = Column(String(20), nullable=False, default=DocumentStatus.UPLOADED, index=True)
    processing_started_at = Column(DateTime(timezone=True), nullable=True)
    processing_completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Content extraction
    raw_text = Column(Text, nullable=True)
    clean_text = Column(Text, nullable=True)
    word_count = Column(Integer, nullable=True)
    
    # Metadata
    document_metadata = Column(JSON, nullable=True)  # Flexible metadata storage
    tags = Column(JSON, nullable=True)  # List of tags
    
    # Relationships
    analyses = relationship("DocumentAnalysis", back_populates="document", cascade="all, delete-orphan")
    insights = relationship("DocumentInsight", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Document(filename='{self.filename}', status='{self.status}')>"


class DocumentAnalysis(BaseModel):
    """
    Analysis results for documents using various AI models and techniques.
    """
    
    __tablename__ = "document_analyses"
    
    # Foreign key to document
    document_id = Column(GUID(), ForeignKey("case_documents.id"), nullable=False, index=True)
    
    # Analysis configuration
    analysis_type = Column(String(50), nullable=False, index=True)
    model_name = Column(String(100), nullable=False)  # e.g., "gpt-4", "claude-3"
    model_version = Column(String(50), nullable=True)
    
    # Processing information
    started_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    processing_time = Column(Float, nullable=True)  # seconds
    
    # Results
    status = Column(String(20), nullable=False, default="processing", index=True)
    result = Column(JSON, nullable=True)  # Analysis results as JSON
    confidence_score = Column(Float, nullable=True)  # 0.0 to 1.0
    
    # Error handling
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    
    # Cost tracking
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    cost_usd = Column(Float, nullable=True)
    
    # Relationships
    document = relationship("Document", back_populates="analyses")
    
    def __repr__(self) -> str:
        return f"<DocumentAnalysis(type='{self.analysis_type}', status='{self.status}')>"


class DocumentInsight(BaseModel):
    """
    High-level insights extracted from document analysis.
    
    This represents the "knowledge" that accumulates over time
    from processing multiple documents.
    """
    
    # Foreign key to document
    document_id = Column(GUID(), ForeignKey("case_documents.id"), nullable=False, index=True)
    
    # Insight metadata
    insight_type = Column(String(50), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Content
    content = Column(Text, nullable=False)  # The actual insight text
    summary = Column(Text, nullable=True)   # Brief summary
    
    # Relevance and quality
    relevance_score = Column(Float, nullable=True)  # 0.0 to 1.0
    confidence_score = Column(Float, nullable=True)  # 0.0 to 1.0
    importance_score = Column(Float, nullable=True)  # 0.0 to 1.0
    
    # Categorization
    categories = Column(JSON, nullable=True)  # List of category tags
    entities = Column(JSON, nullable=True)    # Extracted entities
    keywords = Column(JSON, nullable=True)    # Important keywords
    
    # Relationships
    document = relationship("Document", back_populates="insights")
    
    def __repr__(self) -> str:
        return f"<DocumentInsight(title='{self.title[:50]}...', type='{self.insight_type}')>"


class DocumentRelationship(BaseModel):
    """
    Relationships between documents discovered through analysis.
    """
    
    # Source and target documents
    source_document_id = Column(GUID(), ForeignKey("case_documents.id"), nullable=False, index=True)
    target_document_id = Column(GUID(), ForeignKey("case_documents.id"), nullable=False, index=True)
    
    # Relationship metadata
    relationship_type = Column(String(50), nullable=False, index=True)  # e.g., "references", "contradicts"
    description = Column(Text, nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Analysis that discovered this relationship
    discovered_by_analysis_id = Column(GUID(), ForeignKey("document_analyses.id"), nullable=True)
    
    # Bidirectional relationship indicator
    is_bidirectional = Column(Boolean, nullable=False, default=False)
    
    # Relationships
    source_document = relationship("Document", foreign_keys=[source_document_id])
    target_document = relationship("Document", foreign_keys=[target_document_id])
    
    def __repr__(self) -> str:
        return f"<DocumentRelationship(type='{self.relationship_type}')>"


class AnalysisTemplate(BaseModel):
    """
    Templates for different types of analysis configurations.
    
    Allows for reusable analysis configurations across documents.
    """
    
    name = Column(String(100), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    
    # Analysis configuration
    analysis_type = Column(String(50), nullable=False)
    model_name = Column(String(100), nullable=False)
    
    # Prompt templates and settings
    system_prompt = Column(Text, nullable=True)
    user_prompt_template = Column(Text, nullable=False)
    model_parameters = Column(JSON, nullable=True)  # temperature, max_tokens, etc.
    
    # Processing settings
    batch_size = Column(Integer, nullable=False, default=1)
    timeout_seconds = Column(Integer, nullable=False, default=300)
    max_retries = Column(Integer, nullable=False, default=3)
    
    # Status
    is_active = Column(Boolean, nullable=False, default=True)
    
    def __repr__(self) -> str:
        return f"<AnalysisTemplate(name='{self.name}', type='{self.analysis_type}')>"
