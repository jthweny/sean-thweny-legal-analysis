"""
Knowledge and Insights Models

Models for storing accumulated knowledge, insights, and patterns
discovered across multiple documents and analysis sessions.
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
    Index,
)
from sqlalchemy.orm import relationship

from .base import BaseModel, SoftDeleteMixin, GUID


class InsightType(str, Enum):
    """Types of insights that can be generated."""
    PATTERN = "pattern"           # Recurring patterns across documents
    TREND = "trend"              # Temporal trends in data
    ANOMALY = "anomaly"          # Unusual findings
    RELATIONSHIP = "relationship" # Connections between entities
    SUMMARY = "summary"          # High-level summaries
    PREDICTION = "prediction"    # Predictive insights
    RECOMMENDATION = "recommendation"  # Action recommendations


class EntityType(str, Enum):
    """Types of entities that can be extracted."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    AMOUNT = "amount"
    LEGAL_CASE = "legal_case"
    CONTRACT = "contract"
    REGULATION = "regulation"
    CONCEPT = "concept"


class KnowledgeGraph(BaseModel):
    """
    Central knowledge graph that accumulates insights over time.
    
    This represents the persistent memory of the system,
    growing and evolving with each document processed.
    """
    
    __tablename__ = "knowledge_graph"
    
    name = Column(String(255), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    
    # Versioning for knowledge evolution
    version = Column(Integer, nullable=False, default=1)
    
    # Statistics
    total_documents = Column(Integer, nullable=False, default=0)
    total_insights = Column(Integer, nullable=False, default=0)
    total_entities = Column(Integer, nullable=False, default=0)
    total_relationships = Column(Integer, nullable=False, default=0)
    
    # Metadata
    domain = Column(String(100), nullable=True, index=True)  # e.g., "legal", "medical"
    language = Column(String(10), nullable=False, default="en")
    
    # Configuration
    similarity_threshold = Column(Float, nullable=False, default=0.8)
    auto_merge_enabled = Column(Boolean, nullable=False, default=True)
    
    # Status
    is_active = Column(Boolean, nullable=False, default=True)
    last_updated = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    entities = relationship("Entity", back_populates="knowledge_graph", cascade="all, delete-orphan")
    insights = relationship("Insight", back_populates="knowledge_graph", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<KnowledgeGraph(name='{self.name}', version={self.version})>"


class Entity(BaseModel):
    """
    Entities extracted from documents and accumulated in the knowledge graph.
    """
    
    __tablename__ = "knowledge_entities"
    
    # Foreign key to knowledge graph
    knowledge_graph_id = Column(GUID(), ForeignKey("knowledge_graph.id"), nullable=False, index=True)
    
    # Entity identification
    name = Column(String(255), nullable=False, index=True)
    entity_type = Column(String(50), nullable=False, index=True)
    canonical_name = Column(String(255), nullable=True, index=True)  # Normalized name
    
    # Content
    description = Column(Text, nullable=True)
    aliases = Column(JSON, nullable=True)  # Alternative names/spellings
    
    # Attributes extracted from documents
    attributes = Column(JSON, nullable=True)  # Flexible attribute storage
    
    # Confidence and frequency
    confidence_score = Column(Float, nullable=True)  # Average confidence across mentions
    mention_count = Column(Integer, nullable=False, default=1)
    document_count = Column(Integer, nullable=False, default=1)
    
    # Temporal information
    first_seen = Column(DateTime(timezone=True), nullable=False)
    last_seen = Column(DateTime(timezone=True), nullable=False)
    
    # Status
    is_active = Column(Boolean, nullable=False, default=True)
    
    # Relationships
    knowledge_graph = relationship("KnowledgeGraph", back_populates="entities")
    source_relationships = relationship("EntityRelationship", foreign_keys="EntityRelationship.source_entity_id", back_populates="source_entity")
    target_relationships = relationship("EntityRelationship", foreign_keys="EntityRelationship.target_entity_id", back_populates="target_entity")
    
    # Indexes for better query performance
    __table_args__ = (
        Index('ix_entity_name_type', 'name', 'entity_type'),
        Index('ix_entity_knowledge_graph_type', 'knowledge_graph_id', 'entity_type'),
    )
    
    def __repr__(self) -> str:
        return f"<Entity(name='{self.name}', type='{self.entity_type}')>"


class EntityRelationship(BaseModel):
    """
    Relationships between entities in the knowledge graph.
    """
    
    # Source and target entities
    source_entity_id = Column(GUID(), ForeignKey("knowledge_entities.id"), nullable=False, index=True)
    target_entity_id = Column(GUID(), ForeignKey("knowledge_entities.id"), nullable=False, index=True)
    
    # Relationship metadata
    relationship_type = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Confidence and evidence
    confidence_score = Column(Float, nullable=True)
    evidence_count = Column(Integer, nullable=False, default=1)
    
    # Temporal information
    first_observed = Column(DateTime(timezone=True), nullable=False)
    last_observed = Column(DateTime(timezone=True), nullable=False)
    
    # Properties of the relationship
    properties = Column(JSON, nullable=True)
    
    # Relationships
    source_entity = relationship("Entity", foreign_keys=[source_entity_id], back_populates="source_relationships")
    target_entity = relationship("Entity", foreign_keys=[target_entity_id], back_populates="target_relationships")
    
    def __repr__(self) -> str:
        return f"<EntityRelationship(type='{self.relationship_type}')>"


class Insight(BaseModel, SoftDeleteMixin):
    """
    High-level insights accumulated across multiple documents.
    
    These represent the "learned knowledge" of the system.
    """
    
    __tablename__ = "accumulated_insights"
    
    # Foreign key to knowledge graph
    knowledge_graph_id = Column(GUID(), ForeignKey("knowledge_graph.id"), nullable=False, index=True)
    
    # Insight identification
    title = Column(String(255), nullable=False, index=True)
    insight_type = Column(String(50), nullable=False, index=True)
    
    # Content
    content = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    
    # Supporting evidence
    evidence = Column(JSON, nullable=True)  # References to supporting documents/analyses
    supporting_document_ids = Column(JSON, nullable=True)  # List of document UUIDs
    
    # Quality metrics
    confidence_score = Column(Float, nullable=True)  # 0.0 to 1.0
    relevance_score = Column(Float, nullable=True)   # 0.0 to 1.0
    novelty_score = Column(Float, nullable=True)     # 0.0 to 1.0 (how new/unique is this insight)
    
    # Impact and usage
    view_count = Column(Integer, nullable=False, default=0)
    validation_score = Column(Float, nullable=True)  # Human validation feedback
    
    # Categorization
    categories = Column(JSON, nullable=True)
    tags = Column(JSON, nullable=True)
    related_entities = Column(JSON, nullable=True)  # Entity IDs this insight relates to
    
    # Temporal validity
    valid_from = Column(DateTime(timezone=True), nullable=True)
    valid_until = Column(DateTime(timezone=True), nullable=True)
    
    # Status
    is_active = Column(Boolean, nullable=False, default=True)
    
    # Relationships
    knowledge_graph = relationship("KnowledgeGraph", back_populates="insights")
    
    def __repr__(self) -> str:
        return f"<Insight(title='{self.title[:50]}...', type='{self.insight_type}')>"


class Pattern(BaseModel):
    """
    Patterns discovered across multiple documents and analyses.
    """
    
    # Pattern identification
    name = Column(String(255), nullable=False, index=True)
    pattern_type = Column(String(50), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Pattern definition
    pattern_definition = Column(JSON, nullable=False)  # Formal pattern definition
    
    # Statistics
    occurrence_count = Column(Integer, nullable=False, default=1)
    document_count = Column(Integer, nullable=False, default=1)
    confidence_score = Column(Float, nullable=True)
    
    # Temporal information
    first_detected = Column(DateTime(timezone=True), nullable=False)
    last_detected = Column(DateTime(timezone=True), nullable=False)
    
    # Examples and evidence
    examples = Column(JSON, nullable=True)  # Example instances of the pattern
    
    # Status
    is_validated = Column(Boolean, nullable=False, default=False)
    is_active = Column(Boolean, nullable=False, default=True)
    
    def __repr__(self) -> str:
        return f"<Pattern(name='{self.name}', type='{self.pattern_type}')>"


class KnowledgeEvolution(BaseModel):
    """
    Track how knowledge and insights evolve over time.
    
    This provides an audit trail of how the system's understanding
    changes as more documents are processed.
    """
    
    # What changed
    change_type = Column(String(50), nullable=False, index=True)  # "created", "updated", "merged", "deprecated"
    entity_type = Column(String(50), nullable=False, index=True)  # "insight", "entity", "relationship"
    entity_id = Column(GUID(), nullable=False, index=True)
    
    # Change details
    previous_state = Column(JSON, nullable=True)
    new_state = Column(JSON, nullable=True)
    change_description = Column(Text, nullable=True)
    
    # Change metadata
    triggered_by = Column(String(100), nullable=True)  # What triggered the change
    confidence_change = Column(Float, nullable=True)  # Change in confidence score
    
    # Impact assessment
    impact_score = Column(Float, nullable=True)  # How significant was this change
    affected_entities = Column(JSON, nullable=True)  # Other entities affected by this change
    
    def __repr__(self) -> str:
        return f"<KnowledgeEvolution(type='{self.change_type}', entity_type='{self.entity_type}')>"
