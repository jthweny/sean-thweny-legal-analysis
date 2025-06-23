"""
Background Task Models

Models for tracking and managing long-running background tasks,
including analysis jobs, data processing, and system maintenance.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

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
    func,
)
from sqlalchemy.orm import relationship

from .base import BaseModel, GUID


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"       # Task created but not started
    QUEUED = "queued"        # Task queued for execution
    RUNNING = "running"      # Task currently executing
    COMPLETED = "completed"  # Task completed successfully
    FAILED = "failed"        # Task failed with error
    CANCELLED = "cancelled"  # Task was cancelled
    RETRY = "retry"          # Task is being retried
    TIMEOUT = "timeout"      # Task timed out


class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class TaskType(str, Enum):
    """Types of background tasks."""
    DOCUMENT_ANALYSIS = "document_analysis"
    BATCH_PROCESSING = "batch_processing"
    KNOWLEDGE_UPDATE = "knowledge_update"
    SYSTEM_MAINTENANCE = "system_maintenance"
    DATA_EXPORT = "data_export"
    MODEL_TRAINING = "model_training"


class BackgroundTask(BaseModel):
    """
    Background task tracking and management.
    
    Tracks all async tasks including Celery jobs, scheduled tasks,
    and long-running analysis operations.
    """
    
    __tablename__ = "background_task"
    
    # Task identification
    task_id = Column(String(255), nullable=False, unique=True, index=True)  # Celery task ID
    name = Column(String(255), nullable=False, index=True)
    task_type = Column(String(50), nullable=False, index=True)
    
    # Task configuration
    function_name = Column(String(255), nullable=False)
    arguments = Column(JSON, nullable=True)  # Task arguments as JSON
    kwargs = Column(JSON, nullable=True)     # Task keyword arguments as JSON
    
    # Scheduling
    priority = Column(String(20), nullable=False, default=TaskPriority.NORMAL, index=True)
    scheduled_at = Column(DateTime(timezone=True), nullable=True)
    max_retries = Column(Integer, nullable=False, default=3)
    retry_delay = Column(Integer, nullable=True)  # seconds
    timeout = Column(Integer, nullable=True)      # seconds
    
    # Execution tracking
    status = Column(String(20), nullable=False, default=TaskStatus.PENDING, index=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Results and progress
    result = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    error_traceback = Column(Text, nullable=True)
    progress_percent = Column(Float, nullable=True)  # 0.0 to 100.0
    progress_message = Column(String(500), nullable=True)
    
    # Retry tracking
    retry_count = Column(Integer, nullable=False, default=0)
    next_retry_at = Column(DateTime(timezone=True), nullable=True)
    
    # Performance metrics
    execution_time = Column(Float, nullable=True)  # seconds
    memory_usage = Column(Float, nullable=True)    # MB
    cpu_time = Column(Float, nullable=True)        # seconds
    
    # Dependencies
    depends_on = Column(JSON, nullable=True)  # List of task IDs this task depends on
    
    # Relationships to documents/analyses
    document_id = Column(GUID(), ForeignKey("case_documents.id"), nullable=True, index=True)
    analysis_id = Column(GUID(), ForeignKey("document_analyses.id"), nullable=True, index=True)
    
    # Relationships
    document = relationship("Document")
    analysis = relationship("DocumentAnalysis")
    logs = relationship("TaskLog", back_populates="task", cascade="all, delete-orphan")
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate task duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def is_finished(self) -> bool:
        """Check if task has finished (completed, failed, or cancelled)."""
        return self.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT}
    
    @property
    def is_running(self) -> bool:
        """Check if task is currently running."""
        return self.status == TaskStatus.RUNNING
    
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return (
            self.status == TaskStatus.FAILED and
            self.retry_count < self.max_retries
        )
    
    def __repr__(self) -> str:
        return f"<BackgroundTask(name='{self.name}', status='{self.status}')>"


class TaskLog(BaseModel):
    """
    Detailed logs for background tasks.
    
    Provides granular logging for debugging and monitoring
    task execution.
    """
    
    # Foreign key to task
    task_id = Column(GUID(), ForeignKey("background_task.id"), nullable=False, index=True)
    
    # Log entry details
    level = Column(String(20), nullable=False, index=True)  # DEBUG, INFO, WARNING, ERROR
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Additional context
    context = Column(JSON, nullable=True)  # Additional structured data
    
    # Performance metrics at log time
    memory_usage = Column(Float, nullable=True)  # MB
    cpu_percent = Column(Float, nullable=True)   # CPU usage percentage
    
    # Relationships
    task = relationship("BackgroundTask", back_populates="logs")
    
    def __repr__(self) -> str:
        return f"<TaskLog(level='{self.level}', message='{self.message[:50]}...')>"


class TaskQueue(BaseModel):
    """
    Task queue management and statistics.
    
    Tracks queue performance and helps with load balancing.
    """
    
    # Queue identification
    name = Column(String(100), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    
    # Queue configuration
    max_workers = Column(Integer, nullable=False, default=1)
    max_queue_size = Column(Integer, nullable=True)
    
    # Statistics
    total_tasks = Column(Integer, nullable=False, default=0)
    completed_tasks = Column(Integer, nullable=False, default=0)
    failed_tasks = Column(Integer, nullable=False, default=0)
    
    # Current state
    active_workers = Column(Integer, nullable=False, default=0)
    pending_tasks = Column(Integer, nullable=False, default=0)
    
    # Performance metrics
    avg_execution_time = Column(Float, nullable=True)  # seconds
    avg_wait_time = Column(Float, nullable=True)       # seconds
    throughput = Column(Float, nullable=True)          # tasks per minute
    
    # Status
    is_active = Column(Boolean, nullable=False, default=True)
    last_task_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self) -> str:
        return f"<TaskQueue(name='{self.name}', active_workers={self.active_workers})>"


class ScheduledTask(BaseModel):
    """
    Periodic and scheduled tasks.
    
    Manages recurring tasks like health checks, cleanup jobs,
    and periodic analysis updates.
    """
    
    # Task identification
    name = Column(String(255), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    function_name = Column(String(255), nullable=False)
    
    # Schedule configuration
    schedule_type = Column(String(50), nullable=False)  # "cron", "interval", "once"
    schedule_expression = Column(String(255), nullable=False)  # Cron expression or interval
    timezone = Column(String(50), nullable=False, default="UTC")
    
    # Task configuration
    arguments = Column(JSON, nullable=True)
    kwargs = Column(JSON, nullable=True)
    max_retries = Column(Integer, nullable=False, default=3)
    timeout = Column(Integer, nullable=True)
    
    # Execution tracking
    last_run_at = Column(DateTime(timezone=True), nullable=True)
    next_run_at = Column(DateTime(timezone=True), nullable=True, index=True)
    last_status = Column(String(20), nullable=True)
    
    # Statistics
    total_runs = Column(Integer, nullable=False, default=0)
    successful_runs = Column(Integer, nullable=False, default=0)
    failed_runs = Column(Integer, nullable=False, default=0)
    
    # Status
    is_active = Column(Boolean, nullable=False, default=True)
    
    def __repr__(self) -> str:
        return f"<ScheduledTask(name='{self.name}', schedule='{self.schedule_expression}')>"


class TaskDependency(BaseModel):
    """
    Dependencies between tasks.
    
    Manages task execution order and dependency resolution.
    """
    
    # Dependent task (waits for prerequisite)
    task_id = Column(GUID(), ForeignKey("background_task.id"), nullable=False, index=True)
    
    # Prerequisite task (must complete first)
    prerequisite_task_id = Column(GUID(), ForeignKey("background_task.id"), nullable=False, index=True)
    
    # Dependency configuration
    dependency_type = Column(String(50), nullable=False, default="completion")  # "completion", "success"
    is_blocking = Column(Boolean, nullable=False, default=True)
    
    # Status
    is_satisfied = Column(Boolean, nullable=False, default=False)
    satisfied_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self) -> str:
        return f"<TaskDependency(type='{self.dependency_type}', satisfied={self.is_satisfied})>"
