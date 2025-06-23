"""
Celery Configuration and Task Application

Configures Celery for distributed task processing with Redis broker.
Implements best practices for async task management and monitoring.
"""

import logging
from typing import Any, Dict

from celery import Celery
from celery.signals import after_setup_logger, task_failure, task_success
from kombu import Queue

from src.config import get_celery_settings

logger = logging.getLogger(__name__)

# Get Celery settings
celery_settings = get_celery_settings()

# Create Celery app
celery_app = Celery("ai_analysis_system")

# Configure Celery
celery_app.conf.update(
    # Broker and Result Backend
    broker_url=celery_settings.broker_url,
    result_backend=celery_settings.result_backend,
    
    # Serialization
    task_serializer=celery_settings.task_serializer,
    result_serializer=celery_settings.result_serializer,
    accept_content=celery_settings.accept_content,
    
    # Timezone
    timezone=celery_settings.timezone,
    enable_utc=True,
    
    # Task execution
    task_track_started=celery_settings.task_track_started,
    task_time_limit=celery_settings.task_time_limit,
    task_soft_time_limit=celery_settings.task_soft_time_limit,
    task_acks_late=celery_settings.task_acks_late,
    
    # Worker configuration
    worker_prefetch_multiplier=celery_settings.worker_prefetch_multiplier,
    worker_disable_rate_limits=celery_settings.worker_disable_rate_limits,
    worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',
    worker_task_log_format='[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s',
    
    # Result backend settings
    result_expires=celery_settings.result_expires,
    result_persistent=celery_settings.result_persistent,
    
    # Task routing and queues
    task_default_queue='default',
    task_default_exchange='default',
    task_default_exchange_type='direct',
    task_default_routing_key='default',
    
    # Define task queues
    task_routes={
        'src.tasks.document_tasks.*': {'queue': 'document_processing'},
        'src.tasks.analysis_tasks.*': {'queue': 'analysis'},
        'src.tasks.maintenance_tasks.*': {'queue': 'maintenance'},
    },
    
    # Queue definitions
    task_queues=[
        Queue('default', routing_key='default'),
        Queue('document_processing', routing_key='document_processing'),
        Queue('analysis', routing_key='analysis'),
        Queue('maintenance', routing_key='maintenance'),
    ],
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Error handling
    task_reject_on_worker_lost=True,
    task_ignore_result=False,
    
    # Beat scheduler (for periodic tasks)
    beat_schedule={
        'cleanup-expired-tasks': {
            'task': 'src.tasks.maintenance_tasks.cleanup_expired_tasks',
            'schedule': 3600.0,  # Every hour
        },
        'update-knowledge-graph': {
            'task': 'src.tasks.analysis_tasks.update_knowledge_graph',
            'schedule': 1800.0,  # Every 30 minutes
        },
        'health-check': {
            'task': 'src.tasks.maintenance_tasks.health_check',
            'schedule': 300.0,  # Every 5 minutes
        },
    },
)

# Import task modules to register them
from src.tasks import analysis_tasks, document_tasks, maintenance_tasks

# Task discovery
celery_app.autodiscover_tasks([
    'src.tasks.document_tasks',
    'src.tasks.analysis_tasks',
    'src.tasks.maintenance_tasks',
])


@after_setup_logger.connect
def setup_loggers(logger, *args, **kwargs):
    """Setup custom logging for Celery tasks."""
    formatter = logging.Formatter(
        '[%(asctime)s: %(levelname)s/%(processName)s] %(message)s'
    )
    
    # Add file handler for task logs
    file_handler = logging.FileHandler('/var/log/celery_tasks.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


@task_success.connect
def log_task_success(sender=None, task_id=None, result=None, retval=None, **kwargs):
    """Log successful task completion."""
    logger.info(f"Task {sender.name}[{task_id}] completed successfully")


@task_failure.connect
def log_task_failure(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwargs):
    """Log task failure."""
    logger.error(f"Task {sender.name}[{task_id}] failed: {exception}")


class CeleryTask:
    """Base class for Celery tasks with common functionality."""
    
    @staticmethod
    def update_task_progress(task_id: str, current: int, total: int, message: str = ""):
        """Update task progress in the result backend."""
        from src.tasks.celery_app import celery_app
        
        celery_app.backend.store_result(
            task_id,
            {
                'current': current,
                'total': total,
                'message': message,
                'progress': int((current / total) * 100) if total > 0 else 0,
            },
            'PROGRESS'
        )
    
    @staticmethod
    def get_task_progress(task_id: str) -> Dict[str, Any]:
        """Get task progress from the result backend."""
        from src.tasks.celery_app import celery_app
        
        result = celery_app.AsyncResult(task_id)
        if result.state == 'PROGRESS':
            return result.result
        return {'current': 0, 'total': 1, 'progress': 0, 'message': result.state}


# Utility functions for task management
def get_active_tasks() -> Dict[str, Any]:
    """Get list of active tasks across all workers."""
    inspect = celery_app.control.inspect()
    active_tasks = inspect.active()
    return active_tasks or {}


def get_scheduled_tasks() -> Dict[str, Any]:
    """Get list of scheduled tasks."""
    inspect = celery_app.control.inspect()
    scheduled_tasks = inspect.scheduled()
    return scheduled_tasks or {}


def get_worker_stats() -> Dict[str, Any]:
    """Get worker statistics."""
    inspect = celery_app.control.inspect()
    stats = inspect.stats()
    return stats or {}


def cancel_task(task_id: str) -> bool:
    """Cancel a running task."""
    try:
        celery_app.control.revoke(task_id, terminate=True)
        return True
    except Exception as e:
        logger.error(f"Failed to cancel task {task_id}: {e}")
        return False


def purge_queue(queue_name: str) -> int:
    """Purge all tasks from a specific queue."""
    try:
        return celery_app.control.purge()
    except Exception as e:
        logger.error(f"Failed to purge queue {queue_name}: {e}")
        return 0


# Export the Celery app
__all__ = ['celery_app', 'CeleryTask']
