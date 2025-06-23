"""
Celery Task Manager
Handles background processing tasks for document analysis and knowledge extraction
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

try:
    from celery import Celery
    from celery.result import AsyncResult
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    # Fallback for when Celery is not installed
    class Celery:
        def __init__(self, *args, **kwargs):
            pass
    
    class AsyncResult:
        def __init__(self, *args, **kwargs):
            pass

from src.config import get_settings
from src.services.mcp_document_processor import MCPDocumentProcessor

logger = logging.getLogger(__name__)

settings = get_settings()

# Initialize Celery (with fallback)
if CELERY_AVAILABLE:
    from src.config import settings
    
    celery_app = Celery(
        'legal_analysis_system',
        broker=settings.REDIS_URL,
        backend=settings.REDIS_URL,
        include=['src.services.task_manager']
    )
    
    # Celery configuration
    celery_app.conf.update(
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        task_track_started=True,
        task_time_limit=30 * 60,  # 30 minutes
        task_soft_time_limit=25 * 60,  # 25 minutes
        worker_prefetch_multiplier=1,
        worker_max_tasks_per_child=1000,
    )
else:
    celery_app = None
    logger.warning("Celery not available, using fallback task manager")

# Document processor instance
from src.services.mcp_document_processor import MCPDocumentProcessor
document_processor = MCPDocumentProcessor()

# Only define Celery tasks if Celery is available
if CELERY_AVAILABLE and celery_app:
    @celery_app.task(bind=True, name='analyze_document')
    def analyze_document_task(self, document_id: str, content: str, relevancy_score: int) -> Dict[str, Any]:
        """Celery task for document analysis"""
        try:
            logger.info(f"Starting background analysis for document {document_id}")
            
            # Update task status
            self.update_state(
                state='PROGRESS',
                meta={'stage': 'initializing', 'progress': 0}
            )
            
            # Run async document analysis
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
            result = loop.run_until_complete(
                document_processor.analyze_document(document_id, content, relevancy_score)
            )
            
            return {
                'status': 'completed',
                'document_id': document_id,
                'analysis_results': result,
                'completed_at': datetime.utcnow().isoformat()
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Document analysis task failed for {document_id}: {e}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'document_id': document_id}
        )
        raise e

@celery_app.task(bind=True, name='load_baseline_document')
def load_baseline_document_task(self, file_path: str) -> Dict[str, Any]:
    """Celery task for loading baseline document"""
    try:
        from pathlib import Path
        
        logger.info(f"Loading baseline document: {file_path}")
        
        self.update_state(
            state='PROGRESS',
            meta={'stage': 'loading_baseline', 'progress': 0}
        )
        
        # Run async baseline loading
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            success = loop.run_until_complete(
                document_processor.load_baseline_document(Path(file_path))
            )
            
            return {
                'status': 'completed' if success else 'failed',
                'file_path': file_path,
                'loaded': success,
                'completed_at': datetime.utcnow().isoformat()
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Baseline loading task failed for {file_path}: {e}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'file_path': file_path}
        )
        raise e

@celery_app.task(bind=True, name='batch_analyze_documents')
def batch_analyze_documents_task(self, document_ids: list) -> Dict[str, Any]:
    """Celery task for batch document analysis"""
    try:
        logger.info(f"Starting batch analysis for {len(document_ids)} documents")
        
        results = {}
        total_docs = len(document_ids)
        
        for i, doc_id in enumerate(document_ids):
            try:
                # Update progress
                progress = int((i / total_docs) * 100)
                self.update_state(
                    state='PROGRESS',
                    meta={'stage': 'processing', 'progress': progress, 'current_doc': doc_id}
                )
                
                # Trigger individual document analysis
                task = analyze_document_task.delay(doc_id, "", 50)  # Placeholder values
                results[doc_id] = task.id
                
            except Exception as e:
                logger.error(f"Failed to process document {doc_id}: {e}")
                results[doc_id] = {'error': str(e)}
        
        return {
            'status': 'completed',
            'processed_documents': len(document_ids),
            'results': results,
            'completed_at': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch analysis task failed: {e}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(e)}
        )
        raise e

@celery_app.task(bind=True, name='knowledge_graph_update')
def knowledge_graph_update_task(self) -> Dict[str, Any]:
    """Celery task for updating knowledge graph relationships"""
    try:
        logger.info("Starting knowledge graph update")
        
        self.update_state(
            state='PROGRESS',
            meta={'stage': 'updating_relationships', 'progress': 0}
        )
        
        # Run async knowledge graph update
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Placeholder for knowledge graph logic
            result = loop.run_until_complete(
                _update_knowledge_graph()
            )
            
            return {
                'status': 'completed',
                'updated_relationships': result.get('relationships', 0),
                'updated_entities': result.get('entities', 0),
                'completed_at': datetime.utcnow().isoformat()
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Knowledge graph update task failed: {e}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(e)}
        )
        raise e

async def _update_knowledge_graph() -> Dict[str, Any]:
    """Update knowledge graph relationships"""
    # Placeholder implementation
    return {'relationships': 0, 'entities': 0}

class TaskManager:
    """Manager for Celery background tasks"""
    
    def __init__(self):
        self.celery_app = celery_app
    
    def submit_document_analysis(self, document_id: str, content: str, relevancy_score: int) -> str:
        """Submit document for background analysis"""
        task = analyze_document_task.delay(document_id, content, relevancy_score)
        logger.info(f"Submitted document {document_id} for analysis, task ID: {task.id}")
        return task.id
    
    def submit_baseline_loading(self, file_path: str) -> str:
        """Submit baseline document for loading"""
        task = load_baseline_document_task.delay(file_path)
        logger.info(f"Submitted baseline document {file_path} for loading, task ID: {task.id}")
        return task.id
    
    def submit_batch_analysis(self, document_ids: list) -> str:
        """Submit multiple documents for batch analysis"""
        task = batch_analyze_documents_task.delay(document_ids)
        logger.info(f"Submitted {len(document_ids)} documents for batch analysis, task ID: {task.id}")
        return task.id
    
    def submit_knowledge_graph_update(self) -> str:
        """Submit knowledge graph for update"""
        task = knowledge_graph_update_task.delay()
        logger.info(f"Submitted knowledge graph for update, task ID: {task.id}")
        return task.id
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a background task"""
        try:
            result = AsyncResult(task_id, app=self.celery_app)
            
            status_info = {
                'task_id': task_id,
                'status': result.status,
                'current': result.info.get('progress', 0) if isinstance(result.info, dict) else None,
                'total': 100,
                'stage': result.info.get('stage', 'unknown') if isinstance(result.info, dict) else None
            }
            
            if result.successful():
                status_info['result'] = result.result
            elif result.failed():
                status_info['error'] = str(result.info) if result.info else 'Unknown error'
            
            return status_info
            
        except Exception as e:
            logger.error(f"Failed to get task status for {task_id}: {e}")
            return {
                'task_id': task_id,
                'status': 'ERROR',
                'error': str(e)
            }
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a background task"""
        try:
            result = AsyncResult(task_id, app=self.celery_app)
            result.revoke(terminate=True)
            logger.info(f"Cancelled task {task_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get list of active tasks"""
        try:
            inspect = self.celery_app.control.inspect()
            active_tasks = inspect.active()
            
            if not active_tasks:
                return []
            
            all_tasks = []
            for worker, tasks in active_tasks.items():
                for task in tasks:
                    all_tasks.append({
                        'worker': worker,
                        'task_id': task['id'],
                        'name': task['name'],
                        'args': task['args'],
                        'kwargs': task['kwargs']
                    })
            
            return all_tasks
            
        except Exception as e:
            logger.error(f"Failed to get active tasks: {e}")
            return []
    
    def get_task_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent task history"""
        # This would require additional setup with a result backend
        # For now, return empty list
        return []

# Global task manager instance
task_manager = TaskManager()
