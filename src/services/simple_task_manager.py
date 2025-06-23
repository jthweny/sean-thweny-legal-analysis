"""
Simple Task Manager
Provides basic background task support without Celery dependency
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Simple fallback task manager
CELERY_AVAILABLE = False

class SimpleTaskManager:
    """Simple task manager without Celery dependency"""
    
    def __init__(self):
        self.tasks = {}
        
    def add_task(self, func, *args, **kwargs):
        """Add a task to run in background"""
        task_id = f"task_{datetime.now().timestamp()}"
        self.tasks[task_id] = {
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "status": "pending",
            "created_at": datetime.now()
        }
        
        # Run the task in background
        asyncio.create_task(self._run_task(task_id))
        return task_id
    
    async def _run_task(self, task_id: str):
        """Run a task"""
        try:
            task = self.tasks[task_id]
            task["status"] = "running"
            
            # Run the function
            if asyncio.iscoroutinefunction(task["func"]):
                await task["func"](*task["args"], **task["kwargs"])
            else:
                task["func"](*task["args"], **task["kwargs"])
                
            task["status"] = "completed"
            task["completed_at"] = datetime.now()
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            self.tasks[task_id]["status"] = "failed"
            self.tasks[task_id]["error"] = str(e)
    
    def get_task_status(self, task_id: str):
        """Get task status"""
        return self.tasks.get(task_id, {"status": "not_found"})

# Create global task manager instance
task_manager = SimpleTaskManager()

# Document processor instance
from src.services.mcp_document_processor import MCPDocumentProcessor
document_processor = MCPDocumentProcessor()
