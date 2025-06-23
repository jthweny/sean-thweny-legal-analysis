"""
Document API Routes

FastAPI router for document upload, analysis, and management endpoints.
Provides full CRUD operations with async processing capabilities.
"""

import logging
import mimetypes
import os
import tempfile
from typing import List, Optional
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_db
from src.dependencies import get_current_user
from src.models.document import Document, DocumentAnalysis
from src.schemas.document import (
    AnalysisCreate,
    AnalysisResponse,
    BatchAnalysisCreate,
    BatchAnalysisResponse,
    DocumentFilter,
    DocumentResponse,
    DocumentSearchQuery,
    DocumentSearchResponse,
    DocumentStats,
    DocumentSummary,
    FileUploadResponse,
)
from src.services.document_service import DocumentService
from src.services.analysis_service import AnalysisService
from src.tasks.document_tasks import process_document_task, analyze_document_task
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/upload",
    response_model=FileUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a new document",
    description="Upload a document file for analysis. Supports PDF, text, CSV, JSON, and HTML files.",
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file to upload"),
    tags: Optional[str] = Form(None, description="Comma-separated tags"),
    metadata: Optional[str] = Form(None, description="JSON metadata"),
    auto_analyze: bool = Form(True, description="Automatically start analysis"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
) -> FileUploadResponse:
    """
    Upload a new document for analysis.
    
    - **file**: Document file (PDF, TXT, CSV, JSON, HTML)
    - **tags**: Optional comma-separated tags
    - **metadata**: Optional JSON metadata
    - **auto_analyze**: Whether to automatically start analysis
    
    Returns the created document with upload details.
    """
    document_service = DocumentService(db)
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename is required"
            )
        
        # Check file size
        if file.size and file.size > 100 * 1024 * 1024:  # 100MB limit
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File size exceeds 100MB limit"
            )
        
        # Create document record
        document = await document_service.create_document_from_upload(
            file=file,
            tags=tags.split(",") if tags else None,
            metadata=metadata,
            user_id=current_user.get("id"),
        )
        
        # Schedule background processing
        background_tasks.add_task(
            process_document_task.delay,
            str(document.id),
            auto_analyze
        )
        
        logger.info(f"Document uploaded: {document.filename} ({document.id})")
        
        return FileUploadResponse(document=DocumentResponse.from_orm(document))
        
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@router.get(
    "/",
    response_model=DocumentSearchResponse,
    summary="List documents",
    description="Get a paginated list of documents with optional filtering and search.",
)
async def list_documents(
    query: Optional[str] = Query(None, description="Search query"),
    status_filter: Optional[List[str]] = Query(None, description="Filter by status"),
    mime_types: Optional[List[str]] = Query(None, description="Filter by MIME type"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
    limit: int = Query(20, ge=1, le=100, description="Results per page"),
    offset: int = Query(0, ge=0, description="Results offset"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
) -> DocumentSearchResponse:
    """
    Get a paginated list of documents with filtering and search capabilities.
    """
    document_service = DocumentService(db)
    
    # Build search query
    search_query = DocumentSearchQuery(
        query=query,
        filters=DocumentFilter(
            status=status_filter,
            mime_types=mime_types,
            tags=tags,
        ),
        sort_by=sort_by,
        sort_order=sort_order,
        limit=limit,
        offset=offset,
    )
    
    # Execute search
    documents, total = await document_service.search_documents(
        search_query,
        user_id=current_user.get("id")
    )
    
    return DocumentSearchResponse(
        documents=[DocumentSummary.from_orm(doc) for doc in documents],
        total=total,
        limit=limit,
        offset=offset,
        has_more=offset + limit < total,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document details",
    description="Get detailed information about a specific document.",
)
async def get_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
) -> DocumentResponse:
    """Get detailed information about a specific document."""
    document_service = DocumentService(db)
    
    document = await document_service.get_document(
        document_id,
        user_id=current_user.get("id")
    )
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    return DocumentResponse.from_orm(document)


@router.get(
    "/{document_id}/content",
    summary="Get document content",
    description="Get the extracted text content of a document.",
)
async def get_document_content(
    document_id: UUID,
    format: str = Query("text", description="Content format (text/json)"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Get the extracted text content of a document."""
    document_service = DocumentService(db)
    
    document = await document_service.get_document(
        document_id,
        user_id=current_user.get("id")
    )
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    if not document.clean_text:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document content not available"
        )
    
    if format == "json":
        return {
            "id": document.id,
            "filename": document.filename,
            "content": document.clean_text,
            "word_count": document.word_count,
        }
    else:
        return StreamingResponse(
            iter([document.clean_text]),
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename={document.filename}.txt"}
        )


@router.post(
    "/{document_id}/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start document analysis",
    description="Start a specific type of analysis on a document.",
)
async def analyze_document(
    document_id: UUID,
    analysis_request: AnalysisCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
) -> AnalysisResponse:
    """Start a specific type of analysis on a document."""
    analysis_service = AnalysisService(db)
    
    # Verify document exists
    document_service = DocumentService(db)
    document = await document_service.get_document(
        document_id,
        user_id=current_user.get("id")
    )
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Create analysis record
    analysis = await analysis_service.create_analysis(
        document_id=document_id,
        analysis_type=analysis_request.analysis_type,
        model_name=analysis_request.model_name,
        model_version=analysis_request.model_version,
        user_id=current_user.get("id"),
    )
    
    # Schedule background analysis
    background_tasks.add_task(
        analyze_document_task.delay,
        str(analysis.id)
    )
    
    logger.info(f"Analysis started: {analysis.analysis_type} for document {document_id}")
    
    return AnalysisResponse.from_orm(analysis)


@router.post(
    "/batch-analyze",
    response_model=BatchAnalysisResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start batch analysis",
    description="Start analysis on multiple documents.",
)
async def batch_analyze_documents(
    batch_request: BatchAnalysisCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
) -> BatchAnalysisResponse:
    """Start analysis on multiple documents."""
    analysis_service = AnalysisService(db)
    
    # Create analyses for all documents
    results = await analysis_service.create_batch_analysis(
        document_ids=batch_request.document_ids,
        analysis_type=batch_request.analysis_type,
        model_name=batch_request.model_name,
        model_version=batch_request.model_version,
        user_id=current_user.get("id"),
    )
    
    # Schedule background processing for successful analyses
    for analysis_id in results["created_analyses"]:
        background_tasks.add_task(
            analyze_document_task.delay,
            str(analysis_id)
        )
    
    logger.info(f"Batch analysis started: {len(results['created_analyses'])} documents")
    
    return BatchAnalysisResponse(**results)


@router.get(
    "/{document_id}/analyses",
    response_model=List[AnalysisResponse],
    summary="Get document analyses",
    description="Get all analyses for a specific document.",
)
async def get_document_analyses(
    document_id: UUID,
    analysis_type: Optional[str] = Query(None, description="Filter by analysis type"),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
) -> List[AnalysisResponse]:
    """Get all analyses for a specific document."""
    analysis_service = AnalysisService(db)
    
    analyses = await analysis_service.get_document_analyses(
        document_id=document_id,
        analysis_type=analysis_type,
        status=status_filter,
        user_id=current_user.get("id"),
    )
    
    return [AnalysisResponse.from_orm(analysis) for analysis in analyses]


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete document",
    description="Delete a document and all associated analyses.",
)
async def delete_document(
    document_id: UUID,
    permanent: bool = Query(False, description="Permanent deletion (vs soft delete)"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Delete a document and all associated analyses."""
    document_service = DocumentService(db)
    
    success = await document_service.delete_document(
        document_id=document_id,
        user_id=current_user.get("id"),
        permanent=permanent,
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    logger.info(f"Document deleted: {document_id}")


@router.get(
    "/stats/overview",
    response_model=DocumentStats,
    summary="Get document statistics",
    description="Get overview statistics for documents.",
)
async def get_document_stats(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
) -> DocumentStats:
    """Get overview statistics for documents."""
    document_service = DocumentService(db)
    
    stats = await document_service.get_document_stats(
        user_id=current_user.get("id")
    )
    
    return DocumentStats(**stats)
